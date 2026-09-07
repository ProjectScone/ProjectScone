"""HTTP surface, shaped like the Rust ``scone serve`` so one client
speaks to both.

Every request carries a bearer key, and the key decides the space: there
is no space parameter to get wrong, and a key can never read outside the
space it was issued for. Errors are ``{"error": "..."}`` with the status
the engine's error type maps to.
"""

from __future__ import annotations

import re

from pathlib import Path
from typing import Mapping, Optional

from fastapi import Depends, FastAPI, Query, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import HTMLResponse, JSONResponse, Response
from pydantic import BaseModel, ConfigDict, Field

from contextlib import asynccontextmanager

from ..observability import metrics
from ..memory.engine import MemoryEngine
from ..core.errors import Conflict, InvalidInput, NotFound
from ..core.models import Attachment, Fact, RecallItem


#: Types a browser may render in place. Everything else is handed back as
#: a download: an SVG or an HTML file served inline is script running in
#: this origin, and the bytes are kept as evidence either way.
INLINE_TYPES = ("image/png", "image/jpeg", "image/gif", "image/webp", "application/pdf")
_DIGEST = re.compile(r"[0-9a-f]{64}")
_SAFE_FILENAME = re.compile(r"[^A-Za-z0-9._-]")


class DecideBody(BaseModel):
    """One decision applied to a list of facts a person has reviewed."""

    model_config = ConfigDict(extra="forbid")

    decision: str
    ids: list[int]
    #: Required by decline and exclude, which keep the reason on the fact.
    reason: Optional[str] = None
    #: The revision the reviewed list was read at, when the caller froze one.
    expect_revision: Optional[int] = None


class EpisodeBody(BaseModel):
    """Unknown fields are refused rather than ignored: a client that
    sends a field we silently drop believes it stored something it
    did not."""

    model_config = ConfigDict(extra="forbid")

    content: str
    tags: list[str] = Field(default_factory=list)
    #: Attachments stored earlier by POST /v1/attachments.
    attachment_ids: list[str] = Field(default_factory=list)
    source: Optional[str] = None
    created_at: Optional[str] = None
    kind: str = "note"
    metadata: dict[str, str] = Field(default_factory=dict)


class SourceQuery(BaseModel):
    model_config = ConfigDict(extra="forbid")
    before: Optional[int] = Field(default=None, ge=1, le=2**63-1)
    limit: int = Field(default=25, ge=1, le=100)
    kind: Optional[str] = None


class FactBody(BaseModel):
    model_config = ConfigDict(extra="forbid")

    subject: str
    predicate: str
    object: str
    valid_from: Optional[str] = None
    confidence: float = 1.0
    source_episode_id: Optional[int] = None
    origin: str = "stated"
    proposed: bool = False
    quote: Optional[str] = None
    # A fact this one adds detail to, and the ledger facts it was inferred from.
    extends: Optional[int] = None
    derived_from: list[int] = Field(default_factory=list)


class LinkBody(BaseModel):
    model_config = ConfigDict(extra="forbid")

    to_fact: int
    kind: str
    source_episode_id: Optional[int] = None
    quote: Optional[str] = None


class CloseBody(BaseModel):
    model_config = ConfigDict(extra="forbid")

    reason: str


class ExternalEventBody(BaseModel):
    model_config = ConfigDict(extra="forbid")

    kind: str
    payload: dict


class FeedbackBody(BaseModel):
    model_config = ConfigDict(extra="forbid")

    recall_event_id: int
    chunk_id: int
    useful: bool
    note: Optional[str] = None


CONSOLE = Path(__file__).with_name("console.html")
PLAYGROUND = Path(__file__).with_name("playground.html")
MARK = Path(__file__).with_name("scone-mark.png")


def mark_data_uri() -> str:
    """The Scone mark, embedded so the page loads nothing external. The
    packaged playground carries its own copy; the console shares the file."""
    import base64

    if not MARK.exists():
        return ""
    return "data:image/png;base64," + base64.b64encode(MARK.read_bytes()).decode()


def create_app(
    engine: MemoryEngine,
    keys: Mapping[str, str],
    console: bool = True,
    console_key: Optional[str] = None,
    worker=None,
    reload_pages: bool = False,
    conversations: bool = False,
) -> FastAPI:
    """``console_key`` is baked into the page served at ``/`` so the key
    stays out of the URL and out of anything the user might paste; with
    no baked key the page asks for one and keeps it in the tab.
    ``worker`` is a ConsolidationWorker started with the app and stopped
    with it; None means no distiller runs on this server. ``reload_pages``
    re-reads the console and playground files on every request, for
    editing them with the server running; off in normal use. ``conversations``
    says this app is mounted under a conversation service on the same
    origin, so the capability manifest may advertise it."""

    @asynccontextmanager
    async def lifespan(_app: FastAPI):
        if worker is not None:
            worker.start()
        yield
        if worker is not None:
            await worker.stop()

    app = FastAPI(title="scone-memory", version="0.1.0", docs_url=None, redoc_url=None, lifespan=lifespan)
    app.state.engine = engine
    app.state.keys = dict(keys)
    app.state.worker = worker

    def space_for(request: Request) -> str:
        header = request.headers.get("authorization", "")
        scheme, _, token = header.partition(" ")
        if scheme.lower() != "bearer" or not token.strip():
            raise Unauthorized("missing bearer key")
        space = app.state.keys.get(token.strip())
        if space is None:
            raise Unauthorized("unknown key")
        return space

    def actor_for(request: Request) -> str:
        """Who judged: a fingerprint of the bearer key (never the key) plus an
        optional label the caller sends in X-Scone-Actor, so a review event
        can say "the console" or "playwright smoke" and can always be tied
        to the key that made it."""
        import hashlib

        token = request.headers.get("authorization", "").partition(" ")[2].strip()
        fingerprint = hashlib.sha256(token.encode()).hexdigest()[:12] if token else "anonymous"
        label = request.headers.get("x-scone-actor", "").strip()[:64]
        return f"key:{fingerprint}" + (f" {label}" if label else "")

    @app.exception_handler(Unauthorized)
    async def _unauthorized(_: Request, e: Unauthorized) -> JSONResponse:
        return JSONResponse({"error": str(e)}, status_code=401)

    @app.exception_handler(InvalidInput)
    async def _invalid(_: Request, e: InvalidInput) -> JSONResponse:
        return JSONResponse({"error": str(e)}, status_code=422)

    @app.exception_handler(Conflict)
    async def _moved(_: Request, e: Conflict) -> JSONResponse:
        return JSONResponse({"error": str(e), "revision": e.revision}, status_code=409)

    @app.exception_handler(NotFound)
    async def _missing(_: Request, e: NotFound) -> JSONResponse:
        return JSONResponse({"error": str(e)}, status_code=404)

    @app.exception_handler(RequestValidationError)
    async def _malformed(_: Request, e: RequestValidationError) -> JSONResponse:
        first = e.errors()[0] if e.errors() else {}
        where = ".".join(str(p) for p in first.get("loc", ()) if p != "body")
        return JSONResponse({"error": f"{where}: {first.get('msg', 'invalid request')}"}, status_code=422)

    @app.get("/healthz")
    async def healthz() -> dict:
        return {"ok": True}

    @app.get("/v1/capabilities")
    async def capabilities(_space: str = Depends(space_for)) -> dict:
        """Implemented HTTP operations, not a health check or a ledger read."""
        features = {
            "recall": True, "facts.read": True, "facts.review": True,
            "facts.close": True, "facts.exclude": True, "facts.include": True, "facts.links": True,
            "events.read": True, "metrics.read": True, "scopes.read": True,
            "status.read": True, "episodes.attachments": True,
            "episodes.list": callable(getattr(engine.documents, "page_episodes", None)),
        }
        if conversations:
            # Present only when the service is mounted here; its own manifest
            # at /v1/conversations/capabilities says what it can do.
            features["conversations"] = True
        return {"schema_version": 1, "implementation": "python", "features": features}

    if console:
        def render_console() -> str:
            # Two page generations share this route: the packaged React app,
            # which carries a __SCONE_TOKEN__ placeholder like the playground,
            # and the earlier inline page, which took the key on its first
            # <script> as data-token. Both get the key; neither logs it.
            page = CONSOLE.read_text(encoding="utf-8").replace("__SCONE_MARK__", mark_data_uri())
            if console_key:
                if "__SCONE_TOKEN__" in page:
                    page = page.replace("__SCONE_TOKEN__", console_key)
                else:
                    page = page.replace('data-token=""', "", 1).replace("<script>", f'<script data-token="{console_key}">', 1)
            return page

        def render_playground() -> str:
            # Shared asset owned in ui/playground.html and copied here by
            # scripts/sync-playground.cjs; same key placeholder as the console.
            page = PLAYGROUND.read_text(encoding="utf-8")
            return page.replace("__SCONE_TOKEN__", console_key) if console_key else page

        console_html = None if reload_pages else render_console()
        playground_html = None if (reload_pages or not PLAYGROUND.exists()) else render_playground()

        def revision(path: Path) -> str:
            # File identity only (mtime and size): says "changed", carries no content or key.
            st = path.stat()
            return f'"{st.st_mtime_ns:x}-{st.st_size:x}"'

        def page_response(body: str, path: Path) -> Response:
            headers = {"ETag": revision(path)}
            if reload_pages:
                headers["Cache-Control"] = "no-store"
            return HTMLResponse(body, headers=headers)

        # /memory is the canonical address of the memory console (the
        # playground lives at /playground); / stays as a compatible alias.
        @app.api_route("/memory", methods=["GET", "HEAD"])
        @app.api_route("/", methods=["GET", "HEAD"])
        async def console_page() -> Response:
            return page_response(console_html if console_html is not None else render_console(), CONSOLE)

        # The packaged workspace also owns the conversation addresses, so a
        # refresh or a shared link lands on the page even on a host with no
        # conversation service mounted; the page reads
        # /v1/conversations/capabilities (a JSON 404 here) and says so itself.
        for path in ("/conversations", "/conversations/{sid}"):
            app.add_api_route(path, console_page, methods=["GET", "HEAD"], include_in_schema=False)

        if PLAYGROUND.exists():
            # GET and HEAD: the console probes with HEAD to decide whether to
            # show its Playground link; in development the page polls HEAD
            # and reloads when the ETag changes.
            @app.api_route("/playground", methods=["GET", "HEAD"])
            async def playground_page() -> Response:
                return page_response(playground_html if playground_html is not None else render_playground(), PLAYGROUND)

    @app.post("/v1/attachments")
    async def post_attachment(request: Request, space: str = Depends(space_for)) -> dict:
        """Bytes an episode will carry. The id is the SHA-256 of the body,
        so storing the same screenshot twice stores it once."""
        stored = await engine.attach(
            space,
            await read_bounded(request, engine.max_attachment_bytes),
            media_type=(request.headers.get("content-type") or "").split(";")[0].strip(),
            filename=request.headers.get("x-filename"),
        )
        return stored.model_dump()

    @app.get("/v1/attachments/{attachment_id}")
    async def get_attachment(request: Request, attachment_id: str, space: str = Depends(space_for)) -> Response:
        # The id is a digest, never a path. Anything else names nothing.
        if not _DIGEST.fullmatch(attachment_id):
            raise NotFound(f"attachment {attachment_id!r} not found in {space!r}")
        # The lookup happens before any 304, so losing access to an
        # attachment takes effect on the next request rather than
        # whenever a browser decides its copy has expired.
        stored, data = await engine.attachment(space, attachment_id)
        tag = f'"{attachment_id}"'
        headers = {
            "etag": tag,
            # The bytes never change; who may read them does. So the
            # client revalidates every time and pays a 304 for it.
            "cache-control": "private, no-cache",
            "x-content-type-options": "nosniff",
        }
        if request.headers.get("if-none-match") == tag:
            return Response(status_code=304, headers=headers)
        inline = stored.media_type in INLINE_TYPES
        name = _SAFE_FILENAME.sub("_", stored.filename or attachment_id)[:120]
        return Response(
            content=data,
            media_type=stored.media_type if inline else "application/octet-stream",
            headers={
                **headers,
                "content-disposition": f'{"inline" if inline else "attachment"}; filename="{name}"',
            },
        )

    @app.post("/v1/episodes")
    async def post_episode(body: EpisodeBody, space: str = Depends(space_for)) -> dict:
        added = await engine.remember(
            space,
            body.content,
            kind=body.kind,  # type: ignore[arg-type]
            source=body.source,
            tags=body.tags,
            created_at=body.created_at,
            metadata=body.metadata,
            attachment_ids=body.attachment_ids,
        )
        return added.model_dump()

    @app.get("/v1/episodes")
    async def get_episodes(ids: str, space: str = Depends(space_for)) -> dict:
        """Several episodes in one request, so a page showing the source of
        each row does not fan out one request per row. Episodes come back in
        the order asked for; ids that are confirmed gone are listed under
        ``missing``, which keeps a missing source distinguishable from a
        request that failed."""
        found, missing = [], []
        for episode_id in parse_ids(ids):
            try:
                found.append(episode_json(await engine.episode(space, episode_id)))
            except NotFound:
                missing.append(episode_id)
        return {"episodes": found, "missing": missing}

    @app.get("/v1/sources")
    async def get_sources(query: SourceQuery = Query(), space: str = Depends(space_for)):
        if not callable(getattr(engine.documents, "page_episodes", None)):
            return JSONResponse({"error": "this document store does not implement source inventory"}, status_code=501)
        page = await engine.source_page(space, before=query.before, limit=query.limit, kind=query.kind)
        return {"items": [{"episode_id": e.episode_id, "kind": e.kind, "source": e.source,
                           "created_at": e.created_at, "byte_count": len(e.content.encode("utf-8")),
                           "preview": e.content[:500], "preview_truncated": len(e.content) > 500}
                          for e in page.episodes], "has_more": page.has_more, "next_before": page.next_before}

    @app.get("/v1/episodes/{episode_id}")
    async def get_episode(episode_id: int, space: str = Depends(space_for)) -> dict:
        return episode_json(await engine.episode(space, episode_id))

    @app.delete("/v1/episodes/{episode_id}")
    async def delete_episode(episode_id: int, space: str = Depends(space_for)) -> dict:
        await engine.forget(space, episode_id)
        return {"forgotten": episode_id}

    @app.get("/v1/recall")
    async def get_recall(
        q: str,
        limit: int = 5,
        as_of: Optional[str] = None,
        tags: Optional[str] = None,
        where: Optional[str] = None,
        history: bool = False,
        kind: Optional[str] = None,
        source_prefix: Optional[str] = None,
        since: Optional[str] = None,
        until: Optional[str] = None,
        space: str = Depends(space_for),
    ) -> dict:
        tag_list = [t for t in (tags or "").split(",") if t.strip()]
        result = await engine.recall(
            space, q, limit=limit, as_of=as_of, tags=tag_list, where=parse_where(where), history=history,
            kind=kind, source_prefix=source_prefix, since=since, until=until,
        )
        return {
            "event_id": result.event_id,
            "items": [item_json(i) for i in result.items],
            "facts": [fact_json(f) for f in result.facts],
            "history": [fact_json(f) for f in result.history],
            "top_similarity": result.top_similarity,
            "low_confidence": result.low_confidence,
            "degraded": result.degraded,
            "returned_bytes": result.returned_bytes,
            "space_bytes": result.space_bytes,
            "context_reduction": round(result.context_reduction, 6),
        }

    @app.get("/v1/facts")
    async def get_facts(
        all: bool = False,
        as_of: Optional[str] = None,
        status: Optional[str] = None,
        excluded: bool = False,
        space: str = Depends(space_for),
    ) -> dict:
        facts = await engine.facts(space, include_closed=all, as_of=as_of, status=status, include_excluded=excluded)
        return {"facts": [fact_json(f) for f in facts], "revision": await engine.revision(space)}

    @app.post("/v1/facts")
    async def post_fact(body: FactBody, space: str = Depends(space_for)) -> dict:
        fact = await engine.assert_fact(
            space,
            body.subject,
            body.predicate,
            body.object,
            valid_from=body.valid_from,
            confidence=body.confidence,
            source_episode_id=body.source_episode_id,
            origin=body.origin,
            proposed=body.proposed,
            quote=body.quote,
            extends=body.extends,
            derived_from=body.derived_from,
        )
        return fact_json(fact)

    @app.get("/v1/facts/{fact_id}")
    async def get_fact(fact_id: int, space: str = Depends(space_for)) -> dict:
        """One fact with the relations it takes part in and the ids of the
        episodes it rests on. Ids, not the episodes themselves: a source
        may have been forgotten since, and GET /v1/episodes says so."""
        fact = await engine.fact(space, fact_id)
        return {
            "fact": fact_json(fact),
            "links": [link_json(link) for link in await engine.fact_links(space, fact_id)],
            "sources": [fact.source_episode_id] if fact.source_episode_id is not None else [],
        }

    @app.post("/v1/facts/{fact_id}/links")
    async def post_link(fact_id: int, body: LinkBody, space: str = Depends(space_for)) -> dict:
        link = await engine.link_facts(space, fact_id, body.to_fact, body.kind,
                                       source_episode_id=body.source_episode_id, quote=body.quote)
        return link_json(link)

    @app.get("/v1/facts/audit")
    async def get_facts_audit(status: str = "active", flagged: bool = False,
                              space: str = Depends(space_for)) -> dict:
        """Every extracted fact judged against the text it came from. Read
        only: a flagged claim is one whose own source cannot support it,
        which is a question for a person, and the repair is exclude with a
        reason through the decision routes."""
        from collections import Counter
        from dataclasses import asdict

        from ..observability.audit import audit_grounding

        findings = await audit_grounding(engine, space, statuses=(status,))
        shown = [f for f in findings if f.flagged] if flagged else findings
        return {
            "findings": [asdict(f) for f in shown],
            # Always the whole picture for these statuses, so a filtered
            # page can still say how much of the ledger it is showing.
            "counts": dict(Counter(f.verdict for f in findings).most_common()),
            "revision": await engine.revision(space),
        }

    @app.post("/v1/facts/decide")
    async def post_decide(body: DecideBody, space: str = Depends(space_for), actor: str = Depends(actor_for)) -> dict:
        """One reviewed batch, settled together: applied oldest first, and
        refused whole if the space moved since the batch was built."""
        decided = await engine.decide(
            space, body.decision, body.ids,
            reason=body.reason, actor=actor, expect_revision=body.expect_revision,
        )
        return decided.model_dump()

    @app.post("/v1/facts/{fact_id}/approve")
    async def post_fact_approve(fact_id: int, space: str = Depends(space_for), actor: str = Depends(actor_for)) -> dict:
        return fact_json(await engine.approve(space, fact_id, actor=actor))

    @app.post("/v1/facts/{fact_id}/decline")
    async def post_fact_decline(fact_id: int, body: CloseBody, space: str = Depends(space_for), actor: str = Depends(actor_for)) -> dict:
        return fact_json(await engine.decline(space, fact_id, body.reason, actor=actor))

    @app.post("/v1/facts/{fact_id}/exclude")
    async def post_fact_exclude(fact_id: int, body: CloseBody, space: str = Depends(space_for), actor: str = Depends(actor_for)) -> dict:
        return fact_json(await engine.exclude(space, fact_id, body.reason, actor=actor))

    @app.post("/v1/facts/{fact_id}/include")
    async def post_fact_include(fact_id: int, space: str = Depends(space_for), actor: str = Depends(actor_for)) -> dict:
        return fact_json(await engine.include(space, fact_id, actor=actor))

    @app.post("/v1/facts/{fact_id}/close")
    async def post_fact_close(fact_id: int, body: CloseBody, space: str = Depends(space_for), actor: str = Depends(actor_for)) -> dict:
        closed = await engine.close_fact(space, fact_id, body.reason, actor=actor)
        return {"closed": closed.fact_id, "reason": closed.closed_reason}

    @app.get("/v1/profile")
    async def get_profile(limit: int = 10, space: str = Depends(space_for)) -> dict:
        profile = await engine.profile(space, limit)
        return {"static_facts": [fact_json(f) for f in profile.static_facts], "dynamic": profile.dynamic}

    @app.get("/v1/tags")
    async def get_tags(space: str = Depends(space_for)) -> dict:
        return {"tags": [{"name": n, "count": c} for n, c in (await engine.tags(space)).items()]}

    @app.get("/v1/events")
    async def get_events(
        kind: Optional[str] = None, since: Optional[str] = None, limit: int = 100, after_id: Optional[int] = None,
        space: str = Depends(space_for),
    ) -> dict:
        """Newest first by default. With after_id, oldest first and strictly
        after that id: a cursor a live reader can follow without losing a
        burst. next_after_id is the last id returned, or the cursor itself
        when nothing new arrived."""
        if engine.events is None:
            return {"events": [], "evidence": "none: no event log attached"}
        limit = max(1, min(limit, 1000))
        events = await engine.events.query(space, kind=kind, since=since, limit=limit, after_id=after_id)
        body = {
            "events": [event_json(e) for e in events],
            "evidence": engine.events.name,
            "queries_recorded": "text" if engine.record_queries else "hash",
            "truncated": len(events) >= limit,
        }
        if after_id is not None:
            body["next_after_id"] = events[-1].event_id if events else after_id
        return body

    @app.post("/v1/events")
    async def post_event(body: ExternalEventBody, space: str = Depends(space_for)) -> dict:
        event = await engine.record(space, body.kind, body.payload)
        return {"recorded": event.event_id}

    @app.post("/v1/feedback")
    async def post_feedback(body: FeedbackBody, space: str = Depends(space_for)) -> dict:
        event = await engine.feedback(space, body.recall_event_id, body.chunk_id, body.useful, body.note)
        return {"recorded": event.event_id}

    @app.get("/v1/metrics")
    async def get_metrics(
        since: Optional[str] = None, until: Optional[str] = None, limit: int = 5000, space: str = Depends(space_for)
    ) -> dict:
        """Computed from the retained events in [since, until). The reply
        says how many events it saw and whether the read was truncated."""
        if engine.events is None:
            return {"evidence": "none: no event log attached", "metrics": [], "coverage": None}
        limit = max(1, min(limit, 20000))
        events = await engine.events.query(space, since=since, limit=limit)
        report = metrics.compute(events, since=since, until=until, truncated=len(events) >= limit)
        return {"evidence": engine.events.name, **report.as_dict()}

    @app.get("/v1/graph")
    async def get_graph(
        session_id: Optional[str] = None, episode_id: Optional[int] = None, since: Optional[str] = None,
        limit: int = 400, space: str = Depends(space_for),
    ) -> dict:
        """Recorded relations only; see scone_memory/graph.py."""
        g = await engine.graph(space, session_id=session_id, episode_id=episode_id, since=since, limit=limit)
        return {"evidence": engine.events.name if engine.events else "none", **g.as_dict()}

    @app.get("/v1/scopes")
    async def get_scopes(space: str = Depends(space_for)) -> dict:
        return {"scopes": await engine.scopes(space)}

    @app.get("/v1/status")
    async def get_status(space: str = Depends(space_for)) -> dict:
        status = await engine.status(space)
        pending = await engine.pending_distillation(space)
        if worker is None:
            lane = "manual"  # claims arrive through POST /v1/facts, MCP, or the CLI
        elif worker.running:
            lane = "active"
        else:
            lane = "stopped"
        last = worker.last.get(space) if worker is not None else None
        return {
            **status.model_dump(),
            "semantic_lane": lane,
            "pending_distill": pending,
            "last_distill": last.as_payload() if last else None,
        }

    return app


class Unauthorized(Exception):
    pass


#: Ids one batch episode read may ask for. A review page asks for the
#: sources of the rows on screen, not for the space.
MAX_IDS = 100


async def read_bounded(request: Request, limit: int) -> bytes:
    """The request body, refused as soon as it passes ``limit``.

    Reading it whole and then measuring it lets the caller decide how much
    memory the server spends, which is the cap not existing. A declared
    length over the limit is refused before a byte is read.
    """
    declared = request.headers.get("content-length")
    if declared and declared.isdigit() and int(declared) > limit:
        raise InvalidInput(f"an attachment takes at most {limit} bytes, got {declared}")
    read, total = [], 0
    async for chunk in request.stream():
        total += len(chunk)
        if total > limit:
            raise InvalidInput(f"an attachment takes at most {limit} bytes")
        read.append(chunk)
    return b"".join(read)


def parse_ids(text: str) -> list[int]:
    """``3,9,14`` from the query string, in the order asked for, each id
    read once. Over the bound the answer is a refusal, never a truncation."""
    parts = [p.strip() for p in text.split(",") if p.strip()]
    if not parts:
        raise InvalidInput("ids needs at least one episode id")
    if len(parts) > MAX_IDS:
        raise InvalidInput(f"ids takes at most {MAX_IDS} ids, got {len(parts)}")
    seen: dict[int, None] = {}
    for part in parts:
        try:
            seen.setdefault(int(part), None)
        except ValueError:
            raise InvalidInput(f"ids are whole numbers, got {part!r}") from None
    return list(seen)


def parse_where(text: Optional[str]) -> dict[str, str]:
    """``user_id:alice,agent_id:planner`` from the query string."""
    where: dict[str, str] = {}
    for entry in (text or "").split(","):
        entry = entry.strip()
        if not entry:
            continue
        key, sep, value = entry.partition(":")
        if not sep:
            raise InvalidInput(f"where entries are key:value, got {entry!r}")
        where[key.strip()] = value.strip()
    return where


def item_json(item: RecallItem) -> dict:
    return {
        "chunk_id": item.chunk_id,
        "episode_id": item.episode_id,
        "text": item.text,
        "score": item.score,
        "similarity": item.similarity,
        "lanes": item.lanes,
        "created_at": item.created_at,
        "source": item.source,
        "tags": list(item.tags),
        "metadata": dict(item.metadata),
    }


def episode_json(episode) -> dict:
    return {
        "episode_id": episode.episode_id, "kind": episode.kind, "content": episode.content,
        "source": episode.source, "tags": list(episode.tags), "metadata": dict(episode.metadata),
        "created_at": episode.created_at, "ingested_at": episode.ingested_at,
        "attachments": [attachment_json(a) for a in episode.attachments],
    }


def attachment_json(attachment: Attachment) -> dict:
    return {
        "attachment_id": attachment.attachment_id, "media_type": attachment.media_type,
        "bytes": attachment.bytes, "filename": attachment.filename,
    }


def event_json(event) -> dict:
    return {
        "event_id": event.event_id,
        "ts": event.ts,
        "kind": event.kind,
        "schema_version": event.schema_version,
        "payload": dict(event.payload),
    }


def link_json(link) -> dict:
    return {
        "link_id": link.link_id, "from_fact": link.from_fact, "to_fact": link.to_fact, "kind": link.kind,
        "created_at": link.created_at, "source_episode_id": link.source_episode_id, "quote": link.quote,
    }


def fact_json(fact: Fact) -> dict:
    return {
        "fact_id": fact.fact_id,
        "subject": fact.subject,
        "predicate": fact.predicate,
        "object": fact.object,
        "confidence": fact.confidence,
        "valid_from": fact.valid_from,
        "valid_until": fact.valid_until,
        "status": fact.status,
        "closed_reason": fact.closed_reason,
        "source_episode_id": fact.source_episode_id,
        "origin": fact.origin,
        "excluded_reason": fact.excluded_reason,
        "superseded_by": fact.superseded_by,
        "quote": fact.quote,
        "grounded": fact.grounded,
    }
