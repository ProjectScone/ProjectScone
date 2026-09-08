"""HTTP surface, shaped like the Rust ``scone serve`` so one client
speaks to both.

Every request carries a bearer key, and the key decides the space: there
is no space parameter to get wrong, and a key can never read outside the
space it was issued for. Errors are ``{"error": "..."}`` with the status
the engine's error type maps to.
"""

from __future__ import annotations

from dataclasses import asdict

import asyncio
import json

import re

from pathlib import Path
from typing import Literal, Mapping, Optional

from fastapi import Depends, FastAPI, Query, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import HTMLResponse, JSONResponse, Response
from pydantic import BaseModel, ConfigDict, Field

from contextlib import asynccontextmanager

from ..observability import metrics
from ..memory.engine import Record, MemoryEngine
from ..core.errors import Gone, Conflict, InvalidInput, NotFound
from ..retrieval.filters import read_conditions
from ..core.models import Attachment, Fact, RecallItem


#: Types a browser may render in place. Everything else is handed back as
#: a download: an SVG or an HTML file served inline is script running in
#: this origin, and the bytes are kept as evidence either way.
INLINE_TYPES = ("image/png", "image/jpeg", "image/gif", "image/webp", "application/pdf")
_DIGEST = re.compile(r"[0-9a-f]{64}")
_SAFE_FILENAME = re.compile(r"[^A-Za-z0-9._-]")


class ConsolidateBody(BaseModel):
    """One pass by hand: ``distill`` runs the worker's pass (extraction,
    retention and, when configured, derivation); ``derive`` runs only the
    derivation pass."""

    model_config = ConfigDict(extra="forbid")

    scope: Literal["distill", "derive"]


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
    #: Identity across writes; with replace, changed content under a known
    #: key is an update instead of a reported duplicate.
    dedup_key: Optional[str] = None
    replace: bool = False
    kind: str = "note"
    metadata: dict[str, str] = Field(default_factory=dict)


class SourceQuery(BaseModel):
    model_config = ConfigDict(extra="forbid")
    before: Optional[int] = Field(default=None, ge=1, le=2**63-1)
    limit: int = Field(default=25, ge=1, le=100)
    kind: Optional[str] = None
    conditions: Optional[str] = None


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


#: What each role may do, by request. Reads are open to every role; the
#: decisions of review belong to review and full; every other write belongs
#: to write and full. A key with no role recorded is full.
_REVIEW_PATHS = ("/approve", "/decline", "/exclude", "/include", "/v1/facts/decide")


def _is_decision(path: str) -> bool:
    return path.rstrip("/").endswith(_REVIEW_PATHS) or path.startswith("/v1/facts/decide")


def _is_space_delete(method: str, path: str) -> bool:
    return method == "DELETE" and path.startswith("/v1/spaces/")


def permitted(role: str, method: str, path: str) -> bool:
    if method in ("GET", "HEAD", "OPTIONS") or role == "full":
        return True
    if _is_space_delete(method, path):
        return False  # a whole space goes only on the full role's word
    return role == ("review" if _is_decision(path) else "write")


def _refused_verb(method: str, path: str) -> str:
    if _is_space_delete(method, path):
        return "delete a space"
    return "decide" if _is_decision(path) else "write"


def _own_space(name: str, space: str) -> None:
    """A key reaches its own space and no other; another name is as
    unknown as a space that never existed."""
    if name != space:
        raise NotFound(f"space {name!r} is not this key's")


class Forbidden(Exception):
    """The key is known and scoped to this space, but its role does not
    allow this request. Maps to HTTP 403."""


#: Records one batch request may carry. A batch is bounded work, not an import.
MAX_BATCH_RECORDS = 500


class BatchBody(BaseModel):
    model_config = ConfigDict(extra="forbid")

    records: list[EpisodeBody] = Field(min_length=1, max_length=MAX_BATCH_RECORDS)
    #: The caller's id for this request. Sending it again returns the job
    #: the first attempt made instead of ingesting the batch a second time.
    request_id: Optional[str] = Field(default=None, max_length=128)


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


#: The public concept pages the packaged workspace renders without a key.
LEARN_PAGES = (
    "/learn", "/learn/quickstart", "/learn/how-it-works", "/learn/graph-memory",
    "/learn/sources", "/learn/search", "/learn/review", "/learn/profiles",
    "/learn/conversations", "/learn/spaces", "/learn/api",
)
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
    ingest_concurrency: int = 4,
    roles: Optional[Mapping[str, str]] = None,
    model_connections_available: bool = False,
    vision_available=None,
) -> FastAPI:
    """``console_key`` is baked into the page served at ``/`` so the key
    stays out of the URL and out of anything the user might paste; with
    no baked key the page asks for one and keeps it in the tab.
    ``worker`` is a ConsolidationWorker started with the app and stopped
    with it; None means no distiller runs on this server. ``reload_pages``
    re-reads the console and playground files on every request, for
    editing them with the server running; off in normal use. ``conversations``
    says this app is mounted under a conversation service on the same
    origin, so the capability manifest may advertise it. ``ingest_concurrency``
    is how many writes may be embedding at once; one more is answered 429
    with Retry-After instead of queued without limit. Reads are not gated.
    ``roles`` maps a key to read | write | review | full (see ``permitted``);
    a key not in it is full."""

    @asynccontextmanager
    async def lifespan(_app: FastAPI):
        if worker is not None:
            worker.start()
        yield
        if worker is not None:
            await worker.stop()

    app = FastAPI(title="scone-memory", version="0.1.0", docs_url=None, redoc_url=None, lifespan=lifespan)
    if isinstance(ingest_concurrency, bool) or not isinstance(ingest_concurrency, int) or not 1 <= ingest_concurrency <= 64:
        raise ValueError("ingest_concurrency must be an integer in 1..64")
    app.state.engine = engine
    app.state.roles = dict(roles or {})
    app.state.ingest_lane_width = ingest_concurrency
    ingest_lane = asyncio.Semaphore(ingest_concurrency)

    class IngestBusy(Exception):
        pass

    @asynccontextmanager
    async def ingest_slot(records: int):
        # Backpressure, not a queue: a write that finds every slot taken is
        # told to come back, with the number it is waiting behind.
        if ingest_lane.locked():
            raise IngestBusy(f"ingest is busy: {ingest_concurrency} record(s) are being embedded; try again shortly")
        async with ingest_lane:
            yield

    @app.exception_handler(IngestBusy)
    async def _busy(_: Request, e: IngestBusy) -> JSONResponse:
        return JSONResponse({"error": str(e), "code": "ingest_busy"}, status_code=429, headers={"Retry-After": "1"})
    app.state.keys = dict(keys)
    app.state.worker = worker

    async def space_for(request: Request) -> str:
        header = request.headers.get("authorization", "")
        scheme, _, token = header.partition(" ")
        if scheme.lower() != "bearer" or not token.strip():
            raise Unauthorized("missing bearer key")
        space = app.state.keys.get(token.strip())
        if space is None:
            raise Unauthorized("unknown key")
        role = app.state.roles.get(token.strip(), "full")
        if not permitted(role, request.method, request.url.path):
            raise Forbidden(f"key role {role} cannot {_refused_verb(request.method, request.url.path)}")
        # A deleted space's key answers 404 on every route: the space is
        # gone, and a key left in config must not bring it back.
        when = await engine.space_deleted(space)
        if when is not None:
            raise NotFound(f"space {space!r} was deleted at {when}")
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

    @app.exception_handler(Forbidden)
    async def _forbidden(_: Request, e: Forbidden) -> JSONResponse:
        return JSONResponse({"error": str(e)}, status_code=403)

    @app.exception_handler(InvalidInput)
    async def _invalid(_: Request, e: InvalidInput) -> JSONResponse:
        return JSONResponse({"error": str(e)}, status_code=422)

    @app.exception_handler(Conflict)
    async def _moved(_: Request, e: Conflict) -> JSONResponse:
        return JSONResponse({"error": str(e), "revision": e.revision}, status_code=409)

    @app.exception_handler(Gone)
    async def _gone(_: Request, e: Gone) -> JSONResponse:
        return JSONResponse({"error": str(e), "forgotten_at": e.forgotten_at}, status_code=410)

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
            "recall": True, "recall.conditions": True, "recall.evidence_graph": True, "facts.read": True, "facts.review": True,
            "recall.candidate_budget": True, "recall.reranking": engine.reranker is not None,
            "recall.structural_context": True,
            "recall.multi_hop": all(callable(getattr(engine.documents, name, None))
                                    for name in ("fact_links_from", "facts_by_subject")),
            "facts.close": True, "facts.exclude": True, "facts.include": True, "facts.links": True,
            "events.read": True, "metrics.read": True, "scopes.read": True,
            "status.read": True, "episodes.attachments": True,
            "integrity.read": True,
            "profile.read": True,
            "episodes.list": callable(getattr(engine.documents, "page_episodes", None)),
            "episodes.read": True,
            "jobs.read": all(callable(getattr(engine.documents, name, None)) for name in MemoryEngine.READS_JOBS),
        }
        if conversations:
            # Present only when the service is mounted here; its own manifest
            # at /v1/conversations/capabilities says what it can do.
            features["conversations"] = True
        if model_connections_available:
            features["models.manage"] = True
        if vision_available is not None and vision_available():
            features["images.understand"] = True
        # Manual passes must share a server-side guard with scheduled work
        # before the console advertises them as an available workflow.
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
        # Each address is named: there is no catch-all that would turn a
        # mistyped /v1 path into HTML.
        for path in ("/conversations", "/conversations/{sid}", "/memory/sources/{episode_id}"):
            app.add_api_route(path, console_page, methods=["GET", "HEAD"], include_in_schema=False)

        def render_guide() -> str:
            # The concept pages are public: the same bundle, with no key put
            # into it, so anyone may read them and nothing configured leaks.
            source = PLAYGROUND if PLAYGROUND.exists() else CONSOLE
            return source.read_text(encoding="utf-8").replace("__SCONE_MARK__", mark_data_uri())

        guide_html = None if reload_pages else render_guide()

        async def guide_page() -> Response:
            return page_response(guide_html if guide_html is not None else render_guide(), PLAYGROUND if PLAYGROUND.exists() else CONSOLE)

        for path in LEARN_PAGES:
            app.add_api_route(path, guide_page, methods=["GET", "HEAD"], include_in_schema=False)

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
        async with ingest_slot(1):
            return await _post_episode(body, space)

    async def _post_episode(body: EpisodeBody, space: str) -> dict:
        added = await engine.remember(
            space,
            body.content,
            kind=body.kind,  # type: ignore[arg-type]
            source=body.source,
            tags=body.tags,
            created_at=body.created_at,
            metadata=body.metadata,
            attachment_ids=body.attachment_ids,
            dedup_key=body.dedup_key,
            replace=body.replace,
        )
        return added.model_dump()

    @app.post("/v1/episodes/batch")
    async def post_episodes(body: BatchBody, space: str = Depends(space_for)) -> dict:
        """One bounded request, one outcome per record in the order sent;
        the batch lands whole or not at all. Replacement is one record at
        a time, because it forgets before it stores."""
        if any(r.replace for r in body.records):
            raise InvalidInput("replace is one record at a time: use POST /v1/episodes")
        if body.request_id:
            replayed = await engine.job_for_request(space, body.request_id)
            if replayed is not None:
                # The batch already landed under this id. Nothing is ingested
                # again, and the answer says plainly that this is the receipt
                # of the first attempt rather than a second one.
                return {"items": [{"episode_id": i.episode_id, "outcome": i.outcome} for i in replayed.items],
                        "counts": {outcome: sum(1 for i in replayed.items if i.outcome == outcome)
                                   for outcome in ("accepted", "duplicate", "updated")},
                        "job": job_json(replayed), "replayed": True}
        records = [
            Record(r.content, r.kind, r.source, tuple(r.tags), r.created_at, dict(r.metadata), dedup_key=r.dedup_key)
            for r in body.records
        ]
        async with ingest_slot(len(records)):
            added = await engine.remember_many(space, records)
        for record, item in zip(body.records, added):
            for attachment_id in dict.fromkeys(record.attachment_ids):
                await engine.blobs.link(space, attachment_id, item.episode_id)
        counts = {outcome: sum(1 for a in added if a.outcome == outcome) for outcome in ("accepted", "duplicate", "updated")}
        job = await engine.record_job(space, added, request_id=body.request_id)
        return {"items": [a.model_dump() for a in added], "counts": counts, "job": job_json(job)}

    @app.get("/v1/jobs")
    async def get_jobs(limit: int = 20, before: Optional[str] = None, space: str = Depends(space_for)) -> dict:
        """Recent batches, newest first. ``next`` names the cursor for the
        page after this one, or is null when there is nothing older, so a
        reader can tell a short page from the end of the list."""
        found = await engine.jobs(space, limit, before)
        more = len(found) == limit and bool(await engine.jobs(space, 1, found[-1].job_id)) if found else False
        return {"jobs": [job_json(job) for job in found], "next": found[-1].job_id if more else None}

    @app.get("/v1/jobs/{job_id}")
    async def get_job(job_id: str, space: str = Depends(space_for)) -> dict:
        """One batch: what each record became, and how far it has got."""
        return job_json(await engine.job(space, job_id))

    @app.post("/v1/jobs/{job_id}/cancel")
    async def post_job_cancel(job_id: str, space: str = Depends(space_for)) -> dict:
        """Stop expecting more of this batch. Records already stored stay
        stored: this abandons work, it does not delete memory."""
        return job_json(await engine.cancel_job(space, job_id))

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
        page = await engine.source_page(space, before=query.before, limit=query.limit, kind=query.kind,
                                        conditions=read_conditions(query.conditions))
        # Where consolidation left each source, in the words the API means:
        # cited (a claim rests on it), parked (the distiller gave up on it in
        # this process), pending (no claim cites it yet; a pass, the worker's
        # or `distill`'s, visits it). The Rust host says the same words and
        # adds "done" for a source its queue has visited and found nothing in.
        cited = await engine.cited_episode_ids(space)
        distiller = getattr(worker, "distiller", None) if worker is not None else None
        parked = distiller.parked(space) if distiller is not None and callable(getattr(distiller, "parked", None)) else {}

        def status_of(episode_id: int) -> dict:
            if episode_id in cited:
                return {"status": "cited"}
            if episode_id in parked:
                return {"status": "parked", "parked_reason": str(parked[episode_id])[:200]}
            return {"status": "pending"}

        return {"items": [{"episode_id": e.episode_id, "kind": e.kind, "source": e.source,
                           "created_at": e.created_at, "byte_count": len(e.content.encode("utf-8")),
                           "preview": e.content[:500], "preview_truncated": len(e.content) > 500,
                           **status_of(e.episode_id)}
                          for e in page.episodes], "has_more": page.has_more, "next_before": page.next_before}

    @app.get("/v1/episodes/{episode_id}")
    async def get_episode(episode_id: int, space: str = Depends(space_for)) -> dict:
        return episode_json(await engine.episode(space, episode_id))

    @app.get("/v1/episodes/{episode_id}/impact")
    async def episode_impact(episode_id: int, space: str = Depends(space_for)) -> dict:
        """What forgetting would take and leave; removes nothing."""
        return (await engine.impact(space, episode_id)).model_dump()

    @app.delete("/v1/episodes/{episode_id}")
    async def delete_episode(episode_id: int, space: str = Depends(space_for)) -> dict:
        receipt = await engine.forget(space, episode_id)
        return {"forgotten": episode_id, **receipt.model_dump()}

    @app.get("/v1/spaces/{name}/impact")
    async def space_impact(name: str, space: str = Depends(space_for)) -> dict:
        """What deleting the space would take with it; nothing is removed."""
        _own_space(name, space)
        return (await engine.space_impact(space)).model_dump()

    @app.delete("/v1/spaces/{name}")
    async def delete_space(name: str, confirm: Optional[str] = None, space: str = Depends(space_for)) -> dict:
        """Remove everything the space holds. ``confirm`` must repeat the
        space name; afterwards this key answers 404 on every route."""
        _own_space(name, space)
        if confirm != name:
            raise InvalidInput("confirm must repeat the space name; a whole space is not deleted by accident")
        receipt = await engine.delete_space(space)
        return {"deleted": name, **receipt.model_dump()}

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
        conditions: Optional[str] = None,
        evidence_graph: bool = False,
        candidate_limit: Optional[int] = Query(default=None, ge=1, le=1000),
        rerank: bool = True,
        structural_context: bool = False,
        multi_hop: bool = False,
        max_hops: int = Query(default=3, ge=1, le=6),
        expansion_max_bytes: int = Query(default=16000, ge=512, le=256000,
                                          description="Byte budget per enabled expansion stage."),
        space: str = Depends(space_for),
    ) -> dict:
        tag_list = [t for t in (tags or "").split(",") if t.strip()]
        result = await engine.recall(
            space, q, limit=limit, as_of=as_of, tags=tag_list, where=parse_where(where), history=history,
            kind=kind, source_prefix=source_prefix, since=since, until=until,
            conditions=read_conditions(conditions),
            candidate_limit=candidate_limit, rerank=rerank,
        )
        response: dict[str, object] = {
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
        if result.rerank is not None:
            response["rerank"] = result.rerank.model_dump(mode="json")
        if evidence_graph or structural_context or multi_hop:
            from ..core.ports import TextFilter
            from ..memory.engine import normalise_metadata, normalise_tags, normalise_time
            from ..retrieval.evidence_graph import QueryEvidenceGraph, build_query_evidence_graph
            from ..retrieval.filters import parse_filter

            parsed_conditions = read_conditions(conditions)
            scope = TextFilter(
                as_of=normalise_time(as_of) if as_of else None,
                tags=normalise_tags(tag_list), where=normalise_metadata(parse_where(where) or {}),
                kind=kind, source_prefix=source_prefix,
                since=normalise_time(since) if since else None,
                until=normalise_time(until) if until else None,
                conditions=parse_filter(parsed_conditions) if parsed_conditions is not None else None,
            )
            if evidence_graph:
                try:
                    graph = await asyncio.wait_for(
                        build_query_evidence_graph(engine.documents, space, q, result, scope=scope), timeout=1.0,
                    )
                except Exception:
                    graph = QueryEvidenceGraph(notices=["Query evidence graph is unavailable; recall results are still shown."])
                response["evidence_graph"] = graph.model_dump(mode="json")
            if structural_context:
                from ..retrieval.structural import StructuralLimits, expand_structural_context
                try:
                    structural = await asyncio.wait_for(expand_structural_context(
                        engine.documents, space, result, scope=scope,
                        limits=StructuralLimits(max_bytes=expansion_max_bytes)), timeout=1.0)
                    response["structural_context"] = structural.model_dump(mode="json")
                except Exception as error:
                    response["structural_context"] = {"status": "unavailable", "error": type(error).__name__}
            if multi_hop:
                from ..retrieval.multihop import MultiHopLimits, expand_multihop
                try:
                    expanded = await asyncio.wait_for(expand_multihop(
                        engine.documents, space, seeds=result, scope=scope, include_history=history,
                        limits=MultiHopLimits(max_hops=max_hops, max_bytes=expansion_max_bytes)), timeout=1.0)
                    response["multi_hop"] = expanded.model_dump(mode="json")
                except Exception as error:
                    response["multi_hop"] = {"status": "unavailable", "error": type(error).__name__}
        return response

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


    @app.get("/v1/facts/audit")
    async def get_facts_audit(status: str = "active", flagged: bool = False,
                              space: str = Depends(space_for)) -> dict:
        """Every extracted fact judged against the text it came from. Read
        only: a flagged claim is one whose own source cannot support it,
        which is a question for a person, and the repair is exclude with a
        reason through the decision routes."""
        from collections import Counter

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

    @app.post("/v1/consolidate")
    async def post_consolidate(body: ConsolidateBody, space: str = Depends(space_for)) -> JSONResponse:
        """One consolidation pass by hand over this key's space."""
        if body.scope == "derive":
            deriver = getattr(worker, "deriver", None)
            if deriver is None:
                return JSONResponse({"error": "no derivation model configured (SCONE_DERIVE=1 with SCONE_CHAT_URL and SCONE_CHAT_MODEL)"}, status_code=501)
            outcome = await deriver.derive(space, limit_groups=worker.batch)
            return JSONResponse({"space": space, "scope": "derive", **outcome.as_payload()})
        if worker is None:
            return JSONResponse({"error": "no consolidation worker configured (SCONE_CHAT_URL and SCONE_CHAT_MODEL, or SCONE_RETAIN)"}, status_code=501)
        report = await worker.run_once(space)
        return JSONResponse({"space": space, "scope": "distill", **report.as_payload()})

    @app.post("/v1/facts/decide")
    async def post_decide(body: DecideBody, space: str = Depends(space_for), actor: str = Depends(actor_for)) -> dict:
        """One reviewed batch, settled together: applied oldest first, and
        refused whole if the space moved since the batch was built."""
        decided = await engine.decide(
            space, body.decision, body.ids,
            reason=body.reason, actor=actor, expect_revision=body.expect_revision,
        )
        return decided.model_dump()

    # Parametrised fact routes come after every fixed /v1/facts/... address,
    # so /v1/facts/audit and /v1/facts/decide are never read as an id.
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
        return {"static_facts": [fact_json(f) for f in profile.static_facts], "dynamic": profile.dynamic,
                "recent": [asdict(r) for r in profile.recent]}

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

    @app.get("/v1/doctor")
    async def get_doctor(space: str = Depends(space_for)) -> dict:
        """What references what across the stores, read only; see doctor()."""
        return (await engine.doctor(space)).model_dump()

    @app.get("/v1/scopes")
    async def get_scopes(space: str = Depends(space_for)) -> dict:
        return {"scopes": await engine.scopes(space)}

    @app.get("/v1/status")
    async def get_status(space: str = Depends(space_for)) -> dict:
        status = await engine.status(space)
        pending = await engine.pending_distillation(space)
        # The lane is the model's work; a worker that only applies retention
        # has no lane, and claims arrive through POST /v1/facts, MCP or the CLI.
        if worker is None or getattr(worker, "distiller", None) is None:
            lane = "manual"
        elif worker.running:
            lane = "active"
        else:
            lane = "stopped"
        last = worker.last.get(space) if worker is not None else None
        return {
            **status.model_dump(),
            "semantic_lane": lane,
            "pending_distill": pending,
            "pending_derivation": await engine.pending_derivation(space),
            "derivation": "on" if getattr(worker, "deriver", None) is not None else "off",
            "last_distill": last.as_payload() if last else None,
            "retention": dict(getattr(worker, "retention", None) or {}) if worker is not None else {},
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
    result = {
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
    if item.rerank_score is not None:
        result["rerank_score"] = item.rerank_score
    return result


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


def job_json(job) -> dict:
    """A job as the API reports it: the stored fields, plus the two counts
    and the state that are read off them."""
    return {**job.model_dump(), "searchable": job.searchable,
            "consolidated": job.consolidated, "state": job.state}


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
