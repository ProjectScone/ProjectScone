"""HTTP surface, shaped like the Rust ``scone serve`` so one client
speaks to both.

Every request carries a bearer key, and the key decides the space: there
is no space parameter to get wrong, and a key can never read outside the
space it was issued for. Errors are ``{"error": "..."}`` with the status
the engine's error type maps to.
"""

from __future__ import annotations

from pathlib import Path
from typing import Mapping, Optional

from fastapi import Depends, FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel, ConfigDict, Field

from contextlib import asynccontextmanager

from .. import metrics
from ..engine import MemoryEngine
from ..errors import InvalidInput, NotFound
from ..models import Fact, RecallItem


class EpisodeBody(BaseModel):
    """Unknown fields are refused rather than ignored: a client that
    sends a field we silently drop believes it stored something it
    did not."""

    model_config = ConfigDict(extra="forbid")

    content: str
    tags: list[str] = Field(default_factory=list)
    source: Optional[str] = None
    created_at: Optional[str] = None
    kind: str = "note"
    metadata: dict[str, str] = Field(default_factory=dict)


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


def create_app(
    engine: MemoryEngine, keys: Mapping[str, str], console: bool = True, console_key: Optional[str] = None, worker=None
) -> FastAPI:
    """``console_key`` is baked into the page served at ``/`` so the key
    stays out of the URL and out of anything the user might paste; with
    no baked key the page asks for one and keeps it in the tab.
    ``worker`` is a ConsolidationWorker started with the app and stopped
    with it; None means no distiller runs on this server."""

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

    @app.exception_handler(Unauthorized)
    async def _unauthorized(_: Request, e: Unauthorized) -> JSONResponse:
        return JSONResponse({"error": str(e)}, status_code=401)

    @app.exception_handler(InvalidInput)
    async def _invalid(_: Request, e: InvalidInput) -> JSONResponse:
        return JSONResponse({"error": str(e)}, status_code=422)

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

    if console:
        page = CONSOLE.read_text(encoding="utf-8")
        if console_key:
            page = page.replace('data-token=""', "", 1).replace("<script>", f'<script data-token="{console_key}">', 1)

        @app.get("/", response_class=HTMLResponse)
        async def console_page() -> str:
            return page

        if PLAYGROUND.exists():
            # Shared asset owned in ui/playground.html and copied here by
            # scripts/sync-playground.cjs; same key placeholder as the console.
            playground = PLAYGROUND.read_text(encoding="utf-8")
            if console_key:
                playground = playground.replace("__SCONE_TOKEN__", console_key)

            # GET and HEAD: the console probes with HEAD to decide whether
            # to show its Playground link.
            @app.api_route("/playground", methods=["GET", "HEAD"], response_class=HTMLResponse)
            async def playground_page() -> str:
                return playground

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
        )
        return added.model_dump()

    @app.get("/v1/episodes/{episode_id}")
    async def get_episode(episode_id: int, space: str = Depends(space_for)) -> dict:
        episode = await engine.episode(space, episode_id)
        return {
            "episode_id": episode.episode_id, "kind": episode.kind, "content": episode.content,
            "source": episode.source, "tags": list(episode.tags), "metadata": dict(episode.metadata),
            "created_at": episode.created_at, "ingested_at": episode.ingested_at,
        }

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
        space: str = Depends(space_for),
    ) -> dict:
        tag_list = [t for t in (tags or "").split(",") if t.strip()]
        result = await engine.recall(
            space, q, limit=limit, as_of=as_of, tags=tag_list, where=parse_where(where)
        )
        return {
            "event_id": result.event_id,
            "items": [item_json(i) for i in result.items],
            "facts": [fact_json(f) for f in result.facts],
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
        return {"facts": [fact_json(f) for f in facts]}

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
        )
        return fact_json(fact)

    @app.post("/v1/facts/{fact_id}/approve")
    async def post_fact_approve(fact_id: int, space: str = Depends(space_for)) -> dict:
        return fact_json(await engine.approve(space, fact_id))

    @app.post("/v1/facts/{fact_id}/decline")
    async def post_fact_decline(fact_id: int, body: CloseBody, space: str = Depends(space_for)) -> dict:
        return fact_json(await engine.decline(space, fact_id, body.reason))

    @app.post("/v1/facts/{fact_id}/exclude")
    async def post_fact_exclude(fact_id: int, body: CloseBody, space: str = Depends(space_for)) -> dict:
        return fact_json(await engine.exclude(space, fact_id, body.reason))

    @app.post("/v1/facts/{fact_id}/include")
    async def post_fact_include(fact_id: int, space: str = Depends(space_for)) -> dict:
        return fact_json(await engine.include(space, fact_id))

    @app.post("/v1/facts/{fact_id}/close")
    async def post_fact_close(fact_id: int, body: CloseBody, space: str = Depends(space_for)) -> dict:
        closed = await engine.close_fact(space, fact_id, body.reason)
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


def event_json(event) -> dict:
    return {
        "event_id": event.event_id,
        "ts": event.ts,
        "kind": event.kind,
        "schema_version": event.schema_version,
        "payload": dict(event.payload),
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
    }
