"""HTTP surface, shaped like the Rust ``scone serve`` so one client
speaks to both.

Every request carries a bearer key, and the key decides the space: there
is no space parameter to get wrong, and a key can never read outside the
space it was issued for. Errors are ``{"error": "..."}`` with the status
the engine's error type maps to.
"""

from __future__ import annotations

from typing import Mapping, Optional

from fastapi import Depends, FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ConfigDict, Field

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


class FactBody(BaseModel):
    model_config = ConfigDict(extra="forbid")

    subject: str
    predicate: str
    object: str
    valid_from: Optional[str] = None
    confidence: float = 1.0
    source_episode_id: Optional[int] = None


class CloseBody(BaseModel):
    model_config = ConfigDict(extra="forbid")

    reason: str


def create_app(engine: MemoryEngine, keys: Mapping[str, str]) -> FastAPI:
    app = FastAPI(title="scone-memory", version="0.1.0", docs_url=None, redoc_url=None)
    app.state.engine = engine
    app.state.keys = dict(keys)

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

    @app.post("/v1/episodes")
    async def post_episode(body: EpisodeBody, space: str = Depends(space_for)) -> dict:
        added = await engine.remember(
            space,
            body.content,
            kind=body.kind,  # type: ignore[arg-type]
            source=body.source,
            tags=body.tags,
            created_at=body.created_at,
        )
        return added.model_dump()

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
        space: str = Depends(space_for),
    ) -> dict:
        tag_list = [t for t in (tags or "").split(",") if t.strip()]
        result = await engine.recall(space, q, limit=limit, as_of=as_of, tags=tag_list)
        return {
            "items": [item_json(i) for i in result.items],
            "facts": [fact_json(f) for f in result.facts],
            "degraded": result.degraded,
            "returned_bytes": result.returned_bytes,
            "space_bytes": result.space_bytes,
            "context_reduction": round(result.context_reduction, 6),
        }

    @app.get("/v1/facts")
    async def get_facts(all: bool = False, as_of: Optional[str] = None, space: str = Depends(space_for)) -> dict:
        facts = await engine.facts(space, include_closed=all, as_of=as_of)
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
        )
        return fact_json(fact)

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

    @app.get("/v1/status")
    async def get_status(space: str = Depends(space_for)) -> dict:
        status = await engine.status(space)
        return {
            **status.model_dump(),
            # No distiller runs in this stack yet; facts arrive through
            # POST /v1/facts. Named so the shared client can tell.
            "semantic_lane": "manual",
            "pending_distill": 0,
        }

    return app


class Unauthorized(Exception):
    pass


def item_json(item: RecallItem) -> dict:
    return {
        "chunk_id": item.chunk_id,
        "episode_id": item.episode_id,
        "text": item.text,
        "score": item.score,
        "similarity": item.similarity,
        "created_at": item.created_at,
        "source": item.source,
        "tags": list(item.tags),
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
    }
