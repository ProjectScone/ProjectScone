"""Authorized external image metadata indexing and image/entity retrieval."""
from __future__ import annotations

from collections.abc import Callable
from dataclasses import asdict
from typing import AsyncContextManager, cast, Literal

from fastapi import Depends, FastAPI, Query, Request
from fastapi.encoders import jsonable_encoder
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ConfigDict, Field, ValidationError

from ..core.errors import InvalidInput
from ..ingestion.images import ImageContext, ingest_image, recall_images
from ..memory.engine import MemoryEngine


class _RebuildBody(BaseModel):
    """Unknown fields are refused: a pass told something it silently ignores did not do what was asked."""
    model_config = ConfigDict(extra='forbid')
    #: File episodes this pass reads, 1..1000.
    limit: int = 100
    #: Walk on from a previous pass's ``resume_before``.
    before: int | None = None


class _ImageBody(BaseModel):
    model_config = ConfigDict(strict=True, extra='forbid', hide_input_in_errors=True)
    attachment_id: str = Field(pattern=r'^[a-f0-9]{64}$')
    context: ImageContext


def mount_image_context_routes(app: FastAPI, engine: MemoryEngine,
                               space_for: Callable[..., object],
                               ingest_slot: Callable[[int], AsyncContextManager[None]]) -> None:
    @app.post('/v1/images')
    async def index_image(request: Request, space: str = Depends(space_for)) -> JSONResponse:
        data = bytearray()
        async for part in request.stream():
            if len(data) + len(part) > 132_000:
                return JSONResponse({'error': 'image context exceeds its byte limit'}, status_code=413)
            data.extend(part)
        try:
            body = _ImageBody.model_validate_json(data)
        except ValidationError:
            return JSONResponse({'error': 'invalid image context or attachment ID'}, status_code=400)
        async with ingest_slot(1):
            attachment, raw = await engine.attachment(space, body.attachment_id)
            if attachment.media_type not in ('image/png', 'image/jpeg', 'image/webp'):
                raise InvalidInput('image context supports still PNG, JPEG and WebP images')
            media_type = cast(Literal['image/png', 'image/jpeg', 'image/webp'], attachment.media_type)
            saved = await ingest_image(engine, space, raw, media_type=media_type,
                context=body.context, filename=attachment.filename)
        return JSONResponse(jsonable_encoder(asdict(saved)))

    @app.post('/v1/images/reembed')
    async def reembed_images(body: _RebuildBody, space: str = Depends(space_for)) -> JSONResponse:
        """One bounded pass of embedding the space's stored images again with the
        image lane's embedder (``MemoryEngine.reembed_images``); the report says
        where to walk on and what the image index records."""
        async with ingest_slot(1):  # a pass embeds as an ingest does, so it waits its turn
            report = await engine.reembed_images(space, limit=body.limit, before=body.before)
        return JSONResponse(report.model_dump())

    @app.get('/v1/images/search')
    async def search_images(query: str = Query(min_length=1, max_length=16_000),
                            limit: int = Query(default=5, ge=1, le=25),
                            entity_id: str | None = Query(default=None, min_length=1, max_length=256),
                            space: str = Depends(space_for)) -> JSONResponse:
        found = await recall_images(engine, space, query, limit=limit, entity_id=entity_id)
        matches = [{**asdict(match), 'download_path': f'/v1/attachments/{match.image.attachment_id}'}
            for match in found.matches]
        return JSONResponse(jsonable_encoder({'matches': matches, 'recall': found.recall,
            'unresolved_episode_ids': found.unresolved_episode_ids}))
