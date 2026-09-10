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
