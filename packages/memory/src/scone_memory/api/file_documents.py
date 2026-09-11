"""Authorized file extraction, indexing and original-backed evidence routes."""
from __future__ import annotations

from collections.abc import Callable
from dataclasses import asdict
from typing import AsyncContextManager

from fastapi import Depends, FastAPI, Path, Query, Request
from fastapi.encoders import jsonable_encoder
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ConfigDict, Field, ValidationError

from ..ingestion.files import document_provenance, extraction_filename, prepare_document, store_document
from ..ingestion.formats.registry import BuiltinDocumentParser
from ..ingestion.formats.types import DocumentLimits
from ..memory.engine import MemoryEngine


class _FileBody(BaseModel):
    model_config = ConfigDict(strict=True, extra='forbid', hide_input_in_errors=True)
    attachment_id: str = Field(pattern=r'^[a-f0-9]{64}$')
    filename: str | None = Field(default=None, min_length=1, max_length=1024)


def mount_file_document_routes(app: FastAPI, engine: MemoryEngine,
                               space_for: Callable[..., object],
                               ingest_slot: Callable[[int], AsyncContextManager[None]]) -> None:
    @app.get('/v1/documents/formats')
    async def formats(_space: str = Depends(space_for)) -> dict[str, object]:
        from ..ingestion.formats.capabilities import document_formats
        return {'formats': document_formats(), 'max_input_bytes': min(engine.max_attachment_bytes, 25*1024*1024)}

    @app.post('/v1/documents')
    async def index_file(request: Request, space: str = Depends(space_for)) -> JSONResponse:
        data = bytearray()
        async for part in request.stream():
            if len(data) + len(part) > 4096:
                return JSONResponse({'error': 'document request exceeds its byte limit'}, status_code=413)
            data.extend(part)
        try:
            body = _FileBody.model_validate_json(data)
        except ValidationError:
            return JSONResponse({'error': 'invalid document attachment request'}, status_code=400)
        async with ingest_slot(1):
            original, raw = await engine.attachment(space, body.attachment_id)
            filename = extraction_filename(original, body.filename)
            manifest = await prepare_document(raw, filename, parser=BuiltinDocumentParser(), limits=DocumentLimits())
            saved = await store_document(engine, space, original, manifest)
        return JSONResponse(jsonable_encoder(asdict(saved)))

    @app.get('/v1/episodes/{episode_id}/document')
    async def evidence(episode_id: int = Path(ge=1, le=2**63 - 1),
                       chunk_id: int | None = Query(default=None, ge=1, le=2**63 - 1),
                       space: str = Depends(space_for)) -> JSONResponse:
        result = await document_provenance(engine, space, episode_id, chunk_id=chunk_id)
        return JSONResponse(jsonable_encoder({**asdict(result),
            'download_path': f'/v1/attachments/{result.original.attachment_id}'}))
