"""Authenticated PDF text ingestion and retained page evidence over HTTP."""
from __future__ import annotations

from collections.abc import Callable
from dataclasses import asdict
from importlib.util import find_spec
from typing import AsyncContextManager

from fastapi import Depends, FastAPI, Path, Query, Request
from fastapi.encoders import jsonable_encoder
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ConfigDict, Field, ValidationError

from ..core.errors import InvalidInput
from ..ingestion.documents import ingest_pdf, pdf_provenance
from ..memory.engine import MemoryEngine


class _PdfBody(BaseModel):
    model_config = ConfigDict(strict=True, extra='forbid', hide_input_in_errors=True)
    attachment_id: str = Field(pattern=r'^[a-f0-9]{64}$')


def pdf_available() -> bool:
    """Dependency availability, not a promise that every PDF can be parsed."""
    return find_spec('pypdf') is not None


def mount_pdf_document_routes(app: FastAPI, engine: MemoryEngine,
                              space_for: Callable[..., object],
                              ingest_slot: Callable[[int], AsyncContextManager[None]]) -> None:
    @app.post('/v1/documents/pdf')
    async def index_pdf(request: Request, space: str = Depends(space_for)) -> JSONResponse:
        data = bytearray()
        async for part in request.stream():
            if len(data) + len(part) > 4096:
                return JSONResponse({'error': 'PDF request exceeds its byte limit'}, status_code=413)
            data.extend(part)
        try:
            body = _PdfBody.model_validate_json(data)
        except ValidationError:
            return JSONResponse({'error': 'invalid PDF attachment request'}, status_code=400)
        if not pdf_available():
            return JSONResponse({'error': 'PDF parsing requires the optional scone-memory[pdf] extra'},
                                status_code=501)
        async with ingest_slot(1):
            attachment, raw = await engine.attachment(space, body.attachment_id)
            if attachment.media_type != 'application/pdf':
                raise InvalidInput('PDF ingestion requires an application/pdf attachment')
            saved = await ingest_pdf(engine, space, raw, filename=attachment.filename)
        return JSONResponse(jsonable_encoder(asdict(saved)))

    @app.get('/v1/episodes/{episode_id}/pdf')
    async def page_evidence(episode_id: int = Path(ge=1, le=2**63 - 1),
                            chunk_id: int | None = Query(default=None, ge=1, le=2**63 - 1),
                            space: str = Depends(space_for)) -> JSONResponse:
        evidence = await pdf_provenance(engine, space, episode_id, chunk_id=chunk_id)
        return JSONResponse(jsonable_encoder({**asdict(evidence),
            'download_path': f'/v1/attachments/{evidence.original.attachment_id}'}))
