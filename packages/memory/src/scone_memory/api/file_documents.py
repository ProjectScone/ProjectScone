"""Authorized file extraction, indexing and original-backed evidence routes."""
from __future__ import annotations

from collections.abc import Callable
from dataclasses import asdict
from typing import AsyncContextManager

from fastapi import Depends, FastAPI, Path, Query, Request
from fastapi.encoders import jsonable_encoder
from fastapi.responses import JSONResponse, Response
from pydantic import BaseModel, ConfigDict, Field, ValidationError

from ..ingestion.files import document_provenance, extraction_filename, prepare_document, store_document, digest
from ..ocr.tables import infer_tables
from ..core.errors import InvalidInput
from ..ingestion.document_media import DocumentMedia, MEDIA_DOCUMENT_EXTENSIONS
from ..ingestion.document_ocr import DocumentOcr, PdfOcrSelection, ocr_choices
from ..ingestion.formats.registry import BuiltinDocumentParser, DocumentParser
from ..ingestion.formats.types import DocumentLimits
from ..memory.engine import MemoryEngine


class _FileBody(BaseModel):
    model_config = ConfigDict(strict=True, extra='forbid', hide_input_in_errors=True)
    attachment_id: str = Field(pattern=r'^[a-f0-9]{64}$')
    filename: str | None = Field(default=None, min_length=1, max_length=1024)
    pdf_ocr: PdfOcrSelection | None = None


def mount_file_document_routes(app: FastAPI, engine: MemoryEngine,
                               space_for: Callable[..., object],
                               ingest_slot: Callable[[int], AsyncContextManager[None]],
                               document_ocr: DocumentOcr | None = None, *,
                               assert_current_space: Callable[[Request, str], None],
                               document_media: DocumentMedia | None = None) -> None:
    @app.get('/v1/documents/formats')
    async def formats(_space: str = Depends(space_for)) -> dict[str, object]:
        from ..ingestion.formats.capabilities import document_formats
        return {'formats': {**document_formats(), **(document_media.formats() if document_media else {})}, 'max_input_bytes': min(engine.max_attachment_bytes, 25*1024*1024),
                'pdf_ocr': ocr_choices(document_ocr)}

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
        assert_current_space(request, space)
        parser: DocumentParser = document_media.parser() if document_media else BuiltinDocumentParser()
        if body.pdf_ocr is not None:
            if document_ocr is None:
                raise InvalidInput('document OCR is not configured on this server')
            parser = document_ocr.parser(body.pdf_ocr)
        async with ingest_slot(1):
            original, raw = await engine.attachment(space, body.attachment_id)
            assert_current_space(request, space)
            filename = extraction_filename(original, body.filename)
            manifest = await prepare_document(raw, filename, parser=parser, limits=DocumentLimits())
            assert_current_space(request, space)
            saved = await store_document(engine, space, original, manifest)
        assert_current_space(request, space)
        return JSONResponse(jsonable_encoder({**asdict(saved),
            **({'pdf_ocr': body.pdf_ocr.model_dump()} if body.pdf_ocr is not None else {})}))

    @app.get('/v1/episodes/{episode_id}/document')
    async def evidence(request: Request, episode_id: int = Path(ge=1, le=2**63 - 1),
                       chunk_id: int | None = Query(default=None, ge=1, le=2**63 - 1),
                       space: str = Depends(space_for)) -> JSONResponse:
        result = await document_provenance(engine, space, episode_id, chunk_id=chunk_id)
        assert_current_space(request, space)
        return JSONResponse(jsonable_encoder({**asdict(result),
            'download_path': f'/v1/attachments/{result.original.attachment_id}'}))


    @app.get('/v1/episodes/{episode_id}/document/ocr-tables')
    async def ocr_tables(request: Request, episode_id: int = Path(ge=1, le=2**63 - 1),
                         page: int = Query(ge=1, le=1000),
                         space: str = Depends(space_for)) -> JSONResponse:
        result = await document_provenance(engine, space, episode_id)
        selected = [segment for segment in result.segments
                    if segment.locator == f'page:{page}' and segment.metadata.get('page') == str(page)]
        if (result.format != 'pdf' or len(selected) != 1
                or selected[0].metadata.get('extraction') != 'ocr'
                or not selected[0].regions
                or any(r.coordinate_space != 'normalized_displayed_page_top_left' for r in selected[0].regions)):
            raise InvalidInput('table analysis requires a retained PDF page with OCR regions')
        segment = selected[0]
        layout = infer_tables(segment.regions)
        current = await engine.episode(space, episode_id)
        linked = {attachment.attachment_id for attachment in current.attachments}
        if (current.metadata.get('document_original') != result.original.attachment_id
                or current.metadata.get('document_manifest') != result.manifest.attachment_id
                or not {result.original.attachment_id, result.manifest.attachment_id} <= linked
                or current.content != '\n\n'.join(s.text for s in result.segments)):
            raise InvalidInput('document changed during OCR table analysis')
        assert_current_space(request, space)
        return JSONResponse({'space': space, 'episode_id': episode_id, 'page': page,
            'original_sha256': result.original.attachment_id,
            'manifest_sha256': result.manifest.attachment_id,
            'page_text_sha256': digest(segment.text.encode('utf-8')),
            'layout': layout.model_dump(mode='json')})


    @app.get('/v1/episodes/{episode_id}/document/audio')
    async def audio_evidence(request: Request, episode_id: int = Path(ge=1, le=2**63 - 1),
                             space: str = Depends(space_for)) -> Response:
        if document_media is None:
            raise InvalidInput('document media decoding is not configured on this server')
        async with ingest_slot(1):
            result = await document_provenance(engine, space, episode_id)
            metadata = result.metadata
            if ('.' + result.format not in MEDIA_DOCUMENT_EXTENSIONS or result.parser != 'media-transcription'
                    or metadata.get('extraction') != 'audio-only'
                    or metadata.get('transcriber_revision') != document_media.revision
                    or not metadata.get('audio_wav_sha256') or not metadata.get('audio_wav_bytes')):
                raise InvalidInput('document has no matching normalized audio evidence')
            _, raw = await engine.attachment(space, result.original.attachment_id)
            assert_current_space(request, space)
            audio = await document_media.media_parser.decode_audio(raw, result.filename)
            if (digest(audio) != metadata['audio_wav_sha256']
                    or str(len(audio)) != metadata['audio_wav_bytes']):
                raise InvalidInput('decoded audio does not match the retained transcription input')
            current = await engine.episode(space, episode_id)
            linked = {attachment.attachment_id for attachment in current.attachments}
            if (current.kind != 'file'
                    or current.metadata.get('document_original') != result.original.attachment_id
                    or current.metadata.get('document_manifest') != result.manifest.attachment_id
                    or current.metadata.get('document_format') != result.format
                    or not {result.original.attachment_id, result.manifest.attachment_id} <= linked
                    or current.content != '\n\n'.join(s.text for s in result.segments)):
                raise InvalidInput('document changed during audio verification')
            assert_current_space(request, space)
        return Response(audio, media_type='audio/wav', headers={
            'Cache-Control': 'no-store', 'X-Content-Type-Options': 'nosniff',
            'Content-Disposition': 'attachment; filename="transcription-audio.wav"',
        })
