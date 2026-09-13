"""Lossless browser representation of retained video document evidence."""
from __future__ import annotations

from collections.abc import Callable
from dataclasses import asdict
from typing import AsyncContextManager

from fastapi import Depends, FastAPI, Path, Request
from fastapi.encoders import jsonable_encoder
from fastapi.responses import JSONResponse

from ..core.errors import Gone, InvalidInput, NotFound
from ..ingestion.files import DocumentProvenance, digest, document_provenance
from ..memory.engine import MemoryEngine


def _browser_evidence(evidence: DocumentProvenance) -> dict[str, object]:
    retained = evidence.video
    if retained is None or evidence.parser != 'video-frame-ocr':
        raise InvalidInput('document has no retained video frame evidence')
    video = retained.model_dump(mode='json')
    video['start_timestamp'] = str(retained.start_timestamp)
    video['duration_ticks'] = str(retained.duration_ticks)
    video['frames'] = [dict(frame.model_dump(mode='json'),
                           presentation_timestamp=str(frame.presentation_timestamp))
                       for frame in retained.frames]
    return {**asdict(evidence), 'video': video,
            'download_path': f'/v1/attachments/{evidence.original.attachment_id}'}


def mount_video_catalogue_route(app: FastAPI, engine: MemoryEngine,
                                space_for: Callable[..., object],
                                ingest_slot: Callable[[int], AsyncContextManager[None]], *,
                                assert_current_space: Callable[[Request, str], None]) -> None:
    @app.get('/v1/episodes/{episode_id}/document/video/catalogue')
    async def catalogue(request: Request, episode_id: int = Path(ge=1, le=2**63 - 1),
                        space: str = Depends(space_for)) -> JSONResponse:
        async with ingest_slot(1):
            evidence = await document_provenance(engine, space, episode_id)
            assert_current_space(request, space)
            payload = _browser_evidence(evidence)
            original, raw = await engine.attachment(space, evidence.original.attachment_id)
            manifest, encoded = await engine.attachment(space, evidence.manifest.attachment_id)
            # Finish storage observations with links and the raw source row.
            # Hydration can otherwise retain a row forgotten during a blob read.
            links = await engine.blobs.for_episode(space, episode_id)
            current = await engine.documents.get_episode(space, episode_id)
            if current is None:
                tombstone = await engine.tombstone(space, episode_id)
                if tombstone is not None:
                    raise Gone('This source was forgotten', tombstone.forgotten_at)
                raise NotFound('Video source unavailable in this space')
            assert_current_space(request, space)
            if (current.kind != 'file' or original != evidence.original or manifest != evidence.manifest
                    or digest(raw) != original.attachment_id or digest(encoded) != manifest.attachment_id
                    or current.metadata.get('document_original') != original.attachment_id
                    or current.metadata.get('document_manifest') != manifest.attachment_id
                    or current.metadata.get('document_format') != evidence.format
                    or current.content != '\n\n'.join(segment.text for segment in evidence.segments)
                    or not {original.attachment_id, manifest.attachment_id}
                    <= {item.attachment_id for item in links}):
                raise InvalidInput('video source changed during catalogue verification')
        try:
            return JSONResponse(jsonable_encoder({'schema_version': 1, 'timestamp_encoding': 'decimal-string',
                'space': space, 'episode_id': str(episode_id), 'evidence': payload}), headers={
                    'Cache-Control': 'no-store', 'X-Content-Type-Options': 'nosniff'})
        except UnicodeEncodeError:
            raise InvalidInput('video catalogue contains invalid Unicode text') from None
