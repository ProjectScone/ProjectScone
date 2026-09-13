"""Read exact retained video frame pixels under current source authorization."""
from __future__ import annotations

from collections.abc import Callable
from typing import AsyncContextManager

from fastapi import Depends, FastAPI, Path, Request
from fastapi.responses import Response

from ..core.errors import Gone, InvalidInput, NotFound
from ..ingestion.document_video import DocumentVideo
from ..ingestion.files import digest, document_provenance
from ..ingestion.formats.types import DocumentLimits
from ..memory.engine import MemoryEngine


def mount_video_frame_routes(app: FastAPI, engine: MemoryEngine,
                             space_for: Callable[..., object],
                             ingest_slot: Callable[[int], AsyncContextManager[None]],
                             document_video: DocumentVideo | None, *,
                             assert_current_space: Callable[[Request, str], None]) -> None:
    @app.get('/v1/episodes/{episode_id}/document/video/frames/{ordinal}')
    async def frame(request: Request, episode_id: int = Path(ge=1, le=2**63 - 1),
                    ordinal: int = Path(ge=0, le=99999), space: str = Depends(space_for)) -> Response:
        if document_video is None:
            raise InvalidInput('video frame decoding is not configured on this server')
        async with ingest_slot(1):
            evidence = await document_provenance(engine, space, episode_id)
            assert_current_space(request, space)
            retained = evidence.video
            if retained is None or evidence.parser != 'video-frame-ocr':
                raise InvalidInput('document has no retained video frame evidence')
            if ordinal not in {item.ordinal for item in retained.frames}:
                raise InvalidInput('requested frame is not in the retained video evidence')
            original, raw = await engine.attachment(space, evidence.original.attachment_id)
            manifest, encoded = await engine.attachment(space, evidence.manifest.attachment_id)
            source = await engine.episode(space, episode_id)
            snapshot = (source.kind, source.source, source.content, dict(source.metadata))
            required_links = {original.attachment_id, manifest.attachment_id}
            if (source.kind != 'file' or original != evidence.original or manifest != evidence.manifest
                    or digest(raw) != original.attachment_id or digest(encoded) != manifest.attachment_id
                    or source.metadata.get('document_original') != original.attachment_id
                    or source.metadata.get('document_manifest') != manifest.attachment_id
                    or source.metadata.get('document_format') != evidence.format
                    or source.content != '\n\n'.join(segment.text for segment in evidence.segments)
                    or not required_links <= {item.attachment_id for item in source.attachments}):
                raise InvalidInput('video document changed before frame decoding')
            assert_current_space(request, space)
            selected = await document_video.parser.read_frame(raw, evidence.filename, retained, ordinal,
                                                               limits=DocumentLimits())
            assert_current_space(request, space)
            # Hydration itself awaits storage. Read the raw source row last so a
            # forget during hydration cannot leave a stale source snapshot valid.
            await engine.episode(space, episode_id)
            current_original, current_raw = await engine.attachment(space, original.attachment_id)
            current_manifest, current_encoded = await engine.attachment(space, manifest.attachment_id)
            await engine.episode(space, episode_id)
            links = await engine.blobs.for_episode(space, episode_id)
            current = await engine.documents.get_episode(space, episode_id)
            if current is None:
                tombstone = await engine.tombstone(space, episode_id)
                if tombstone is not None:
                    raise Gone('This source was forgotten', tombstone.forgotten_at)
                raise NotFound('Video source unavailable in this space')
            assert_current_space(request, space)
            if (not required_links <= {item.attachment_id for item in links}
                    or (current.kind, current.source, current.content, current.metadata) != snapshot
                    or current_original != original or current_manifest != manifest
                    or current_raw != raw or current_encoded != encoded):
                raise InvalidInput('video source changed during frame verification')
        return Response(selected.png, media_type='image/png', headers={
            'Cache-Control': 'no-store', 'X-Content-Type-Options': 'nosniff',
            'Content-Disposition': f'inline; filename="video-frame-{ordinal}.png"',
            'X-Scone-Video-Frame-SHA256': selected.sha256,
            'X-Scone-Video-Frame-Ordinal': str(ordinal),
            'X-Scone-Video-PTS': str(selected.selection.presentation_timestamp),
            'X-Scone-Video-Time-Base': retained.time_base,
        })
