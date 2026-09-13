"""Read exact retained video frame pixels under current source authorization."""
from __future__ import annotations

from collections.abc import Callable
from typing import AsyncContextManager

from fastapi import Depends, FastAPI, Path, Request
from fastapi.responses import Response

from ..core.errors import InvalidInput
from ..ingestion.document_video import DocumentVideo
from ..ingestion.video_source import read_video_source, verify_video_source
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
            source = await read_video_source(engine, space, episode_id, ordinal)
            assert_current_space(request, space)
            retained = source.evidence.video
            assert retained is not None  # read_video_source requires video evidence.
            selected = await document_video.parser.read_frame(source.raw, source.evidence.filename, retained, ordinal,
                                                               limits=DocumentLimits())
            assert_current_space(request, space)
            await verify_video_source(engine, space, episode_id, source)
            assert_current_space(request, space)
        return Response(selected.png, media_type='image/png', headers={
            'Cache-Control': 'no-store', 'X-Content-Type-Options': 'nosniff',
            'Content-Disposition': f'inline; filename="video-frame-{ordinal}.png"',
            'X-Scone-Video-Frame-SHA256': selected.sha256,
            'X-Scone-Video-Frame-Ordinal': str(ordinal),
            'X-Scone-Video-PTS': str(selected.selection.presentation_timestamp),
            'X-Scone-Video-Time-Base': retained.time_base,
        })
