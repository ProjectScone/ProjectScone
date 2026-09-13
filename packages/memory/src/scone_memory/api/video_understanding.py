"""Explicit, unsaved interpretations of verified retained video frames."""
from __future__ import annotations

import asyncio
from asyncio import CancelledError, current_task, get_running_loop
from collections.abc import Callable
from dataclasses import asdict
import json
from typing import AsyncContextManager, Literal

from fastapi import Depends, FastAPI, Path, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ConfigDict, Field, field_validator

from ..ingestion.document_video import DocumentVideo
from ..ingestion.formats.types import DocumentLimits
from ..ingestion.video_source import read_video_source, verify_video_source
from ..ingestion.video_frames import VideoFrame
from ..memory.engine import MemoryEngine
from ..providers.llm import ChatError
from ..providers.vision import ImageUnderstanding, VisionModel


class _Task(BaseModel):
    model_config = ConfigDict(strict=True, extra='forbid', hide_input_in_errors=True)
    prompt: str = Field(min_length=1, max_length=16000)

    @field_validator('prompt')
    @classmethod
    def valid_prompt(cls, value: str) -> str:
        if not value.strip() or '\x00' in value:
            raise ValueError('prompt must contain text')
        value.encode('utf-8')
        return value


class _Understanding(BaseModel):
    model_config = ConfigDict(strict=True, extra='forbid')
    text: str = Field(min_length=1, max_length=64000)
    attachment_id: str = Field(pattern=r'^[a-f0-9]{64}$')
    source: str = Field(min_length=1, max_length=4096)
    media_type: Literal['image/png']
    model: str = Field(min_length=1, max_length=256)
    width: int = Field(ge=1, le=100000)
    height: int = Field(ge=1, le=100000)
    origin: Literal['model_generated']

    @field_validator('text', 'model', 'source')
    @classmethod
    def valid_text(cls, value: str) -> str:
        if not value.strip() or '\x00' in value:
            raise ValueError('description must contain text')
        value.encode('utf-8')
        return value


def _response(body: dict[str, object], status: int = 200) -> JSONResponse:
    return JSONResponse(body, status_code=status, headers={'Cache-Control': 'no-store',
                                                         'X-Content-Type-Options': 'nosniff'})


async def _interpret(vision: VisionModel, frame: VideoFrame, prompt: str, locator: str) -> _Understanding:
    declared_model = getattr(vision, 'model', None)
    result = await vision.describe(frame.png, 'image/png', prompt=prompt,
                                   source=locator, attachment_id=frame.sha256)
    if not isinstance(result, ImageUnderstanding):
        raise ChatError('invalid frame description')
    try:
        checked = _Understanding.model_validate(asdict(result))
    except (ValueError, UnicodeError):
        raise ChatError('invalid frame description') from None
    if (checked.attachment_id != frame.sha256 or checked.source != locator
            or checked.width != frame.width or checked.height != frame.height
            or (declared_model is not None and checked.model != declared_model)):
        raise ChatError('frame or model identity changed')
    return checked


def _check_budget(budget: asyncio.Timeout) -> None:
    deadline = budget.when()
    if budget.expired() or (deadline is not None and get_running_loop().time() >= deadline):
        raise TimeoutError
    task = current_task()
    if task is not None and task.cancelling():
        raise CancelledError


def mount_video_understanding_routes(app: FastAPI, engine: MemoryEngine,
        space_for: Callable[..., object], ingest_slot: Callable[[int], AsyncContextManager[None]],
        document_video: DocumentVideo | None, vision_factory: Callable[[], VisionModel | None], *,
        assert_current_space: Callable[[Request, str], None]) -> None:
    @app.post('/v1/episodes/{episode_id}/document/video/frames/{ordinal}/understand')
    async def understand(request: Request, episode_id: int = Path(ge=1, le=2**63 - 1),
                         ordinal: int = Path(ge=0, le=99999), space: str = Depends(space_for)) -> JSONResponse:
        content = bytearray()
        async for part in request.stream():
            if len(content) + len(part) > 200000:
                return _response({'error': 'Video frame task exceeds its size limit'}, 400)
            content.extend(part)
        try:
            task = _Task.model_validate(json.loads(content))
        except (ValueError, UnicodeError, RecursionError):
            return _response({'error': 'Provide a nonempty frame task of at most 16000 characters'}, 400)
        assert_current_space(request, space)
        if document_video is None:
            return _response({'error': 'Video frame decoding is not configured on this server'}, 503)
        async with ingest_slot(1):
            try:
                async with asyncio.timeout(120) as budget:
                    source = await read_video_source(engine, space, episode_id, ordinal)
                    assert_current_space(request, space)
                    try:
                        vision = vision_factory()
                    except (RuntimeError, ValueError, ImportError):
                        return _response({'error': 'The configured self-hosted vision model is unavailable'}, 503)
                    if vision is None:
                        return _response({'error': 'Configure an image-capable self-hosted model in Models first'}, 503)
                    retained = source.evidence.video
                    assert retained is not None
                    selected = await document_video.parser.read_frame(source.raw, source.evidence.filename,
                        retained, ordinal, limits=DocumentLimits())
                    await verify_video_source(engine, space, episode_id, source)
                    assert_current_space(request, space)
                    _check_budget(budget)
                    locator = f'video:episode:{episode_id}/stream:{retained.stream_index}/frame:{ordinal}'
                    try:
                        checked = await _interpret(vision, selected, task.prompt, locator)
                    except ImportError:
                        return _response({'error': 'Vision dependencies are unavailable on this server'}, 503)
                    except (ChatError, RuntimeError, ValueError):
                        return _response({'error': 'The self-hosted vision model did not return a complete, frame-bound interpretation'}, 502)
                    await verify_video_source(engine, space, episode_id, source)
                    assert_current_space(request, space)
                    _check_budget(budget)
                    return _response({'schema_version': 1, 'space': space, 'episode_id': str(episode_id),
                        'original_sha256': source.original.attachment_id, 'manifest_sha256': source.manifest.attachment_id,
                        'frame': {'ordinal': ordinal, 'presentation_timestamp': str(selected.selection.presentation_timestamp),
                                  'time_base': retained.time_base, 'png_sha256': selected.sha256,
                                  'width': selected.width, 'height': selected.height},
                        'understanding': checked.model_dump(), 'persisted': False})
            except TimeoutError:
                return _response({'error': 'Video frame interpretation exceeded its deadline'}, 504)
