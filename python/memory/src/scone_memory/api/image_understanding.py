"""Explicit image interpretation of an authorized retained source; no writes."""

from __future__ import annotations

import asyncio
import json
import logging
import re
import time
from collections.abc import Callable
from dataclasses import asdict

from fastapi import Depends, FastAPI, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator

from ..core.errors import Gone, NotFound
from ..memory.engine import MemoryEngine
from ..providers.llm import ChatError
from ..providers.vision import VisionModel


class _Task(BaseModel):
    model_config = ConfigDict(strict=True, extra='forbid', hide_input_in_errors=True)
    prompt: str = Field(min_length=1, max_length=16000)

    @field_validator('prompt')
    @classmethod
    def nonblank(cls, value: str) -> str:
        if not value.strip() or '\x00' in value:
            raise ValueError('prompt must contain text')
        return value


def mount_image_understanding_routes(app: FastAPI, engine: MemoryEngine,
                                     space_for: Callable[..., object],
                                     vision_factory: Callable[[], VisionModel | None]) -> None:
    """The host dependency authorizes an inference action and returns its space.

    A fresh provider is selected on each explicit request, so settings changes
    affect subsequent requests. The generated result is not stored or approved.
    """
    if not callable(space_for) or not callable(vision_factory):
        raise ValueError('space authorization and a vision factory are required')
    slot = asyncio.Semaphore(1)
    logger = logging.getLogger(__name__)

    @app.post('/v1/episodes/{episode_id}/attachments/{attachment_id}/understand')
    async def understand(episode_id: int, attachment_id: str, request: Request,
                         space: str = Depends(space_for)) -> JSONResponse:
        if not 0 < episode_id <= 2**53 - 1 or not re.fullmatch(r'[0-9a-f]{64}', attachment_id):
            return JSONResponse({'error': 'Invalid source image address'}, status_code=400)
        content = bytearray()
        async for part in request.stream():
            if len(content) + len(part) > 68000:
                return JSONResponse({'error': 'Image task exceeds its size limit'}, status_code=400)
            content.extend(part)
        try:
            task = _Task.model_validate(json.loads(content))
        except (ValueError, ValidationError):
            return JSONResponse({'error': 'Provide a nonempty image task of at most 16000 characters'}, status_code=400)
        try:
            episode = await engine.episode(space, episode_id)
            if attachment_id not in {item.attachment_id for item in episode.attachments}:
                raise NotFound('Image is not attached to this source')
            attachment, data = await engine.attachment(space, attachment_id)
        except Gone:
            return JSONResponse({'error': 'This source was forgotten'}, status_code=410)
        except NotFound:
            return JSONResponse({'error': 'Source image unavailable in this space'}, status_code=404)
        if slot.locked():
            return JSONResponse({'error': 'Image understanding is busy; retry after the current request'}, status_code=429)
        async with slot:
            try:
                vision = vision_factory()
            except (RuntimeError, ValueError, ImportError):
                return JSONResponse({'error': 'The configured self-hosted vision model is unavailable'}, status_code=503)
            if vision is None:
                return JSONResponse({'error': 'Configure an image-capable self-hosted model in Models first'}, status_code=503)
            source = episode.source or f'episode:{episode_id}'
            started = time.perf_counter()
            model_name = getattr(vision, 'model', type(vision).__name__)
            outcome, exception_type = 'failed', None
            logger.info('Image understanding started', extra={'event': 'image_understanding.started',
                        'episode_id': episode_id, 'model_name': model_name})
            try:
                try:
                    result = await vision.describe(data, attachment.media_type, prompt=task.prompt,
                                                   source=source, attachment_id=attachment_id)
                except ValueError as error:
                    outcome, exception_type = 'invalid_image', type(error).__name__
                    return JSONResponse({'error': 'This image cannot be understood; use a valid still PNG, JPEG or WebP within the configured limits'}, status_code=400)
                except ImportError as error:
                    outcome, exception_type = 'unavailable', type(error).__name__
                    return JSONResponse({'error': 'Image understanding dependencies are unavailable on this server'}, status_code=503)
                except (ChatError, RuntimeError) as error:
                    exception_type = type(error).__name__
                    return JSONResponse({'error': 'The self-hosted vision model did not return a complete description'}, status_code=502)
                except asyncio.CancelledError:
                    outcome, exception_type = 'cancelled', 'CancelledError'
                    raise
                if result.attachment_id != attachment_id or result.source != source or result.media_type != attachment.media_type:
                    outcome = 'source_mismatch'
                    return JSONResponse({'error': 'Vision result did not match the retained source image'}, status_code=502)
                outcome = 'completed'
            finally:
                logger.info('Image understanding finished', extra={'event': 'image_understanding.finished',
                            'episode_id': episode_id, 'model_name': model_name, 'outcome': outcome,
                            'exception_type': exception_type, 'elapsed_ms': round((time.perf_counter()-started)*1000, 3)})
        return JSONResponse({'schema_version': 1, 'episode_id': episode_id,
                             'attachment_id': attachment_id, 'understanding': asdict(result), 'persisted': False})
