"""Authenticated ephemeral dictation; transcripts are returned, never stored."""
import asyncio
from collections.abc import Awaitable, Callable

from fastapi import Depends, FastAPI, Request
from fastapi.responses import JSONResponse

from ..agents.workflow import WorkflowError
from ..speech.dictation import LocalDictation, MAX_AUDIO_BYTES, MAX_SECONDS, validate_dictation_format


async def _transcribe_connected(request: Request, service: LocalDictation,
                                content: bytes, media_type: str, language: str) -> str:
    async def disconnected() -> None:
        while not await request.is_disconnected():
            await asyncio.sleep(0.1)

    work = asyncio.create_task(service.transcribe(content, media_type, language))
    watcher = asyncio.create_task(disconnected())
    try:
        async with asyncio.timeout(120):
            await asyncio.wait({work, watcher}, return_when=asyncio.FIRST_COMPLETED)
            if watcher.done():
                watcher.result()
                raise WorkflowError('dictation_cancelled')
            return work.result()
    finally:
        async def join() -> None:
            for task in (work, watcher):
                if not task.done():
                    task.cancel()
            await asyncio.gather(work, watcher, return_exceptions=True)
        cleanup = asyncio.create_task(join())
        interrupted = False
        while not cleanup.done():
            try:
                await asyncio.shield(cleanup)
            except asyncio.CancelledError:
                interrupted = True
        cleanup.result()
        if interrupted:
            raise asyncio.CancelledError()


def mount_dictation_routes(app: FastAPI, service: LocalDictation | None,
                           space_for: Callable[..., Awaitable[str]]) -> None:
    admission = asyncio.Semaphore(4)

    def response(value: object, status: int = 200) -> JSONResponse:
        return JSONResponse(value, status_code=status, headers={'Cache-Control': 'no-store'})

    @app.get('/v1/dictation/status')
    async def status(space: str = Depends(space_for)) -> JSONResponse:
        return response({'available': service is not None and service.available, 'mode': 'local',
                         'max_duration_seconds': MAX_SECONDS, 'max_bytes': MAX_AUDIO_BYTES})

    @app.post('/v1/dictation/transcribe')
    async def transcribe(request: Request, language: str = 'en', space: str = Depends(space_for)) -> JSONResponse:
        if admission.locked():
            return response({'error': 'dictation_busy', 'code': 'dictation_busy'}, 429)
        await admission.acquire()
        try:
            media_type = request.headers.get('content-type', '').split(';')[0].strip().lower()
            validate_dictation_format(media_type, language)
            if service is None or not service.available:
                raise WorkflowError('dictation_unavailable')
            declared = request.headers.get('content-length', '')
            if declared.isdigit() and int(declared) > MAX_AUDIO_BYTES:
                raise WorkflowError('dictation_request_limit')
            content = bytearray()
            async with asyncio.timeout(15):
                async for chunk in request.stream():
                    if len(content) + len(chunk) > MAX_AUDIO_BYTES:
                        raise WorkflowError('dictation_request_limit')
                    content.extend(chunk)
            if not content:
                raise WorkflowError('invalid_dictation_size')
            if await space_for(request) != space:
                raise WorkflowError('dictation_scope_changed')
            text = await _transcribe_connected(request, service, bytes(content), media_type, language)
            if await space_for(request) != space:
                raise WorkflowError('dictation_scope_changed')
            return response({'text': text, 'language': language, 'mode': 'local'})
        except WorkflowError as error:
            code = error.code
            status = 499 if code == 'dictation_cancelled' else 403 if code == 'dictation_scope_changed' else 413 if code == 'dictation_request_limit' else 429 if code == 'dictation_busy' else 422 if code.startswith('invalid_') else 503
            return response({'error': code, 'code': code}, status)
        except TimeoutError:
            return response({'error': 'dictation_timeout', 'code': 'dictation_timeout'}, 408)

        finally:
            admission.release()
