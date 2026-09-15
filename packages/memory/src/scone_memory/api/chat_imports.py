"""Chat exports over HTTP: an uploaded export becomes conversation memories.

The export is uploaded first (`POST /v1/attachments`, as a document is),
then imported by its attachment id, so the bytes travel once and the
import request stays a small bounded JSON body. The receipt is the same
one `import-chat` prints: what was stored, what was already known, and
what was counted instead of stored.
"""
from __future__ import annotations

from collections.abc import Callable
from dataclasses import asdict
from typing import AsyncContextManager, Literal

from fastapi import Depends, FastAPI, Request
from fastapi.encoders import jsonable_encoder
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ConfigDict, Field, ValidationError

from ..ingestion.chat_exports import DEFAULT_SESSION_GAP_SECONDS, ingest_chat_export
from ..ingestion.files import extraction_filename
from ..memory.engine import MemoryEngine

#: A day and a half of silence is the longest gap a session may span here.
MAX_GAP_SECONDS = 36 * 3600


class _ImportBody(BaseModel):
    model_config = ConfigDict(strict=True, extra='forbid', hide_input_in_errors=True)
    attachment_id: str = Field(pattern=r'^[a-f0-9]{64}$')
    #: The export's file name, when the upload did not carry one; the
    #: suffix decides how it is read.
    filename: str | None = Field(default=None, min_length=1, max_length=1024)
    chat: str | None = Field(default=None, min_length=1, max_length=200)
    time_zone: str | None = Field(default=None, min_length=1, max_length=64)
    date_order: Literal['day-first', 'month-first'] | None = None
    gap_seconds: int = Field(default=DEFAULT_SESSION_GAP_SECONDS, ge=0, le=MAX_GAP_SECONDS)
    metadata: dict[str, str] = Field(default_factory=dict, max_length=9)


def mount_chat_import_routes(app: FastAPI, engine: MemoryEngine, space_for: Callable[..., object],
                             ingest_slot: Callable[[int], AsyncContextManager[None]], *,
                             assert_current_space: Callable[[Request, str], None]) -> None:
    @app.post('/v1/chat-imports')
    async def import_chat(request: Request, space: str = Depends(space_for)) -> JSONResponse:
        data = bytearray()
        async for part in request.stream():
            if len(data) + len(part) > 8192:
                return JSONResponse({'error': 'chat import request exceeds its byte limit'}, status_code=413)
            data.extend(part)
        try:
            body = _ImportBody.model_validate_json(data)
        except ValidationError:
            return JSONResponse({'error': 'invalid chat import request'}, status_code=400)
        assert_current_space(request, space)
        async with ingest_slot(1):
            original, raw = await engine.attachment(space, body.attachment_id)
            assert_current_space(request, space)
            filename = extraction_filename(original, body.filename)
            imported = await ingest_chat_export(engine, space, raw, filename=filename, chat=body.chat,
                                                time_zone=body.time_zone, date_order=body.date_order,
                                                gap_seconds=body.gap_seconds, metadata=body.metadata)
        assert_current_space(request, space)
        receipt = {key: value for key, value in asdict(imported).items() if key != 'episode_ids'}
        return JSONResponse(jsonable_encoder({**receipt, 'episodes': len(imported.episode_ids),
                                              'attachment_id': body.attachment_id, 'filename': filename}))
