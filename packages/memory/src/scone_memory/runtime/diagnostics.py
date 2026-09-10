"""Bounded local diagnostics: timings and identifiers, never request content."""

from __future__ import annotations

import json
import logging
import os
import stat
import time
from contextvars import ContextVar
from datetime import datetime, timezone
from logging.handlers import RotatingFileHandler
from pathlib import Path
from uuid import uuid4

from starlette.types import ASGIApp, Message, Receive, Scope, Send

context: ContextVar[dict[str, str]] = ContextVar("scone_diagnostics", default={})
_FIELDS = frozenset({
    "event", "request_id", "session_id", "call_id", "model_name", "mode", "outcome",
    "elapsed_ms", "first_token_ms", "exception_type", "timeout_s", "stage", "role",
    "episode_id", "context_bytes", "reference_count", "status_code", "method", "route",
    "episodes", "proposed", "accepted", "rejected", "parked", "failed", "expired",
    "records_considered", "has_more",
    "protocol", "phase", "failure_kind", "headers_ms", "http_status",
    "request_bytes", "response_bytes", "output_token_limit", "finish_reason",
    "prompt_tokens", "completion_tokens", "total_tokens",
})


class DiagnosticFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, object] = {
            "time": datetime.fromtimestamp(record.created, timezone.utc).isoformat(),
            "level": record.levelname, "logger": record.name,
        }
        values = {**context.get(), **record.__dict__}
        for key in _FIELDS:
            value = values.get(key)
            if isinstance(value, (str, int, float, bool)):
                payload[key] = value[:256] if isinstance(value, str) else value
        # Arbitrary existing messages, exception strings and stack locals are
        # deliberately excluded: providers may embed credentials or source text.
        payload.setdefault("event", "diagnostic")
        return json.dumps(payload, ensure_ascii=True, allow_nan=False)


class PrivateRotatingHandler(RotatingFileHandler):
    def _open(self):
        descriptor = os.open(self.baseFilename, os.O_WRONLY | os.O_APPEND | os.O_CREAT | os.O_NOFOLLOW, 0o600)
        try:
            info = os.fstat(descriptor)
            if not stat.S_ISREG(info.st_mode) or info.st_nlink != 1:
                raise ValueError("diagnostic log must be a regular, unlinked file")
            os.fchmod(descriptor, 0o600)
            return os.fdopen(descriptor, "a", encoding="utf-8")
        except BaseException:
            os.close(descriptor)
            raise


def configure_diagnostics(path: str | None) -> None:
    """Opt-in private JSONL; five 5 MiB backups, no third-party HTTP logging."""
    logger = logging.getLogger("scone_memory")
    existing = [handler for handler in logger.handlers if isinstance(handler, PrivateRotatingHandler)]
    if path:
        target = Path(path).expanduser().absolute()
        target.parent.mkdir(parents=True, exist_ok=True)
        if len(existing) == 1 and existing[0].baseFilename == str(target):
            return
        handler = PrivateRotatingHandler(target, maxBytes=5 * 1024 * 1024, backupCount=5)
        handler.setFormatter(DiagnosticFormatter())
    for previous in existing:
        logger.removeHandler(previous)
        previous.close()
    if path:
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)


class HttpDiagnostics:
    def __init__(self, app: ASGIApp):
        self.app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return
        token = context.set({"request_id": uuid4().hex})
        started, status, error_type = time.perf_counter(), 500, None

        async def observed(message: Message) -> None:
            nonlocal status
            if message["type"] == "http.response.start":
                status = message["status"]
            await send(message)

        try:
            await self.app(scope, receive, observed)
        except BaseException as error:
            error_type = type(error).__name__
            raise
        finally:
            elapsed = round((time.perf_counter() - started) * 1000, 3)
            # Successful polling stays quiet; writes, failures and slow reads
            # retain evidence. Route templates exclude IDs, queries and keys.
            if scope["method"] not in {"GET", "HEAD"} or status >= 400 or elapsed >= 1000:
                logging.getLogger(__name__).log(logging.WARNING if status >= 500 else logging.INFO,
                    "http_request.finished", extra={"event": "http_request.finished", "method": scope["method"],
                    "route": getattr(scope.get("route"), "path", "unmatched"), "status_code": status,
                    "elapsed_ms": elapsed, "exception_type": error_type})
            context.reset(token)


def install_http_diagnostics(app) -> None:
    app.add_middleware(HttpDiagnostics)
