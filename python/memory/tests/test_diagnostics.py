import json
import logging
import stat

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from scone_memory.runtime.diagnostics import (
    DiagnosticFormatter, PrivateRotatingHandler, configure_diagnostics, context, install_http_diagnostics,
)


def test_formatter_excludes_messages_exceptions_and_unapproved_fields():
    record = logging.LogRecord("scone_memory.test", logging.ERROR, "", 1, "private key=%s", ("SECRET",), None)
    record.event = "capture.failed"
    record.exception_type = "ReadTimeout"
    record.prompt = "PRIVATE PROMPT"
    token = context.set({"request_id": "request-one", "session_id": "session-one"})
    try:
        line = DiagnosticFormatter().format(record)
    finally:
        context.reset(token)
    result = json.loads(line)
    assert result["request_id"] == "request-one" and result["session_id"] == "session-one"
    assert result["exception_type"] == "ReadTimeout"
    assert all(secret not in line for secret in ("SECRET", "private", "PRIVATE", "prompt"))


def test_logs_rotate_privately_and_configuration_is_idempotent(tmp_path):
    path = tmp_path / "diagnostics.jsonl"
    logger = logging.getLogger("scone_memory")
    old_level = logger.level
    try:
        configure_diagnostics(str(path))
        configure_diagnostics(str(path))
        [handler] = [item for item in logger.handlers if isinstance(item, PrivateRotatingHandler)]
        handler.maxBytes = 300
        for _ in range(10):
            logger.info("PRIVATE CONTENT", extra={"event": "test.finished", "elapsed_ms": 12.0})
        files = list(tmp_path.glob("diagnostics.jsonl*"))
        assert 1 < len(files) <= 6
        for saved in files:
            assert stat.S_IMODE(saved.stat().st_mode) == 0o600
            assert "PRIVATE" not in saved.read_text()
            assert all(json.loads(line)["event"] == "test.finished" for line in saved.read_text().splitlines())
    finally:
        configure_diagnostics(None)
        logger.setLevel(old_level)


def test_symlink_log_is_rejected_without_touching_target(tmp_path):
    target = tmp_path / "secret"
    target.write_text("unchanged")
    link = tmp_path / "diagnostics.jsonl"
    link.symlink_to(target)
    with pytest.raises(OSError):
        configure_diagnostics(str(link))
    assert target.read_text() == "unchanged"


def test_http_logs_template_and_correlation_without_url_or_body(caplog):
    caplog.set_level(logging.INFO, logger="scone_memory")
    app = FastAPI()
    install_http_diagnostics(app)

    @app.post("/items/{item_id}")
    async def item(item_id: str):
        return {"ok": True}

    with TestClient(app) as client:
        assert client.post("/items/PRIVATE-ID?key=SECRET", json={"text": "PRIVATE-TEXT"}).status_code == 200
    [record] = [record for record in caplog.records if getattr(record, "event", None) == "http_request.finished"]
    assert record.route == "/items/{item_id}" and record.status_code == 200
    assert record.elapsed_ms >= 0
    assert not context.get(), "the request context is restored"
    assert "PRIVATE" not in DiagnosticFormatter().format(record)
