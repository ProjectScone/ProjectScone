"""Latest means acceptance order, never UUID order or a truncated list."""

import pytest

from scone_memory.api.conversations import create_conversation_app
from scone_memory.core.errors import NotFound
from scone_memory.realtime.session_journal import SessionJournal
from test_conversations_api import client_for, engine  # noqa: F401


def test_latest_turn_survives_reopen_and_ignores_uuid_order_clock_ties_and_retries(tmp_path, monkeypatch):
    path = tmp_path / "journal.db"
    monkeypatch.setattr("scone_memory.realtime.session_journal._now", lambda: "2026-09-06T10:00:00.000Z")
    with SessionJournal(path) as journal:
        sid = journal.create("alpha", "create")["session_id"]
        assert journal.latest_turn_id("alpha", sid) is None
        for number in range(205):
            journal.start_turn("alpha", sid, f"z-{number:03}", {"text": "fixture"})
        # A clock adjustment must not make a newly accepted turn look older.
        monkeypatch.setattr("scone_memory.realtime.session_journal._now", lambda: "2026-09-05T10:00:00.000Z")
        journal.start_turn("alpha", sid, "a-newest", {"text": "last"})
        journal.start_turn("alpha", sid, "z-204", {"text": "fixture"})
        assert journal.latest_turn_id("alpha", sid) == "a-newest"
        assert journal.turns("alpha", sid, limit=200)["has_more"] is True
        with pytest.raises(NotFound):
            journal.latest_turn_id("beta", sid)
        with pytest.raises(NotFound):
            journal.latest_turn_id("alpha", "missing")
    with SessionJournal(path) as reopened:
        assert reopened.latest_turn_id("alpha", sid) == "a-newest"


async def test_recreated_api_exposes_latest_without_running_a_model(engine, tmp_path):
    path = tmp_path / "journal.db"
    with SessionJournal(path) as journal:
        sid = journal.create("alpha", "create")["session_id"]
        empty = journal.create("alpha", "empty")["session_id"]
        journal.start_turn("alpha", sid, "z-earlier", {"text": "earlier"})
        journal.start_turn("alpha", sid, "a-newer", {"text": "newer"})
    app = create_conversation_app(engine, {"alpha-key": "alpha", "beta-key": "beta"}, path, None)
    async with client_for(app) as client:
        response = await client.get("/v1/conversations/" + sid)
        assert response.status_code == 200
        assert response.json()["latest_request_id"] == "a-newer"
        assert response.json()["active_request_id"] is None
        assert (await client.get("/v1/conversations/" + empty)).json()["latest_request_id"] is None
        assert (await client.get("/v1/conversations/" + sid, headers={"Authorization": "Bearer beta-key"})).status_code == 404
        capability = (await client.get("/v1/conversations/capabilities")).json()
        assert capability["reply_replay"] == "durable_receipts"
        assert capability["text_configured"] is False
