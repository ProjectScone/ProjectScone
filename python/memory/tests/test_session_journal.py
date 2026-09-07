"""Real SQLite lifecycle receipts; no provider or live memory fixtures."""

import sqlite3
from concurrent.futures import ThreadPoolExecutor
from threading import Barrier

import pytest

from scone_memory.errors import Conflict, InvalidInput, NotFound
from scone_memory.session_journal import SessionJournal


def test_lifecycle_and_ordered_receipts_survive_reopen(tmp_path):
    path = tmp_path / "sessions.db"
    with SessionJournal(path) as journal:
        created = journal.create("alpha", "create-1")
        sid = created["session_id"]
        running = journal.transition("alpha", sid, "start-1", "start", 1)
        assert running["state"] == "running" and running["revision"] == 2
    with SessionJournal(path) as journal:
        assert journal.get("alpha", sid)["state"] == "running"
        stopping = journal.transition("alpha", sid, "stop-1", "stop", 2)
        assert stopping["state"] == "stopping"
        ended = journal.transition("alpha", sid, "end-1", "end", 3)
        assert ended["state"] == "ended"
        first = journal.events("alpha", sid, limit=2)
        assert [item["revision"] for item in first["events"]] == [1, 2]
        assert first["next_after"] == 2 and first["has_more"] is True
        second = journal.events("alpha", sid, after=2, limit=2)
        assert [item["revision"] for item in second["events"]] == [3, 4]
        assert second["has_more"] is False


def test_retries_return_original_receipts_even_after_state_changes(tmp_path):
    with SessionJournal(tmp_path / "sessions.db") as journal:
        created = journal.create("alpha", "create-1")
        sid = created["session_id"]
        running = journal.transition("alpha", sid, "start-1", "start", 1)
        journal.transition("alpha", sid, "end-1", "end", 2)
        assert journal.create("alpha", "create-1") == created
        assert journal.transition("alpha", sid, "start-1", "start", 1) == running
        with pytest.raises(Conflict):
            journal.create("alpha", "create-1", mode="voice")
        with pytest.raises(Conflict):
            journal.transition("alpha", sid, "start-1", "stop", 1)
        with pytest.raises(Conflict):
            journal.transition("alpha", sid, "start-1", "start", 2)
        assert len(journal.events("alpha", sid)["events"]) == 3


def test_scope_does_not_leak_sessions_or_reuse_another_spaces_key(tmp_path):
    with SessionJournal(tmp_path / "sessions.db") as journal:
        created = journal.create("alpha", "shared-key")
        other = journal.create("beta", "shared-key")
        assert created["session_id"] != other["session_id"]
        sid = created["session_id"]
        for operation in [lambda: journal.get("beta", sid),
                          lambda: journal.events("beta", sid),
                          lambda: journal.transition("beta", sid, "start", "start", 1)]:
            with pytest.raises(NotFound):
                operation()


@pytest.mark.parametrize("terminal_action,expected", [("stop", "ended"), ("fail", "failed"), ("interrupt", "interrupted")])
def test_created_can_terminate_but_terminal_session_cannot_restart(tmp_path, terminal_action, expected):
    with SessionJournal(tmp_path / "sessions.db") as journal:
        sid = journal.create("alpha", "new")["session_id"]
        assert journal.transition("alpha", sid, "finish", terminal_action, 1)["state"] == expected
        with pytest.raises(Conflict):
            journal.transition("alpha", sid, "restart", "start", 2)
        assert len(journal.events("alpha", sid)["events"]) == 2


def test_two_connections_reject_stale_revision_without_appending(tmp_path):
    path = tmp_path / "sessions.db"
    with SessionJournal(path) as first, SessionJournal(path) as second:
        sid = first.create("alpha", "new")["session_id"]
        first.transition("alpha", sid, "start", "start", 1)
        with pytest.raises(Conflict):
            second.transition("alpha", sid, "stale-stop", "stop", 1)
        assert second.get("alpha", sid)["revision"] == 2
        second.transition("alpha", sid, "stop", "stop", 2)
        assert first.get("alpha", sid)["state"] == "stopping"


def test_unrelated_database_is_refused_without_changing_it(tmp_path):
    path = tmp_path / "memory.db"
    with sqlite3.connect(path) as db:
        db.execute("CREATE TABLE valuable_data (value TEXT)")
        db.execute("INSERT INTO valuable_data VALUES ('keep me')")
    before = path.read_bytes()
    with pytest.raises(InvalidInput, match="journal"):
        SessionJournal(path)
    assert path.read_bytes() == before


def test_newer_schema_is_refused(tmp_path):
    path = tmp_path / "sessions.db"
    with SessionJournal(path):
        pass
    with sqlite3.connect(path) as db:
        db.execute("PRAGMA user_version=999")
    with pytest.raises(InvalidInput, match="version"):
        SessionJournal(path)


@pytest.mark.parametrize("schema", [
    "CREATE VIEW existing_report AS SELECT 'keep me' AS value",
    "CREATE TABLE sqlitex_owned_by_other_app (value TEXT)",
])
def test_unowned_schema_objects_are_refused_without_changing_bytes(tmp_path, schema):
    path = tmp_path / "other-app.db"
    with sqlite3.connect(path) as db:
        db.execute(schema)
    before = path.read_bytes()
    with pytest.raises(InvalidInput, match="refusing"):
        SessionJournal(path)
    assert path.read_bytes() == before


@pytest.mark.parametrize("after,limit", [(-1, 10), (True, 10), (0, 0), (0, 201), (0, True)])
def test_invalid_replay_bounds_are_rejected(tmp_path, after, limit):
    with SessionJournal(tmp_path / "sessions.db") as journal:
        sid = journal.create("alpha", "new")["session_id"]
        with pytest.raises(InvalidInput):
            journal.events("alpha", sid, after=after, limit=limit)


def test_invalid_commands_and_keys_do_not_create_events(tmp_path):
    with SessionJournal(tmp_path / "sessions.db") as journal:
        for key in ["", "a" * 129, "contains secret prose", None]:
            with pytest.raises(InvalidInput):
                journal.create("alpha", key)
        with pytest.raises(InvalidInput):
            journal.create("alpha", "new", mode="invented")
        sid = journal.create("alpha", "new")["session_id"]
        with pytest.raises(InvalidInput):
            journal.transition("alpha", sid, "bad", "invented", 1)
        with pytest.raises(InvalidInput):
            journal.transition("alpha", sid, "bad", "start", True)
        with pytest.raises(Conflict):
            journal.transition("alpha", sid, "bad", "end", 1)
        assert journal.get("alpha", sid)["revision"] == 1


def test_failed_event_insert_rolls_back_session_revision(tmp_path):
    path = tmp_path / "sessions.db"
    with SessionJournal(path) as journal:
        sid = journal.create("alpha", "new")["session_id"]
        with sqlite3.connect(path) as db:
            db.execute("CREATE TRIGGER simulate_disk_failure BEFORE INSERT ON session_events WHEN NEW.revision > 1 BEGIN SELECT RAISE(ABORT, 'injected write failure'); END")
        with pytest.raises(sqlite3.DatabaseError):
            journal.transition("alpha", sid, "start", "start", 1)
        assert journal.get("alpha", sid)["revision"] == 1
        assert len(journal.events("alpha", sid)["events"]) == 1


def test_simultaneous_controls_have_one_committed_winner(tmp_path):
    path = tmp_path / "sessions.db"
    with SessionJournal(path) as journal:
        sid = journal.create("alpha", "new")["session_id"]
    barrier = Barrier(2)

    def act(action):
        with SessionJournal(path) as journal:
            barrier.wait(timeout=3)
            try:
                return journal.transition("alpha", sid, action, action, 1)
            except Conflict as conflict:
                return {"conflict_revision": conflict.revision}

    with ThreadPoolExecutor(max_workers=2) as pool:
        results = list(pool.map(act, ["start", "stop"]))
    assert sum("conflict_revision" in result for result in results) == 1
    assert next(result for result in results if "conflict_revision" in result)["conflict_revision"] == 2
    with SessionJournal(path) as journal:
        assert journal.get("alpha", sid)["revision"] == 2
        assert len(journal.events("alpha", sid)["events"]) == 2


def test_simultaneous_create_retries_return_one_identity(tmp_path):
    path = tmp_path / "sessions.db"
    with SessionJournal(path):
        pass
    barrier = Barrier(2)

    def create(_):
        with SessionJournal(path) as journal:
            barrier.wait(timeout=3)
            return journal.create("alpha", "same-request")

    with ThreadPoolExecutor(max_workers=2) as pool:
        results = list(pool.map(create, [0, 1]))
    assert results[0] == results[1]
    with SessionJournal(path) as journal:
        assert len(journal.events("alpha", results[0]["session_id"])["events"]) == 1


def test_session_pages_are_scoped_and_do_not_expose_create_keys(tmp_path):
    with SessionJournal(tmp_path / "sessions.db") as journal:
        expected = sorted(journal.create("alpha", key)["session_id"] for key in ["a", "b"])
        journal.create("beta", "c")
        first = journal.sessions("alpha", limit=1)
        second = journal.sessions("alpha", after=first["next_after"], limit=1)
        assert [first["items"][0]["session_id"], second["items"][0]["session_id"]] == expected
        assert first["has_more"] is True and second["has_more"] is False
        assert all("create_key" not in item for item in [*first["items"], *second["items"]])
