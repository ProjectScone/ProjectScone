"""Session recall constraints must survive retries, listings and recreation."""
import sqlite3

import pytest

from scone_memory.errors import Conflict, InvalidInput, NotFound
from scone_memory.session_journal import SessionJournal


def test_scope_is_copied_persisted_and_bound_to_the_create_request(tmp_path):
    path = tmp_path / "sessions.db"
    requested = {"where": {"collection": "manuals"}, "kind": "file", "source_prefix": "docs/",
                 "since": "2026-09-01T01:00:00+01:00"}
    expected = {"where": {"collection": "manuals"}, "kind": "file", "source_prefix": "docs/",
                "since": "2026-09-01T00:00:00.000Z"}
    with SessionJournal(path) as journal:
        receipt = journal.create("alpha", "new", recall_scope=requested)
        sid = receipt["session_id"]
        requested["where"]["collection"] = "private"
        assert journal.get("alpha", sid)["recall_scope"] == expected
        assert journal.create("alpha", "new", recall_scope=expected) == receipt
        with pytest.raises(Conflict):
            journal.create("alpha", "new", recall_scope=requested)
        with pytest.raises(Conflict):
            journal.create("alpha", "new")
        assert len(journal.events("alpha", sid)["events"]) == 1
    with SessionJournal(path) as journal:
        assert journal.sessions("alpha")["items"][0]["recall_scope"] == expected
        assert journal.create("alpha", "new", recall_scope=expected) == receipt
        with pytest.raises(NotFound):
            journal.get("beta", sid)


def test_empty_source_prefix_is_preserved_as_a_real_constraint(tmp_path):
    # Native recall distinguishes any present source (including "") from None.
    with SessionJournal(tmp_path / "sessions.db") as journal:
        saved = journal.create("alpha", "new", recall_scope={"source_prefix": ""})
        assert journal.get("alpha", saved["session_id"])["recall_scope"] == {"source_prefix": ""}
        with pytest.raises(Conflict):
            journal.create("alpha", "new", recall_scope={})


@pytest.mark.parametrize("scope", [
    {"space": "beta"}, {"kind": "invented"}, {"where": {"team": 7}},
    {"since": "not-a-date"}, {"since": "2026-09-02", "until": "2026-09-01"},
    {"source_prefix": False}, {"where": []}, [],
])
def test_invalid_scope_creates_no_session(tmp_path, scope):
    with SessionJournal(tmp_path / "sessions.db") as journal:
        with pytest.raises(InvalidInput):
            journal.create("alpha", "new", recall_scope=scope)
        assert journal.sessions("alpha")["items"] == []


@pytest.mark.parametrize("version", [1, 2])
def test_legacy_journal_migrates_without_changing_create_replays(tmp_path, version):
    # Construct the historical schema directly, not a newer schema relabelled v1.
    path = tmp_path / "sessions.db"
    with sqlite3.connect(path) as db:
        db.executescript('''
            PRAGMA application_id=1396918833;
            CREATE TABLE sessions (space TEXT NOT NULL, session_id TEXT NOT NULL,
                create_key TEXT NOT NULL, mode TEXT NOT NULL, state TEXT NOT NULL,
                revision INTEGER NOT NULL, created_at TEXT NOT NULL, updated_at TEXT NOT NULL,
                PRIMARY KEY(space,session_id), UNIQUE(space,create_key));
            CREATE TABLE session_events (space TEXT NOT NULL, session_id TEXT NOT NULL,
                revision INTEGER NOT NULL, request_id TEXT NOT NULL, signature TEXT NOT NULL,
                receipt TEXT NOT NULL, PRIMARY KEY(space,session_id,revision),
                UNIQUE(space,session_id,request_id));
            INSERT INTO sessions VALUES ('alpha','old','old-create','text','created',1,'then','then');
            INSERT INTO session_events VALUES ('alpha','old',1,'old-create','["create", "text"]','{"session_id":"old"}');
        ''')
        if version == 2:
            db.execute('''CREATE TABLE session_turns (space TEXT NOT NULL, session_id TEXT NOT NULL,
                request_id TEXT NOT NULL, signature TEXT NOT NULL, status TEXT NOT NULL,
                episode_id INTEGER, error TEXT, accepted_at TEXT NOT NULL, settled_at TEXT,
                PRIMARY KEY(space,session_id,request_id))''')
            db.execute("INSERT INTO session_turns VALUES ('alpha','old','turn','{}','completed',7,NULL,'then','then')")
        db.execute(f"PRAGMA user_version={version}")
    with SessionJournal(path) as journal:
        assert journal.get("alpha", "old")["recall_scope"] == {}
        assert journal.create("alpha", "old-create", recall_scope={}) == {"session_id": "old"}
        with pytest.raises(Conflict):
            journal.create("alpha", "old-create", recall_scope={"kind": "file"})
        if version == 2:
            assert journal.turn("alpha", "old", "turn")["episode_id"] == 7
        assert journal.create("alpha", "new", recall_scope={"kind": "file"})["state"] == "created"
