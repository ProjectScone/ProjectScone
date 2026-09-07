"""Turn receipts that outlive the process that made them.

Codex's conversation service keeps a turn's result in a per-process dict,
so a restart loses every receipt and a client that reconnects cannot find
out what happened to the turn it sent. These are the durable half: the
journal records that a turn was accepted, and later what became of it,
and a receipt names the episode its text lives in rather than carrying a
second copy of that text.
"""

import sqlite3

import pytest

from scone_memory.core.errors import Conflict, NotFound
from scone_memory.realtime.session_journal import SessionJournal


def running_session(journal, space="alpha"):
    created = journal.create(space, "create-1")
    sid = created["session_id"]
    journal.transition(space, sid, "start-1", "start", created["revision"])
    return sid


def test_a_turn_receipt_survives_the_process_that_made_it(tmp_path):
    path = tmp_path / "sessions.db"
    with SessionJournal(path) as journal:
        sid = running_session(journal)
        accepted = journal.start_turn("alpha", sid, "turn-1", {"text": "hello"})
        assert accepted["status"] == "accepted" and accepted["episode_id"] is None
        journal.finish_turn("alpha", sid, "turn-1", "completed", episode_id=7)

    with SessionJournal(path) as journal:
        found = journal.turn("alpha", sid, "turn-1")
        assert (found["status"], found["episode_id"]) == ("completed", 7)
        assert [t["request_id"] for t in journal.turns("alpha", sid)["turns"]] == ["turn-1"]


def test_a_receipt_names_the_episode_instead_of_copying_its_text(tmp_path):
    """Forgetting an episode has to remove the text everywhere. A receipt
    that carried its own copy would leave the transcript readable after
    the memory holding it was deleted."""
    with SessionJournal(tmp_path / "sessions.db") as journal:
        sid = running_session(journal)
        journal.start_turn("alpha", sid, "turn-1", {"text": "what did I say about Lisbon"})
        finished = journal.finish_turn("alpha", sid, "turn-1", "completed", episode_id=11)

    assert finished["episode_id"] == 11
    assert "Lisbon" not in repr(finished)


def test_a_turn_left_pending_by_a_crash_is_not_reported_as_still_running(tmp_path):
    """The process that owned the turn is gone, so whether the provider
    answered is unknowable. Recovery says exactly that rather than leaving
    a receipt that reads as in flight forever, and it does not touch a
    turn that already finished."""
    path = tmp_path / "sessions.db"
    with SessionJournal(path) as journal:
        sid = running_session(journal)
        journal.start_turn("alpha", sid, "turn-1", {"text": "one"})
        journal.start_turn("alpha", sid, "turn-2", {"text": "two"})
        journal.finish_turn("alpha", sid, "turn-2", "completed", episode_id=3)

    with SessionJournal(path) as journal:
        assert journal.recover("alpha") == 1
        lost = journal.turn("alpha", sid, "turn-1")
        assert lost["status"] == "interrupted"
        assert "did not finish" in lost["error"]
        assert journal.turn("alpha", sid, "turn-2")["status"] == "completed"
        assert journal.recover("alpha") == 0  # nothing left to recover


def test_a_retried_turn_returns_its_first_receipt_and_a_changed_one_conflicts(tmp_path):
    """The idempotency the in-process dict gave, kept across a restart."""
    path = tmp_path / "sessions.db"
    with SessionJournal(path) as journal:
        sid = running_session(journal)
        journal.start_turn("alpha", sid, "turn-1", {"text": "hello"})
        journal.finish_turn("alpha", sid, "turn-1", "completed", episode_id=5)

    with SessionJournal(path) as journal:
        again = journal.start_turn("alpha", sid, "turn-1", {"text": "hello"})
        assert (again["status"], again["episode_id"]) == ("completed", 5)
        with pytest.raises(Conflict):
            journal.start_turn("alpha", sid, "turn-1", {"text": "something else"})


def test_turns_do_not_cross_spaces_or_sessions(tmp_path):
    with SessionJournal(tmp_path / "sessions.db") as journal:
        mine = running_session(journal, "alpha")
        journal.start_turn("alpha", mine, "turn-1", {"text": "mine"})
        with pytest.raises(NotFound):
            journal.turn("beta", mine, "turn-1")
        # Two different absences, said differently: a client that asked
        # about the wrong session should not be told its turn is missing.
        with pytest.raises(NotFound, match="session not found"):
            journal.turn("alpha", "conversation-nobody-has", "turn-1")
        with pytest.raises(NotFound, match="turn not found"):
            journal.turn("alpha", mine, "turn-nobody-sent")


def test_a_journal_written_before_turns_existed_opens_and_keeps_its_rows(tmp_path):
    """A v1 journal is a real one, not a foreign database: it migrates in
    place, and the sessions already in it are still there afterwards."""
    path = tmp_path / "sessions.db"
    with SessionJournal(path) as journal:
        sid = running_session(journal)
    db = sqlite3.connect(path)
    db.execute("DROP TABLE session_turns")
    db.execute("ALTER TABLE sessions DROP COLUMN recall_scope")
    db.execute("ALTER TABLE sessions DROP COLUMN persona")
    db.execute("ALTER TABLE sessions DROP COLUMN persona_fingerprint")
    db.execute("PRAGMA user_version=1")
    db.commit()
    db.close()

    with SessionJournal(path) as journal:
        assert journal.get("alpha", sid)["state"] == "running"
        assert journal.turns("alpha", sid)["turns"] == []
        journal.start_turn("alpha", sid, "turn-1", {"text": "after the migration"})
        assert journal.turn("alpha", sid, "turn-1")["status"] == "accepted"
