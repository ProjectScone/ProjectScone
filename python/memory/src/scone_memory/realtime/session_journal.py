"""Durable conversation lifecycle receipts, not a conversation runtime.

Use a separate database, never a native memory database. This synchronous
storage boundary requires its caller to authorize the space. Opening it neither
starts providers nor infers that previously running sessions ended successfully.
"""

from __future__ import annotations

from contextlib import contextmanager
from datetime import datetime, timezone
import json
import re
import sqlite3
from pathlib import Path
from uuid import uuid4

from ..memory.engine import check_space
from ..core.errors import Conflict, InvalidInput, NotFound
from ..retrieval.recall_scope import RecallScope

_APPLICATION_ID = 0x53434A31
_VERSION = 4
_KEY = re.compile(r"[A-Za-z0-9._:-]{1,128}\Z")
_TRANSITIONS = {
    "created": {"start": "running", "stop": "ended", "fail": "failed", "interrupt": "interrupted"},
    "running": {"stop": "stopping", "end": "ended", "fail": "failed", "interrupt": "interrupted"},
    "stopping": {"end": "ended", "fail": "failed", "interrupt": "interrupted"},
}


#: A turn's durable half. The reply's text is not here: it lives in the
#: episode this row names, so forgetting that episode removes the text
#: rather than leaving a second copy behind in a receipt.
_TURNS_TABLE = """CREATE TABLE session_turns (
    space TEXT NOT NULL, session_id TEXT NOT NULL, request_id TEXT NOT NULL,
    signature TEXT NOT NULL, status TEXT NOT NULL, episode_id INTEGER,
    error TEXT, accepted_at TEXT NOT NULL, settled_at TEXT,
    PRIMARY KEY(space, session_id, request_id))"""
#: What a turn can be. "accepted" is the only one a process is still
#: working on, which is what makes recovery decidable after a restart.
_TURN_STATUSES = ("accepted", "completed", "interrupted", "failed", "cancelled")


def _key(value: str) -> None:
    if not isinstance(value, str) or not _KEY.fullmatch(value):
        raise InvalidInput("session/request IDs must be 1..128 ASCII letters, digits, '.', '_', ':' or '-'")


def _integer(value: int, minimum: int, maximum: int = 2**63 - 2) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or not minimum <= value <= maximum:
        raise InvalidInput(f"expected an integer in {minimum}..{maximum}")


def _turn_receipt(row: sqlite3.Row) -> dict:
    return {"request_id": row["request_id"], "session_id": row["session_id"],
            "status": row["status"], "episode_id": row["episode_id"], "error": row["error"],
            "accepted_at": row["accepted_at"], "settled_at": row["settled_at"]}


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds").replace("+00:00", "Z")


class SessionJournal:
    """One owned connection; use a serialized service boundary, not shared threads.

    Multiple connections coordinate writes through SQLite transactions. Runtime
    ownership and provider-task recovery are separate concerns: this class never
    claims a task is alive merely because its recorded state says ``running``.
    """

    def __init__(self, path: str | Path):
        self._db = sqlite3.connect(str(path), timeout=5, isolation_level=None)
        self._db.row_factory = sqlite3.Row
        try:
            with self._transaction():
                app_id = self._db.execute("PRAGMA application_id").fetchone()[0]
                version = self._db.execute("PRAGMA user_version").fetchone()[0]
                objects = self._db.execute(
                    "SELECT type, name FROM sqlite_master WHERE name NOT GLOB 'sqlite_*'"
                ).fetchall()
                tables = {row["name"] for row in objects if row["type"] == "table"}
                if app_id == _APPLICATION_ID:
                    if not 1 <= version <= _VERSION:
                        raise InvalidInput("unsupported session journal schema version")
                    if version == 1:
                        # A journal written before turn receipts existed. Its
                        # sessions are real and stay exactly as they are; only
                        # the new table is added.
                        if tables != {"sessions", "session_events"}:
                            raise InvalidInput("invalid session journal tables")
                        self._db.execute(_TURNS_TABLE)
                    elif tables != {"sessions", "session_events", "session_turns"}:
                        raise InvalidInput("invalid session journal tables")
                    if version < 3:
                        self._db.execute("ALTER TABLE sessions ADD COLUMN recall_scope TEXT NOT NULL DEFAULT '{}'")
                    if version < 4:
                        # Empty means no persona; rows from before the columns stay unbound.
                        self._db.execute("ALTER TABLE sessions ADD COLUMN persona TEXT NOT NULL DEFAULT ''")
                        self._db.execute("ALTER TABLE sessions ADD COLUMN persona_fingerprint TEXT NOT NULL DEFAULT ''")
                    if version < _VERSION:
                        self._db.execute(f"PRAGMA user_version={_VERSION}")
                elif app_id or version or objects:
                    raise InvalidInput("path is not a session journal; refusing to modify another database")
                else:
                    self._db.execute("""CREATE TABLE sessions (
                        space TEXT NOT NULL, session_id TEXT NOT NULL,
                        create_key TEXT NOT NULL, mode TEXT NOT NULL,
                        state TEXT NOT NULL, revision INTEGER NOT NULL,
                        created_at TEXT NOT NULL, updated_at TEXT NOT NULL,
                        recall_scope TEXT NOT NULL DEFAULT '{}',
                        persona TEXT NOT NULL DEFAULT '',
                        persona_fingerprint TEXT NOT NULL DEFAULT '',
                        PRIMARY KEY(space, session_id), UNIQUE(space, create_key))""")
                    self._db.execute("""CREATE TABLE session_events (
                        space TEXT NOT NULL, session_id TEXT NOT NULL,
                        revision INTEGER NOT NULL, request_id TEXT NOT NULL,
                        signature TEXT NOT NULL, receipt TEXT NOT NULL,
                        PRIMARY KEY(space, session_id, revision),
                        UNIQUE(space, session_id, request_id))""")
                    self._db.execute(_TURNS_TABLE)
                    self._db.execute(f"PRAGMA application_id={_APPLICATION_ID}")
                    self._db.execute(f"PRAGMA user_version={_VERSION}")
            self._db.execute("PRAGMA journal_mode=WAL")
            self._db.execute("PRAGMA synchronous=FULL")
        except BaseException:
            self._db.close()
            raise

    @contextmanager
    def _transaction(self):
        self._db.execute("BEGIN IMMEDIATE")
        try:
            yield
            self._db.execute("COMMIT")
        except BaseException:
            if self._db.in_transaction:
                self._db.execute("ROLLBACK")
            raise

    def close(self) -> None:
        self._db.close()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()

    def _session(self, space: str, session_id: str) -> sqlite3.Row:
        found = self._db.execute("SELECT * FROM sessions WHERE space=? AND session_id=?", (space, session_id)).fetchone()
        if found is None:
            raise NotFound("session not found in this space")
        return found

    def _event(self, space, session_id, request_id, signature):
        found = self._db.execute("SELECT signature, receipt FROM session_events WHERE space=? AND session_id=? AND request_id=?", (space, session_id, request_id)).fetchone()
        if found is None:
            return None
        if found["signature"] != signature:
            raise Conflict("request ID was already used for a different session command", self._session(space, session_id)["revision"])
        return json.loads(found["receipt"])

    def _append(self, space, sid, revision, request_id, signature, action, previous, state, now):
        receipt = {"session_id": sid, "request_id": request_id, "revision": revision,
                   "action": action, "previous_state": previous, "state": state, "recorded_at": now}
        self._db.execute("INSERT INTO session_events VALUES (?, ?, ?, ?, ?, ?)",
                         (space, sid, revision, request_id, signature, json.dumps(receipt)))
        return receipt

    def create(self, space: str, request_id: str, mode: str = "text", *, recall_scope=None, persona=None,
               persona_fingerprint=None) -> dict:
        check_space(space)
        _key(request_id)
        if mode not in ("text", "voice"):
            raise InvalidInput("session mode must be text or voice; runtime support is configured separately")
        if persona is not None and (not isinstance(persona, str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}", persona)):
            raise InvalidInput("persona must be a catalog id")
        if persona_fingerprint is not None and (persona is None or not isinstance(persona_fingerprint, str)
                                                or not re.fullmatch(r"[0-9a-f]{16}", persona_fingerprint)):
            raise InvalidInput("persona_fingerprint must accompany a persona as 16 hex characters")
        scope = RecallScope.from_mapping(recall_scope).as_dict()
        # Preserve replay signatures from pre-scope journals for empty scopes;
        # a persona always signs with the scope so the two never read alike.
        parts = ["create", mode, scope] if scope else ["create", mode]
        signature = json.dumps([*parts, persona] if persona else parts, sort_keys=True)
        with self._transaction():
            found = self._db.execute("SELECT session_id FROM sessions WHERE space=? AND create_key=?", (space, request_id)).fetchone()
            if found is not None:
                return self._event(space, found["session_id"], request_id, signature)
            sid, now = uuid4().hex, _now()
            self._db.execute("INSERT INTO sessions VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                             (space, sid, request_id, mode, "created", 1, now, now, json.dumps(scope, sort_keys=True),
                              persona or "", persona_fingerprint or ""))
            return self._append(space, sid, 1, request_id, signature, "create", None, "created", now)

    def created(self, space: str, request_id: str) -> str | None:
        """The session a create request already made, by its request id."""
        check_space(space)
        _key(request_id)
        found = self._db.execute("SELECT session_id FROM sessions WHERE space=? AND create_key=?", (space, request_id)).fetchone()
        return found["session_id"] if found is not None else None

    def get(self, space: str, session_id: str) -> dict:
        check_space(space)
        _key(session_id)
        found = dict(self._session(space, session_id))
        del found["create_key"]
        found["recall_scope"] = json.loads(found["recall_scope"])
        found["persona"] = found["persona"] or None
        found["persona_fingerprint"] = found["persona_fingerprint"] or None
        return found

    def sessions(self, space: str, after: str = "", limit: int = 100) -> dict:
        check_space(space)
        if after != "":
            _key(after)
        _integer(limit, 1, 200)
        rows = self._db.execute(
            "SELECT space, session_id, mode, state, revision, created_at, updated_at, recall_scope, persona, persona_fingerprint "
            "FROM sessions WHERE space=? AND session_id>? ORDER BY session_id LIMIT ?",
            (space, after, limit + 1),
        ).fetchall()
        items = [dict(row) for row in rows[:limit]]
        for item in items:
            item["recall_scope"] = json.loads(item["recall_scope"])
            item["persona"] = item["persona"] or None
            item["persona_fingerprint"] = item["persona_fingerprint"] or None
        return {"items": items, "next_after": items[-1]["session_id"] if items else after,
                "has_more": len(rows) > limit}

    def transition(self, space: str, session_id: str, request_id: str, action: str, expected_revision: int) -> dict:
        check_space(space)
        _key(session_id)
        _key(request_id)
        _integer(expected_revision, 1)
        if action not in ("start", "stop", "end", "fail", "interrupt"):
            raise InvalidInput("unknown session lifecycle action")
        signature = json.dumps([action, expected_revision])
        with self._transaction():
            session = self._session(space, session_id)
            replay = self._event(space, session_id, request_id, signature)
            if replay is not None:
                return replay
            if session["revision"] != expected_revision:
                raise Conflict("session revision changed; inspect current state before retrying", session["revision"])
            state = _TRANSITIONS.get(session["state"], {}).get(action)
            if state is None:
                raise Conflict("action is not allowed in this session state", session["revision"])
            revision, now = expected_revision + 1, _now()
            self._db.execute("UPDATE sessions SET state=?, revision=?, updated_at=? WHERE space=? AND session_id=?",
                             (state, revision, now, space, session_id))
            return self._append(space, session_id, revision, request_id, signature, action, session["state"], state, now)

    # -- turns ------------------------------------------------------------

    def _turn_row(self, space, session_id, request_id):
        return self._db.execute(
            "SELECT * FROM session_turns WHERE space=? AND session_id=? AND request_id=?",
            (space, session_id, request_id),
        ).fetchone()

    def start_turn(self, space: str, session_id: str, request_id: str, signature: dict) -> dict:
        """Record that a turn was accepted, before anything is asked of a
        provider. A repeat of the same request returns what that turn has
        become; a repeat that changed its mind is a conflict, because the
        first one may already have been delivered."""
        check_space(space)
        _key(session_id)
        _key(request_id)
        signed = json.dumps(signature, sort_keys=True)
        with self._transaction():
            session = self._session(space, session_id)
            found = self._turn_row(space, session_id, request_id)
            if found is not None:
                if found["signature"] != signed:
                    raise Conflict("request ID was already used for a different turn", session["revision"])
                return _turn_receipt(found)
            now = _now()
            self._db.execute(
                "INSERT INTO session_turns VALUES (?, ?, ?, ?, 'accepted', NULL, NULL, ?, NULL)",
                (space, session_id, request_id, signed, now),
            )
            return _turn_receipt(self._turn_row(space, session_id, request_id))

    def finish_turn(self, space: str, session_id: str, request_id: str, status: str,
                    episode_id: int | None = None, error: str | None = None) -> dict:
        """What became of an accepted turn. ``episode_id`` names where the
        reply's text is; the text itself is never stored here."""
        check_space(space)
        _key(session_id)
        _key(request_id)
        if status not in _TURN_STATUSES or status == "accepted":
            raise InvalidInput(f"a turn settles as one of {_TURN_STATUSES[1:]}")
        if episode_id is not None:
            _integer(episode_id, 1)
        with self._transaction():
            session = self._session(space, session_id)
            found = self._turn_row(space, session_id, request_id)
            if found is None:
                raise NotFound("turn not found in this session")
            if found["status"] != "accepted":
                # Settling twice is the retry case: the first answer stands.
                if found["status"] != status:
                    raise Conflict("turn already settled differently", session["revision"])
                return _turn_receipt(found)
            self._db.execute(
                "UPDATE session_turns SET status=?, episode_id=?, error=?, settled_at=? "
                "WHERE space=? AND session_id=? AND request_id=?",
                (status, episode_id, error, _now(), space, session_id, request_id),
            )
            return _turn_receipt(self._turn_row(space, session_id, request_id))

    def turn(self, space: str, session_id: str, request_id: str) -> dict:
        check_space(space)
        _key(session_id)
        _key(request_id)
        self._session(space, session_id)
        found = self._turn_row(space, session_id, request_id)
        if found is None:
            raise NotFound("turn not found in this session")
        return _turn_receipt(found)

    def turns(self, space: str, session_id: str, after: str = "", limit: int = 100) -> dict:
        check_space(space)
        _key(session_id)
        if after != "":
            _key(after)
        _integer(limit, 1, 200)
        self._session(space, session_id)
        rows = self._db.execute(
            "SELECT * FROM session_turns WHERE space=? AND session_id=? AND request_id>? "
            "ORDER BY request_id LIMIT ?",
            (space, session_id, after, limit + 1),
        ).fetchall()
        items = [_turn_receipt(row) for row in rows[:limit]]
        return {"turns": items, "next_after": items[-1]["request_id"] if items else after,
                "has_more": len(rows) > limit}

    def latest_turn_id(self, space: str, session_id: str) -> str | None:
        """Latest insertion in this journal, not UUID or wall-clock order.

        New turns append a SQLite row; idempotent retries and outcome updates
        keep that row. Read it afresh so no cached pointer can become stale.
        This does not imply the turn is running, completed or delivered.
        """
        check_space(space)
        _key(session_id)
        self._session(space, session_id)
        row = self._db.execute(
            "SELECT request_id FROM session_turns WHERE space=? AND session_id=? "
            "ORDER BY rowid DESC LIMIT 1", (space, session_id),
        ).fetchone()
        return row["request_id"] if row else None

    def episodes_held_elsewhere(self, space: str, session_id: str, episode_ids) -> set[int]:
        """Which of these episodes another conversation's receipt names.

        Identical replies deduplicate to one episode, so two conversations
        can hold the same one between them. Deleting either must leave the
        other's transcript readable, and this says which ids are not this
        session's alone to remove.
        """
        check_space(space)
        _key(session_id)
        wanted = {int(found) for found in episode_ids if found is not None}
        if not wanted:
            return set()
        marks = ",".join("?" * len(wanted))
        rows = self._db.execute(
            f"SELECT DISTINCT episode_id FROM session_turns "
            f"WHERE space=? AND session_id!=? AND episode_id IN ({marks})",
            (space, session_id, *sorted(wanted)),
        ).fetchall()
        return {row["episode_id"] for row in rows}

    def delete_session(self, space: str, session_id: str) -> None:
        """Remove a session, its lifecycle events and its turn receipts.

        The transcript's episodes are the caller's to forget first: a
        receipt naming an episode that is gone reads as forgotten, which
        is true, while an episode with no session left to explain it is
        an orphan nothing can account for.
        """
        check_space(space)
        _key(session_id)
        with self._transaction():
            self._session(space, session_id)
            for table in ("session_turns", "session_events", "sessions"):
                self._db.execute(f"DELETE FROM {table} WHERE space=? AND session_id=?", (space, session_id))

    def recover(self, space: str) -> int:
        """Settle turns whose process is gone.

        A turn still marked accepted after a restart was owned by a
        process that no longer exists, so whether the provider answered
        cannot be known from here. It is recorded as interrupted saying
        exactly that, rather than left reading as in flight forever. The
        caller decides when no process owns these sessions any more; the
        journal will not guess that for a server sharing this file.
        """
        check_space(space)
        with self._transaction():
            settled = self._db.execute(
                "UPDATE session_turns SET status='interrupted', error=?, settled_at=? "
                "WHERE space=? AND status='accepted'",
                ("the process handling this turn did not finish it; whether the provider "
                 "answered is unknown, so it is not retried automatically", _now(), space),
            ).rowcount
        return settled

    def events(self, space: str, session_id: str, after: int = 0, limit: int = 100) -> dict:
        check_space(space)
        _key(session_id)
        _integer(after, 0)
        _integer(limit, 1, 200)
        self._session(space, session_id)
        rows = self._db.execute("SELECT receipt FROM session_events WHERE space=? AND session_id=? AND revision>? ORDER BY revision LIMIT ?",
                                (space, session_id, after, limit + 1)).fetchall()
        events = [json.loads(row["receipt"]) for row in rows[:limit]]
        return {"events": events, "next_after": events[-1]["revision"] if events else after, "has_more": len(rows) > limit}
