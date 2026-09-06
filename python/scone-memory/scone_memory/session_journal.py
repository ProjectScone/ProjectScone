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

from .engine import check_space
from .errors import Conflict, InvalidInput, NotFound

_APPLICATION_ID = 0x53434A31
_VERSION = 1
_KEY = re.compile(r"[A-Za-z0-9._:-]{1,128}\Z")
_TRANSITIONS = {
    "created": {"start": "running", "stop": "ended", "fail": "failed", "interrupt": "interrupted"},
    "running": {"stop": "stopping", "end": "ended", "fail": "failed", "interrupt": "interrupted"},
    "stopping": {"end": "ended", "fail": "failed", "interrupt": "interrupted"},
}


def _key(value: str) -> None:
    if not isinstance(value, str) or not _KEY.fullmatch(value):
        raise InvalidInput("session/request IDs must be 1..128 ASCII letters, digits, '.', '_', ':' or '-'")


def _integer(value: int, minimum: int, maximum: int = 2**63 - 2) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or not minimum <= value <= maximum:
        raise InvalidInput(f"expected an integer in {minimum}..{maximum}")


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
                    if version != _VERSION:
                        raise InvalidInput("unsupported session journal schema version")
                    if tables != {"sessions", "session_events"}:
                        raise InvalidInput("invalid session journal tables")
                elif app_id or version or objects:
                    raise InvalidInput("path is not a session journal; refusing to modify another database")
                else:
                    self._db.execute("""CREATE TABLE sessions (
                        space TEXT NOT NULL, session_id TEXT NOT NULL,
                        create_key TEXT NOT NULL, mode TEXT NOT NULL,
                        state TEXT NOT NULL, revision INTEGER NOT NULL,
                        created_at TEXT NOT NULL, updated_at TEXT NOT NULL,
                        PRIMARY KEY(space, session_id), UNIQUE(space, create_key))""")
                    self._db.execute("""CREATE TABLE session_events (
                        space TEXT NOT NULL, session_id TEXT NOT NULL,
                        revision INTEGER NOT NULL, request_id TEXT NOT NULL,
                        signature TEXT NOT NULL, receipt TEXT NOT NULL,
                        PRIMARY KEY(space, session_id, revision),
                        UNIQUE(space, session_id, request_id))""")
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

    def create(self, space: str, request_id: str, mode: str = "text") -> dict:
        check_space(space)
        _key(request_id)
        if mode not in ("text", "voice"):
            raise InvalidInput("session mode must be text or voice; runtime support is configured separately")
        signature = json.dumps(["create", mode])
        with self._transaction():
            found = self._db.execute("SELECT session_id FROM sessions WHERE space=? AND create_key=?", (space, request_id)).fetchone()
            if found is not None:
                return self._event(space, found["session_id"], request_id, signature)
            sid, now = uuid4().hex, _now()
            self._db.execute("INSERT INTO sessions VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                             (space, sid, request_id, mode, "created", 1, now, now))
            return self._append(space, sid, 1, request_id, signature, "create", None, "created", now)

    def get(self, space: str, session_id: str) -> dict:
        check_space(space)
        _key(session_id)
        found = dict(self._session(space, session_id))
        del found["create_key"]
        return found

    def sessions(self, space: str, after: str = "", limit: int = 100) -> dict:
        check_space(space)
        if after != "":
            _key(after)
        _integer(limit, 1, 200)
        rows = self._db.execute(
            "SELECT space, session_id, mode, state, revision, created_at, updated_at "
            "FROM sessions WHERE space=? AND session_id>? ORDER BY session_id LIMIT ?",
            (space, after, limit + 1),
        ).fetchall()
        items = [dict(row) for row in rows[:limit]]
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
