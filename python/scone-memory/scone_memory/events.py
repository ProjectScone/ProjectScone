"""Event log sinks: where the evidence goes.

``InMemoryEventLog`` keeps the newest N events; ``SqliteEventLog`` keeps
them in the same file as the SQLite stores, with an optional max age.
Both return newest first and isolate spaces. Retention is a stated
policy, not an accident: what was dropped is not evidence any more and
metrics computed afterwards say so through their n.
"""

from __future__ import annotations

import json
import sqlite3
from collections import deque
from itertools import count
from pathlib import Path
from typing import Callable, Optional

from .ports import Event, NewEvent
from .timeutil import format_rfc3339, now, parse_rfc3339


class InMemoryEventLog:
    name = "memory"

    def __init__(self, max_events: int = 10_000) -> None:
        self.max_events = max_events
        self._events: deque[Event] = deque(maxlen=max_events)
        self._ids = count(1)

    async def append(self, new: NewEvent) -> Event:
        event = Event(event_id=next(self._ids), **new.__dict__)
        self._events.append(event)
        return event

    async def get(self, space: str, event_id: int) -> Optional[Event]:
        for event in self._events:
            if event.event_id == event_id and event.space == space:
                return event
        return None

    async def query(
        self, space: str, kind: Optional[str] = None, since: Optional[str] = None, limit: int = 100
    ) -> list[Event]:
        out: list[Event] = []
        floor = parse_rfc3339(since) if since else None
        for event in reversed(self._events):
            if event.space != space or (kind and event.kind != kind):
                continue
            if floor is not None and parse_rfc3339(event.ts) < floor:
                continue
            out.append(event)
            if len(out) == limit:
                break
        return out


class SqliteEventLog:
    name = "sqlite"

    SCHEMA = """
    CREATE TABLE IF NOT EXISTS events (
        id INTEGER PRIMARY KEY, ts TEXT NOT NULL, space TEXT NOT NULL, kind TEXT NOT NULL,
        schema_version INTEGER NOT NULL, payload TEXT NOT NULL);
    CREATE INDEX IF NOT EXISTS events_space_kind ON events(space, kind, id);
    CREATE INDEX IF NOT EXISTS events_ts ON events(ts);
    """
    SWEEP_EVERY = 100

    def __init__(
        self,
        path: str | Path = "~/.scone-memory/memory.db",
        max_age_days: Optional[float] = None,
        clock: Callable[[], str] = lambda: format_rfc3339(now()),
    ) -> None:
        self.path = path
        self.max_age_days = max_age_days
        self.clock = clock
        target = str(Path(path).expanduser()) if str(path) != ":memory:" else path
        if target != ":memory:":
            Path(target).parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(target)
        self.conn.row_factory = sqlite3.Row
        self.conn.executescript(self.SCHEMA)
        self._appends = 0

    async def close(self) -> None:
        self.conn.close()

    async def append(self, new: NewEvent) -> Event:
        cur = self.conn.execute(
            "INSERT INTO events (ts, space, kind, schema_version, payload) VALUES (?, ?, ?, ?, ?)",
            (new.ts, new.space, new.kind, new.schema_version, json.dumps(dict(new.payload), ensure_ascii=False)),
        )
        self.conn.commit()
        self._appends += 1
        if self.max_age_days is not None and self._appends % self.SWEEP_EVERY == 0:
            self.sweep()
        return Event(event_id=cur.lastrowid, **new.__dict__)

    def sweep(self) -> int:
        """Delete events older than max_age_days; returns how many."""
        if self.max_age_days is None:
            return 0
        cutoff = parse_rfc3339(self.clock()).timestamp() - self.max_age_days * 86400
        from datetime import datetime, timezone

        cutoff_text = format_rfc3339(datetime.fromtimestamp(cutoff, tz=timezone.utc))
        cur = self.conn.execute("DELETE FROM events WHERE ts < ?", (cutoff_text,))
        self.conn.commit()
        return cur.rowcount

    async def get(self, space: str, event_id: int) -> Optional[Event]:
        row = self.conn.execute("SELECT * FROM events WHERE id = ? AND space = ?", (event_id, space)).fetchone()
        return _event(row) if row else None

    async def query(
        self, space: str, kind: Optional[str] = None, since: Optional[str] = None, limit: int = 100
    ) -> list[Event]:
        sql = "SELECT * FROM events WHERE space = ?"
        params: list[object] = [space]
        if kind:
            sql += " AND kind = ?"
            params.append(kind)
        if since:
            sql += " AND ts >= ?"
            params.append(format_rfc3339(parse_rfc3339(since)))
        sql += " ORDER BY id DESC LIMIT ?"
        params.append(limit)
        return [_event(r) for r in self.conn.execute(sql, params)]


def _event(row: sqlite3.Row) -> Event:
    return Event(
        event_id=row["id"],
        ts=row["ts"],
        space=row["space"],
        kind=row["kind"],
        payload=json.loads(row["payload"]),
        schema_version=row["schema_version"],
    )
