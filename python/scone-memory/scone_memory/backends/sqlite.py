"""One-file persistence with nothing to run: SQLite from the standard
library, FTS5 for the lexical lane, vectors as blobs scanned in Python.

This is the default for the CLI and for any environment without a
database server, and it is what "local-first" means in this stack. The
vector scan is linear; it is fine to a few hundred thousand chunks, and
above that Qdrant is the answer. Calls block the event loop briefly,
which is acceptable for a file on local disk.
"""

from __future__ import annotations

import json
import math
import sqlite3
from array import array
from pathlib import Path
from typing import Mapping, Optional, Sequence

from ..errors import SconeError
from ..lexical import tokenize
from ..models import Chunk, Episode, Fact
from ..ports import NewChunk, NewEpisode, NewFact, SpaceCounts, TextFilter, VectorPoint
from .validation import validate_vector

SCHEMA = """
CREATE TABLE IF NOT EXISTS episodes (
    id INTEGER PRIMARY KEY, space TEXT NOT NULL, kind TEXT NOT NULL,
    content TEXT NOT NULL, content_hash TEXT NOT NULL, source TEXT,
    tags TEXT NOT NULL, metadata TEXT NOT NULL,
    created_at TEXT NOT NULL, ingested_at TEXT NOT NULL);
CREATE UNIQUE INDEX IF NOT EXISTS episodes_hash ON episodes(space, content_hash);
CREATE INDEX IF NOT EXISTS episodes_recent ON episodes(space, created_at);
CREATE TABLE IF NOT EXISTS chunks (
    id INTEGER PRIMARY KEY,
    episode_id INTEGER NOT NULL REFERENCES episodes(id) ON DELETE CASCADE,
    space TEXT NOT NULL, ordinal INTEGER NOT NULL,
    start INTEGER NOT NULL, "end" INTEGER NOT NULL,
    text TEXT NOT NULL, created_at TEXT NOT NULL);
CREATE INDEX IF NOT EXISTS chunks_episode ON chunks(episode_id);
CREATE VIRTUAL TABLE IF NOT EXISTS chunks_fts USING fts5(text, content='chunks', content_rowid='id');
CREATE TRIGGER IF NOT EXISTS chunks_ai AFTER INSERT ON chunks BEGIN
    INSERT INTO chunks_fts(rowid, text) VALUES (new.id, new.text); END;
CREATE TRIGGER IF NOT EXISTS chunks_ad AFTER DELETE ON chunks BEGIN
    INSERT INTO chunks_fts(chunks_fts, rowid, text) VALUES ('delete', old.id, old.text); END;
CREATE TABLE IF NOT EXISTS facts (
    id INTEGER PRIMARY KEY, space TEXT NOT NULL,
    subject TEXT NOT NULL, predicate TEXT NOT NULL, object TEXT NOT NULL,
    confidence REAL NOT NULL, valid_from TEXT NOT NULL, valid_until TEXT,
    status TEXT NOT NULL, closed_reason TEXT, source_episode_id INTEGER,
    origin TEXT NOT NULL DEFAULT 'stated', excluded_reason TEXT, superseded_by INTEGER);
CREATE INDEX IF NOT EXISTS facts_key ON facts(space, subject, predicate);
CREATE TABLE IF NOT EXISTS revisions (space TEXT PRIMARY KEY, revision INTEGER NOT NULL);
CREATE TABLE IF NOT EXISTS vectors (
    chunk_id INTEGER PRIMARY KEY, space TEXT NOT NULL, episode_id INTEGER NOT NULL,
    created_at TEXT NOT NULL, tags TEXT NOT NULL, metadata TEXT NOT NULL, vector BLOB NOT NULL);
CREATE INDEX IF NOT EXISTS vectors_space ON vectors(space, created_at);
CREATE TABLE IF NOT EXISTS vector_meta (key TEXT PRIMARY KEY, value TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS meta (key TEXT PRIMARY KEY, value TEXT NOT NULL);
"""

#: Shared spec 3.6. Bumped on any incompatible change; while the package
#: is pre-release nothing is migrated: a file from an older build is
#: refused with a message, not rewritten.
SCHEMA_VERSION = 5  # 5: facts carry superseded_by


def connect(path: str | Path) -> sqlite3.Connection:
    if str(path) != ":memory:":
        Path(path).expanduser().parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(Path(path).expanduser()) if str(path) != ":memory:" else path)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    conn.execute("PRAGMA journal_mode = WAL")
    conn.executescript(SCHEMA)
    check_schema(conn)
    return conn


def schema_version(conn: sqlite3.Connection) -> int:
    row = conn.execute("SELECT value FROM meta WHERE key = 'schema_version'").fetchone()
    if row:
        return int(row["value"])
    has_rows = conn.execute("SELECT 1 FROM episodes LIMIT 1").fetchone() is not None
    return 1 if has_rows else SCHEMA_VERSION


def check_schema(conn: sqlite3.Connection) -> None:
    """Stamp a fresh file; refuse a file another build wrote."""
    version = schema_version(conn)
    if version != SCHEMA_VERSION:
        raise SchemaMismatch(
            f"this file holds schema v{version}; this build writes v{SCHEMA_VERSION}. "
            "scone-memory is pre-release and does not migrate: export with the build that "
            "wrote the file, delete it, and import again."
        )
    conn.execute("INSERT OR REPLACE INTO meta (key, value) VALUES ('schema_version', ?)", (str(SCHEMA_VERSION),))
    conn.commit()


class SchemaMismatch(SconeError):
    pass


def _episode(row: sqlite3.Row) -> Episode:
    return Episode(
        episode_id=row["id"],
        space=row["space"],
        kind=row["kind"],
        content=row["content"],
        content_hash=row["content_hash"],
        source=row["source"],
        tags=tuple(json.loads(row["tags"])),
        metadata=json.loads(row["metadata"]),
        created_at=row["created_at"],
        ingested_at=row["ingested_at"],
    )


def _chunk(row: sqlite3.Row) -> Chunk:
    return Chunk(
        chunk_id=row["id"],
        episode_id=row["episode_id"],
        space=row["space"],
        ordinal=row["ordinal"],
        start=row["start"],
        end=row["end"],
        text=row["text"],
        created_at=row["created_at"],
    )


def _fact(row: sqlite3.Row) -> Fact:
    return Fact(
        fact_id=row["id"],
        space=row["space"],
        subject=row["subject"],
        predicate=row["predicate"],
        object=row["object"],
        confidence=row["confidence"],
        valid_from=row["valid_from"],
        valid_until=row["valid_until"],
        status=row["status"],
        closed_reason=row["closed_reason"],
        source_episode_id=row["source_episode_id"],
        origin=row["origin"],
        excluded_reason=row["excluded_reason"],
        superseded_by=row["superseded_by"],
    )


class SqliteDocumentStore:
    name = "sqlite"

    def __init__(self, path: str | Path = "~/.scone-memory/memory.db") -> None:
        self.path = path
        self.conn = connect(path)

    async def close(self) -> None:
        self.conn.close()

    async def insert_episode(self, new: NewEpisode) -> Episode:
        cur = self.conn.execute(
            "INSERT INTO episodes (space, kind, content, content_hash, source, tags, metadata, created_at, ingested_at)"
            " VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                new.space, new.kind, new.content, new.content_hash, new.source,
                json.dumps(list(new.tags)), json.dumps(dict(new.metadata)), new.created_at, new.ingested_at,
            ),
        )
        self.conn.commit()
        return Episode(episode_id=cur.lastrowid, **new.__dict__)

    async def episode_by_hash(self, space: str, content_hash: str) -> Optional[Episode]:
        row = self.conn.execute(
            "SELECT * FROM episodes WHERE space = ? AND content_hash = ?", (space, content_hash)
        ).fetchone()
        return _episode(row) if row else None

    async def get_episode(self, space: str, episode_id: int) -> Optional[Episode]:
        row = self.conn.execute("SELECT * FROM episodes WHERE id = ? AND space = ?", (episode_id, space)).fetchone()
        return _episode(row) if row else None

    async def delete_episode(self, space: str, episode_id: int) -> list[int]:
        removed = [
            r["id"] for r in self.conn.execute("SELECT id FROM chunks WHERE episode_id = ? AND space = ?", (episode_id, space))
        ]
        self.conn.execute("DELETE FROM episodes WHERE id = ? AND space = ?", (episode_id, space))
        self.conn.commit()
        return removed

    async def insert_chunks(self, new: Sequence[NewChunk]) -> list[Chunk]:
        out = []
        for n in new:
            cur = self.conn.execute(
                'INSERT INTO chunks (episode_id, space, ordinal, start, "end", text, created_at) VALUES (?, ?, ?, ?, ?, ?, ?)',
                (n.episode_id, n.space, n.ordinal, n.start, n.end, n.text, n.created_at),
            )
            out.append(Chunk(chunk_id=cur.lastrowid, **n.__dict__))
        self.conn.commit()
        return out

    async def get_chunks(self, space: str, chunk_ids: Sequence[int]) -> list[Chunk]:
        if not chunk_ids:
            return []
        marks = ",".join("?" * len(chunk_ids))
        rows = self.conn.execute(
            f"SELECT * FROM chunks WHERE space = ? AND id IN ({marks})", (space, *chunk_ids)
        ).fetchall()
        return [_chunk(r) for r in rows]

    async def search_text(
        self, space: str, query: str, limit: int, filter: TextFilter
    ) -> list[tuple[int, float]]:
        terms = tokenize(query)
        if not terms:
            return []
        match = " OR ".join('"' + t.replace('"', '""') + '"' for t in terms)
        sql = (
            "SELECT c.id AS id, bm25(chunks_fts) AS rank, e.tags AS tags, e.metadata AS metadata"
            " FROM chunks_fts JOIN chunks c ON c.id = chunks_fts.rowid"
            " JOIN episodes e ON e.id = c.episode_id"
            " WHERE chunks_fts MATCH ? AND c.space = ?"
        )
        params: list[object] = [match, space]
        if filter.as_of:
            sql += " AND c.created_at <= ?"
            params.append(filter.as_of)
        # Apply every scope condition before LIMIT so unrelated memories
        # cannot crowd all matching candidates out of the lexical lane.
        for tag in filter.tags:
            sql += " AND EXISTS (SELECT 1 FROM json_each(e.tags) AS tag WHERE tag.value = ?)"
            params.append(tag)
        for key, value in filter.where.items():
            sql += " AND EXISTS (SELECT 1 FROM json_each(e.metadata) AS meta WHERE meta.key = ? AND meta.value = ?)"
            params.extend((key, value))
        sql += " ORDER BY rank, c.id LIMIT ?"
        params.append(limit)
        return [(row["id"], -row["rank"]) for row in self.conn.execute(sql, params)]

    async def recent_episodes(self, space: str, limit: int) -> list[Episode]:
        rows = self.conn.execute(
            "SELECT * FROM episodes WHERE space = ? ORDER BY created_at DESC, id DESC LIMIT ?", (space, limit)
        ).fetchall()
        return [_episode(r) for r in rows]

    async def counts(self, space: str) -> SpaceCounts:
        counts = SpaceCounts()
        row = self.conn.execute(
            "SELECT COUNT(*) AS n, COALESCE(SUM(LENGTH(CAST(content AS BLOB))), 0) AS b FROM episodes WHERE space = ?",
            (space,),
        ).fetchone()
        counts.episodes, counts.bytes = row["n"], row["b"]
        counts.chunks = self.conn.execute("SELECT COUNT(*) FROM chunks WHERE space = ?", (space,)).fetchone()[0]
        for (tags,) in self.conn.execute("SELECT tags FROM episodes WHERE space = ?", (space,)):
            for tag in json.loads(tags):
                counts.tags[tag] = counts.tags.get(tag, 0) + 1
        return counts

    async def insert_fact(self, new: NewFact) -> Fact:
        cur = self.conn.execute(
            "INSERT INTO facts (space, subject, predicate, object, confidence, valid_from, valid_until, status,"
            " closed_reason, source_episode_id, origin, excluded_reason, superseded_by) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                new.space, new.subject, new.predicate, new.object, new.confidence, new.valid_from,
                new.valid_until, new.status, new.closed_reason, new.source_episode_id, new.origin,
                new.excluded_reason, new.superseded_by,
            ),
        )
        self.conn.commit()
        return Fact(fact_id=cur.lastrowid, **new.__dict__)

    async def update_fact(self, fact: Fact) -> None:
        cur = self.conn.execute(
            "UPDATE facts SET object = ?, confidence = ?, valid_from = ?, valid_until = ?, status = ?,"
            " closed_reason = ?, origin = ?, excluded_reason = ?, superseded_by = ? WHERE id = ? AND space = ?",
            (
                fact.object, fact.confidence, fact.valid_from, fact.valid_until, fact.status,
                fact.closed_reason, fact.origin, fact.excluded_reason, fact.superseded_by, fact.fact_id, fact.space,
            ),
        )
        self.conn.commit()
        if cur.rowcount == 0:
            raise KeyError(fact.fact_id)

    async def get_fact(self, space: str, fact_id: int) -> Optional[Fact]:
        row = self.conn.execute("SELECT * FROM facts WHERE id = ? AND space = ?", (fact_id, space)).fetchone()
        return _fact(row) if row else None

    async def list_facts(self, space: str, include_closed: bool) -> list[Fact]:
        sql = "SELECT * FROM facts WHERE space = ?" + ("" if include_closed else " AND status = 'active'")
        return [_fact(r) for r in self.conn.execute(sql + " ORDER BY id", (space,))]

    async def facts_for(self, space: str, subject: str, predicate: str) -> list[Fact]:
        rows = self.conn.execute(
            "SELECT * FROM facts WHERE space = ? AND subject = ? AND predicate = ? ORDER BY id",
            (space, subject, predicate),
        )
        return [_fact(r) for r in rows]

    async def bump_revision(self, space: str) -> int:
        self.conn.execute(
            "INSERT INTO revisions (space, revision) VALUES (?, 1)"
            " ON CONFLICT(space) DO UPDATE SET revision = revision + 1",
            (space,),
        )
        self.conn.commit()
        return await self.revision(space)

    async def revision(self, space: str) -> int:
        row = self.conn.execute("SELECT revision FROM revisions WHERE space = ?", (space,)).fetchone()
        return row["revision"] if row else 0


class SqliteVectorIndex:
    name = "sqlite"

    def __init__(self, path: str | Path = "~/.scone-memory/memory.db") -> None:
        self.path = path
        self.conn = connect(path)
        self.dim: Optional[int] = None
        row = self.conn.execute("SELECT value FROM vector_meta WHERE key = 'dim'").fetchone()
        if row:
            self.dim = int(row["value"])

    async def close(self) -> None:
        self.conn.close()

    async def ensure(self, dim: int) -> None:
        if self.dim is not None and self.dim != dim:
            raise ValueError(f"index holds {self.dim}-d vectors, embedder makes {dim}-d")
        self.dim = dim
        self.conn.execute("INSERT OR REPLACE INTO vector_meta (key, value) VALUES ('dim', ?)", (str(dim),))
        self.conn.commit()

    async def upsert(self, points: Sequence[VectorPoint]) -> None:
        for point in points:
            validate_vector(point.vector, self.dim)
        self.conn.executemany(
            "INSERT OR REPLACE INTO vectors (chunk_id, space, episode_id, created_at, tags, metadata, vector)"
            " VALUES (?, ?, ?, ?, ?, ?, ?)",
            [
                (
                    p.chunk_id, p.space, p.episode_id, p.created_at, json.dumps(list(p.tags)),
                    json.dumps(dict(p.metadata)), array("f", p.vector).tobytes(),
                )
                for p in points
            ],
        )
        self.conn.commit()

    async def search(
        self,
        space: str,
        vector: Sequence[float],
        limit: int,
        as_of: Optional[str] = None,
        tags: tuple[str, ...] = (),
        where: Mapping[str, str] | None = None,
    ) -> list[tuple[int, float]]:
        validate_vector(vector, self.dim)
        sql = "SELECT chunk_id, tags, metadata, vector FROM vectors WHERE space = ?"
        params: list[object] = [space]
        if as_of:
            sql += " AND created_at <= ?"
            params.append(as_of)
        query = array("f", vector)
        qnorm = math.sqrt(sum(x * x for x in query))
        scored: list[tuple[int, float]] = []
        for row in self.conn.execute(sql, params):
            if tags and not set(tags) <= set(json.loads(row["tags"])):
                continue
            if where:
                meta = json.loads(row["metadata"])
                if any(meta.get(k) != v for k, v in where.items()):
                    continue
            stored = array("f")
            stored.frombytes(row["vector"])
            dot = sum(a * b for a, b in zip(query, stored))
            snorm = math.sqrt(sum(x * x for x in stored))
            scored.append((row["chunk_id"], dot / (qnorm * snorm) if qnorm and snorm else 0.0))
        scored.sort(key=lambda pair: (-pair[1], pair[0]))
        return scored[:limit]

    async def delete(self, chunk_ids: Sequence[int]) -> None:
        self.conn.executemany("DELETE FROM vectors WHERE chunk_id = ?", [(c,) for c in chunk_ids])
        self.conn.commit()
