"""Documents, vectors and evidence in PostgreSQL, with pgvector.

One schema (``scone`` by default) holds everything: episodes, chunks with
a generated ``tsvector`` for the lexical lane, facts, revisions, events,
and a ``vectors`` table with a pgvector column and an HNSW cosine index.
Tags and metadata are ``jsonb`` and every filter runs in SQL. Timestamps
are stored as the engine's RFC 3339 strings, which sort as text, the same
as every other store here.

Three classes share a connection pool: ``PostgresDocumentStore``,
``PostgresVectorIndex`` and ``PostgresEventLog``. Any of them may be used
alone, with its own pool, or together through
``PostgresDocumentStore.vectors()`` / ``.events()`` so one database
serves the whole engine. Needs ``pip install 'scone-memory[postgres]'``
(psycopg 3 with a pool, and the pgvector adapter) and a server with the
``vector`` extension available.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from contextlib import asynccontextmanager
from contextvars import ContextVar
from typing import Any, AsyncIterator, Callable, Mapping, Optional, Sequence

from ..core.affirmations import Affirmation, NewAffirmation, read_links, stored_links
from ..core.errors import SconeError
from ..retrieval.lexical import tokenize
from ..core.models import Chunk, Episode, Fact, FactLink, Tombstone
from ..core.ports import DeletedSpace, DuplicateEvent, Event, NewChunk, NewEpisode, NewEvent, NewFact, NewFactLink, NewTombstone, SpaceCounts, TextFilter, VectorPoint
from ..core.timeutil import epoch_seconds, format_rfc3339, now_rfc3339, parse_rfc3339
from .validation import validate_vector

#: Shared spec 3.6. Pre-release: a database another build wrote is
#: refused, not migrated.
SCHEMA_VERSION = 6


class SchemaMismatch(SconeError):
    pass


def _ident(name: str) -> str:
    if not name.replace("_", "").isalnum() or not name[:1].isalpha():
        raise ValueError(f"schema name must be an identifier, got {name!r}")
    return name


class Pool:
    """A lazily opened psycopg async pool with pgvector registered on
    every connection. Shared by the three stores when they are built from
    one document store."""

    def __init__(self, url: str, min_size: int = 1, max_size: int = 8) -> None:
        try:
            from pgvector.psycopg import register_vector_async
            from psycopg_pool import AsyncConnectionPool
        except ImportError as e:  # pragma: no cover
            raise ImportError("Postgres stores need psycopg[pool] and pgvector: pip install 'scone-memory[postgres]'") from e

        async def configure(conn) -> None:
            await conn.execute("CREATE EXTENSION IF NOT EXISTS vector")
            await register_vector_async(conn)

        self.pool = AsyncConnectionPool(url, min_size=min_size, max_size=max_size, open=False, configure=configure, kwargs={"autocommit": True})
        self.opened = False
        #: Stores sharing this pool. The pool closes when the last one does.
        self.users = 0

    async def open(self) -> None:
        if not self.opened:
            await self.pool.open()
            self.opened = True

    async def release(self) -> None:
        self.users = max(0, self.users - 1)
        if self.opened and self.users == 0:
            await self.pool.close()
            self.opened = False

    def connection(self):
        return self.pool.connection()


def _episode(row: Mapping) -> Episode:
    return Episode(
        episode_id=row["id"], space=row["space"], kind=row["kind"], content=row["content"], content_hash=row["content_hash"],
        source=row["source"], tags=tuple(row["tags"]), metadata=dict(row["metadata"]), created_at=row["created_at"],
        ingested_at=row["ingested_at"],
    )


def _chunk(row: Mapping) -> Chunk:
    return Chunk(
        chunk_id=row["id"], episode_id=row["episode_id"], space=row["space"], ordinal=row["ordinal"], start=row["start_off"],
        end=row["end_off"], text=row["text"], created_at=row["created_at"],
    )


def _tombstone(row: Mapping) -> Tombstone:
    return Tombstone(space=row["space"], episode_id=int(row["episode_id"]), content_hash=row["content_hash"],
                     forgotten_at=row["forgotten_at"], reason=row["reason"])


def _fact_link(row: Mapping) -> FactLink:
    return FactLink(link_id=int(row["id"]), space=row["space"], from_fact=int(row["from_fact"]), to_fact=int(row["to_fact"]),
                    kind=row["kind"], created_at=row["created_at"], source_episode_id=row["source_episode_id"], quote=row["quote"])


def _affirmation(row: Mapping) -> Affirmation:
    return Affirmation(affirmation_id=int(row["id"]), space=row["space"], fact_id=int(row["fact_id"]),
                       valid_from=row["valid_from"], recorded_at=row["recorded_at"],
                       confidence=float(row["confidence"]), source_episode_id=row["source_episode_id"],
                       origin=row["origin"], quote=row["quote"], links=read_links(json.loads(row["links"])))


def _fact(row: Mapping) -> Fact:
    return Fact(
        fact_id=row["id"], space=row["space"], subject=row["subject"], predicate=row["predicate"], object=row["object"],
        confidence=row["confidence"], valid_from=row["valid_from"], valid_until=row["valid_until"], status=row["status"],
        closed_reason=row["closed_reason"], source_episode_id=row["source_episode_id"], origin=row["origin"],
        excluded_reason=row["excluded_reason"], superseded_by=row["superseded_by"], quote=row["quote"],
    )


def _event(row: Mapping) -> Event:
    return Event(
        event_id=row["id"], ts=row["ts"], space=row["space"], kind=row["kind"], payload=dict(row["payload"]),
        schema_version=row["schema_version"], dedup_key=row["dedup_key"],
    )


def _same_payload(a, b) -> bool:
    return json.dumps(dict(a), sort_keys=True, default=str) == json.dumps(dict(b), sort_keys=True, default=str)


def _json(value) -> object:
    from psycopg.types.json import Jsonb

    return Jsonb(value, dumps=lambda v: json.dumps(v, ensure_ascii=False, default=str))


class PostgresDocumentStore:
    name = "postgres"

    DDL = """
    CREATE SCHEMA IF NOT EXISTS {s};
    CREATE TABLE IF NOT EXISTS {s}.meta (key TEXT PRIMARY KEY, value TEXT NOT NULL);
    CREATE TABLE IF NOT EXISTS {s}.episodes (
        id BIGSERIAL PRIMARY KEY, space TEXT NOT NULL, kind TEXT NOT NULL, content TEXT NOT NULL, content_hash TEXT NOT NULL,
        source TEXT, tags JSONB NOT NULL DEFAULT '[]', metadata JSONB NOT NULL DEFAULT '{{}}',
        created_at TEXT NOT NULL, ingested_at TEXT NOT NULL, UNIQUE (space, content_hash));
    CREATE INDEX IF NOT EXISTS episodes_recent ON {s}.episodes (space, created_at DESC, id DESC);
    CREATE TABLE IF NOT EXISTS {s}.chunks (
        id BIGSERIAL PRIMARY KEY, episode_id BIGINT NOT NULL REFERENCES {s}.episodes(id) ON DELETE CASCADE,
        space TEXT NOT NULL, ordinal INTEGER NOT NULL, start_off INTEGER NOT NULL, end_off INTEGER NOT NULL,
        text TEXT NOT NULL, created_at TEXT NOT NULL,
        tsv TSVECTOR GENERATED ALWAYS AS (to_tsvector('simple', text)) STORED);
    CREATE INDEX IF NOT EXISTS chunks_tsv ON {s}.chunks USING GIN (tsv);
    CREATE INDEX IF NOT EXISTS chunks_episode ON {s}.chunks (episode_id);
    CREATE INDEX IF NOT EXISTS chunks_window ON {s}.chunks (space, episode_id, ordinal, id);
    CREATE INDEX IF NOT EXISTS chunks_space_created ON {s}.chunks (space, created_at);
    CREATE TABLE IF NOT EXISTS {s}.facts (
        id BIGSERIAL PRIMARY KEY, space TEXT NOT NULL, subject TEXT NOT NULL, predicate TEXT NOT NULL, object TEXT NOT NULL,
        confidence DOUBLE PRECISION NOT NULL, valid_from TEXT NOT NULL, valid_until TEXT, status TEXT NOT NULL,
        closed_reason TEXT, source_episode_id BIGINT, origin TEXT NOT NULL DEFAULT 'stated', superseded_by BIGINT,
        excluded_reason TEXT, quote TEXT);
    CREATE INDEX IF NOT EXISTS facts_key ON {s}.facts (space, subject, predicate);
    CREATE INDEX IF NOT EXISTS facts_subject_id ON {s}.facts (space, subject, id);
    CREATE INDEX IF NOT EXISTS facts_space_id ON {s}.facts (space, id);
    CREATE INDEX IF NOT EXISTS facts_source_id ON {s}.facts (space, source_episode_id, id);
    CREATE TABLE IF NOT EXISTS {s}.fact_links (
        id BIGSERIAL PRIMARY KEY, space TEXT NOT NULL, from_fact BIGINT NOT NULL, to_fact BIGINT NOT NULL, kind TEXT NOT NULL,
        created_at TEXT NOT NULL, source_episode_id BIGINT, quote TEXT, UNIQUE (space, from_fact, to_fact, kind));
    CREATE INDEX IF NOT EXISTS fact_links_to ON {s}.fact_links (space, to_fact);
    CREATE INDEX IF NOT EXISTS fact_links_from_id ON {s}.fact_links (space, from_fact, id);
    CREATE INDEX IF NOT EXISTS fact_links_to_id ON {s}.fact_links (space, to_fact, id);
    CREATE TABLE IF NOT EXISTS {s}.fact_affirmations (
        id BIGSERIAL PRIMARY KEY, space TEXT NOT NULL, fact_id BIGINT NOT NULL, valid_from TEXT NOT NULL,
        recorded_at TEXT NOT NULL, confidence DOUBLE PRECISION NOT NULL, source_episode_id BIGINT,
        origin TEXT NOT NULL, quote TEXT, links TEXT NOT NULL DEFAULT '[]', UNIQUE (space, fact_id, valid_from));
    -- A schema from the build that first kept affirmations, before they
    -- carried links: each it holds has none.
    ALTER TABLE {s}.fact_affirmations ADD COLUMN IF NOT EXISTS links TEXT NOT NULL DEFAULT '[]';
    CREATE TABLE IF NOT EXISTS {s}.tombstones (
        space TEXT NOT NULL, episode_id BIGINT NOT NULL, content_hash TEXT NOT NULL, forgotten_at TEXT NOT NULL,
        reason TEXT, PRIMARY KEY (space, episode_id));
    CREATE INDEX IF NOT EXISTS tombstones_hash ON {s}.tombstones (space, content_hash);
    CREATE TABLE IF NOT EXISTS {s}.revisions (space TEXT PRIMARY KEY, revision BIGINT NOT NULL);
    CREATE TABLE IF NOT EXISTS {s}.inflight (space TEXT NOT NULL, content_hash TEXT NOT NULL, PRIMARY KEY (space, content_hash));
    CREATE TABLE IF NOT EXISTS {s}.erased_spaces (space TEXT PRIMARY KEY, erased_at TEXT NOT NULL);
    """

    def __init__(self, url: str, schema: str = "scone", pool: Optional[Pool] = None) -> None:
        #: The connection a transaction holds for the task inside ``atomic``;
        #: other tasks keep borrowing their own.
        self._held: ContextVar[Optional[Any]] = ContextVar(f"scone_postgres_{id(self)}", default=None)
        self.schema = _ident(schema)
        self.pool = pool or Pool(url)
        self.pool.users += 1
        self.url = url

    def vectors(self) -> "PostgresVectorIndex":
        return PostgresVectorIndex(self.url, self.schema, pool=self.pool)

    def events(self, max_age_days: Optional[float] = None, clock: Callable[[], str] = now_rfc3339) -> "PostgresEventLog":
        return PostgresEventLog(self.url, self.schema, max_age_days=max_age_days, pool=self.pool, clock=clock)

    async def open(self) -> "PostgresDocumentStore":
        await self.pool.open()
        async with self.pool.connection() as conn:
            await conn.execute(self.DDL.format(s=self.schema))
        await self.check_schema()
        return self

    async def schema_version(self) -> int:
        async with self.pool.connection() as conn:
            row = await (await conn.execute(f"SELECT value FROM {self.schema}.meta WHERE key = 'schema_version'")).fetchone()
            if row:
                return int(row[0])
            has_rows = await (await conn.execute(f"SELECT 1 FROM {self.schema}.episodes LIMIT 1")).fetchone() is not None
        return 1 if has_rows else SCHEMA_VERSION

    async def check_schema(self) -> None:
        version = await self.schema_version()
        if version != SCHEMA_VERSION:
            raise SchemaMismatch(
                f"schema {self.schema!r} holds schema v{version}; this build writes v{SCHEMA_VERSION}. "
                "scone-memory is pre-release and does not migrate: export with the build that wrote it, "
                "drop the schema, and import again."
            )
        async with self.pool.connection() as conn:
            await conn.execute(
                f"INSERT INTO {self.schema}.meta (key, value) VALUES ('schema_version', %s) ON CONFLICT (key) DO UPDATE SET value = EXCLUDED.value",
                (str(SCHEMA_VERSION),),
            )

    async def drop(self) -> None:
        async with self.pool.connection() as conn:
            await conn.execute(f"DROP SCHEMA IF EXISTS {self.schema} CASCADE")

    async def close(self) -> None:
        await self.pool.release()

    @asynccontextmanager
    async def atomic(self) -> AsyncIterator[None]:
        """Writes made inside, by this task, commit together at the end or
        roll back together if anything inside fails."""
        if self._held.get() is not None:
            yield
            return
        async with self.pool.connection() as conn:
            async with conn.transaction():
                token = self._held.set(conn)
                try:
                    yield
                finally:
                    self._held.reset(token)

    async def _rows(self, sql: str, params: Sequence = ()) -> list[dict]:
        from psycopg.rows import dict_row

        held = self._held.get()
        if held is not None:
            cur = await held.execute(sql, params)
            cur.row_factory = dict_row
            return await cur.fetchall()
        async with self.pool.connection() as conn:
            cur = await conn.execute(sql, params)
            cur.row_factory = dict_row  # type: ignore[assignment]
            return await cur.fetchall()

    async def _row(self, sql: str, params: Sequence = ()) -> Optional[dict]:
        rows = await self._rows(sql, params)
        return rows[0] if rows else None

    async def _required_row(self, sql: str, params: Sequence = ()) -> dict:
        row = await self._row(sql, params)
        if row is None:
            raise RuntimeError("PostgreSQL operation did not return its expected row")
        return row

    async def insert_episode(self, new: NewEpisode) -> Episode:
        row = await self._required_row(
            f"INSERT INTO {self.schema}.episodes (space, kind, content, content_hash, source, tags, metadata, created_at, ingested_at)"
            " VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s) RETURNING *",
            (new.space, new.kind, new.content, new.content_hash, new.source, _json(list(new.tags)), _json(dict(new.metadata)),
             new.created_at, new.ingested_at),
        )
        return _episode(row)

    async def episode_by_hash(self, space: str, content_hash: str) -> Optional[Episode]:
        row = await self._row(f"SELECT * FROM {self.schema}.episodes WHERE space = %s AND content_hash = %s", (space, content_hash))
        return _episode(row) if row else None

    async def get_episode(self, space: str, episode_id: int) -> Optional[Episode]:
        row = await self._row(f"SELECT * FROM {self.schema}.episodes WHERE space = %s AND id = %s", (space, episode_id))
        return _episode(row) if row else None

    async def delete_space(self, space: str, erased_at: str) -> DeletedSpace:
        s = self.schema
        chunk_ids = tuple(r["id"] for r in await self._rows(f"SELECT id FROM {s}.chunks WHERE space = %s ORDER BY id", (space,)))

        async def count(table: str) -> int:
            row = await self._row(f"SELECT count(*) AS n FROM {s}.{table} WHERE space = %s", (space,))
            return int(row["n"]) if row else 0

        gone = DeletedSpace(chunk_ids=chunk_ids, episodes=await count("episodes"), facts=await count("facts"),
                            links=await count("fact_links"), tombstones=await count("tombstones"))
        for table in ("chunks", "episodes", "fact_links", "fact_affirmations", "facts", "tombstones", "inflight",
                      "revisions"):
            await self._rows(f"DELETE FROM {s}.{table} WHERE space = %s RETURNING space", (space,))
        await self._rows(
            f"INSERT INTO {s}.erased_spaces (space, erased_at) VALUES (%s, %s)"
            " ON CONFLICT (space) DO UPDATE SET erased_at = EXCLUDED.erased_at RETURNING space",
            (space, erased_at),
        )
        return gone

    async def space_deleted(self, space: str) -> Optional[str]:
        row = await self._row(f"SELECT erased_at FROM {self.schema}.erased_spaces WHERE space = %s", (space,))
        return row["erased_at"] if row else None

    async def delete_episode(self, space: str, episode_id: int) -> list[int]:
        if await self.get_episode(space, episode_id) is None:
            return []
        chunk_ids = [r["id"] for r in await self._rows(f"SELECT id FROM {self.schema}.chunks WHERE episode_id = %s ORDER BY id", (episode_id,))]
        await self._rows(f"DELETE FROM {self.schema}.episodes WHERE id = %s AND space = %s RETURNING id", (episode_id, space))
        return chunk_ids

    async def insert_chunks(self, new: Sequence[NewChunk]) -> list[Chunk]:
        out = []
        for n in new:
            row = await self._required_row(
                f"INSERT INTO {self.schema}.chunks (episode_id, space, ordinal, start_off, end_off, text, created_at)"
                " VALUES (%s, %s, %s, %s, %s, %s, %s) RETURNING *",
                (n.episode_id, n.space, n.ordinal, n.start, n.end, n.text, n.created_at),
            )
            out.append(_chunk(row))
        return out

    async def get_chunks(self, space: str, chunk_ids: Sequence[int]) -> list[Chunk]:
        if not chunk_ids:
            return []
        rows = await self._rows(f"SELECT * FROM {self.schema}.chunks WHERE space = %s AND id = ANY(%s)", (space, list(chunk_ids)))
        return [_chunk(r) for r in rows]

    async def chunks_of(self, space: str, episode_id: int) -> list[Chunk]:
        rows = await self._rows(
            f"SELECT * FROM {self.schema}.chunks WHERE space = %s AND episode_id = %s ORDER BY ordinal", (space, episode_id)
        )
        return [_chunk(r) for r in rows]

    async def page_chunks(self, space: str, episode_id: int, *, start_ordinal: int, limit: int) -> list[Chunk]:
        from ..core.chunk_window import validate_chunk_window
        validate_chunk_window(episode_id, start_ordinal, limit)
        rows = await self._rows(
            f'SELECT * FROM {self.schema}.chunks WHERE space = %s AND episode_id = %s AND ordinal >= %s ORDER BY ordinal, id LIMIT %s',
            (space, episode_id, start_ordinal, limit))
        return [_chunk(row) for row in rows]

    async def mark_inflight(self, space: str, content_hash: str) -> None:
        await self._rows(
            f"INSERT INTO {self.schema}.inflight (space, content_hash) VALUES (%s, %s) ON CONFLICT DO NOTHING RETURNING space",
            (space, content_hash),
        )

    async def clear_inflight(self, space: str, content_hash: str) -> None:
        await self._rows(
            f"DELETE FROM {self.schema}.inflight WHERE space = %s AND content_hash = %s RETURNING space", (space, content_hash)
        )

    async def inflight(self) -> list[tuple[str, str]]:
        rows = await self._rows(f"SELECT space, content_hash FROM {self.schema}.inflight ORDER BY space, content_hash")
        return [(r["space"], r["content_hash"]) for r in rows]

    async def search_text(self, space: str, query: str, limit: int, filter: TextFilter) -> list[tuple[int, float]]:
        terms = tokenize(query)
        if not terms:
            return []
        # Terms are plain lowercase words from the shared tokenizer, so a
        # websearch query of "a OR b" is safe to build from them.
        sql = (
            f"SELECT c.id AS id, ts_rank(c.tsv, q) AS rank FROM {self.schema}.chunks c"
            f" JOIN {self.schema}.episodes e ON e.id = c.episode_id, websearch_to_tsquery('simple', %s) q"
            " WHERE c.space = %s AND c.tsv @@ q"
        )
        params: list[object] = [" OR ".join(terms), space]
        if filter.as_of:
            sql += " AND c.created_at <= %s"
            params.append(filter.as_of)
        # Every scope condition applies before LIMIT so unrelated memories
        # cannot crowd matching candidates out of the lexical lane.
        for tag in filter.tags:
            sql += " AND e.tags @> %s"
            params.append(_json([tag]))
        if filter.where:
            sql += " AND e.metadata @> %s"
            params.append(_json(dict(filter.where)))
        sql += " ORDER BY rank DESC, c.id ASC LIMIT %s"
        params.append(limit)
        return [(r["id"], float(r["rank"])) for r in await self._rows(sql, params)]

    async def recent_episodes(self, space: str, limit: int) -> list[Episode]:
        rows = await self._rows(
            f"SELECT * FROM {self.schema}.episodes WHERE space = %s ORDER BY created_at DESC, id DESC LIMIT %s", (space, limit)
        )
        return [_episode(r) for r in rows]

    async def page_episodes(self, space, before, limit, kind):
        sql, params = f"SELECT * FROM {self.schema}.episodes WHERE space = %s", [space]
        if before is not None:
            sql += " AND id < %s"
            params.append(before)
        if kind is not None:
            sql += " AND kind = %s"
            params.append(kind)
        sql += " ORDER BY id DESC LIMIT %s"
        params.append(limit)
        return [_episode(r) for r in await self._rows(sql, params)]

    async def counts(self, space: str) -> SpaceCounts:
        counts = SpaceCounts()
        row = await self._required_row(
            f"SELECT COUNT(*) AS n, COALESCE(SUM(octet_length(content)), 0) AS b FROM {self.schema}.episodes WHERE space = %s", (space,)
        )
        counts.episodes, counts.bytes = int(row["n"]), int(row["b"])
        counts.chunks = int((await self._required_row(f"SELECT COUNT(*) AS n FROM {self.schema}.chunks WHERE space = %s", (space,)))["n"])
        for r in await self._rows(
            f"SELECT t AS tag, COUNT(*) AS n FROM {self.schema}.episodes, jsonb_array_elements_text(tags) t WHERE space = %s GROUP BY t",
            (space,),
        ):
            counts.tags[r["tag"]] = int(r["n"])
        return counts

    async def insert_fact(self, new: NewFact) -> Fact:
        row = await self._required_row(
            f"INSERT INTO {self.schema}.facts (space, subject, predicate, object, confidence, valid_from, valid_until, status,"
            " closed_reason, source_episode_id, origin, superseded_by, excluded_reason, quote)"
            " VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s) RETURNING *",
            (new.space, new.subject, new.predicate, new.object, new.confidence, new.valid_from, new.valid_until, new.status,
             new.closed_reason, new.source_episode_id, new.origin, new.superseded_by, new.excluded_reason, new.quote),
        )
        return _fact(row)

    async def update_fact(self, fact: Fact) -> None:
        rows = await self._rows(
            f"UPDATE {self.schema}.facts SET object = %s, confidence = %s, valid_from = %s, valid_until = %s, status = %s,"
            " closed_reason = %s, origin = %s, excluded_reason = %s, superseded_by = %s WHERE id = %s AND space = %s RETURNING id",
            (fact.object, fact.confidence, fact.valid_from, fact.valid_until, fact.status, fact.closed_reason, fact.origin,
             fact.excluded_reason, fact.superseded_by, fact.fact_id, fact.space),
        )
        if not rows:
            raise KeyError(fact.fact_id)

    async def get_fact(self, space: str, fact_id: int) -> Optional[Fact]:
        row = await self._row(f"SELECT * FROM {self.schema}.facts WHERE space = %s AND id = %s", (space, fact_id))
        return _fact(row) if row else None

    async def list_facts(self, space: str, include_closed: bool) -> list[Fact]:
        sql = f"SELECT * FROM {self.schema}.facts WHERE space = %s"
        if not include_closed:
            sql += " AND status = 'active'"
        return [_fact(r) for r in await self._rows(sql + " ORDER BY id", (space,))]

    async def page_facts(self, space: str, before_id: int | None, limit: int) -> list[Fact]:
        """Newest first below the cursor, on facts_space_id."""
        from ..core.graph_read import ledger_page_limit
        cap = ledger_page_limit(limit)
        if not cap:
            return []
        if before_id is None:
            rows = await self._rows(f'SELECT * FROM {self.schema}.facts WHERE space = %s ORDER BY id DESC LIMIT %s',
                (space, cap))
        else:
            rows = await self._rows(f'SELECT * FROM {self.schema}.facts WHERE space = %s AND id < %s '
                'ORDER BY id DESC LIMIT %s', (space, before_id, cap))
        return [_fact(row) for row in rows]

    async def facts_for_graph(self, space: str, source_episode_id: int | None, limit: int) -> list[Fact]:
        from ..core.graph_read import graph_fact_read_limit
        cap = graph_fact_read_limit(source_episode_id, limit)
        if source_episode_id is None:
            rows = await self._rows(f'SELECT * FROM {self.schema}.facts WHERE space = %s ORDER BY id LIMIT %s',
                (space, cap))
        else:
            rows = await self._rows(f'SELECT * FROM {self.schema}.facts WHERE space = %s '
                'AND source_episode_id = %s ORDER BY id LIMIT %s', (space, source_episode_id, cap))
        return [_fact(row) for row in rows]

    async def facts_for(self, space: str, subject: str, predicate: str) -> list[Fact]:
        rows = await self._rows(
            f"SELECT * FROM {self.schema}.facts WHERE space = %s AND subject = %s AND predicate = %s ORDER BY id", (space, subject, predicate)
        )
        return [_fact(r) for r in rows]

    async def facts_by_subject(self, space: str, subject: str, limit: int) -> list[Fact]:
        rows = await self._rows(
            f"SELECT * FROM {self.schema}.facts WHERE space = %s AND subject = %s ORDER BY id LIMIT %s",
            (space, subject, max(0, min(limit, 129))),
        )
        return [_fact(row) for row in rows]

    async def record_tombstone(self, new: NewTombstone) -> Tombstone:
        await self._rows(
            f"INSERT INTO {self.schema}.tombstones (space, episode_id, content_hash, forgotten_at, reason)"
            " VALUES (%s, %s, %s, %s, %s) ON CONFLICT (space, episode_id) DO NOTHING RETURNING space",
            (new.space, new.episode_id, new.content_hash, new.forgotten_at, new.reason),
        )
        row = await self._required_row(f"SELECT * FROM {self.schema}.tombstones WHERE space = %s AND episode_id = %s", (new.space, new.episode_id))
        return _tombstone(row)

    async def tombstone(self, space: str, episode_id: int) -> Optional[Tombstone]:
        row = await self._row(f"SELECT * FROM {self.schema}.tombstones WHERE space = %s AND episode_id = %s", (space, episode_id))
        return _tombstone(row) if row else None

    async def tombstone_by_hash(self, space: str, content_hash: str) -> Optional[Tombstone]:
        row = await self._row(
            f"SELECT * FROM {self.schema}.tombstones WHERE space = %s AND content_hash = %s ORDER BY episode_id DESC LIMIT 1",
            (space, content_hash),
        )
        return _tombstone(row) if row else None

    async def list_tombstones(self, space: str) -> list[Tombstone]:
        rows = await self._rows(f"SELECT * FROM {self.schema}.tombstones WHERE space = %s ORDER BY episode_id", (space,))
        return [_tombstone(r) for r in rows]

    async def add_affirmation(self, new: NewAffirmation) -> Affirmation:
        row = await self._row(
            f"INSERT INTO {self.schema}.fact_affirmations (space, fact_id, valid_from, recorded_at, confidence,"
            " source_episode_id, origin, quote, links) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)"
            " ON CONFLICT (space, fact_id, valid_from) DO NOTHING RETURNING *",
            (new.space, new.fact_id, new.valid_from, new.recorded_at, new.confidence, new.source_episode_id,
             new.origin, new.quote, json.dumps(stored_links(new.links))),
        )
        if row is None:
            row = await self._required_row(
                f"SELECT * FROM {self.schema}.fact_affirmations WHERE space = %s AND fact_id = %s AND valid_from = %s",
                (new.space, new.fact_id, new.valid_from),
            )
        return _affirmation(row)

    async def affirmations(self, space: str, fact_id: int) -> list[Affirmation]:
        rows = await self._rows(
            f"SELECT * FROM {self.schema}.fact_affirmations WHERE space = %s AND fact_id = %s ORDER BY valid_from, id",
            (space, fact_id),
        )
        return [_affirmation(row) for row in rows]

    async def drop_affirmations(self, space: str, affirmation_ids: Sequence[int]) -> None:
        if affirmation_ids:
            await self._rows(
                f"DELETE FROM {self.schema}.fact_affirmations WHERE space = %s AND id = ANY(%s) RETURNING id",
                (space, list(affirmation_ids)),
            )

    async def space_affirmations(self, space: str) -> list[Affirmation]:
        rows = await self._rows(f"SELECT * FROM {self.schema}.fact_affirmations WHERE space = %s ORDER BY id", (space,))
        return [_affirmation(row) for row in rows]

    async def insert_fact_link(self, new: NewFactLink) -> FactLink:
        row = await self._row(
            f"INSERT INTO {self.schema}.fact_links (space, from_fact, to_fact, kind, created_at, source_episode_id, quote)"
            " VALUES (%s, %s, %s, %s, %s, %s, %s) ON CONFLICT (space, from_fact, to_fact, kind) DO NOTHING RETURNING *",
            (new.space, new.from_fact, new.to_fact, new.kind, new.created_at, new.source_episode_id, new.quote),
        )
        if row is None:
            row = await self._required_row(
                f"SELECT * FROM {self.schema}.fact_links WHERE space = %s AND from_fact = %s AND to_fact = %s AND kind = %s",
                (new.space, new.from_fact, new.to_fact, new.kind),
            )
        return _fact_link(row)

    async def fact_links(self, space: str, fact_id: int) -> list[FactLink]:
        rows = await self._rows(
            f"SELECT * FROM {self.schema}.fact_links WHERE space = %s AND (from_fact = %s OR to_fact = %s) ORDER BY id",
            (space, fact_id, fact_id),
        )
        return [_fact_link(r) for r in rows]

    async def fact_links_between(self, space: str, fact_ids: Sequence[int], limit: int) -> list[FactLink]:
        """Bounded induced graph over returned facts, never expanding to
        neighbours: the first 16 ids, at most 49 links, ascending id."""
        wanted = list(dict.fromkeys(fact_ids[:16]))
        cap = max(0, min(limit, 49))
        if not wanted or not cap:
            return []
        rows = await self._rows(f"SELECT * FROM {self.schema}.fact_links WHERE space = %s AND from_fact = ANY(%s) "
                                "AND to_fact = ANY(%s) ORDER BY id LIMIT %s", (space, wanted, wanted, cap))
        return [_fact_link(row) for row in rows]

    async def fact_links_from(self, space: str, fact_id: int, limit: int) -> list[FactLink]:
        cap = max(0, min(limit, 129))
        rows = await self._rows(
            f"(SELECT * FROM {self.schema}.fact_links WHERE space = %s AND from_fact = %s ORDER BY id LIMIT %s)"
            f" UNION (SELECT * FROM {self.schema}.fact_links WHERE space = %s AND to_fact = %s ORDER BY id LIMIT %s)"
            " ORDER BY id LIMIT %s", (space, fact_id, cap, space, fact_id, cap, cap),
        )
        return [_fact_link(row) for row in rows]

    async def get_fact_link(self, space: str, link_id: int) -> FactLink | None:
        row = await self._row(f"SELECT * FROM {self.schema}.fact_links WHERE space = %s AND id = %s", (space, link_id))
        return _fact_link(row) if row is not None else None

    async def bump_revision(self, space: str) -> int:
        row = await self._required_row(
            f"INSERT INTO {self.schema}.revisions (space, revision) VALUES (%s, 1)"
            f" ON CONFLICT (space) DO UPDATE SET revision = {self.schema}.revisions.revision + 1 RETURNING revision",
            (space,),
        )
        return int(row["revision"])

    async def revision(self, space: str) -> int:
        row = await self._row(f"SELECT revision FROM {self.schema}.revisions WHERE space = %s", (space,))
        return int(row["revision"]) if row else 0


class PostgresVectorIndex:
    """Chunk vectors in a pgvector column with an HNSW cosine index. The
    width is fixed when the table is created and recorded in ``meta``;
    another width is refused."""

    name = "postgres"

    def __init__(self, url: str, schema: str = "scone", pool: Optional[Pool] = None) -> None:
        self.schema = _ident(schema)
        self.pool = pool or Pool(url)
        self.pool.users += 1
        self.dim: Optional[int] = None

    async def ensure(self, dim: int) -> None:
        await self.pool.open()
        async with self.pool.connection() as conn:
            await conn.execute(f"CREATE SCHEMA IF NOT EXISTS {self.schema}")
            await conn.execute(f"CREATE TABLE IF NOT EXISTS {self.schema}.meta (key TEXT PRIMARY KEY, value TEXT NOT NULL)")
            row = await (await conn.execute(f"SELECT value FROM {self.schema}.meta WHERE key = 'vector_dim'")).fetchone()
            if row is not None and int(row[0]) != dim:
                raise ValueError(f"schema {self.schema!r} holds {row[0]}-d vectors, embedder makes {dim}-d")
            await conn.execute(
                f"CREATE TABLE IF NOT EXISTS {self.schema}.vectors ("
                " chunk_id BIGINT PRIMARY KEY, space TEXT NOT NULL, episode_id BIGINT NOT NULL, created_at TEXT NOT NULL,"
                " created_ts DOUBLE PRECISION NOT NULL, tags JSONB NOT NULL, meta JSONB NOT NULL,"
                f" embedding VECTOR({int(dim)}) NOT NULL)"
            )
            await conn.execute(f"CREATE INDEX IF NOT EXISTS vectors_space ON {self.schema}.vectors (space, created_ts)")
            await conn.execute(f"CREATE INDEX IF NOT EXISTS vectors_hnsw ON {self.schema}.vectors USING hnsw (embedding vector_cosine_ops)")
            await conn.execute(
                f"INSERT INTO {self.schema}.meta (key, value) VALUES ('vector_dim', %s) ON CONFLICT (key) DO NOTHING", (str(dim),)
            )
        self.dim = dim

    async def upsert(self, points: Sequence[VectorPoint]) -> None:
        from pgvector import Vector

        if not points:
            return
        for point in points:
            validate_vector(point.vector, self.dim)
        async with self.pool.connection() as conn:
            async with conn.cursor() as cur:
                await cur.executemany(
                    f"INSERT INTO {self.schema}.vectors (chunk_id, space, episode_id, created_at, created_ts, tags, meta, embedding)"
                    " VALUES (%s, %s, %s, %s, %s, %s, %s, %s) ON CONFLICT (chunk_id) DO UPDATE SET space = EXCLUDED.space,"
                    " episode_id = EXCLUDED.episode_id, created_at = EXCLUDED.created_at, created_ts = EXCLUDED.created_ts,"
                    " tags = EXCLUDED.tags, meta = EXCLUDED.meta, embedding = EXCLUDED.embedding",
                    [
                        (p.chunk_id, p.space, p.episode_id, p.created_at, epoch_seconds(p.created_at), _json(list(p.tags)),
                         _json(dict(p.metadata)), Vector(list(p.vector)))
                        for p in points
                    ],
                )

    async def search(
        self,
        space: str,
        vector: Sequence[float],
        limit: int,
        as_of: Optional[str] = None,
        tags: tuple[str, ...] = (),
        where: Mapping[str, str] | None = None,
    ) -> list[tuple[int, float]]:
        from pgvector import Vector

        validate_vector(vector, self.dim)
        sql = f"SELECT chunk_id, 1 - (embedding <=> %s) AS similarity FROM {self.schema}.vectors WHERE space = %s"
        params: list[object] = [Vector(list(vector)), space]
        if as_of:
            sql += " AND created_ts <= %s"
            params.append(epoch_seconds(as_of))
        for tag in tags:
            sql += " AND tags @> %s"
            params.append(_json([tag]))
        if where:
            sql += " AND meta @> %s"
            params.append(_json(dict(where)))
        sql += " ORDER BY similarity DESC, chunk_id ASC LIMIT %s"
        params.append(limit)
        async with self.pool.connection() as conn:
            rows = await (await conn.execute(sql, params)).fetchall()
        ranked = [(int(r[0]), float(r[1])) for r in rows]
        ranked.sort(key=lambda pair: (-pair[1], pair[0]))
        return ranked

    async def delete(self, chunk_ids: Sequence[int]) -> None:
        if not chunk_ids:
            return
        async with self.pool.connection() as conn:
            await conn.execute(f"DELETE FROM {self.schema}.vectors WHERE chunk_id = ANY(%s)", ([int(c) for c in chunk_ids],))

    async def delete_space(self, space: str) -> None:
        async with self.pool.connection() as conn:
            await conn.execute(f"DELETE FROM {self.schema}.vectors WHERE space = %s", (space,))

    async def drop(self) -> None:
        if not self.pool.opened:
            return  # the schema went with the store that owned the pool
        async with self.pool.connection() as conn:
            await conn.execute(f"DROP TABLE IF EXISTS {self.schema}.vectors")
            await conn.execute(
                f"DO $$ BEGIN IF to_regclass('{self.schema}.meta') IS NOT NULL THEN"
                f" DELETE FROM {self.schema}.meta WHERE key = 'vector_dim'; END IF; END $$"
            )

    async def close(self) -> None:
        await self.pool.release()


class PostgresEventLog:
    """Evidence in an ``events`` table. ``dedup_key`` is unique per space
    through a partial index, so the conflict check is the database's.
    Retention is a sweep: events older than ``max_age_days`` are deleted
    on open and every ``SWEEP_EVERY`` appends, by the log's own clock."""

    name = "postgres"
    SWEEP_EVERY = 500

    DDL = """
    CREATE SCHEMA IF NOT EXISTS {s};
    CREATE TABLE IF NOT EXISTS {s}.events (
        id BIGSERIAL PRIMARY KEY, ts TEXT NOT NULL, space TEXT NOT NULL, kind TEXT NOT NULL,
        schema_version INTEGER NOT NULL, payload JSONB NOT NULL, dedup_key TEXT);
    CREATE UNIQUE INDEX IF NOT EXISTS events_dedup ON {s}.events (space, dedup_key) WHERE dedup_key IS NOT NULL;
    CREATE INDEX IF NOT EXISTS events_space_kind ON {s}.events (space, kind, id DESC);
    CREATE INDEX IF NOT EXISTS events_space_ts ON {s}.events (space, ts);
    """

    def __init__(
        self,
        url: str,
        schema: str = "scone",
        max_age_days: Optional[float] = None,
        pool: Optional[Pool] = None,
        clock: Callable[[], str] = now_rfc3339,
    ) -> None:
        self.schema = _ident(schema)
        self.pool = pool or Pool(url)
        self.pool.users += 1
        self.max_age_days = max_age_days
        self.clock = clock
        self._appends = 0

    async def open(self) -> "PostgresEventLog":
        await self.pool.open()
        async with self.pool.connection() as conn:
            await conn.execute(self.DDL.format(s=self.schema))
        await self.sweep()
        return self

    async def drop(self) -> None:
        if not self.pool.opened:
            return  # the schema went with the store that owned the pool
        async with self.pool.connection() as conn:
            await conn.execute(f"DROP TABLE IF EXISTS {self.schema}.events")

    async def close(self) -> None:
        await self.pool.release()

    async def _rows(self, sql: str, params: Sequence = ()) -> list[dict]:
        from psycopg.rows import dict_row

        async with self.pool.connection() as conn:
            cur = await conn.execute(sql, params)
            cur.row_factory = dict_row  # type: ignore[assignment]
            return await cur.fetchall() if cur.description else []

    async def append(self, new: NewEvent) -> Event:
        rows = await self._rows(
            f"INSERT INTO {self.schema}.events (ts, space, kind, schema_version, payload, dedup_key) VALUES (%s, %s, %s, %s, %s, %s)"
            " ON CONFLICT (space, dedup_key) WHERE dedup_key IS NOT NULL DO NOTHING RETURNING id",
            (new.ts, new.space, new.kind, new.schema_version, _json(dict(new.payload)), new.dedup_key),
        )
        if not rows:
            [existing] = await self._rows(
                f"SELECT * FROM {self.schema}.events WHERE space = %s AND dedup_key = %s", (new.space, new.dedup_key)
            )
            stored = _event(existing)
            if _same_payload(stored.payload, new.payload):
                return stored
            raise DuplicateEvent(stored)
        self._appends += 1
        if self.max_age_days is not None and self._appends % self.SWEEP_EVERY == 0:
            await self.sweep()
        return Event(event_id=int(rows[0]["id"]), **new.__dict__)

    async def purge(self, space: str, *, preview: bool = False) -> int:
        if preview:
            rows = await self._rows(f"SELECT count(*) AS n FROM {self.schema}.events WHERE space = %s", (space,))
            return int(rows[0]["n"]) if rows else 0
        return len(await self._rows(f"DELETE FROM {self.schema}.events WHERE space = %s RETURNING id", (space,)))

    async def sweep(self) -> int:
        """Delete events older than max_age_days; returns how many."""
        if self.max_age_days is None:
            return 0
        cutoff = parse_rfc3339(self.clock()).timestamp() - self.max_age_days * 86400
        cutoff_text = format_rfc3339(datetime.fromtimestamp(cutoff, tz=timezone.utc))
        rows = await self._rows(f"DELETE FROM {self.schema}.events WHERE ts < %s RETURNING id", (cutoff_text,))
        return len(rows)

    async def get(self, space: str, event_id: int) -> Optional[Event]:
        rows = await self._rows(f"SELECT * FROM {self.schema}.events WHERE space = %s AND id = %s", (space, event_id))
        return _event(rows[0]) if rows else None

    async def query(
        self, space: str, kind: Optional[str] = None, since: Optional[str] = None, limit: int = 100, after_id: Optional[int] = None
    ) -> list[Event]:
        sql = f"SELECT * FROM {self.schema}.events WHERE space = %s"
        params: list[object] = [space]
        if kind:
            sql += " AND kind = %s"
            params.append(kind)
        if since:
            sql += " AND ts >= %s"
            params.append(format_rfc3339(parse_rfc3339(since)))
        if after_id is not None:
            sql += " AND id > %s ORDER BY id ASC"
            params.append(after_id)
        else:
            sql += " ORDER BY id DESC"
        sql += " LIMIT %s"
        params.append(limit)
        return [_event(r) for r in await self._rows(sql, params)]
