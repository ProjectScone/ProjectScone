"""Derived exact-token postings for SQLite's optional bounded fact lookup.

Pure SQL triggers keep a dirty queue compatible with old clients and raw writes.
Only the queried space is tokenized, in batches, before an indexed lookup. The
cache is disposable and does not change the semantic schema version. Savepoints
preserve caller transactions and roll back partial cache work on any failure.

SQL orders posting matches; exact Python validity/scope checks precede the
result limit. Only selected IDs are hydrated as full Fact records by the store.
Source timestamp comparisons intentionally match the engine's existing string
comparisons; fact validity uses parsed RFC3339 offsets and microseconds.
"""
from __future__ import annotations

from collections.abc import Iterator, Mapping
from contextlib import contextmanager
import json
import sqlite3
from typing import Protocol, cast
from uuid import uuid4

from ..core.ports import TextFilter
from ..core.timeutil import parse_rfc3339
from ..retrieval.lexical import tokenize

_VERSION_KEY = "fact_search_postings_version"
_DDL = (
    "CREATE TABLE IF NOT EXISTS fact_search_postings (space TEXT NOT NULL, term TEXT NOT NULL, fact_id INTEGER NOT NULL, PRIMARY KEY(space,term,fact_id)) WITHOUT ROWID",
    "CREATE INDEX IF NOT EXISTS fact_search_by_fact ON fact_search_postings(fact_id)",
    "CREATE TABLE IF NOT EXISTS fact_search_dirty (fact_id INTEGER PRIMARY KEY, space TEXT NOT NULL)",
    "CREATE INDEX IF NOT EXISTS fact_search_dirty_space ON fact_search_dirty(space,fact_id)",
    """CREATE TRIGGER IF NOT EXISTS fact_search_insert AFTER INSERT ON facts BEGIN
        DELETE FROM fact_search_postings WHERE fact_id=NEW.id;
        INSERT OR REPLACE INTO fact_search_dirty(fact_id,space) VALUES(NEW.id,NEW.space);
        END""",
    """CREATE TRIGGER IF NOT EXISTS fact_search_update AFTER UPDATE OF id,space,subject,predicate,object ON facts BEGIN
        DELETE FROM fact_search_postings WHERE fact_id=OLD.id;
        DELETE FROM fact_search_dirty WHERE fact_id=OLD.id;
        INSERT OR REPLACE INTO fact_search_dirty(fact_id,space) VALUES(NEW.id,NEW.space);
        END""",
    """CREATE TRIGGER IF NOT EXISTS fact_search_delete AFTER DELETE ON facts BEGIN
        DELETE FROM fact_search_postings WHERE fact_id=OLD.id;
        DELETE FROM fact_search_dirty WHERE fact_id=OLD.id;
        END""",
)

# CROSS JOIN keeps posting matches outermost. SQLite otherwise may choose a
# space-wide facts scan and probe matches for every fact, even for a rare term.
RANK_SQL = """WITH matched AS (
    SELECT p.fact_id,COUNT(*) AS overlap
    FROM json_each(?) AS q CROSS JOIN fact_search_postings AS p
    ON p.space=? AND p.term=q.value
    GROUP BY p.fact_id
)
SELECT f.id,f.valid_from,f.valid_until,
    e.id AS episode_id,e.space AS episode_space,e.tags,e.metadata,
    e.kind,e.source,e.created_at
FROM matched AS m CROSS JOIN facts AS f ON f.id=m.fact_id
LEFT JOIN episodes AS e ON e.id=f.source_episode_id AND e.space=f.space
WHERE f.space=? AND f.excluded_reason IS NULL AND f.status IN ('active','closed')
ORDER BY m.overlap DESC,f.confidence DESC,f.id ASC"""

# Preserve the same point-lookup order when hydrating the bounded result set.
HYDRATE_SQL = """SELECT f.* FROM json_each(?) AS selected
CROSS JOIN facts AS f ON f.id=selected.value WHERE f.space=?"""


class _Conditions(Protocol):
    def matches(self, metadata: Mapping[str,str]) -> bool: ...


@contextmanager
def _savepoint(conn: sqlite3.Connection) -> Iterator[None]:
    name = "fact_search_"+uuid4().hex
    conn.execute(f"SAVEPOINT {name}")
    try:
        yield
    except BaseException:
        conn.execute(f"ROLLBACK TO {name}")
        conn.execute(f"RELEASE {name}")
        raise
    else:
        conn.execute(f"RELEASE {name}")


def _normalized_sql(sql: str) -> str:
    return " ".join(sql.casefold().replace("if not exists ", "").split()).rstrip(";")


def _derived_schema(conn: sqlite3.Connection) -> tuple[bool, list[sqlite3.Row]]:
    expected = {(statement.split()[1].lower(), statement.split()[5]): _normalized_sql(statement)
                for statement in _DDL}
    names = [name for _, name in expected]
    rows = conn.execute("SELECT type,name,sql FROM sqlite_master WHERE name COLLATE NOCASE IN (SELECT value FROM json_each(?))",
                        (json.dumps(names),)).fetchall()
    complete = len(rows) == len(expected) and all(
        expected.get((row["type"], row["name"].casefold())) == _normalized_sql(row["sql"] or "") for row in rows)
    return complete, rows


def initialize_fact_search(conn: sqlite3.Connection) -> None:
    """Trust the marker only when every derived definition already matches.

    Missing objects or changed triggers can hide writes that happened while the
    index was broken. Recreating an object alone cannot restore that history:
    rebuild all disposable state and queue the ledger once in the savepoint.
    """
    with _savepoint(conn):
        complete, objects = _derived_schema(conn)
        row = conn.execute("SELECT value FROM meta WHERE key=?",(_VERSION_KEY,)).fetchone()
        if complete and row is not None and row[0] == "1":
            return
        if not complete:
            # Remove only our reserved names, using each object's actual type.
            # A same-name index on a different table or an incompatible table
            # must be removed before CREATE IF NOT EXISTS can be meaningful.
            priority = {"trigger": 0, "index": 1, "view": 2, "table": 3}
            for existing in sorted(objects, key=lambda item: priority[item["type"]]):
                kind, name = existing["type"], existing["name"]
                conn.execute(f"DROP {kind} IF EXISTS {name}")
        for statement in _DDL:
            conn.execute(statement)
        conn.execute("DELETE FROM fact_search_postings")
        conn.execute("DELETE FROM fact_search_dirty")
        conn.execute("INSERT INTO fact_search_dirty(fact_id,space) SELECT id,space FROM facts")
        conn.execute("INSERT OR REPLACE INTO meta(key,value) VALUES(?,?)",(_VERSION_KEY,"1"))


def _synchronize(conn: sqlite3.Connection, space: str) -> None:
    while True:
        rows = conn.execute("""SELECT f.id,f.subject,f.predicate,f.object
            FROM fact_search_dirty AS d INDEXED BY fact_search_dirty_space
            JOIN facts AS f ON f.id=d.fact_id AND f.space=d.space
            WHERE d.space=? ORDER BY d.fact_id LIMIT 256""",(space,)).fetchall()
        if not rows:
            return
        for row in rows:
            fact_id = cast(int,row["id"])
            terms = set(tokenize(f'{row["subject"]} {row["predicate"]} {row["object"]}'))
            conn.execute("DELETE FROM fact_search_postings WHERE fact_id=?",(fact_id,))
            conn.executemany("INSERT INTO fact_search_postings(space,term,fact_id) VALUES(?,?,?)",
                ((space,term,fact_id) for term in sorted(terms)))
            conn.execute("DELETE FROM fact_search_dirty WHERE fact_id=?",(fact_id,))


def _fits(row: sqlite3.Row, space: str, scope: TextFilter | None) -> bool:
    if scope is None:
        return True
    if row["episode_id"] is None or row["episode_space"] != space:
        return False
    tags = cast(list[str],json.loads(row["tags"]))
    metadata = cast(dict[str,str],json.loads(row["metadata"]))
    created = cast(str,row["created_at"])
    source = cast(str | None,row["source"])
    if (not set(scope.tags).issubset(tags) or not all(metadata.get(key)==value for key,value in scope.where.items())
            or (scope.as_of is not None and created > scope.as_of)):
        return False
    conditions = cast(_Conditions | None,scope.conditions)
    if conditions is not None and not conditions.matches(metadata):
        return False
    if scope.kind is not None and row["kind"] != scope.kind:
        return False
    if scope.source_prefix is not None and (source is None or not source.startswith(scope.source_prefix)):
        return False
    if scope.since is not None and created < scope.since:
        return False
    return not (scope.until is not None and created > scope.until)


def search_fact_rows(conn: sqlite3.Connection, space: str, query: str, when: str, limit: int,
                     scope: TextFilter | None = None) -> list[sqlite3.Row]:
    """Return at most limit full rows after exact match/time/scope filtering."""
    from ..memory.engine import MAX_QUERY, check_space

    if type(space) is not str:
        raise ValueError("invalid fact search space")
    check_space(space)
    if type(query) is not str or not query or len(query) > MAX_QUERY:
        raise ValueError("fact search query must be 1 to 1000 characters")
    if type(limit) is not int or not 1 <= limit <= 100:
        raise ValueError("fact search limit must be an integer from 1 to 100")
    if scope is not None and not isinstance(scope,TextFilter):
        raise ValueError("invalid fact search scope")
    if type(when) is not str:
        raise ValueError("invalid fact search time")
    boundary = parse_rfc3339(when)
    terms = sorted(set(tokenize(query)))
    if not terms:
        return []
    with _savepoint(conn):
        _synchronize(conn,space)
        ids: list[int] = []
        cursor = conn.execute(RANK_SQL,(json.dumps(terms),space,space))
        try:
            for row in cursor:
                if parse_rfc3339(row["valid_from"]) > boundary:
                    continue
                if row["valid_until"] is not None and parse_rfc3339(row["valid_until"]) <= boundary:
                    continue
                if _fits(row,space,scope):
                    ids.append(cast(int,row["id"]))
                    if len(ids) == limit:
                        break
        finally:
            cursor.close()
        if not ids:
            return []
        rows = conn.execute(HYDRATE_SQL,(json.dumps(ids),space)).fetchall()
        by_id = {cast(int,row["id"]):row for row in rows}
        return [by_id[identifier] for identifier in ids]
