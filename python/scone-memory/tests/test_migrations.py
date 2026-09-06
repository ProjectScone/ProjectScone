"""Shared spec 3.6. A store stamps the schema version it writes. A file
exactly one known step behind is brought forward additively, in one
transaction, after a backup; anything else is refused, never rewritten.
The first step exists because a live store held v5 with real data."""

from __future__ import annotations

import os
import sqlite3

import pytest

from scone_memory.backends.sqlite import SCHEMA, SCHEMA_VERSION, STEPS, SchemaMismatch, SqliteDocumentStore, schema_version
from scone_memory.events import SqliteEventLog

ROW = ("INSERT INTO episodes (id, space, kind, content, content_hash, tags, metadata, created_at, ingested_at)"
       " VALUES (?, 'default', 'note', ?, ?, '[]', '{}', '2025-01-01T00:00:00.000Z', '2025-01-01T00:00:00.000Z')")


def write_v5_file(path):
    """A file as the 2026-09-06 live store looked: schema 5, facts without a
    quote column, plus events and an index, so the step must leave all of
    them exactly as they were."""
    conn = sqlite3.connect(path)
    v5_schema = SCHEMA.replace(", superseded_by INTEGER, quote TEXT);", ", superseded_by INTEGER);")
    assert "quote" not in v5_schema, "the v5 shape must not carry the new column"
    conn.executescript(v5_schema)
    conn.executescript(SqliteEventLog.SCHEMA)  # the live file holds its event log in the same database
    conn.execute("INSERT OR REPLACE INTO meta (key, value) VALUES ('schema_version', '5')")
    conn.execute(ROW, (1, "first", "h1"))
    conn.execute(ROW, (7, "seventh", "h7"))  # a gap in ids: they must survive as they are
    conn.execute("INSERT INTO facts (id, space, subject, predicate, object, confidence, valid_from, status, origin, superseded_by)"
                 " VALUES (3, 'default', 'ana', 'lives_in', 'Lisbon', 0.9, '2024-03-02T00:00:00.000Z', 'active', 'extracted', NULL)")
    conn.execute("INSERT INTO facts (id, space, subject, predicate, object, confidence, valid_from, valid_until, status, closed_reason, origin, superseded_by)"
                 " VALUES (2, 'default', 'ana', 'lives_in', 'Austin', 1.0, '2022-01-01T00:00:00.000Z', '2024-03-02T00:00:00.000Z', 'closed', 'superseded by fact 3', 'stated', 3)")
    conn.execute("INSERT INTO events (id, ts, space, kind, schema_version, payload) VALUES (41, '2025-01-01T00:00:00.000Z', 'default', 'recall', 1, '{\"n\": 1}')")
    conn.execute("CREATE INDEX IF NOT EXISTS facts_extra ON facts(space, status)")
    conn.commit()
    conn.close()


def snapshot(path):
    conn = sqlite3.connect(path)
    conn.row_factory = sqlite3.Row
    out = {
        "episodes": [tuple(r) for r in conn.execute("SELECT id, content FROM episodes ORDER BY id")],
        "facts": [tuple(r) for r in conn.execute("SELECT id, subject, object, status, closed_reason, superseded_by FROM facts ORDER BY id")],
        "events": [tuple(r) for r in conn.execute("SELECT id, kind, payload FROM events ORDER BY id")],
        "indexes": sorted(r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type='index' AND name NOT LIKE 'sqlite_%'")),
    }
    conn.close()
    return out


def test_a_fresh_file_is_stamped_with_the_current_version(tmp_path):
    store = SqliteDocumentStore(tmp_path / "new.db")
    assert schema_version(store.conn) == SCHEMA_VERSION
    again = SqliteDocumentStore(tmp_path / "new.db")
    assert schema_version(again.conn) == SCHEMA_VERSION


def test_the_known_step_brings_a_v5_file_forward_and_keeps_everything(tmp_path):
    path = tmp_path / "live.db"
    write_v5_file(path)
    before = snapshot(path)

    store = SqliteDocumentStore(path)  # opening applies the step
    assert schema_version(store.conn) == SCHEMA_VERSION == 6
    cols = [r[1] for r in store.conn.execute("PRAGMA table_info(facts)")]
    assert "quote" in cols
    after = snapshot(path)
    assert after["episodes"] == before["episodes"], "episode rows and ids untouched"
    assert after["facts"] == before["facts"], "fact rows, ids, statuses, reasons untouched"
    assert after["events"] == before["events"], "events untouched"
    assert set(before["indexes"]) <= set(after["indexes"]), "no index lost"
    assert store.conn.execute("SELECT quote FROM facts WHERE id = 3").fetchone()[0] is None, "new column is null for old rows"

    backup = tmp_path / "live.db.v5.bak"
    assert backup.exists(), "a backup is written before the step"
    assert snapshot(backup) == before
    assert sqlite3.connect(backup).execute("SELECT value FROM meta WHERE key='schema_version'").fetchone()[0] == "5"

    # The upgraded store works: it reads the old rows and writes the new column.
    import asyncio

    facts = asyncio.run(store.list_facts("default", include_closed=True))
    assert sorted(f.fact_id for f in facts) == [2, 3] and all(f.quote is None for f in facts)
    again = SqliteDocumentStore(path)  # reopening does nothing further
    assert schema_version(again.conn) == 6 and not (tmp_path / "live.db.v6.bak").exists()


def test_the_step_is_atomic(tmp_path, monkeypatch):
    import scone_memory.backends.sqlite as mod

    path = tmp_path / "live.db"
    write_v5_file(path)
    before = snapshot(path)
    monkeypatch.setattr(mod, "STEPS", {5: ("ALTER TABLE facts ADD COLUMN quote TEXT", "INSERT INTO no_such_table VALUES (1)")})
    with pytest.raises(sqlite3.OperationalError):
        SqliteDocumentStore(path)
    probe = sqlite3.connect(path)
    assert probe.execute("SELECT value FROM meta WHERE key='schema_version'").fetchone()[0] == "5", "version not bumped"
    assert "quote" not in [r[1] for r in probe.execute("PRAGMA table_info(facts)")], "column not added"
    assert snapshot(path) == before


def test_other_versions_are_still_refused_not_rewritten(tmp_path):
    for version in ("2", "4", "7"):
        path = tmp_path / f"v{version}.db"
        conn = sqlite3.connect(path)
        conn.executescript(SCHEMA)
        conn.execute("INSERT OR REPLACE INTO meta (key, value) VALUES ('schema_version', ?)", (version,))
        conn.execute(ROW, (1, "old row", "h"))
        conn.commit()
        conn.close()
        with pytest.raises(SchemaMismatch, match=f"holds schema v{version}"):
            SqliteDocumentStore(path)
        probe = sqlite3.connect(path)
        assert probe.execute("SELECT value FROM meta WHERE key='schema_version'").fetchone()[0] == version
        assert probe.execute("SELECT content FROM episodes").fetchone()[0] == "old row", "the file was left as it was"
        assert not (tmp_path / f"v{version}.db.v{version}.bak").exists(), "no backup for a refused file"


def test_an_unversioned_file_with_rows_counts_as_v1_and_is_refused(tmp_path):
    path = tmp_path / "unversioned.db"
    conn = sqlite3.connect(path)
    conn.executescript(SCHEMA)
    conn.execute(ROW, (1, "row", "h"))
    conn.commit()
    conn.close()
    with pytest.raises(SchemaMismatch, match="holds schema v1"):
        SqliteDocumentStore(path)


def test_steps_only_ever_add():
    """The contract of a step: additive statements only."""
    for version, statements in STEPS.items():
        for stmt in statements:
            assert stmt.upper().startswith(("ALTER TABLE", "CREATE INDEX", "CREATE TABLE")), (version, stmt)
            assert "DROP" not in stmt.upper() and "DELETE" not in stmt.upper()


@pytest.mark.mongo
@pytest.mark.skipif("SCONE_TEST_MONGO_URL" not in os.environ, reason="needs a live MongoDB")
async def test_mongo_refuses_another_builds_database():
    from pymongo import AsyncMongoClient

    from scone_memory.backends.mongo import MongoDocumentStore, SchemaMismatch as MongoMismatch

    client = AsyncMongoClient(os.environ["SCONE_TEST_MONGO_URL"])
    name = "scone_test_schema"
    await client.drop_database(name)
    await client[name]["meta"].insert_one({"_id": "schema", "version": 1})
    store = MongoDocumentStore(os.environ["SCONE_TEST_MONGO_URL"], name, client=client)
    try:
        with pytest.raises(MongoMismatch, match="holds schema v1"):
            await store.open()
    finally:
        await client.drop_database(name)
        await client.close()
