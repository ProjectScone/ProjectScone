"""Shared spec 3.6. A store stamps the schema version it writes. A file
behind by known steps is walked forward additively, one transaction and
one backup per step; anything else is refused, never rewritten. The
first step exists because a live store held v5 with real data; the
second adds the inflight table for crash recovery."""

from __future__ import annotations

import os
import re
import sqlite3

import pytest

from scone_memory.backends.sqlite import SCHEMA, SCHEMA_VERSION, STEPS, SchemaMismatch, SqliteDocumentStore, schema_version
from scone_memory.observability.events import SqliteEventLog

ROW = ("INSERT INTO episodes (id, space, kind, content, content_hash, tags, metadata, created_at, ingested_at)"
       " VALUES (?, 'default', 'note', ?, ?, '[]', '{}', '2025-01-01T00:00:00.000Z', '2025-01-01T00:00:00.000Z')")


def write_v5_file(path):
    """A file as the 2026-09-06 live store looked: schema 5, facts without a
    quote column and no inflight table, plus events and an index, so the
    steps must leave all of them exactly as they were."""
    conn = sqlite3.connect(path)
    v5_schema = SCHEMA.replace(", superseded_by INTEGER, quote TEXT);", ", superseded_by INTEGER);")
    v5_schema = re.sub(r"CREATE TABLE IF NOT EXISTS fact_links \(.*?\);\n", "", v5_schema, flags=re.S)
    v5_schema = "\n".join(line for line in v5_schema.splitlines() if "inflight" not in line and "fact_links" not in line)
    assert "quote" not in v5_schema and "inflight" not in v5_schema and "fact_links" not in v5_schema, "the v5 shape must not carry the new parts"
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

    store = SqliteDocumentStore(path)  # opening applies every step, 5 -> 6 -> 7 -> 8
    assert schema_version(store.conn) == SCHEMA_VERSION == 8
    cols = [r[1] for r in store.conn.execute("PRAGMA table_info(facts)")]
    assert "quote" in cols
    assert store.conn.execute("SELECT count(*) FROM sqlite_master WHERE name = 'inflight'").fetchone()[0] == 1
    assert store.conn.execute("SELECT count(*) FROM sqlite_master WHERE name = 'fact_links'").fetchone()[0] == 1
    after = snapshot(path)
    assert after["episodes"] == before["episodes"], "episode rows and ids untouched"
    assert after["facts"] == before["facts"], "fact rows, ids, statuses, reasons untouched"
    assert after["events"] == before["events"], "events untouched"
    assert set(before["indexes"]) <= set(after["indexes"]), "no index lost"
    assert store.conn.execute("SELECT quote FROM facts WHERE id = 3").fetchone()[0] is None, "new column is null for old rows"

    backup = tmp_path / "live.db.v5.bak"
    assert backup.exists(), "a backup is written before the first step"
    assert snapshot(backup) == before
    assert sqlite3.connect(backup).execute("SELECT value FROM meta WHERE key='schema_version'").fetchone()[0] == "5"
    second = tmp_path / "live.db.v6.bak"
    assert second.exists(), "and another before the second step"
    assert sqlite3.connect(second).execute("SELECT value FROM meta WHERE key='schema_version'").fetchone()[0] == "6"
    assert snapshot(second) == before, "the v6 copy holds the same rows, one step further along"
    third = tmp_path / "live.db.v7.bak"
    assert third.exists() and snapshot(third) == before, "and one before the third"
    assert sqlite3.connect(third).execute("SELECT value FROM meta WHERE key='schema_version'").fetchone()[0] == "7"

    # The upgraded store works: it reads the old rows and writes the new column.
    import asyncio

    facts = asyncio.run(store.list_facts("default", include_closed=True))
    assert sorted(f.fact_id for f in facts) == [2, 3] and all(f.quote is None for f in facts)
    again = SqliteDocumentStore(path)  # reopening does nothing further
    assert schema_version(again.conn) == 8 and not (tmp_path / "live.db.v8.bak").exists()


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


def test_two_openers_racing_on_a_v5_file_apply_the_step_once(tmp_path):
    """Two processes starting against the same v5 file must not both ALTER.
    The second waits on the write lock, re-reads the version under it, finds
    the step done, and carries on."""
    import threading

    path = tmp_path / "live.db"
    write_v5_file(path)
    results: dict[str, object] = {}

    def opener(name):
        try:
            store = SqliteDocumentStore(path)
            results[name] = schema_version(store.conn)
        except Exception as e:  # noqa: BLE001
            results[name] = f"{type(e).__name__}: {e}"

    threads = [threading.Thread(target=opener, args=(f"t{i}",)) for i in range(4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=30)
    assert all(v == SCHEMA_VERSION for v in results.values()), results
    probe = sqlite3.connect(path)
    assert [r[1] for r in probe.execute("PRAGMA table_info(facts)")].count("quote") == 1, "the column was added exactly once"
    assert probe.execute("SELECT value FROM meta WHERE key='schema_version'").fetchone()[0] == str(SCHEMA_VERSION)


def test_a_step_finished_by_someone_else_is_recognised_under_the_lock(tmp_path):
    """The re-read inside the transaction is what makes the race safe; this
    pins it without threads by pre-applying the step and stale-calling."""
    import scone_memory.backends.sqlite as mod

    path = tmp_path / "live.db"
    write_v5_file(path)
    first = SqliteDocumentStore(path)  # applies both steps
    assert schema_version(first.conn) == SCHEMA_VERSION
    stale = sqlite3.connect(path)
    stale.row_factory = sqlite3.Row
    mod.apply_step(stale, 5, path)  # a caller that still believed the file was v5
    assert schema_version(stale) == SCHEMA_VERSION and [r[1] for r in stale.execute("PRAGMA table_info(facts)")].count("quote") == 1


def test_other_versions_are_still_refused_not_rewritten(tmp_path):
    for version in ("2", "4", "9"):  # 9 stands for a build newer than this one
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


def test_an_opener_retries_the_wal_switch_while_another_connection_is_writing(tmp_path):
    """Switching a rollback-journal file to WAL opens a read transaction and
    then upgrades it to a write. If another connection holds the reserved
    lock at that moment, SQLite refuses the upgrade at once instead of
    calling the busy handler (its deadlock rule), so several openers
    starting on such a file together failed in CI with "database is
    locked". The opener retries until the writer is gone. A connection
    holding an exclusive lock is the other case: there SQLite does wait,
    which is why this test holds a reserved lock (BEGIN IMMEDIATE)."""
    import threading
    import time

    path = tmp_path / "rollback.db"
    write_v5_file(path)  # a plain rollback-journal file, as a foreign tool would leave it
    holder = sqlite3.connect(path, check_same_thread=False)  # released from another thread below
    holder.execute("BEGIN IMMEDIATE")

    def release():
        time.sleep(0.3)
        holder.execute("COMMIT")

    threading.Thread(target=release).start()
    store = SqliteDocumentStore(path)
    assert store.conn.execute("PRAGMA journal_mode").fetchall()[0][0] == "wal"
    assert schema_version(store.conn) == SCHEMA_VERSION
