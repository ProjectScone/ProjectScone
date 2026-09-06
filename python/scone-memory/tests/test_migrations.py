"""Shared spec 3.6, pre-release form: every store stamps the schema version
it writes and refuses a store another build wrote. Nothing is migrated
until there is a release to migrate from."""

from __future__ import annotations

import os
import sqlite3

import pytest

from scone_memory.backends.sqlite import SCHEMA, SCHEMA_VERSION, SchemaMismatch, SqliteDocumentStore, schema_version


def test_a_fresh_file_is_stamped_with_the_current_version(tmp_path):
    store = SqliteDocumentStore(tmp_path / "new.db")
    assert schema_version(store.conn) == SCHEMA_VERSION
    again = SqliteDocumentStore(tmp_path / "new.db")  # reopening the same build's file is fine
    assert schema_version(again.conn) == SCHEMA_VERSION


def test_a_file_from_another_build_is_refused_not_rewritten(tmp_path):
    path = tmp_path / "old.db"
    conn = sqlite3.connect(path)
    conn.executescript(SCHEMA)
    conn.execute("INSERT OR REPLACE INTO meta (key, value) VALUES ('schema_version', '2')")
    conn.execute(
        "INSERT INTO episodes (id, space, kind, content, content_hash, tags, metadata, created_at, ingested_at)"
        " VALUES (1, 'default', 'chat', 'old row', 'h', '[]', '{}', '2025-01-01T00:00:00.000Z', '2025-01-01T00:00:00.000Z')"
    )
    conn.commit()
    conn.close()
    with pytest.raises(SchemaMismatch, match="holds schema v2"):
        SqliteDocumentStore(path)
    probe = sqlite3.connect(path)
    assert probe.execute("SELECT kind FROM episodes").fetchone()[0] == "chat", "the file was left as it was"


def test_an_unversioned_file_with_rows_counts_as_v1(tmp_path):
    path = tmp_path / "unversioned.db"
    conn = sqlite3.connect(path)
    conn.executescript(SCHEMA)
    conn.execute(
        "INSERT INTO episodes (id, space, kind, content, content_hash, tags, metadata, created_at, ingested_at)"
        " VALUES (1, 'default', 'note', 'row', 'h', '[]', '{}', '2025-01-01T00:00:00.000Z', '2025-01-01T00:00:00.000Z')"
    )
    conn.commit()
    conn.close()
    with pytest.raises(SchemaMismatch, match="holds schema v1"):
        SqliteDocumentStore(path)


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
