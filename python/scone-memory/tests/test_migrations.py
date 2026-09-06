"""Shared spec 3.6: rows written before a rule changed are brought forward
on open, and the file records which version it holds."""

from __future__ import annotations

import json
import os
import sqlite3

import pytest

from scone_memory.backends.sqlite import SCHEMA, SCHEMA_VERSION, SqliteDocumentStore, schema_version

CONTENT = "São Tomé ☕ harbour. " * 3  # non-ASCII: code points and bytes differ


def write_v1_file(path):
    """A file as the first release wrote it: no version, code-point spans."""
    conn = sqlite3.connect(path)
    conn.executescript(SCHEMA)
    conn.execute("DELETE FROM meta")
    conn.execute(
        "INSERT INTO episodes (id, space, kind, content, content_hash, source, tags, metadata, created_at, ingested_at)"
        " VALUES (1, 'default', 'note', ?, 'h', NULL, '[]', '{}', '2025-01-01T00:00:00.000Z', '2025-01-01T00:00:00.000Z')",
        (CONTENT,),
    )
    cut = len(CONTENT) // 2
    for i, (a, b) in enumerate(((0, cut), (cut, len(CONTENT)))):
        conn.execute(
            'INSERT INTO chunks (id, episode_id, space, ordinal, start, "end", text, created_at)'
            " VALUES (?, 1, 'default', ?, ?, ?, ?, '2025-01-01T00:00:00.000Z')",
            (i + 1, i, a, b, CONTENT[a:b]),
        )
    conn.commit()
    conn.close()
    return cut


def test_sqlite_v1_spans_become_byte_offsets_on_open(tmp_path):
    path = tmp_path / "old.db"
    cut = write_v1_file(path)
    probe = sqlite3.connect(path)
    probe.row_factory = sqlite3.Row
    assert schema_version(probe) == 1
    probe.close()

    store = SqliteDocumentStore(path)
    raw = CONTENT.encode()
    rows = store.conn.execute('SELECT start, "end", text FROM chunks ORDER BY id').fetchall()
    assert [(r["start"], r["end"]) for r in rows] == [(0, len(CONTENT[:cut].encode())), (len(CONTENT[:cut].encode()), len(raw))]
    for r in rows:
        assert raw[r["start"] : r["end"]].decode() == r["text"]
    assert schema_version(store.conn) == SCHEMA_VERSION
    assert rows[1]["start"] != cut, "code-point index would have been kept"


def test_a_fresh_file_starts_at_the_current_version(tmp_path):
    store = SqliteDocumentStore(tmp_path / "new.db")
    assert schema_version(store.conn) == SCHEMA_VERSION


@pytest.mark.mongo
@pytest.mark.skipif("SCONE_TEST_MONGO_URL" not in os.environ, reason="needs a live MongoDB")
async def test_mongo_v1_spans_become_byte_offsets_on_open():
    from pymongo import AsyncMongoClient

    from scone_memory.backends.mongo import MongoDocumentStore

    client = AsyncMongoClient(os.environ["SCONE_TEST_MONGO_URL"])
    db = client["scone_test_migration"]
    await client.drop_database(db.name)
    cut = len(CONTENT) // 2
    await db["episodes"].insert_one({"_id": 1, "space": "default", "kind": "note", "content": CONTENT, "content_hash": "h",
                                     "tags": [], "metadata": {}, "created_at": "2025-01-01T00:00:00.000Z",
                                     "ingested_at": "2025-01-01T00:00:00.000Z"})
    await db["chunks"].insert_many([
        {"_id": 1, "episode_id": 1, "space": "default", "ordinal": 0, "start": 0, "end": cut, "text": CONTENT[:cut],
         "created_at": "2025-01-01T00:00:00.000Z"},
        {"_id": 2, "episode_id": 1, "space": "default", "ordinal": 1, "start": cut, "end": len(CONTENT), "text": CONTENT[cut:],
         "created_at": "2025-01-01T00:00:00.000Z"},
    ])
    store = MongoDocumentStore(os.environ["SCONE_TEST_MONGO_URL"], db.name, client=client)
    try:
        assert await store.schema_version() == 1
        await store.open()
        raw = CONTENT.encode()
        chunks = [c async for c in db["chunks"].find({}).sort("_id", 1)]
        assert [(c["start"], c["end"]) for c in chunks] == [(0, len(CONTENT[:cut].encode())), (len(CONTENT[:cut].encode()), len(raw))]
        assert await store.schema_version() == 2
    finally:
        await store.drop()
        await store.close()
