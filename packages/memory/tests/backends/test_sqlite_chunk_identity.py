"""Delayed vector cleanup must never target a later source's chunk."""
from __future__ import annotations

import asyncio
import sqlite3
from concurrent.futures import ThreadPoolExecutor

import pytest

from scone_memory import HashEmbedder, MemoryEngine
from scone_memory.backends.sqlite import SqliteDocumentStore, SqliteVectorIndex
from scone_memory.core.ports import NewChunk, NewEpisode

NOW = "2026-09-11T00:00:00Z"


async def source(documents, digest="first"):
    return await documents.insert_episode(NewEpisode(
        space="alpha", kind="note", content="persisted source", content_hash=digest,
        created_at=NOW, ingested_at=NOW,
    ))


def chunk(episode_id):
    return NewChunk(episode_id=episode_id, space="alpha", ordinal=0,
                    start=0, end=16, text="persisted source", created_at=NOW)


async def test_delayed_cleanup_after_reopen_cannot_delete_new_source_vector(tmp_path):
    path = tmp_path / "catalog.db"
    memory = await MemoryEngine(SqliteDocumentStore(path), SqliteVectorIndex(path), HashEmbedder()).open()
    first = await memory.remember("alpha", "old evidence")
    old_ids = await memory.documents.delete_episode("alpha", first.episode_id)
    await memory.close()
    memory = await MemoryEngine(SqliteDocumentStore(path), SqliteVectorIndex(path), HashEmbedder()).open()
    try:
        second = await memory.remember("alpha", "new unrelated evidence")
        new_ids = [c.chunk_id for c in await memory.documents.chunks_of("alpha", second.episode_id)]
        assert min(new_ids) > max(old_ids)
        await memory.vectors.delete(old_ids)
        assert set(new_ids) <= set(await memory.vectors.ids("alpha"))
    finally:
        await memory.close()


async def test_counter_stays_ahead_after_all_chunks_and_vectors_are_deleted(tmp_path):
    path = tmp_path / "catalog.db"
    documents = SqliteDocumentStore(path)
    first = await source(documents)
    [old] = await documents.insert_chunks([chunk(first.episode_id)])
    await documents.delete_episode("alpha", first.episode_id)
    await documents.close()
    documents = SqliteDocumentStore(path)
    try:
        second = await source(documents, "second")
        [new] = await documents.insert_chunks([chunk(second.episode_id)])
        assert new.chunk_id > old.chunk_id
    finally:
        await documents.close()


@pytest.mark.parametrize("legacy_counter", [None, "1"])
@pytest.mark.parametrize("row_id,vector_id", [(50, 80), (80, 50)])
async def test_legacy_counter_starts_above_existing_rows_and_orphan_vectors(tmp_path, legacy_counter, row_id, vector_id):
    documents = SqliteDocumentStore(tmp_path / "legacy.db")
    try:
        episode = await source(documents)
        documents.conn.execute('INSERT INTO chunks VALUES (?, ?, ?, 0, 0, 16, ?, ?)',
                               (row_id, episode.episode_id, "alpha", "persisted source", NOW))
        documents.conn.execute('INSERT INTO vectors VALUES (?, ?, ?, ?, ?, ?, ?)',
                               (vector_id, "alpha", episode.episode_id, NOW, "[]", "{}", b""))
        documents.conn.execute("DELETE FROM meta WHERE key = 'next_chunk_id'")
        if legacy_counter is not None:
            documents.conn.execute("INSERT INTO meta VALUES ('next_chunk_id', ?)", (legacy_counter,))
        documents.conn.commit()
        [new] = await documents.insert_chunks([chunk(episode.episode_id)])
        assert new.chunk_id > 80
    finally:
        await documents.close()


async def test_invalid_batch_rolls_back_chunks_and_leaves_connection_usable(tmp_path):
    documents = SqliteDocumentStore(tmp_path / "catalog.db")
    try:
        episode = await source(documents)
        with pytest.raises(sqlite3.IntegrityError):
            await documents.insert_chunks([chunk(episode.episode_id), chunk(99999)])
        assert not documents.conn.in_transaction
        assert await documents.chunks_of("alpha", episode.episode_id) == []
        [inserted] = await documents.insert_chunks([chunk(episode.episode_id)])
        assert inserted.episode_id == episode.episode_id
    finally:
        await documents.close()


async def test_open_preserves_legacy_high_water_before_first_delete(tmp_path):
    path = tmp_path / "legacy.db"
    documents = SqliteDocumentStore(path)
    episode = await source(documents)
    [old] = await documents.insert_chunks([chunk(episode.episode_id)])
    documents.conn.execute("DELETE FROM meta WHERE key = 'next_chunk_id'")
    documents.conn.commit()
    await documents.close()
    documents = SqliteDocumentStore(path)
    try:
        await documents.delete_episode("alpha", episode.episode_id)
        replacement = await source(documents, "second")
        [new] = await documents.insert_chunks([chunk(replacement.episode_id)])
        assert new.chunk_id > old.chunk_id
    finally:
        await documents.close()


def test_separate_connections_allocate_disjoint_committed_ids(tmp_path):
    path = tmp_path / "catalog.db"

    async def initialize():
        documents = SqliteDocumentStore(path)
        try:
            return (await source(documents)).episode_id
        finally:
            await documents.close()

    episode_id = asyncio.run(initialize())

    def write_batch(_):
        async def write():
            documents = SqliteDocumentStore(path)
            try:
                return [c.chunk_id for c in await documents.insert_chunks([chunk(episode_id)] * 8)]
            finally:
                await documents.close()
        return asyncio.run(write())

    with ThreadPoolExecutor(max_workers=2) as executor:
        batches = list(executor.map(write_batch, range(4)))
    ids = [identity for batch in batches for identity in batch]
    assert len(set(ids)) == 32
    assert sorted(ids) == list(range(min(ids), min(ids) + 32))
