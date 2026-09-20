"""The SQLite text lane ranks our own tokens, kept current by triggers and rebuilt when the tokenizer moves."""

from __future__ import annotations

import pytest

from scone_memory import HashEmbedder, InMemoryVectorIndex, MemoryEngine
from scone_memory.backends import SqliteDocumentStore
from scone_memory.backends import sqlite_lexical as module
from scone_memory.core.ports import TextFilter


async def test_failed_lexical_write_cannot_publish_a_partial_chunk_batch(tmp_path, monkeypatch):
    from scone_memory.backends import sqlite as sqlite_store
    from scone_memory.core.ports import NewChunk
    store = SqliteDocumentStore(tmp_path/'atomic.db')
    engine = await MemoryEngine(store, InMemoryVectorIndex(), HashEmbedder()).open()
    try:
        saved = await engine.remember('s', 'Existing retained passage.')
        original = sqlite_store.index_lexical_chunk
        calls = 0
        def fail_second(*args):
            nonlocal calls
            calls += 1
            if calls == 2:
                raise RuntimeError('index write failed')
            return original(*args)
        monkeypatch.setattr(sqlite_store, 'index_lexical_chunk', fail_second)
        with pytest.raises(RuntimeError, match='index write failed'):
            await store.insert_chunks([NewChunk(episode_id=saved.episode_id, space='s', ordinal=i,
                start=0, end=12, text='Unpublished.', created_at='2026-09-08') for i in (1, 2)])
        assert store.conn.execute('SELECT count(*) FROM chunks').fetchone()[0] == 1
        assert store.conn.execute('SELECT count(*) FROM chunk_lexical').fetchone()[0] == 1
        assert store.conn.execute('SELECT count(*) FROM chunk_lexical_dirty').fetchone()[0] == 0
    finally:
        await engine.close()


async def test_first_search_after_bulk_ingestion_can_find_the_last_document(tmp_path, monkeypatch):
    from scone_memory.memory.engine import Record
    monkeypatch.setattr(module, 'MAX_SYNC_ROWS', 2)
    store = SqliteDocumentStore(tmp_path / 'first-query.db')
    engine = await MemoryEngine(store, InMemoryVectorIndex(), HashEmbedder()).open()
    try:
        await engine.remember_many('s', [Record(f'Unrelated invoice {i}.') for i in range(5)]
                                   + [Record('The lighthouse custodian is Morgan.')])
        result = await engine.recall('s', 'lighthouse custodian', lanes=('text',))
        assert result.items and 'Morgan' in result.items[0].text
        assert not result.degraded
    finally:
        await engine.close()


async def test_rows_follow_writes_and_deletions_and_a_query_finds_a_part_of_an_unspaced_run(tmp_path):
    store = SqliteDocumentStore(str(tmp_path / "l.db"))
    engine = await MemoryEngine(store, InMemoryVectorIndex(), HashEmbedder()).open()
    kept = await engine.remember("s", "東京タワーは1958年に完成した。")
    await engine.remember("s", "Don't forget the résumé for the interview.")
    assert store.conn.execute("SELECT count(*) FROM chunk_lexical_dirty").fetchone()[0] == 0, "new writes are already searchable"
    found = await store.search_terms("s", "東京", 5, TextFilter(), prefixes=())
    assert len(found) == 1 and store.conn.execute("SELECT count(*) FROM chunk_lexical_dirty").fetchone()[0] == 0
    assert [c for c, _ in await store.search_terms("s", "resume", 5, TextFilter(), prefixes=())] != [], "accents folded on both sides"
    assert [c for c, _ in await store.search_terms("s", "don't", 5, TextFilter(), prefixes=())] != [], "an apostrophe inside a token survives"
    await engine.forget("s", kept.episode_id)
    assert await store.search_terms("s", "東京", 5, TextFilter(), prefixes=()) == []
    assert store.conn.execute("SELECT count(*) FROM chunk_lexical").fetchone()[0] == 1, "the deleted chunk's row went with it"
    assert store.conn.execute("SELECT count(*) FROM chunk_lexical_fts WHERE chunk_lexical_fts MATCH '\"東京\"'").fetchone()[0] == 0


async def test_a_new_tokenizer_version_rebuilds_the_index_lazily(tmp_path, monkeypatch):
    path = str(tmp_path / "v.db")
    store = SqliteDocumentStore(path)
    engine = await MemoryEngine(store, InMemoryVectorIndex(), HashEmbedder()).open()
    await engine.remember("s", "The billing run went out late.")
    assert await store.search_terms("s", "billing", 5, TextFilter(), prefixes=()) != []
    await engine.close()
    monkeypatch.setattr(module, "_VERSION", module._VERSION + ";changed")
    reopened = SqliteDocumentStore(path)
    assert reopened.conn.execute("SELECT value FROM meta WHERE key=?", (module._VERSION_KEY,)).fetchone()[0].endswith(";changed")
    assert reopened.conn.execute("SELECT count(*) FROM chunk_lexical").fetchone()[0] == 0, "rebuilt whole: rows dropped"
    assert reopened.conn.execute("SELECT count(*) FROM chunk_lexical_dirty").fetchone()[0] == 1, "every chunk marked to be re-read"
    assert await reopened.search_terms("s", "billing", 5, TextFilter(), prefixes=()) != [], "and re-read at the next query"


def test_the_match_expression_folds_and_quotes_like_the_index():
    assert module.lexical_match("Résumé, don't", ("bill",)) == '"resume" OR "don\u02bct" OR "bill"*'
    assert module.lexical_match("   ") is None


async def test_a_query_word_its_own_family_covers_is_one_signal_in_both_stores(tmp_path):
    """"billing" asked with the prefix "bill" is scored once, through the
    family, in the in-memory lane and in the SQLite expression alike."""
    from scone_memory.retrieval.lexical import Bm25

    index = Bm25()
    index.add(1, "The billing run went out late.")
    index.add(2, "Two bills were paid on time.")
    index.add(3, "Nothing about money here at all.")
    exact = dict(index.search("billing", limit=5))
    family = dict(index.search("billing", limit=5, prefixes=("bill",)))
    assert 1 in exact and 2 not in exact, "without the family only the exact word matches"
    assert 2 in family, "the family reaches bills"
    assert family[1] <= exact[1] * 1.0001 + 1e-9, "the exact word is not counted again beside its family"
    assert module.lexical_match("billing", ("bill",)) == '"bill"*', "the SQLite phrase for a covered word is the prefix alone"
    assert module.lexical_match("billing run", ("bill",)) == '"run" OR "bill"*'


async def test_the_synchronisation_is_bounded_per_query_and_the_backlog_is_said(tmp_path, monkeypatch):
    from scone_memory import HashEmbedder, InMemoryVectorIndex, MemoryEngine
    from scone_memory.backends import sqlite_lexical

    monkeypatch.setattr(sqlite_lexical, "MAX_SYNC_ROWS", 4)
    store = SqliteDocumentStore(str(tmp_path / "b.db"))
    engine = await MemoryEngine(store, InMemoryVectorIndex(), HashEmbedder()).open()
    try:
        for number in range(10):
            await engine.remember("s", f"Invoice number {number} for the harbour lights, billing due in March.")
        # Old tokenizer data is still migrated in bounded query-time batches.
        monkeypatch.setattr(module, '_VERSION', module._VERSION + ';test-migration')
        module.initialize_lexical(store.conn)
        first = await engine.recall("s", "billing harbour", limit=3)
        assert store.lexical_backlog("s") == 6, "four chunks synchronised, six still behind"
        assert any("6 chunk(s) not yet in the lexical index" in note for note in first.degraded), first.degraded
        second = await engine.recall("s", "billing harbour", limit=3)
        assert store.lexical_backlog("s") == 2 and any("2 chunk(s)" in note for note in second.degraded)
        third = await engine.recall("s", "billing harbour", limit=3)
        assert store.lexical_backlog("s") == 0 and not any("lexical index" in note for note in third.degraded)
        assert len(third.items) == 3
    finally:
        await engine.close()
