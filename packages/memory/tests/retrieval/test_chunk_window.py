"""Chunk navigation uses bounded episode/ordinal reads, never a corpus scan."""
import pytest


async def document(engine, space='alpha', **kwargs):
    engine.chunk_target = 100
    content = '\n\n'.join(f'Section {i}: calibration record ' + ('sample data ' * 12) for i in range(12))
    added = await engine.remember(space, content, **kwargs)
    chunks = await engine.documents.chunks_of(space, added.episode_id)
    assert len(chunks) > 8
    return added, chunks


async def test_window_is_ordered_bounded_and_bound_to_space_and_episode(engine):
    added, chunks = await document(engine)
    await document(engine, 'other')
    page = await engine.documents.page_chunks('alpha', added.episode_id, start_ordinal=chunks[3].ordinal, limit=3)
    assert page == chunks[3:6]
    assert await engine.documents.page_chunks('other', added.episode_id, start_ordinal=0, limit=3) == []
    await engine.forget('alpha', added.episode_id)
    assert await engine.documents.page_chunks('alpha', added.episode_id, start_ordinal=0, limit=3) == []


@pytest.mark.parametrize('changes', [{'episode_id':True}, {'episode_id':0}, {'start_ordinal':-1},
                                    {'start_ordinal':True}, {'limit':0}, {'limit':21}, {'limit':True}])
async def test_bad_window_bounds_are_rejected(engine, changes):
    args = {'episode_id':1, 'start_ordinal':0, 'limit':3, **changes}
    with pytest.raises(ValueError):
        await engine.documents.page_chunks('alpha', **args)


async def test_memory_window_does_not_iterate_the_chunk_corpus():
    from scone_memory import HashEmbedder, InMemoryDocumentStore, InMemoryVectorIndex, MemoryEngine
    store = InMemoryDocumentStore()
    engine = await MemoryEngine(store, InMemoryVectorIndex(), HashEmbedder()).open()
    added, chunks = await document(engine)

    class NoScan(dict):
        def values(self):
            raise AssertionError('window read scanned all chunks')
        def __iter__(self):
            raise AssertionError('window read scanned all chunks')

    store._chunks = NoScan(store._chunks)
    assert await store.page_chunks('alpha', added.episode_id, start_ordinal=chunks[5].ordinal, limit=2) == chunks[5:7]
    await engine.close()


async def test_memory_space_deletion_clears_window_index_without_touching_other_spaces():
    from scone_memory import HashEmbedder, InMemoryDocumentStore, InMemoryVectorIndex, MemoryEngine
    store = InMemoryDocumentStore()
    engine = await MemoryEngine(store, InMemoryVectorIndex(), HashEmbedder()).open()
    removed, _ = await document(engine)
    retained, chunks = await document(engine, 'other')
    try:
        await store.delete_space('alpha', engine.clock())
        assert ('alpha', removed.episode_id) not in store._chunks_by_episode
        assert await store.page_chunks('alpha', removed.episode_id, start_ordinal=0, limit=2) == []
        assert await store.page_chunks('other', retained.episode_id, start_ordinal=0, limit=2) == chunks[:2]
    finally:
        await engine.close()


async def test_sqlite_window_uses_ordered_range_index(tmp_path):
    from scone_memory.backends import SqliteDocumentStore
    store = SqliteDocumentStore(tmp_path / 'window.db')
    try:
        plan = store.conn.execute(
            'EXPLAIN QUERY PLAN SELECT * FROM chunks WHERE space = ? AND episode_id = ? '
            'AND ordinal >= ? ORDER BY ordinal, id LIMIT ?', ('alpha', 1, 3, 5)).fetchall()
        detail = ' '.join(row['detail'] for row in plan)
        assert 'SEARCH chunks USING INDEX chunks_window' in detail
        assert 'TEMP B-TREE' not in detail
    finally:
        await store.close()
