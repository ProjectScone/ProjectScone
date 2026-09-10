"""Missing required rows fail explicitly; optional reads still mean not found."""
from __future__ import annotations
import importlib.util
import pytest
from scone_memory.core.ports import NewTombstone, VectorPoint


@pytest.mark.parametrize('operation', ['counts', 'bump_revision'])
async def test_postgres_missing_required_result_does_not_become_a_success(operation, monkeypatch):
    pytest.importorskip('pgvector.psycopg')
    pytest.importorskip('psycopg_pool')
    from scone_memory.backends.postgres import PostgresDocumentStore
    store = PostgresDocumentStore('postgresql://unused')
    async def missing(sql, params=()):
        return None
    monkeypatch.setattr(store, '_row', missing)
    with pytest.raises(RuntimeError, match='expected row'):
        await getattr(store, operation)('alpha')
    assert await store.get_episode('alpha', 1) is None


async def test_elasticsearch_tombstone_disappearing_after_write_is_explicit(monkeypatch):
    pytest.importorskip('elasticsearch')
    from scone_memory.backends.elastic import ElasticsearchDocumentStore
    class Client:
        async def index(self, **kwargs):
            return {'result': 'created'}
    store = ElasticsearchDocumentStore(client=Client())
    async def missing(kind, doc_id):
        return None
    monkeypatch.setattr(store, '_doc', missing)
    tombstone = NewTombstone('alpha', 1, 'abc', '2026-09-10T00:00:00.000Z')
    with pytest.raises(RuntimeError, match='tombstone disappeared'):
        await store.record_tombstone(tombstone)
    assert await store.tombstone('alpha', 1) is None


@pytest.mark.parametrize('backend', ['chroma', 'lancedb'])
async def test_embedded_collection_requires_ensure_then_accepts_operations(backend, tmp_path):
    dependency = 'chromadb' if backend == 'chroma' else 'lancedb'
    if importlib.util.find_spec(dependency) is None:
        pytest.skip(f'{dependency} is not installed')
    if backend == 'chroma':
        from scone_memory.backends.chroma import ChromaVectorIndex
        index = ChromaVectorIndex(path=str(tmp_path / 'chroma'))
    else:
        from scone_memory.backends.lancedb import LanceDBVectorIndex
        index = LanceDBVectorIndex(str(tmp_path / 'lance'))
    with pytest.raises(RuntimeError, match='ensure first'):
        await index.delete([1])
    await index.ensure(2)
    try:
        await index.upsert([VectorPoint(1, 'alpha', 1, '2026-09-10T00:00:00.000Z', [1.0, 0.0])])
        assert (await index.search('alpha', [1.0, 0.0], 5))[0][0] == 1
        await index.delete([1])
        assert await index.search('alpha', [1.0, 0.0], 5) == []
    finally:
        await index.close()


async def test_chroma_requires_requested_distances_in_query_results(tmp_path, monkeypatch):
    pytest.importorskip('chromadb')
    from scone_memory.backends.chroma import ChromaVectorIndex
    index = ChromaVectorIndex(path=str(tmp_path / 'chroma'))
    await index.ensure(2)
    await index.upsert([VectorPoint(1, 'alpha', 1, '2026-09-10T00:00:00.000Z', [1.0, 0.0])])
    def without_distances(self, **kwargs):
        return {'ids': [['1']], 'distances': None}
    monkeypatch.setattr(type(index.collection), 'query', without_distances)
    try:
        with pytest.raises(RuntimeError, match='requested distances'):
            await index.search('alpha', [1.0, 0.0], 5)
    finally:
        await index.close()
