import json

import httpx
import pytest

from scone_memory import InMemoryDocumentStore, InMemoryVectorIndex, MemoryEngine
from scone_memory.embedders.remote import RemoteEmbedder


async def test_query_instruction_reaches_vector_search_without_changing_documents():
    requests = []

    def serve(request):
        body = json.loads(request.content)
        requests.append(body['input'])
        return httpx.Response(200, json={'data': [
            {'index': i, 'embedding': [0.0, 1.0] if text.startswith('Instruct:') else [1.0, 0.0]}
            for i, text in reversed(list(enumerate(body['input'])))
        ]})

    prefix = 'Instruct: Find passages that support an answer.\nQuery:'
    embedder = RemoteEmbedder('https://openrouter.ai/api/v1', 'qwen/qwen3-embedding-8b',
        dim=2, query_prefix=prefix, transport=httpx.MockTransport(serve))
    plain = RemoteEmbedder('https://openrouter.ai/api/v1', 'qwen/qwen3-embedding-8b', dim=2)
    assert embedder.id != plain.id
    engine = await MemoryEngine(InMemoryDocumentStore(), InMemoryVectorIndex(), embedder).open()
    try:
        await engine.remember('test', 'Morgan maintains Cedar.')
        requests.clear()
        result = await engine.recall('test', 'Who maintains Cedar?', lanes=('vector',))
        assert result.items
        assert requests == [[prefix + 'Who maintains Cedar?']]
        requests.clear()
        await embedder.embed(['document one', 'document two'])
        assert requests == [['document one', 'document two']]
    finally:
        await engine.close()


@pytest.mark.parametrize('rows', [
    [{'index': 0, 'embedding': [1, 0]}, {'index': 0, 'embedding': [0, 1]}],
    [{'index': 0, 'embedding': [1, 0]}, {'index': 2, 'embedding': [0, 1]}],
    [{'index': 0, 'embedding': [1, 0]}, {'index': 1, 'embedding': [1]}],
])
async def test_remote_rejects_misaligned_vectors(rows):
    embedder = RemoteEmbedder('https://example.test/v1', 'test', dim=2,
        transport=httpx.MockTransport(lambda _: httpx.Response(200, json={'data': rows})))
    with pytest.raises((ValueError, RuntimeError)):
        await embedder.embed(['first', 'second'])


async def test_legacy_embedder_still_handles_queries():
    from scone_memory.core.embedding import embed_queries
    class Legacy:
        id = 'legacy'
        dim = 2
        async def embed(self, texts):
            return [[float(len(text)), 1.0] for text in texts]
    assert await embed_queries(Legacy(), ['abc']) == [[3.0, 1.0]]


async def test_cached_queries_never_reuse_document_vectors_for_same_text():
    from scone_memory.bench.comparative import CachedEmbedder
    from scone_memory.core.embedding import embed_queries
    from scone_memory.ingestion.embedding_cache import InMemoryEmbeddingCache
    calls = []
    def serve(request):
        texts = json.loads(request.content)['input']
        calls.append(texts)
        return httpx.Response(200, json={'data': [
            {'index': i, 'embedding': [0, 1] if text.startswith('query:') else [1, 0]}
            for i, text in enumerate(texts)]})
    remote = RemoteEmbedder('https://example.test', 'test', dim=2,
        query_prefix='query:', transport=httpx.MockTransport(serve))
    cached = CachedEmbedder(remote, InMemoryEmbeddingCache(10))
    assert await cached.embed(['same']) == [[1, 0]]
    assert await embed_queries(cached, ['same']) == [[0, 1]]
    assert await embed_queries(cached, ['same']) == [[0, 1]]
    assert await cached.embed(['same']) == [[1, 0]]
    assert calls == [['same'], ['query:same']]
