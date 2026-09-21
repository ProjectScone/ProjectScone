import httpx
import pytest

from scone_memory import InMemoryDocumentStore, InMemoryVectorIndex, MemoryEngine
from scone_memory.embedders.remote import RemoteEmbedder


async def test_remote_embedding_reuses_client_after_error_and_engine_closes_it(monkeypatch):
    clients = []
    requests = []
    original = httpx.AsyncClient

    def respond(request):
        requests.append(request)
        if len(requests) == 2:
            return httpx.Response(503, text='temporarily unavailable')
        return httpx.Response(200, json={'data': [{'index': 0, 'embedding': [3, 4]}]})

    def client(**options):
        instance = original(transport=httpx.MockTransport(respond), **options)
        clients.append(instance)
        return instance

    monkeypatch.setattr(httpx, 'AsyncClient', client)
    embedder = RemoteEmbedder('https://embedding.example/v1', 'qwen', dim=2)
    engine = await MemoryEngine(InMemoryDocumentStore(), InMemoryVectorIndex(), embedder).open()
    try:
        assert await embedder.embed(['first']) == [[.6, .8]]
        with pytest.raises(RuntimeError, match='503'):
            await embedder.embed(['second'])
        assert await embedder.embed(['third']) == [[.6, .8]]
        assert len(clients) == 1 and not clients[0].is_closed
    finally:
        await engine.close()
    assert clients[0].is_closed
    with pytest.raises(RuntimeError, match='closed'):
        await embedder.embed(['fourth'])
