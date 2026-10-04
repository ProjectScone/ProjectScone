import asyncio
import pytest
from scone_memory import HashEmbedder, MemoryEngine
from scone_memory.backends import SqliteDocumentStore, SqliteVectorIndex
from scone_memory.api.conversations import create_conversation_app
from scone_memory.realtime.deferred_capture import DeferredTextCapture
from scone_memory.realtime.text import TextConversation
from scone_memory.realtime.events import TextDelta, ReplyCompleted
from ..api.test_conversations_api import client_for, create

class BlockedEmbedder(HashEmbedder):
    def __init__(self):
        super().__init__()
        self.entered = asyncio.Event()
        self.exited = asyncio.Event()
    async def embed(self, texts):
        self.entered.set()
        try:
            await asyncio.Event().wait()
        finally:
            self.exited.set()

class Model:
    async def respond(self, messages):
        yield TextDelta('A local reply.')
        yield ReplyCompleted()
    async def aclose(self):
        pass

async def test_served_completion_does_not_wait_for_indexing_and_shutdown_joins_worker(tmp_path):
    embedder = BlockedEmbedder()
    engine = await MemoryEngine(SqliteDocumentStore(tmp_path/'memory.db'), SqliteVectorIndex(tmp_path/'memory.db'), embedder).open()
    capture = DeferredTextCapture(engine)
    def factory(space, sid, **options):
        return TextConversation(engine, space, sid, Model, **options)
    app = create_conversation_app(engine, {'alpha-key':'alpha'}, tmp_path/'sessions.db', factory, deferred_capture=capture)
    try:
        async with client_for(app) as client:
            assert (await client.get('/v1/conversations/capabilities')).json()['capture_mode'] == 'deferred'
            session = await create(client)
            route = '/v1/conversations/'+session['session_id']+'/turns/turn-1'
            submitted = await client.post(route.rsplit('/',1)[0], json={'request_id':'turn-1','text':'Hello','expected_revision':session['revision']})
            assert submitted.status_code == 202
            async def completed():
                while True:
                    result = (await client.get(route)).json()
                    if result['status'] == 'completed':
                        return result
                    assert result['status'] != 'failed', result
                    await asyncio.sleep(.005)
            result = await asyncio.wait_for(completed(), 2)
            assert result['result']['text'] == 'A local reply.'
            assert result['result']['capture_indexing'] == {'mode':'deferred','user':'pending','assistant':'pending'}
            await asyncio.wait_for(embedder.entered.wait(), 1)
        assert embedder.exited.is_set()
        assert len(await engine.documents.inflight()) == 2
    finally:
        await engine.close()

async def test_default_capabilities_and_capture_binding_validation(tmp_path):
    engine = await MemoryEngine(SqliteDocumentStore(tmp_path/'m.db'), SqliteVectorIndex(tmp_path/'m.db'), HashEmbedder()).open()
    other = await MemoryEngine(SqliteDocumentStore(tmp_path/'o.db'), SqliteVectorIndex(tmp_path/'o.db'), HashEmbedder()).open()
    try:
        with pytest.raises(ValueError, match='deferred_capture'):
            create_conversation_app(engine, {'alpha-key':'alpha'},tmp_path/'s.db',None,deferred_capture=object())
        with pytest.raises(ValueError, match='same engine'):
            create_conversation_app(engine, {'alpha-key':'alpha'},tmp_path/'s.db',None,deferred_capture=DeferredTextCapture(other))
        app = create_conversation_app(engine, {'alpha-key':'alpha'},tmp_path/'s.db',None)
        async with client_for(app) as client:
            assert (await client.get('/v1/conversations/capabilities')).json()['capture_mode'] == 'inline'
    finally:
        await engine.close()
        await other.close()

async def test_startup_failure_closes_started_capture_worker(tmp_path):
    engine = await MemoryEngine(SqliteDocumentStore(tmp_path/'m.db'), SqliteVectorIndex(tmp_path/'m.db'), HashEmbedder()).open()
    capture = DeferredTextCapture(engine)
    class FailingWorker:
        def start(self):
            raise RuntimeError('startup fixture failure')
        async def stop(self):
            pass
    app = create_conversation_app(engine, {'alpha-key':'alpha'}, tmp_path/'s.db',None,
        deferred_capture=capture, worker=FailingWorker())
    try:
        with pytest.raises(RuntimeError, match='startup fixture failure'):
            async with client_for(app):
                pytest.fail('failed startup must not serve')
        with pytest.raises(RuntimeError, match='closed'):
            capture.start()
        assert not any(task.get_name() == 'scone-text-indexing' and not task.done() for task in asyncio.all_tasks())
    finally:
        await engine.close()
