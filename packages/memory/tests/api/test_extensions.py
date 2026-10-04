import httpx
import pytest
from fastapi import Depends
from scone_memory import MemoryEngine, InMemoryDocumentStore, InMemoryVectorIndex, HashEmbedder
from scone_memory.api.app import create_app
from scone_memory.api.extensions import HttpExtensionContext

@pytest.mark.asyncio
async def test_extension_uses_current_core_authorization_and_no_product_routes():
    engine = await MemoryEngine(InMemoryDocumentStore(), InMemoryVectorIndex(), HashEmbedder()).open()
    def install(context: HttpExtensionContext) -> None:
        @context.app.get('/v1/example')
        async def read(space: str = Depends(context.space_for)):
            return {'space': space}
        @context.app.post('/v1/example')
        async def write(space: str = Depends(context.space_for)):
            return {'space': space}
    app = create_app(engine, {'writer':'a', 'reader':'a'}, roles={'reader':'read'}, extensions=(install,))
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url='http://local') as client:
        assert (await client.get('/v1/example')).status_code == 401
        assert (await client.get('/v1/example', headers={'Authorization':'Bearer reader'})).json() == {'space':'a'}
        assert (await client.post('/v1/example', headers={'Authorization':'Bearer reader'})).status_code == 403
        app.state.keys.pop('writer')
        assert (await client.get('/v1/example', headers={'Authorization':'Bearer writer'})).status_code == 401
        assert (await client.get('/v1/writing/drafts', headers={'Authorization':'Bearer reader'})).status_code == 404
    await engine.close()

@pytest.mark.parametrize('conversations', [False, True])
@pytest.mark.asyncio
async def test_public_host_forwards_extension_and_owns_no_engine(tmp_path, conversations):
    from scone_memory.api.host import build_app
    from scone_memory.runtime.config import Settings
    engine = await MemoryEngine(InMemoryDocumentStore(), InMemoryVectorIndex(), HashEmbedder()).open()
    contexts = []
    def install(context: HttpExtensionContext) -> None:
        contexts.append(context)
        @context.app.get('/v1/example')
        async def read(space: str = Depends(context.space_for)):
            return {'space': space}
    settings = Settings(keys={'key': 'scope'}, conversations_journal=str(tmp_path / 'journal.db') if conversations else None)
    app = build_app(settings, engine, extensions=(install,))
    assert len(contexts) == 1 and contexts[0].engine is engine
    async with app.router.lifespan_context(app):
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url='http://local') as client:
            assert (await client.get('/v1/example', headers={'Authorization': 'Bearer key'})).json() == {'space': 'scope'}
    assert await engine.space_deleted('scope') is None
    await engine.close()

@pytest.mark.asyncio
async def test_extension_checks_deleted_space_and_reauthorizes_after_await():
    import asyncio
    from fastapi import Request
    engine = await MemoryEngine(InMemoryDocumentStore(), InMemoryVectorIndex(), HashEmbedder()).open()
    entered, release = asyncio.Event(), asyncio.Event()
    def install(context: HttpExtensionContext) -> None:
        @context.app.get('/v1/example')
        async def read(request: Request, space: str = Depends(context.space_for)):
            entered.set()
            await release.wait()
            context.assert_current_space(request, space)
            return {'space': space}
    app = create_app(engine, {'key':'a'}, extensions=(install,))
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url='http://local') as client:
        task = asyncio.create_task(client.get('/v1/example', headers={'Authorization':'Bearer key'}))
        await entered.wait()
        app.state.keys['key'] = 'b'
        release.set()
        assert (await task).status_code == 401
        await engine.delete_space('b')
        assert (await client.get('/v1/example', headers={'Authorization':'Bearer key'})).status_code == 404
    await engine.close()

def test_public_serve_factory_preserves_engine_cleanup(monkeypatch):
    from fastapi import FastAPI
    from scone_memory.api import __main__ as host
    from scone_memory.api.host import serve
    from scone_memory.runtime.config import Settings
    events = []
    class Engine:
        documents = type('Store', (), {'name':'documents'})()
        vectors = type('Store', (), {'name':'vectors'})()
        embedder = type('Embedder', (), {'id':'test'})()
        async def close(self):
            events.append('closed')
    engine = Engine()
    async def build_engine(settings):
        return engine
    def factory(settings, actual):
        assert actual is engine
        events.append('factory')
        app = FastAPI()
        app.state.worker = None
        return app
    class Server:
        async def serve(self):
            events.append('served')
    monkeypatch.setattr(host, 'build_engine', build_engine)
    monkeypatch.setattr(host, 'build_server', lambda settings, app: Server())
    serve(Settings(keys={'key':'scope'}), application_factory=factory)
    assert events == ['factory', 'served', 'closed']

@pytest.mark.asyncio
async def test_long_operation_ends_on_revocation_and_joins_cleanup():
    import asyncio
    from scone_memory.api.authentication import run_authenticated
    from scone_memory.core.bearer_keys import Unauthorized
    entered,cleaned=asyncio.Event(),asyncio.Event()
    revoked=False
    closed=[]
    async def operation():
        entered.set()
        try:
            await asyncio.Event().wait()
        finally:
            cleaned.set()
    def authorize():
        if revoked:
            raise Unauthorized('revoked')
    async def close():
        closed.append(True)
    task=asyncio.create_task(run_authenticated(operation,authorize,close,interval=.01))
    await entered.wait();revoked=True
    with pytest.raises(asyncio.CancelledError):
        await task
    assert cleaned.is_set() and closed==[True]

@pytest.mark.asyncio
async def test_authorization_revocation_cancels_work_before_stalled_socket_close():
    import asyncio
    from scone_memory.api.authentication import run_authenticated
    from scone_memory.core.bearer_keys import Unauthorized
    entered,cancelled,closing,release=asyncio.Event(),asyncio.Event(),asyncio.Event(),asyncio.Event()
    revoked=False
    async def operation():
        entered.set()
        try:
            await asyncio.Event().wait()
        finally:
            cancelled.set()
    def authorize():
        if revoked:
            raise Unauthorized('revoked')
    async def close():
        closing.set()
        await release.wait()
    task=asyncio.create_task(run_authenticated(operation,authorize,close,interval=.01))
    await entered.wait();revoked=True
    await asyncio.wait_for(closing.wait(),1)
    await asyncio.wait_for(cancelled.wait(),1)
    assert not release.is_set()
    with pytest.raises(asyncio.CancelledError):
        await task

@pytest.mark.asyncio
async def test_revoked_authorization_never_starts_operation():
    from scone_memory.api.authentication import run_authenticated
    from scone_memory.core.bearer_keys import Unauthorized
    async def operation():
        pytest.fail('private operation must not start')
    def authorize():
        raise Unauthorized('revoked')
    async def close():
        pass
    with pytest.raises(Unauthorized):
        await run_authenticated(operation,authorize,close)
