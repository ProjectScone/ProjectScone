import httpx
import pytest
from scone_memory import HashEmbedder, InMemoryDocumentStore, InMemoryVectorIndex, MemoryEngine
from scone_memory.api.app import create_app
from scone_memory.runtime.config import Settings
from scone_memory.core.errors import InvalidInput


class Speech:
    available=True
    calls=[]
    after=None
    async def transcribe(self,content,media_type,language):
        self.calls.append((content,media_type,language))
        if self.after: self.after()
        return 'The garden was quiet.'


@pytest.fixture
async def client():
    engine=await MemoryEngine(InMemoryDocumentStore(),InMemoryVectorIndex(),HashEmbedder()).open()
    speech=Speech();speech.calls=[]
    app=create_app(engine,{'key':'alpha','reader':'alpha'},roles={'reader':'read'},dictation=speech)
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app),base_url='http://local') as client:
        yield client,speech,app,engine
    await engine.close()


async def test_authentication_and_local_capability(client):
    c,speech,app,engine=client
    assert (await c.get('/v1/dictation/status')).status_code==401
    result=await c.get('/v1/dictation/status',headers={'Authorization':'Bearer key'})
    assert result.json()=={'available':True,'mode':'local','max_duration_seconds':60,'max_bytes':5242880}
    assert result.headers['cache-control']=='no-store'
    assert (await c.post('/v1/dictation/transcribe',headers={'Authorization':'Bearer reader'},content=b'input')).status_code==403
    assert speech.calls==[]


async def test_raw_audio_roundtrip_is_ephemeral(client):
    c,speech,app,engine=client
    response=await c.post('/v1/dictation/transcribe?language=en',headers={'Authorization':'Bearer key','Content-Type':'audio/webm;codecs=opus'},content=b'recorded')
    assert response.status_code==200,response.text
    assert response.json()=={'text':'The garden was quiet.','language':'en','mode':'local'}
    assert speech.calls==[(b'recorded','audio/webm','en')]
    assert await engine.documents.recent_episodes('alpha',10)==[]


async def test_auth_scope_rechecked_after_transcription(client):
    c,speech,app,_=client
    speech.after=lambda:app.state.keys.update({'key':'other'})
    response=await c.post('/v1/dictation/transcribe',headers={'Authorization':'Bearer key','Content-Type':'audio/webm'},content=b'audio')
    assert response.status_code==403
    assert 'garden' not in response.text


async def test_stream_body_limit(client):
    c,speech,_,_=client
    async def chunks():
        for _ in range(6): yield b'x'*(1024*1024)
    response=await c.post('/v1/dictation/transcribe',headers={'Authorization':'Bearer key','Content-Type':'audio/webm'},content=chunks())
    assert response.status_code==413
    assert speech.calls==[]


def test_dictation_settings_require_all_explicit_absolute_paths():
    assert Settings.from_env({}).dictation_model is None
    for env in [{'SCONE_DICTATION_MODEL':'/local/model'}, {'SCONE_DICTATION_MODEL':'model','SCONE_DICTATION_PYTHON':'python','SCONE_DICTATION_FFMPEG':'ffmpeg'}]:
        with pytest.raises(InvalidInput,match='SCONE_DICTATION'):
            Settings.from_env(env)
    settings=Settings.from_env({'SCONE_DICTATION_MODEL':'/local/model','SCONE_DICTATION_PYTHON':'/local/python','SCONE_DICTATION_FFMPEG':'/local/ffmpeg'})
    assert settings.dictation_model=='/local/model'

@pytest.mark.parametrize('language,media,body',[('../en','audio/webm',b'a'),('EN','audio/webm',b'a'),('en','text/plain',b'a'),('en','audio/webm',b'')])
async def test_api_validates_injected_service_inputs(client,language,media,body):
    c,speech,_,_=client
    response=await c.post('/v1/dictation/transcribe',params={'language':language},headers={'Authorization':'Bearer key','Content-Type':media},content=body)
    assert response.status_code==422
    assert speech.calls==[]

async def test_upload_admission_is_bounded_before_reading_body(client):
    import asyncio
    c,speech,_,_=client
    gate=asyncio.Event();ready=asyncio.Event();started=0;extra_read=False
    async def held_body():
        nonlocal started
        started+=1
        if started==4:ready.set()
        await gate.wait()
        yield b'audio'
    headers={'Authorization':'Bearer key','Content-Type':'audio/webm'}
    requests=[asyncio.create_task(c.post('/v1/dictation/transcribe',headers=headers,content=held_body())) for _ in range(4)]
    try:
        await asyncio.wait_for(ready.wait(),2)
        async def extra_body():
            nonlocal extra_read
            extra_read=True
            yield b'audio'
        response=await c.post('/v1/dictation/transcribe',headers=headers,content=extra_body())
        assert response.status_code==429
        assert extra_read is False and speech.calls==[]
        gate.set()
        assert all(response.status_code==200 for response in await asyncio.gather(*requests))
        assert (await c.post('/v1/dictation/transcribe',headers=headers,content=b'audio')).status_code==200
    finally:
        gate.set()
        await asyncio.gather(*requests,return_exceptions=True)

async def test_client_disconnect_cancels_transcriber_and_releases_upload_slot(client):
    import asyncio
    c,speech,app,_=client
    started=asyncio.Event();cancelled=asyncio.Event()
    async def blocked(*args):
        started.set()
        try:await asyncio.Event().wait()
        finally:cancelled.set()
    speech.transcribe=blocked
    messages=asyncio.Queue()
    await messages.put({'type':'http.request','body':b'audio','more_body':False})
    async def receive():return await messages.get()
    sent=[]
    async def send(message):sent.append(message)
    scope={'type':'http','http_version':'1.1','method':'POST','scheme':'http','path':'/v1/dictation/transcribe','raw_path':b'/v1/dictation/transcribe','query_string':b'','headers':[(b'authorization',b'Bearer key'),(b'content-type',b'audio/webm')],'server':('127.0.0.1',0),'client':('127.0.0.1',0),'root_path':''}
    request=asyncio.create_task(app(scope,receive,send))
    try:
        await asyncio.wait_for(started.wait(),2)
        await messages.put({'type':'http.disconnect'})
        await asyncio.wait_for(cancelled.wait(),1)
        await asyncio.wait_for(request,1)
        assert next(message['status'] for message in sent if message['type']=='http.response.start')==499
        async def ready(*args):return 'A retained staged transcript.'
        speech.transcribe=ready
        response=await c.post('/v1/dictation/transcribe',headers={'Authorization':'Bearer key','Content-Type':'audio/webm'},content=b'audio')
        assert response.status_code==200
    finally:
        request.cancel();await asyncio.gather(request,return_exceptions=True)

async def test_repeated_request_cancellation_joins_transcriber_cleanup():
    import asyncio
    from starlette.requests import Request
    from scone_memory.api.dictation import _transcribe_connected
    started=asyncio.Event();cleaning=asyncio.Event();release=asyncio.Event();finished=asyncio.Event()
    class SlowCleanup:
        async def transcribe(self,*args):
            started.set()
            try:await asyncio.Event().wait()
            finally:
                cleaning.set()
                await release.wait()
                finished.set()
    async def receive():
        await asyncio.Event().wait()
        return {'type':'http.disconnect'}
    request=Request({'type':'http'},receive)
    task=asyncio.create_task(_transcribe_connected(request,SlowCleanup(),b'audio','audio/webm','en'))
    try:
        await started.wait();task.cancel();await cleaning.wait();task.cancel();await asyncio.sleep(0)
        assert not task.done() and not finished.is_set()
        release.set()
        with pytest.raises(asyncio.CancelledError):await asyncio.wait_for(task,1)
        assert finished.is_set()
    finally:
        release.set();task.cancel();await asyncio.gather(task,return_exceptions=True)
