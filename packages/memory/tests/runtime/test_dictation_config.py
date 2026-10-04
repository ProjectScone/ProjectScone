import httpx
import pytest
from scone_memory import HashEmbedder, InMemoryDocumentStore, InMemoryVectorIndex, MemoryEngine
from scone_memory.api.__main__ import build_app
from scone_memory.runtime.config import Settings


@pytest.mark.parametrize('conversation',[False,True])
@pytest.mark.parametrize('configured',[False,True])
@pytest.mark.parametrize('backend',['mlx','faster-whisper'])
async def test_optional_dictation_is_composed_into_both_hosts(tmp_path,monkeypatch,conversation,configured,backend):
    from scone_memory.speech import dictation
    calls=[]
    class Speech:
        available=True
        def __init__(self,*paths,backend='mlx'): calls.append((paths,backend))
    monkeypatch.setattr(dictation,'LocalDictation',Speech)
    settings=Settings(keys={'key':'alpha'},conversations_journal=str(tmp_path/'journal.db') if conversation else None,
                      dictation_python='/local/python' if configured else None,
                      dictation_model='/local/model' if configured else None,
                      dictation_ffmpeg='/local/ffmpeg' if configured else None,
                      dictation_backend=backend)
    engine=await MemoryEngine(InMemoryDocumentStore(),InMemoryVectorIndex(),HashEmbedder()).open()
    try:
        app=build_app(settings,engine)
        async with app.router.lifespan_context(app):
            async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app),base_url='http://local',headers={'Authorization':'Bearer key'}) as client:
                response=await client.get('/v1/dictation/status')
                assert response.status_code==200,response.text
                assert response.json()['available'] is configured
        assert calls==[(('/local/python','/local/model','/local/ffmpeg'),backend)] if configured else calls==[]
    finally: await engine.close()


def test_backend_defaults_to_mac_and_rejects_unknown_selection():
    from scone_memory.core.errors import InvalidInput

    assert Settings.from_env({}).dictation_backend == 'mlx'
    assert Settings.from_env({'SCONE_DICTATION_BACKEND': 'faster-whisper'}).dictation_backend == 'faster-whisper'
    with pytest.raises(InvalidInput, match='SCONE_DICTATION_BACKEND'):
        Settings.from_env({'SCONE_DICTATION_BACKEND': 'remote'})
