import asyncio
import io
import sys
import wave
from pathlib import Path

import pytest
from scone_memory.agents.workflow import WorkflowError
from scone_memory.speech import dictation
from scone_memory.speech.dictation import LocalDictation


def audio(seconds=1):
    buffer = io.BytesIO()
    with wave.open(buffer, 'wb') as writer:
        writer.setnchannels(1); writer.setsampwidth(2); writer.setframerate(16000)
        writer.writeframes(b'\x01\x00' * (16000 * seconds))
    return buffer.getvalue()


@pytest.fixture
def service(tmp_path):
    (tmp_path / 'config.json').write_text('{}')
    (tmp_path / 'weights.npz').write_bytes(b'fixture')
    return LocalDictation(sys.executable, str(tmp_path), sys.executable)


@pytest.mark.parametrize('mime,language,content', [('text/plain', 'en', b'a'), ('audio/wav','../en',b'a'), ('audio/wav','en',b''), ('audio/wav','en',b'x'*(5*1024*1024+1))])
async def test_input_rejected_before_any_process(service, monkeypatch, mime, language, content):
    async def forbidden(*args, **kwargs): pytest.fail('must validate before process')
    monkeypatch.setattr(dictation, '_run', forbidden)
    with pytest.raises(WorkflowError, match='invalid_dictation'):
        await service.transcribe(content,mime,language)


async def test_normalized_audio_passed_only_by_pipe(service, monkeypatch):
    calls=[]
    async def run(args, content, **kwargs):
        calls.append((args,content))
        return audio() if len(calls)==1 else b'{"text":"  A quiet garden.  "}'
    monkeypatch.setattr(dictation,'_run',run)
    assert await service.transcribe(b'recorded','audio/webm','en')=='A quiet garden.'
    assert calls[0][0][calls[0][0].index('-protocol_whitelist')+1]=='pipe'
    assert calls[1][1]==audio()
    assert calls[1][0][1]=='-I'
    assert not service._active


@pytest.mark.parametrize('decoded', [b'invalid', audio(61)])
async def test_invalid_and_long_audio_never_reach_model(service, monkeypatch, decoded):
    calls=0
    async def run(*args,**kwargs):
        nonlocal calls
        calls+=1
        return decoded
    monkeypatch.setattr(dictation,'_run',run)
    with pytest.raises(WorkflowError,match='invalid_dictation'):
        await service.transcribe(b'input','audio/wav','en')
    assert calls==1


async def test_cancel_releases_capacity_and_overload_is_bounded(service,monkeypatch):
    started=asyncio.Event()
    async def run(*args,**kwargs):
        started.set(); await asyncio.Event().wait()
    monkeypatch.setattr(dictation,'_run',run)
    task=asyncio.create_task(service.transcribe(b'input','audio/wav','en'))
    await started.wait()
    with pytest.raises(WorkflowError,match='dictation_busy'):
        await service.transcribe(b'input','audio/wav','en')
    task.cancel()
    with pytest.raises(asyncio.CancelledError): await task
    assert not service._active


async def test_worker_timeout_is_sanitized_and_capacity_released(service,monkeypatch):
    async def run(*args,**kwargs): raise TimeoutError('private executable path')
    monkeypatch.setattr(dictation,'_run',run)
    with pytest.raises(WorkflowError,match='dictation_timeout') as error:
        await service.transcribe(b'input','audio/wav','en')
    assert 'private' not in str(error.value)
    assert not service._active


@pytest.fixture
def cpu_model(tmp_path):
    for name in ('config.json', 'model.bin', 'tokenizer.json', 'vocabulary.txt'):
        (tmp_path / name).write_text('{}')
    return tmp_path


async def test_cpu_backend_selects_worker_with_local_model(cpu_model, monkeypatch):
    service = LocalDictation(sys.executable, str(cpu_model), sys.executable, backend='faster-whisper')
    calls = []

    async def run(args, content, **kwargs):
        calls.append((args, content))
        if len(calls) == 1:
            return audio()
        assert args[3:] == [str(cpu_model), 'en', 'faster-whisper']
        return b'{"text":"A local CPU transcript."}'

    monkeypatch.setattr(dictation, '_run', run)
    assert await service.transcribe(b'recorded', 'audio/webm', 'en') == 'A local CPU transcript.'
    assert calls[1][1] == audio()


@pytest.mark.parametrize('missing', ['config.json', 'model.bin', 'tokenizer.json', 'vocabulary.txt'])
def test_cpu_backend_requires_complete_downloaded_model(cpu_model, missing):
    (cpu_model / missing).unlink()
    with pytest.raises(ValueError, match='downloaded model'):
        LocalDictation(sys.executable, str(cpu_model), sys.executable, backend='faster-whisper')


async def test_removed_cpu_tokenizer_disables_service_without_running(cpu_model, monkeypatch):
    service = LocalDictation(sys.executable, str(cpu_model), sys.executable, backend='faster-whisper')
    (cpu_model / 'tokenizer.json').unlink()

    async def forbidden(*args, **kwargs):
        pytest.fail('incomplete models must not run or download')

    monkeypatch.setattr(dictation, '_run', forbidden)
    with pytest.raises(WorkflowError, match='dictation_unavailable'):
        await service.transcribe(b'input', 'audio/wav', 'en')


async def test_failed_worker_releases_cpu_capacity(cpu_model, monkeypatch):
    service = LocalDictation(sys.executable, str(cpu_model), sys.executable, backend='faster-whisper')
    calls = 0

    async def run(args, content, **kwargs):
        nonlocal calls
        calls += 1
        if calls % 2:
            return audio()
        if calls == 2:
            raise WorkflowError('dictation_processing_failed')
        return b'{"text":"Recovered."}'

    monkeypatch.setattr(dictation, '_run', run)
    with pytest.raises(WorkflowError, match='dictation_processing_failed'):
        await service.transcribe(b'input', 'audio/wav', 'en')
    assert await service.transcribe(b'input', 'audio/wav', 'en') == 'Recovered.'


@pytest.mark.parametrize('stop', ['timeout', 'cancel'])
async def test_subprocess_is_reaped_after_timeout_or_cancellation(tmp_path, stop):
    import os

    pid_path = tmp_path / 'pid'
    script = 'import os, pathlib, sys, time; pathlib.Path(sys.argv[1]).write_text(str(os.getpid())); time.sleep(60)'
    task = asyncio.create_task(dictation._run(
        [sys.executable, '-I', '-c', script, str(pid_path)], b'ephemeral audio',
        timeout=0.5 if stop == 'timeout' else 10, limit=100))
    try:
        async with asyncio.timeout(3):
            while not pid_path.exists():
                await asyncio.sleep(0.01)
        pid = int(pid_path.read_text())
        if stop == 'cancel':
            task.cancel()
        with pytest.raises(TimeoutError if stop == 'timeout' else asyncio.CancelledError):
            await task
        with pytest.raises(ProcessLookupError):
            os.kill(pid, 0)
        assert sorted(path.name for path in tmp_path.iterdir()) == ['pid']
    finally:
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)


async def test_worker_environment_is_offline_and_does_not_inherit_credentials(monkeypatch):
    import json

    monkeypatch.setenv('HF_TOKEN', 'test-only-sentinel')
    script = 'import json, os; print(json.dumps(dict(os.environ)))'
    environment = json.loads(await dictation._run(
        [sys.executable, '-I', '-c', script], b'', timeout=5, limit=10000))
    assert environment['HF_HUB_OFFLINE'] == '1'
    assert environment['HF_HUB_DISABLE_IMPLICIT_TOKEN'] == '1'
    assert environment['HF_HUB_DISABLE_TELEMETRY'] == '1'
    assert environment['OMP_NUM_THREADS'] == '2'
    assert environment['OPENBLAS_NUM_THREADS'] == '1'
    assert 'HF_TOKEN' not in environment
