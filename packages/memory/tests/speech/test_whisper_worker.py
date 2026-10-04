import io
import json
import sys
from types import SimpleNamespace
import wave

import pytest

from scone_memory.speech import whisper_worker

np = pytest.importorskip('numpy')


@pytest.fixture
def model(tmp_path):
    for name in ('config.json', 'model.bin', 'tokenizer.json', 'vocabulary.txt'):
        (tmp_path / name).write_text('{}')
    return tmp_path


def invoke(monkeypatch, model, *, backend='faster-whisper', seconds=1):
    audio = io.BytesIO()
    with wave.open(audio, 'wb') as writer:
        writer.setnchannels(1)
        writer.setsampwidth(2)
        writer.setframerate(16000)
        writer.writeframes(b'\x00\x40' * (16000 * seconds))
    monkeypatch.setattr(sys, 'argv', ['whisper_worker.py', str(model), 'en', backend])
    monkeypatch.setattr(sys, 'stdin', SimpleNamespace(buffer=io.BytesIO(audio.getvalue())))
    whisper_worker.main()


def test_cpu_worker_loads_only_local_weights_with_bounded_int8_inference(model, monkeypatch, capsys):
    class Model:
        def __init__(self, path, *, device, compute_type, cpu_threads, num_workers, local_files_only):
            assert path == str(model)
            assert (device, compute_type, cpu_threads, num_workers, local_files_only) == ('cpu', 'int8', 2, 1, True)

        def transcribe(self, samples, *, language, temperature, condition_on_previous_text, beam_size, vad_filter):
            assert language == 'en'
            assert (temperature, condition_on_previous_text, beam_size, vad_filter) == (0, False, 1, False)
            assert samples.dtype == np.float32
            assert samples.shape == (16000,)
            assert float(samples[0]) == 0.5
            return iter([SimpleNamespace(text='  A quiet'), SimpleNamespace(text=' garden.  ')]), None

    # Loading the CPU backend must not depend on an Apple-only package.
    monkeypatch.setitem(sys.modules, 'mlx_whisper', None)
    monkeypatch.setitem(sys.modules, 'faster_whisper', SimpleNamespace(WhisperModel=Model))
    before = sorted(model.iterdir())
    invoke(monkeypatch, model)
    assert json.loads(capsys.readouterr().out) == {'text': 'A quiet garden.'}
    assert sorted(model.iterdir()) == before


def test_worker_rejects_missing_tokenizer_before_importing_backend(model, monkeypatch):
    (model / 'tokenizer.json').unlink()
    monkeypatch.setitem(sys.modules, 'faster_whisper', None)
    monkeypatch.setitem(sys.modules, 'mlx_whisper', None)
    with pytest.raises(ValueError, match='downloaded local model'):
        invoke(monkeypatch, model)


def test_cpu_worker_failure_propagates_without_audio_files_or_fallback(model, monkeypatch, capsys):
    class BrokenModel:
        def __init__(self, *args, **kwargs):
            raise RuntimeError('model unavailable')

    monkeypatch.setitem(sys.modules, 'mlx_whisper', None)
    monkeypatch.setitem(sys.modules, 'faster_whisper', SimpleNamespace(WhisperModel=BrokenModel))
    before = sorted(model.iterdir())
    with pytest.raises(RuntimeError, match='model unavailable'):
        invoke(monkeypatch, model)
    assert capsys.readouterr().out == ''
    assert sorted(model.iterdir()) == before


def test_mac_worker_preserves_existing_transcription_contract(model, monkeypatch, capsys):
    def transcribe(samples, *, path_or_hf_repo, language, temperature, verbose, condition_on_previous_text):
        assert path_or_hf_repo == str(model)
        assert (language, temperature, verbose, condition_on_previous_text) == ('en', 0, None, False)
        return {'text': ' The Mac transcript. '}

    (model / 'weights.npz').write_bytes(b'fixture')
    monkeypatch.setitem(sys.modules, 'mlx_whisper', SimpleNamespace(transcribe=transcribe))
    monkeypatch.setitem(sys.modules, 'faster_whisper', None)
    invoke(monkeypatch, model, backend='mlx')
    assert json.loads(capsys.readouterr().out) == {'text': 'The Mac transcript.'}


def test_unknown_backend_cannot_silently_use_mac_or_cpu(model, monkeypatch):
    monkeypatch.setitem(sys.modules, 'mlx_whisper', None)
    monkeypatch.setitem(sys.modules, 'faster_whisper', None)
    with pytest.raises(ValueError, match='backend'):
        invoke(monkeypatch, model, backend='remote')


def test_cpu_worker_stops_consuming_segments_at_transcript_limit(model, monkeypatch):
    class Model:
        def __init__(self, *args, **kwargs):
            pass

        def transcribe(self, *args, **kwargs):
            def segments():
                yield SimpleNamespace(text='x' * 16001)
                pytest.fail('oversized output must stop inference')

            return segments(), None

    monkeypatch.setitem(sys.modules, 'faster_whisper', SimpleNamespace(WhisperModel=Model))
    with pytest.raises(ValueError, match='Transcript limit'):
        invoke(monkeypatch, model)
