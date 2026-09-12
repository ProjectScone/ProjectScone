"""Sample-aligned windows preserve completed model work and source evidence."""
import asyncio
import io
import json
import struct
import wave

import pytest

from scone_memory.core.errors import InvalidInput
from scone_memory.ingestion.document_media import DocumentMedia
from scone_memory.ingestion.formats.media import MediaDocumentParser, TranscriptionSegment
from scone_memory.ingestion.formats.types import DocumentLimits
from .test_media_formats import ffmpeg  # noqa: F401
from .test_media_transcription_recovery import Receipts


def recording(seconds=2.5):
    buffer = io.BytesIO()
    with wave.open(buffer, 'wb') as output:
        output.setnchannels(1)
        output.setsampwidth(2)
        output.setframerate(16000)
        output.writeframes(struct.pack('<h', 5000) * int(seconds * 16000))
    return buffer.getvalue()


class Observer:
    def __init__(self, *, interrupt=None, empty=()):
        self.calls = []
        self.interrupt = interrupt
        self.empty = empty

    async def transcribe(self, audio):
        self.calls.append(audio)
        if len(self.calls) == self.interrupt:
            raise asyncio.CancelledError()
        if len(self.calls) in self.empty:
            return ()
        return (TranscriptionSegment(text='Observed speech.', start_seconds=0.0,
                                     end_seconds=0.25),)


def parser(ffmpeg, observer, seconds=1):
    return MediaDocumentParser(observer, ffmpeg_executable=ffmpeg, chunk_seconds=seconds)


async def test_completed_windows_survive_interruption_with_original_source_times(ffmpeg):
    receipts, observer = Receipts(), Observer(interrupt=2)
    config = DocumentMedia(parser(ffmpeg, observer), revision='windowed-v1')
    raw = recording()
    with pytest.raises(asyncio.CancelledError):
        await config.parse_checkpointed(raw, 'speech.wav', DocumentLimits(), receipts)
    assert len(observer.calls) == 2
    assert 'media-transcript' not in receipts.values
    resumed_observer = Observer()
    resumed = DocumentMedia(parser(ffmpeg, resumed_observer), revision='windowed-v1')
    parsed = await resumed.parse_checkpointed(raw, 'speech.wav', DocumentLimits(), receipts)
    assert len(resumed_observer.calls) == 2, 'only unfinished windows are observed again'
    assert [s.metadata['start_seconds'] for s in parsed.segments] == ['0.0', '1.0', '2.0']
    direct = await config.parse(raw, 'speech.wav', DocumentLimits())
    assert parsed == direct
    replay = await resumed.parse_checkpointed(raw, 'speech.wav', DocumentLimits(), receipts)
    assert replay == parsed and len(resumed_observer.calls) == 2
    assert all(b'RIFF' not in raw for raw in receipts.values.values())
    decoded = await config.media_parser.decode_audio(raw, 'speech.wav')
    assert decoded[44:] == b''.join(audio[44:] for audio in [observer.calls[0], *resumed_observer.calls])


async def test_silent_windows_are_receipted_without_inventing_text(ffmpeg):
    observer, receipts = Observer(empty=(1, 3)), Receipts()
    config = parser(ffmpeg, observer)
    parsed = await config.parse_checkpointed(recording(), 'speech.wav', DocumentLimits(), receipts)
    assert len(parsed.segments) == 1
    assert parsed.segments[0].metadata['start_seconds'] == '1.0'
    assert await config.parse_checkpointed(recording(), 'speech.wav', DocumentLimits(), receipts) == parsed
    assert len(observer.calls) == 3
    with pytest.raises(InvalidInput, match='nonempty'):
        await parser(ffmpeg, Observer(empty=(1, 2, 3))).parse(recording(), 'silent.wav')


@pytest.mark.parametrize('change', ['policy', 'disable', 'host', 'source', 'receipt', 'hole'])
async def test_partial_window_receipts_fail_closed_before_more_model_calls(ffmpeg, change):
    observer, receipts = Observer(interrupt=3), Receipts()
    config = DocumentMedia(parser(ffmpeg, observer), revision='windowed-v1')
    with pytest.raises(asyncio.CancelledError):
        await config.parse_checkpointed(recording(), 'speech.wav', DocumentLimits(), receipts)
    model = Observer()
    modified = DocumentMedia(parser(ffmpeg, model, seconds=2 if change == 'policy' else
        None if change == 'disable' else 1), revision='windowed-v1')
    if change == 'host':
        del receipts.values['media-host-binding']
    if change == 'receipt':
        value = json.loads(receipts.values['media-window-0001'])
        value['audio_sha256'] = '0' * 64
        receipts.values['media-window-0001'] = json.dumps(value).encode()
    if change == 'hole':
        del receipts.values['media-window-0000']
    with pytest.raises(InvalidInput, match='checkpoint'):
        await modified.parse_checkpointed(recording(2.6) if change == 'source' else recording(),
                                          'speech.wav', DocumentLimits(), receipts)
    assert model.calls == []


@pytest.mark.parametrize('limits', [DocumentLimits(max_segments=1), DocumentLimits(max_text_bytes=20)])
async def test_aggregate_limits_apply_before_saving_the_excess_window(ffmpeg, limits):
    observer, receipts = Observer(), Receipts()
    with pytest.raises(InvalidInput, match='limit'):
        await parser(ffmpeg, observer).parse_checkpointed(recording(), 'speech.wav', limits, receipts)
    assert len(observer.calls) == 2
    assert 'media-window-0000' in receipts.values
    assert 'media-window-0001' not in receipts.values


async def test_times_must_fit_the_window_before_offsetting(ffmpeg):
    async def outside(audio):
        return (TranscriptionSegment(text='Wrong interval', start_seconds=0.0, end_seconds=1.1),)
    receipts = Receipts()
    with pytest.raises(InvalidInput, match='outside'):
        await parser(ffmpeg, outside).parse_checkpointed(recording(), 'speech.wav', DocumentLimits(), receipts)
    assert 'media-window-0000' not in receipts.values


@pytest.mark.parametrize('seconds', [True, 0, -1, 121, 1.5, '30', float('nan'), 10**1000])
def test_invalid_window_configuration(ffmpeg, seconds):
    with pytest.raises(InvalidInput, match='chunk'):
        parser(ffmpeg, Observer(), seconds)


def test_quiet_boundary_preserves_every_sample_and_does_not_skip_silence():
    from scone_memory.ingestion.formats.media_windows import window_ranges
    loud = struct.pack('<h', 5000)
    pcm = loud * 12800 + b'\0\0' * 2560 + loud * 24640
    ranges = window_ranges(pcm, 1)
    assert ranges[0] == (0, 14080), 'cut midway through the 160ms quiet run'
    assert ranges[-1][1] == 40000
    assert all(a[1] == b[0] for a, b in zip(ranges, ranges[1:]))
    assert b''.join(pcm[start * 2:end * 2] for start, end in ranges) == pcm
    assert all(0 < end - start <= 16000 for start, end in ranges)
    assert window_ranges(b'\0\0' * 40000, 1) == ((0, 16000), (16000, 32000), (32000, 40000))


async def test_a_failed_checkpoint_write_repeats_only_the_unsaved_window(ffmpeg):
    class Unavailable(Receipts):
        failing = True

        def put(self, key, raw):
            if self.failing and key == 'media-window-0001':
                raise OSError('window journal unavailable')
            super().put(key, raw)

    receipts, observer = Unavailable(), Observer()
    config = parser(ffmpeg, observer)
    with pytest.raises(OSError, match='journal unavailable'):
        await config.parse_checkpointed(recording(), 'speech.wav', DocumentLimits(), receipts)
    assert len(observer.calls) == 2
    receipts.failing = False
    await config.parse_checkpointed(recording(), 'speech.wav', DocumentLimits(), receipts)
    assert len(observer.calls) == 4


async def test_decoder_changes_refuse_completed_prefix_without_another_observation(ffmpeg, monkeypatch):
    observer, receipts = Observer(interrupt=2), Receipts()
    config = parser(ffmpeg, observer)
    with pytest.raises(asyncio.CancelledError):
        await config.parse_checkpointed(recording(), 'speech.wav', DocumentLimits(), receipts)
    decode = config._decode_audio

    async def changed(data, limits, deadline):
        audio, duration = await decode(data, limits, deadline)
        return audio[:-2] + b'\0\0', duration

    monkeypatch.setattr(config, '_decode_audio', changed)
    with pytest.raises(InvalidInput, match='checkpoint'):
        await config.parse_checkpointed(recording(), 'speech.wav', DocumentLimits(), receipts)
    assert len(observer.calls) == 2


async def test_process_death_reuses_completed_windows_in_the_encrypted_workflow(ffmpeg, tmp_path):
    import os
    from pathlib import Path
    import sys
    import scone_memory

    (tmp_path / 'input.wav').write_bytes(recording())
    worker = Path(__file__).parent / 'fixtures' / 'media_receipt_worker.py'
    env = {key: value for key, value in os.environ.items() if key in {'PATH', 'TMPDIR', 'LANG'}}
    env['PYTHONPATH'] = str(Path(scone_memory.__file__).resolve().parent.parent)
    child = None
    try:
        child = await asyncio.create_subprocess_exec(sys.executable, str(worker), str(tmp_path), ffmpeg, 'hold', 'chunked',
            env=env, stdin=asyncio.subprocess.DEVNULL, stdout=asyncio.subprocess.DEVNULL,
            stderr=asyncio.subprocess.PIPE)

        async def entered():
            while not (tmp_path / 'window-entered').exists():
                if child.returncode is not None:
                    raise AssertionError((await child.stderr.read()).decode())
                await asyncio.sleep(0.02)

        await asyncio.wait_for(entered(), 15)
        assert (tmp_path / 'model-calls').read_text() == 'hold\nhold\n'
        child.kill()
        await asyncio.wait_for(child.wait(), 10)
        child = await asyncio.create_subprocess_exec(sys.executable, str(worker), str(tmp_path), ffmpeg, 'resume', 'chunked',
            env=env, stdin=asyncio.subprocess.DEVNULL, stdout=asyncio.subprocess.DEVNULL,
            stderr=asyncio.subprocess.PIPE)
        await asyncio.wait_for(child.wait(), 20)
        assert child.returncode == 0, (await child.stderr.read()).decode()
        assert (tmp_path / 'model-calls').read_text() == 'hold\nhold\nresume\nresume\n'
        result = json.loads((tmp_path / 'result.json').read_text())
        assert result['text'] == 'Checkpointed Café survives process death.'
        assert len(result['audio_sha256']) == 64
        assert result['starts'] == ['0.0', '1.0', '2.0']
        assert b'Checkpointed' not in (tmp_path / 'journal.db').read_bytes()
    finally:
        if child is not None and child.returncode is None:
            child.kill()
            await child.wait()
