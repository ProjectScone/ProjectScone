"""Completed model work survives a later extraction-stage storage failure."""
import asyncio
import json

import pytest

from scone_memory.agents import WorkflowError
from scone_memory.core.errors import InvalidInput
from scone_memory.ingestion.document_media import DocumentMedia
from scone_memory.ingestion.file_workflow import DocumentIngestionWorkflow
from scone_memory.ingestion.files import document_provenance, encode_manifest, prepare_document
from scone_memory.ingestion.formats.media import MediaDocumentParser, TranscriptionSegment
from scone_memory.ingestion.formats.types import DocumentLimits, DocumentSegment, ParsedDocument
from .test_document_workflow import KEY, open_memory
from .test_media_formats import ffmpeg, silent_wav  # noqa: F401


class Receipts:
    def __init__(self):
        self.values = {}

    def get(self, key):
        return self.values.get(key)

    def put(self, key, value):
        assert len(value) <= 16 * 1024 * 1024
        self.values[key] = value


class Transcriber:
    def __init__(self):
        self.calls = 0

    async def transcribe(self, audio):
        self.calls += 1
        return (TranscriptionSegment(text='Café transcript survives storage interruption.',
                                     start_seconds=-0.0, end_seconds=0.05),)


def media(ffmpeg, transcriber, revision='model-v1', duration=60.0):
    return DocumentMedia(MediaDocumentParser(transcriber, ffmpeg_executable=ffmpeg,
                         max_duration_seconds=duration), revision=revision)


async def extract(config, receipts, *, raw=None, filename='recording.wav', limits=DocumentLimits()):
    return await prepare_document(silent_wav() if raw is None else raw, filename,
        parser=config.parser(), limits=limits, extraction_checkpoint=receipts)


async def test_replay_preserves_manifest_and_observed_negative_zero_without_transcription(ffmpeg):
    transcriber, receipts = Transcriber(), Receipts()
    config = media(ffmpeg, transcriber)
    direct = await extract(config, None)
    completed = await extract(config, receipts)
    replay = await extract(config, receipts)
    assert encode_manifest(direct) == encode_manifest(completed) == encode_manifest(replay)
    assert replay.parsed.segments[0].metadata['start_seconds'] == '-0.0'
    assert transcriber.calls == 2, 'one direct call, one checkpointed call, none on replay'
    assert receipts.values
    assert all(b'RIFF' not in value for value in receipts.values.values()), 'WAV is not journaled'


@pytest.mark.parametrize('failure', ['cancel', 'storage'])
async def test_manifest_failure_reuses_transcription_after_journal_and_storage_reopen(tmp_path, monkeypatch, ffmpeg, failure):
    memory = await open_memory(tmp_path)
    transcriber = Transcriber()
    config = media(ffmpeg, transcriber)
    original = await memory.attach('alpha', silent_wav(), 'audio/wav', filename='recording.wav')
    args = {'space': 'alpha', 'attachment_id': original.attachment_id}
    path = tmp_path / 'media-jobs.db'
    job = DocumentIngestionWorkflow(memory, path, key=KEY, parser=config.parser(),
                                    parser_revision='native-media-v1', automatic_retries=False)
    entered = asyncio.Event()
    attach = memory.attach

    async def interrupt(space, raw, media_type, **kwargs):
        if kwargs.get('filename') == 'document-provenance.json':
            entered.set()
            if failure == 'storage':
                raise OSError('synthetic manifest storage interruption')
            await asyncio.Event().wait()
        return await attach(space, raw, media_type, **kwargs)

    monkeypatch.setattr(memory, 'attach', interrupt)
    task = asyncio.create_task(job.run('recording', **args))
    try:
        await asyncio.wait_for(entered.wait(), 10)
        if failure == 'cancel':
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task
        else:
            with pytest.raises(WorkflowError, match='step_failed'):
                await task
        assert transcriber.calls == 1
        progress = job.status('recording', **args)
        assert progress.completed_steps == () and progress.checkpoint_count >= 1
    finally:
        if not task.done():
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)
        job.close()
        await memory.close()

    memory = await open_memory(tmp_path)
    resumed = DocumentIngestionWorkflow(memory, path, key=KEY, parser=config.parser(),
                                        parser_revision='native-media-v1', automatic_retries=False)
    try:
        result = await resumed.run('recording', **args)
        assert transcriber.calls == 1
        evidence = await document_provenance(memory, 'alpha', result.results['index']['episode_id'])
        assert evidence.segments[0].text == 'Café transcript survives storage interruption.'
        assert evidence.metadata['transcriber_revision'] == 'model-v1'
        assert (await memory.documents.counts('alpha')).episodes == 1
    finally:
        resumed.close()
        await memory.close()
    assert b'Caf' not in path.read_bytes(), 'journal must not expose transcript plaintext'


@pytest.mark.parametrize('change', ['source', 'filename', 'revision', 'limits', 'duration'])
async def test_completed_receipt_refuses_changed_binding_before_another_model_call(ffmpeg, change):
    transcriber, receipts = Transcriber(), Receipts()
    config = media(ffmpeg, transcriber)
    await extract(config, receipts)
    if change == 'revision':
        config = media(ffmpeg, transcriber, revision='model-v2')
    if change == 'duration':
        config = media(ffmpeg, transcriber, duration=30.0)
    with pytest.raises(InvalidInput, match='checkpoint'):
        await extract(config, receipts, raw=silent_wav(0.2) if change == 'source' else None,
            filename='renamed.wav' if change == 'filename' else 'recording.wav',
            limits=DocumentLimits(max_segments=1) if change == 'limits' else DocumentLimits())
    assert transcriber.calls == 1


@pytest.mark.parametrize('damage', ['json', 'oversize', 'schema', 'binding', 'order', 'duration', 'audio_digest'])
async def test_invalid_completed_receipt_does_not_fall_back_to_model(ffmpeg, damage):
    transcriber, receipts = Transcriber(), Receipts()
    await extract(media(ffmpeg, transcriber), receipts)
    key = 'media-transcript'
    payload = json.loads(receipts.values[key])
    if damage == 'schema':
        payload['schema_version'] = True
    if damage == 'binding':
        payload['binding'] = '0' * 64
    if damage == 'order':
        payload['segments'] = [
            {'text': 'Later', 'start_seconds': 0.05, 'end_seconds': 0.09},
            {'text': 'Earlier', 'start_seconds': 0.0, 'end_seconds': 0.04},
        ]
    if damage == 'duration':
        payload['segments'][0]['end_seconds'] = 1.0
    if damage == 'audio_digest':
        payload['audio_sha256'] = '0' * 64
    receipts.values[key] = (b'{' if damage == 'json' else b' ' * (16 * 1024 * 1024 + 1)
                            if damage == 'oversize' else json.dumps(payload).encode())
    with pytest.raises(InvalidInput, match='checkpoint'):
        await extract(media(ffmpeg, transcriber), receipts)
    assert transcriber.calls == 1


async def test_checkpoint_write_failure_never_claims_success_or_durable_model_reuse(ffmpeg):
    transcriber = Transcriber()

    class Unavailable(Receipts):
        def put(self, key, value):
            if key == 'media-transcript':
                raise OSError('synthetic journal outage')
            super().put(key, value)

    receipts = Unavailable()
    with pytest.raises(OSError, match='journal outage'):
        await extract(media(ffmpeg, transcriber), receipts)
    assert transcriber.calls == 1 and 'media-transcript' not in receipts.values


async def test_changed_decoded_audio_refuses_receipt_without_transcription(ffmpeg, monkeypatch):
    transcriber, receipts = Transcriber(), Receipts()
    config = media(ffmpeg, transcriber)
    await extract(config, receipts)
    decode = config.media_parser._decode_audio

    async def changed(data, limits, deadline):
        wav, duration = await decode(data, limits, deadline)
        altered = bytearray(wav)
        altered[44] ^= 1
        return bytes(altered), duration

    monkeypatch.setattr(config.media_parser, '_decode_audio', changed)
    with pytest.raises(InvalidInput, match='checkpoint'):
        await extract(config, receipts)
    assert transcriber.calls == 1


async def test_missing_host_binding_cannot_adopt_a_completed_native_receipt(ffmpeg):
    transcriber, receipts = Transcriber(), Receipts()
    await extract(media(ffmpeg, transcriber), receipts)
    del receipts.values['media-host-binding']
    with pytest.raises(InvalidInput, match='checkpoint'):
        await extract(media(ffmpeg, transcriber, revision='different-model'), receipts)
    assert transcriber.calls == 1


async def test_checkpoint_read_outage_stops_before_another_model_call(ffmpeg):
    transcriber = Transcriber()

    class Unavailable(Receipts):
        def get(self, key):
            raise OSError('synthetic journal read outage')

    with pytest.raises(OSError, match='journal read outage'):
        await extract(media(ffmpeg, transcriber), Unavailable())
    assert transcriber.calls == 0


async def test_inflight_model_cancellation_leaves_no_completed_transcript(ffmpeg):
    entered = asyncio.Event()
    receipts = Receipts()

    async def waiting(audio):
        entered.set()
        await asyncio.Event().wait()

    task = asyncio.create_task(extract(media(ffmpeg, waiting), receipts))
    try:
        await asyncio.wait_for(entered.wait(), 10)
    finally:
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
    assert 'media-transcript' not in receipts.values


async def test_expired_decode_does_not_construct_an_unawaited_provider_call(ffmpeg, monkeypatch, recwarn):
    import gc

    transcriber = Transcriber()
    parser = media(ffmpeg, transcriber).media_parser

    async def late_decode(data, limits, deadline):
        await asyncio.sleep(0.02)
        return silent_wav(), 0.1

    monkeypatch.setattr(parser, '_decode_audio', late_decode)
    with pytest.raises(InvalidInput, match='wall time'):
        await parser.parse(silent_wav(), 'recording.wav', DocumentLimits(timeout_seconds=0.01))
    gc.collect()
    assert transcriber.calls == 0
    assert not [w for w in recwarn if 'was never awaited' in str(w.message)]


async def test_largest_escaped_transcript_fits_one_bounded_receipt(ffmpeg):
    text = 'Café\x00' + '\x01' * (2_000_000 - len('Café\x00'.encode()))

    async def long_transcript(audio):
        return (TranscriptionSegment(text=text, start_seconds=0.0, end_seconds=0.05),)

    receipts = Receipts()
    config = media(ffmpeg, long_transcript)
    first = await extract(config, receipts)
    assert len(receipts.values['media-transcript']) < 16 * 1024 * 1024
    replay = await extract(config, receipts)
    assert encode_manifest(first) == encode_manifest(replay)


async def test_combined_text_and_segment_limits_fit_compact_receipt(ffmpeg):
    segments = tuple(TranscriptionSegment(text='x' + '\x01' * (99 if i == 19_999 else 97),
        start_seconds=0.0, end_seconds=0.05) for i in range(20_000))
    calls = []

    async def overlapping(audio):
        calls.append(len(audio))
        return segments

    config, receipts = media(ffmpeg, overlapping), Receipts()
    first = await extract(config, receipts)
    assert len(first.parsed.segments) == 20_000
    assert sum(len(s.text.encode()) for s in first.parsed.segments) + 2 * 19_999 == 2_000_000
    assert len(receipts.values['media-transcript']) < 16 * 1024 * 1024
    replay = await extract(config, receipts)
    assert encode_manifest(first) == encode_manifest(replay)
    assert len(calls) == 1


@pytest.mark.parametrize('kind', ['native_subclass', 'native_instance', 'wrapper_subclass'])
async def test_checkpoint_dispatch_preserves_custom_parse_overrides(ffmpeg, kind):
    transcriber, calls = Transcriber(), []
    safe = ParsedDocument(format='wav', parser='custom-redactor',
                          segments=(DocumentSegment(text='Public summary', locator='audio:0'),))

    async def redacted(data, filename, limits):
        calls.append('custom')
        return safe

    class CustomParser(MediaDocumentParser):
        async def parse(self, data, filename, limits):
            return await redacted(data, filename, limits)

    class CustomWrapper(DocumentMedia):
        async def parse(self, data, filename, limits):
            return await redacted(data, filename, limits)

    parser = (CustomParser if kind == 'native_subclass' else MediaDocumentParser)(
        transcriber, ffmpeg_executable=ffmpeg)
    if kind == 'native_instance':
        parser.parse = redacted
    config = (CustomWrapper if kind == 'wrapper_subclass' else DocumentMedia)(parser, revision='custom-v1')
    receipts = Receipts()
    for _ in range(2):
        assert (await extract(config, receipts)).parsed.segments[0].text == 'Public summary'
    assert calls == ['custom', 'custom'] and transcriber.calls == 0


async def test_process_death_after_model_work_reuses_receipt(ffmpeg, tmp_path):
    import os
    from pathlib import Path
    import sys
    import scone_memory

    (tmp_path / 'input.wav').write_bytes(silent_wav())
    worker = Path(__file__).parent / 'fixtures' / 'media_receipt_worker.py'
    env = {key: value for key, value in os.environ.items() if key in {'PATH', 'TMPDIR', 'LANG'}}
    env['PYTHONPATH'] = str(Path(scone_memory.__file__).resolve().parent.parent)
    child = None
    try:
        child = await asyncio.create_subprocess_exec(sys.executable, str(worker), str(tmp_path), ffmpeg, 'hold',
            env=env, stdin=asyncio.subprocess.DEVNULL, stdout=asyncio.subprocess.DEVNULL,
            stderr=asyncio.subprocess.PIPE)

        async def entered():
            while not (tmp_path / 'manifest-entered').exists():
                if child.returncode is not None:
                    raise AssertionError((await child.stderr.read()).decode())
                await asyncio.sleep(0.02)

        await asyncio.wait_for(entered(), 15)
        assert (tmp_path / 'model-calls').read_text() == 'hold\n'
        child.kill()
        await asyncio.wait_for(child.wait(), 10)
        child = await asyncio.create_subprocess_exec(sys.executable, str(worker), str(tmp_path), ffmpeg, 'resume',
            env=env, stdin=asyncio.subprocess.DEVNULL, stdout=asyncio.subprocess.DEVNULL,
            stderr=asyncio.subprocess.PIPE)
        await asyncio.wait_for(child.wait(), 20)
        assert child.returncode == 0, (await child.stderr.read()).decode()
        assert (tmp_path / 'model-calls').read_text() == 'hold\n'
        result = json.loads((tmp_path / 'result.json').read_text())
        assert result['text'] == 'Checkpointed Café survives process death.'
        assert len(result['audio_sha256']) == 64
        assert b'Checkpointed' not in (tmp_path / 'journal.db').read_bytes()
    finally:
        if child is not None and child.returncode is None:
            child.kill()
            await child.wait()
