"""Real adapters; only the remote HTTP peer is replaced. No provider calls."""

import asyncio
import json
from contextlib import aclosing

import httpx
import pytest

from scone_memory.providers.speech import CartesiaSpeech, ElevenLabsSpeech, SpeechProviderError


class Stream(httpx.AsyncByteStream):
    def __init__(self, parts):
        self.parts = parts
        self.closed = False
        self.reads = 0

    async def __aiter__(self):
        for part in self.parts:
            self.reads += 1
            yield part

    async def aclose(self):
        self.closed = True


@pytest.fixture(params=[CartesiaSpeech, ElevenLabsSpeech], ids=['cartesia', 'elevenlabs'])
def provider(request):
    return request.param


def make(provider, stream, *, status=200, headers=None, **options):
    requests = []

    async def remote(request):
        requests.append(request)
        return httpx.Response(status, headers=headers or {'content-type': 'application/octet-stream'}, stream=stream)

    adapter = provider(api_key='private-test-key', model='chosen-model', voice='chosen-voice',
                       transport=httpx.MockTransport(remote), **options)
    return adapter, requests


async def test_explicit_model_voice_and_pcm_request_stream_without_buffering_the_reply(provider):
    stream = Stream([b'\x01', b'\x02\x03\x04\x05', b'\x06'])
    adapter, requests = make(provider, stream)
    assert not requests
    async with aclosing(adapter), aclosing(adapter.synthesize('Hello 🌿')) as audio:
        first = await anext(audio)
        assert first.pcm == b'\x01\x02\x03\x04'
        assert stream.reads == 2, 'first complete samples arrive before response EOF'
        chunks = [first] + [chunk async for chunk in audio]
    assert b''.join(c.pcm for c in chunks) == b'\x01\x02\x03\x04\x05\x06'
    assert {(c.sample_rate, c.channels) for c in chunks} == {(24000, 1)}
    assert stream.closed and len(requests) == 1
    request = requests[0]
    assert request.method == 'POST'
    assert request.headers['accept-encoding'] == 'identity'
    body = json.loads(request.content)
    if provider is CartesiaSpeech:
        assert str(request.url) == 'https://api.cartesia.ai/tts/bytes'
        assert request.headers['authorization'] == 'Bearer private-test-key'
        assert request.headers['cartesia-version'] == '2026-08-14'
        assert body == {'model_id': 'chosen-model', 'transcript': 'Hello 🌿', 'voice': 'chosen-voice',
                        'output_format': {'container': 'raw', 'encoding': 'pcm_s16le', 'sample_rate': 24000}}
    else:
        assert str(request.url) == 'https://api.elevenlabs.io/v1/text-to-speech/chosen-voice/stream?output_format=pcm_24000'
        assert request.headers['xi-api-key'] == 'private-test-key'
        assert body == {'model_id': 'chosen-model', 'text': 'Hello 🌿'}


@pytest.mark.parametrize('parts', [[], [b'\x00'], [b'\x00\x00', b'\x01']])
async def test_empty_or_partial_pcm_cannot_be_successful_audio(provider, parts):
    stream = Stream(parts)
    adapter, requests = make(provider, stream)
    async with aclosing(adapter):
        with pytest.raises(SpeechProviderError):
            _ = [chunk async for chunk in adapter.synthesize('Hello')]
    assert stream.closed and len(requests) == 1


@pytest.mark.parametrize('status', [302, 401, 429, 503])
async def test_http_errors_are_not_retried_or_exposed_as_audio_or_secret_messages(provider, status):
    stream = Stream([b'private-test-key and private transcript'])
    adapter, requests = make(provider, stream, status=status, headers={'location': 'https://untrusted.invalid/'})
    async with aclosing(adapter):
        with pytest.raises(SpeechProviderError) as error:
            _ = [chunk async for chunk in adapter.synthesize('private transcript')]
    assert str(status) in str(error.value)
    assert 'private' not in str(error.value)
    assert stream.reads == 0 and stream.closed and len(requests) == 1


@pytest.mark.parametrize('mime', ['application/json', 'audio/mpeg', 'audio/wav', 'audio/L16', 'text/html'])
async def test_non_pcm_success_responses_are_rejected_before_audio_is_emitted(provider, mime):
    stream = Stream([b'not pcm!'])
    adapter, _ = make(provider, stream, headers={'content-type': mime})
    async with aclosing(adapter):
        with pytest.raises(SpeechProviderError):
            _ = [chunk async for chunk in adapter.synthesize('Hello')]
    assert stream.closed and stream.reads == 0


async def test_limits_output_frames_and_total_response_size(provider):
    stream = Stream([b'\x00' * 2050, b'\x01' * 2050])
    adapter, _ = make(provider, stream, chunk_bytes=512, max_audio_bytes=3072)
    chunks = []
    async with aclosing(adapter):
        with pytest.raises(SpeechProviderError, match='limit'):
            async for chunk in adapter.synthesize('Hello'):
                chunks.append(chunk)
    assert chunks and all(len(c.pcm) <= 512 and len(c.pcm) % 2 == 0 for c in chunks)
    assert sum(len(c.pcm) for c in chunks) <= 3072
    assert stream.closed


async def test_cancellation_closes_response_and_closed_adapter_rejects_new_work(provider):
    entered = asyncio.Event()

    class Waiting(Stream):
        async def __aiter__(self):
            entered.set()
            await asyncio.Event().wait()
            yield b'\x00\x00'

    stream = Waiting([])
    adapter, requests = make(provider, stream)
    async with aclosing(adapter):
        audio = adapter.synthesize('Hello')
        task = asyncio.create_task(anext(audio))
        await asyncio.wait_for(entered.wait(), 1)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        await audio.aclose()
        assert stream.closed
    assert len(requests) == 1
    with pytest.raises(SpeechProviderError, match='closed'):
        await anext(adapter.synthesize('Do not send'))
    assert len(requests) == 1


async def test_closing_iterator_early_releases_response_and_reuses_client_for_next_turn(provider):
    stream = Stream([b'\x00\x00', b'\x01\x01'])
    adapter, requests = make(provider, stream)
    async with aclosing(adapter):
        async with aclosing(adapter.synthesize('First')) as audio:
            assert (await anext(audio)).pcm == b'\x00\x00'
            with pytest.raises(SpeechProviderError, match='active'):
                await anext(adapter.synthesize('Overlapping'))
        assert stream.closed and stream.reads == 1
        assert len([c async for c in adapter.synthesize('Next')]) == 2
    assert len(requests) == 2


@pytest.mark.parametrize('text', ['', ' ', None, 'x' * 32001])
async def test_invalid_text_fails_before_network(provider, text):
    adapter, requests = make(provider, Stream([]))
    async with aclosing(adapter):
        with pytest.raises(ValueError):
            await anext(adapter.synthesize(text))
    assert requests == []


@pytest.mark.parametrize('override', [{'api_key': ''}, {'api_key': 'key\nheader'}, {'model': ''},
                                     {'voice': '../elsewhere'}, {'sample_rate': True}, {'sample_rate': 12345},
                                     {'chunk_bytes': 511}, {'max_audio_bytes': 0}, {'timeout': float('inf')}])
def test_invalid_config_is_rejected_without_opening_a_client(provider, override):
    options = {'api_key': 'test-key', 'model': 'chosen', 'voice': 'chosen', **override}
    with pytest.raises(ValueError):
        provider(**options)


async def test_transport_failure_does_not_leak_remote_details_or_retry(provider):
    class Broken(Stream):
        async def __aiter__(self):
            yield b'\x00\x00'
            raise httpx.ReadError('private-test-key and private transcript')

    stream = Broken([])
    adapter, requests = make(provider, stream)
    async with aclosing(adapter):
        with pytest.raises(SpeechProviderError) as error:
            _ = [c async for c in adapter.synthesize('private transcript')]
    assert 'private' not in str(error.value) and error.value.__suppress_context__
    assert stream.closed and len(requests) == 1


async def test_encoded_success_is_not_reinterpreted_as_pcm(provider):
    stream = Stream([b'\x00\x00'])
    adapter, _ = make(provider, stream, headers={'content-type': 'audio/pcm', 'content-encoding': 'gzip'})
    async with aclosing(adapter):
        with pytest.raises(SpeechProviderError):
            _ = [c async for c in adapter.synthesize('Hello')]
    assert stream.closed and stream.reads == 0


async def test_closing_the_adapter_cannot_expose_transport_secrets(provider):
    class BrokenClose(httpx.MockTransport):
        async def aclose(self):
            raise httpx.CloseError('private-test-key')

    adapter = provider(api_key='private-test-key', model='chosen', voice='chosen',
                       transport=BrokenClose(lambda r: httpx.Response(200, headers={'content-type': 'audio/pcm'}, stream=Stream([b'\x00\x00']))))
    _ = [c async for c in adapter.synthesize('Hello')]
    with pytest.raises(SpeechProviderError) as error:
        await adapter.aclose()
    assert 'private' not in str(error.value) and error.value.__suppress_context__


async def test_native_voice_session_uses_selected_speech_adapter_and_retains_only_public_text(provider):
    from scone_memory import HashEmbedder, InMemoryDocumentStore, InMemoryVectorIndex, MemoryEngine, Record
    from scone_memory.realtime.audio import AudioChunk, Transcript
    from scone_memory.realtime.events import TextDelta, ReplyCompleted
    from scone_memory.realtime.voice import VoiceSession

    class Resource:
        async def aclose(self):
            pass

    class Device(Resource):
        def __init__(self):
            self.output = []

        async def receive(self):
            yield AudioChunk(b'\x00\x00' * 160, 16000)
            await asyncio.Event().wait()

        async def send(self, audio, turn_id):
            self.output.append((audio, turn_id))

        async def clear(self, turn_id):
            pass

    class Recognizer(Resource):
        async def transcribe(self, audio):
            async for chunk in audio:
                assert chunk.sample_rate == 16000
                yield Transcript('Where does Juniper point?')

    class Model(Resource):
        async def respond(self, messages):
            assert any('Juniper points to Polaris.' in m['content'] for m in messages)
            yield TextDelta('Juniper points to Polaris.')
            yield ReplyCompleted()

    memory = await MemoryEngine(InMemoryDocumentStore(), InMemoryVectorIndex(), HashEmbedder()).open()
    stream = Stream([b'\x01', b'\x00\x02\x00'])
    adapter, requests = make(provider, stream)
    device = Device()
    await memory.remember_many('speech-test', [Record('Juniper points to Polaris.', source='guide')])
    session = VoiceSession(memory, 'speech-test', 'speech-adapter-test', transport_factory=lambda: device,
                           stt_factory=Recognizer, model_factory=Model, tts_factory=lambda: adapter,
                           capture=True, turn_timeout=2, session_timeout=5)
    running = asyncio.create_task(session.run())
    try:
        async with asyncio.timeout(3):
            while session.stored_count < 2 and not running.done():
                await asyncio.sleep(.005)
        assert session.stored_count == 2
        episodes = await memory.episodes('speech-test', {'session_id': 'speech-adapter-test'})
        assert {e.content for e in episodes} == {'Where does Juniper point?', 'Juniper points to Polaris.'}
        assert b''.join(c.pcm for c, _ in device.output) == b'\x01\x00\x02\x00'
        assert {(c.sample_rate, c.channels) for c, _ in device.output} == {(24000, 1)}
        assert len(requests) == 1
        body = json.loads(requests[0].content)
        assert body.get('text', body.get('transcript')) == 'Juniper points to Polaris.'
    finally:
        await session.close()
        await asyncio.gather(running, return_exceptions=True)
        await memory.close()
    assert stream.closed
