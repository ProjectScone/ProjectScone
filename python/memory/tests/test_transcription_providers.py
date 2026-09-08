"""Words out of audio, over an HTTP transcription API.

The recogniser is asked once per utterance rather than kept on a socket,
so these check the two things that shape: that it asks at the right
moment with the right audio, and that everything which can go wrong at
the far end fails safely.
"""

from __future__ import annotations

import io
import json
import wave

import httpx
import pytest

from scone_memory.providers.transcription import OpenAITranscription, TranscriptionProviderError
from scone_memory.realtime.audio import AudioChunk, SpeechStarted, Transcript

RATE = 16000
#: A fifth of a second, loud enough for the gate to open on.
LOUD = AudioChunk(pcm=b"\x00\x40" * (RATE // 5), sample_rate=RATE)
QUIET = AudioChunk(pcm=b"\x00\x00" * (RATE // 5), sample_rate=RATE)


def recorded(text="the quarterly plan", status=200, body=None, seen=None):
    def handler(request: httpx.Request) -> httpx.Response:
        if seen is not None:
            seen.append(request)
        if status != 200:
            return httpx.Response(status, text=body or "upstream detail nobody should see")
        return httpx.Response(200, json={"text": text})

    return httpx.MockTransport(handler)


def adapter(transport, **options):
    return OpenAITranscription(api_key="sk-not-a-real-key", model="whisper-1",
                               transport=transport, rate=RATE, **options)


async def feed(chunks):
    for chunk in chunks:
        yield chunk


async def heard(stt, chunks):
    return [event async for event in stt.transcribe(feed(chunks))]


async def test_an_utterance_becomes_one_request_and_one_transcript():
    seen = []
    stt = adapter(recorded(seen=seen))
    events = await heard(stt, [QUIET, LOUD, LOUD, QUIET, QUIET, QUIET])

    assert any(isinstance(e, SpeechStarted) for e in events), "the turn opened when speech began"
    finals = [e for e in events if isinstance(e, Transcript)]
    assert [e.text for e in finals] == ["the quarterly plan"]
    assert all(e.final for e in finals)
    assert len(seen) == 1, "asked once, when the speaker stopped"
    await stt.aclose()


async def test_a_room_that_never_speaks_is_never_sent_anywhere():
    """Silence costs nothing and, more to the point, leaves the room."""
    seen = []
    stt = adapter(recorded(seen=seen))
    events = await heard(stt, [QUIET, QUIET, QUIET, QUIET])

    assert events == [] and seen == []
    await stt.aclose()


async def test_speech_still_running_when_the_audio_ends_is_still_transcribed():
    """A caller hanging up mid-sentence has still said something."""
    seen = []
    stt = adapter(recorded(seen=seen))
    events = await heard(stt, [LOUD, LOUD])

    assert [e.text for e in events if isinstance(e, Transcript)] == ["the quarterly plan"]
    assert len(seen) == 1
    await stt.aclose()


async def test_the_audio_is_sent_as_a_wav_at_the_rate_it_was_captured():
    """Sending raw PCM and letting the far end guess the rate is how a
    voice comes back an octave out."""
    seen = []
    stt = adapter(recorded(seen=seen))
    await heard(stt, [LOUD, LOUD, QUIET, QUIET, QUIET])

    body = seen[0].content
    start = body.index(b"RIFF")
    with wave.open(io.BytesIO(body[start:body.index(b"\r\n--", start)]), "rb") as sound:
        assert sound.getframerate() == RATE
        assert sound.getnchannels() == 1
        assert sound.getsampwidth() == 2
    await stt.aclose()


async def test_more_audio_than_was_agreed_is_refused_before_it_is_sent():
    seen = []
    stt = adapter(recorded(seen=seen), max_audio_bytes=len(LOUD.pcm))
    with pytest.raises(TranscriptionProviderError, match="longer than"):
        await heard(stt, [LOUD, LOUD, LOUD])
    assert seen == [], "nothing was uploaded, so nothing was billed"
    await stt.aclose()


async def test_a_refusal_at_the_far_end_says_nothing_it_was_told():
    """The key is in the request and the remote body is somebody else's
    prose. Neither belongs in an error a caller will log."""
    stt = adapter(recorded(status=401, body="invalid api key sk-not-a-real-key"))
    with pytest.raises(TranscriptionProviderError) as raised:
        await heard(stt, [LOUD, LOUD, QUIET, QUIET, QUIET])

    said = str(raised.value)
    assert "401" in said
    assert "sk-not-a-real-key" not in said and "invalid api key" not in said
    await stt.aclose()


async def test_an_answer_that_is_not_a_transcript_is_a_fault_not_an_empty_turn():
    """Empty text would be indistinguishable from the person saying
    nothing, and the conversation would carry on around the hole."""
    def handler(request):
        return httpx.Response(200, json={"nothing": "useful"})

    stt = adapter(httpx.MockTransport(handler))
    with pytest.raises(TranscriptionProviderError, match="no transcript"):
        await heard(stt, [LOUD, LOUD, QUIET, QUIET, QUIET])
    await stt.aclose()
