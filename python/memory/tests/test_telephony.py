"""Phone calls: G.711 codecs and the media-stream dialects carriers use.

Every carrier sends the same thing in a slightly different envelope, so
the differences are data and the handling is written once. Audio arrives
as 8 kHz companded bytes and leaves the same way, whatever rate the rest
of the pipeline runs at.
"""

from __future__ import annotations

import base64
import json
import math

import pytest

from scone_memory.audio import pcm
from scone_memory.realtime.audio import AudioChunk
from scone_memory.telephony import (CallEnded, CallStarted, DIALECTS, Dtmf, MediaStream, g711)


def tone(hz: float, ms: int, rate: int, amplitude: float = 0.5) -> bytes:
    return pcm.to_bytes(int(amplitude * 32767 * math.sin(2 * math.pi * hz * n / rate))
                        for n in range(rate * ms // 1000))


def test_g711_codes_survive_a_round_trip_apart_from_the_law_that_has_two_zeros():
    """μ-law spells zero twice, positive and negative, so one of the two
    has to lose when a sample comes back. Everything else is exact, and
    A-law, which has no duplicate level, is exact throughout."""
    codes = bytes(range(256))
    back = g711.ulaw_encode(g711.ulaw_decode(codes))
    assert [i for i in range(256) if back[i] != codes[i]] == [255], "only the second zero moves"
    assert g711.ulaw_decode(bytes([0x7F])) == g711.ulaw_decode(bytes([0xFF])) == pcm.to_bytes([0])
    assert g711.alaw_encode(g711.alaw_decode(codes)) == codes, "A-law has no duplicate level to lose"
    assert g711.ulaw_decode(codes) != g711.alaw_decode(codes), "the two laws are not the same curve"


def test_speech_through_a_codec_stays_speech():
    speech = tone(440, 40, 8000)
    back = g711.ulaw_decode(g711.ulaw_encode(speech))
    assert len(back) == len(speech)
    assert pcm.rms(back) == pytest.approx(pcm.rms(speech), rel=0.05), "eight bits of companding, not a new sound"
    assert g711.ulaw_encode(pcm.to_bytes([0])) == g711.ulaw_encode(pcm.to_bytes([0])), "silence is stable"


def test_a_carrier_start_message_opens_a_call_and_names_the_stream():
    stream = MediaStream(DIALECTS["twilio"])
    [event] = stream.inbound(json.dumps({
        "event": "start", "streamSid": "MZ123",
        "start": {"streamSid": "MZ123", "callSid": "CA9", "mediaFormat": {"encoding": "audio/x-mulaw", "sampleRate": 8000}},
    }))
    assert isinstance(event, CallStarted) and event.stream_id == "MZ123" and event.call_id == "CA9"
    assert (event.rate, event.codec) == (8000, "ulaw")
    assert stream.stream_id == "MZ123", "the stream is remembered, so replies can name it"


def test_media_arrives_as_audio_and_digits_arrive_as_digits():
    stream = MediaStream(DIALECTS["twilio"])
    stream.inbound(json.dumps({"event": "start", "streamSid": "MZ1", "start": {"streamSid": "MZ1"}}))
    payload = base64.b64encode(g711.ulaw_encode(tone(440, 20, 8000))).decode()

    [audio] = stream.inbound(json.dumps({"event": "media", "media": {"payload": payload, "track": "inbound"}}))
    assert isinstance(audio, AudioChunk) and audio.sample_rate == 8000 and audio.channels == 1
    assert pcm.duration_ms(audio.pcm, 8000) == pytest.approx(20, rel=0.05)

    [digit] = stream.inbound(json.dumps({"event": "dtmf", "dtmf": {"digit": "7"}}))
    assert isinstance(digit, Dtmf) and digit.digit == "7"

    [ended] = stream.inbound(json.dumps({"event": "stop", "streamSid": "MZ1"}))
    assert isinstance(ended, CallEnded) and ended.stream_id == "MZ1"

    assert stream.inbound(json.dumps({"event": "connected"})) == [], "a message we do not act on is not an error"
    assert stream.inbound(json.dumps({"event": "mark", "mark": {"name": "x"}})) == []


def test_what_we_say_goes_back_in_the_carrier_envelope_at_its_rate():
    stream = MediaStream(DIALECTS["twilio"], rate=16000)
    stream.inbound(json.dumps({"event": "start", "streamSid": "MZ2", "start": {"streamSid": "MZ2"}}))

    message = json.loads(stream.outbound(AudioChunk(pcm=tone(440, 20, 16000), sample_rate=16000)))
    assert message["event"] == "media" and message["streamSid"] == "MZ2"
    spoken = g711.ulaw_decode(base64.b64decode(message["media"]["payload"]))
    assert pcm.duration_ms(spoken, 8000) == pytest.approx(20, rel=0.1), "our 16 kHz became the line's 8 kHz"

    cleared = json.loads(stream.clear())
    assert cleared == {"event": "clear", "streamSid": "MZ2"}, "barge-in drops what the carrier still holds"


def test_inbound_audio_is_offered_at_the_rate_the_pipeline_runs_at():
    stream = MediaStream(DIALECTS["twilio"], rate=16000)
    stream.inbound(json.dumps({"event": "start", "streamSid": "MZ3", "start": {"streamSid": "MZ3"}}))
    payload = base64.b64encode(g711.ulaw_encode(tone(440, 20, 8000))).decode()
    [audio] = stream.inbound(json.dumps({"event": "media", "media": {"payload": payload}}))
    assert audio.sample_rate == 16000
    assert pcm.duration_ms(audio.pcm, 16000) == pytest.approx(20, rel=0.1)


def test_each_carrier_reads_its_own_envelope():
    assert set(DIALECTS) >= {"twilio", "telnyx", "plivo", "exotel"}
    telnyx = MediaStream(DIALECTS["telnyx"])
    [event] = telnyx.inbound(json.dumps({"event": "start", "stream_id": "st-1", "start": {"call_control_id": "cc-2"}}))
    assert (event.stream_id, event.call_id) == ("st-1", "cc-2")
    assert json.loads(telnyx.outbound(AudioChunk(pcm=tone(440, 20, 8000), sample_rate=8000)))["stream_id"] == "st-1"

    plivo = MediaStream(DIALECTS["plivo"])
    [event] = plivo.inbound(json.dumps({"event": "start", "start": {"streamId": "pl-1", "callId": "c-1"}}))
    assert event.stream_id == "pl-1"
    assert DIALECTS["exotel"].codec == "pcm", "not every carrier compands"


def test_a_reply_before_the_call_starts_is_refused_rather_than_addressed_to_nobody():
    stream = MediaStream(DIALECTS["twilio"])
    with pytest.raises(ValueError, match="no stream"):
        stream.outbound(AudioChunk(pcm=tone(440, 20, 8000), sample_rate=8000))
    assert stream.clear() is None, "and there is nothing to clear"
