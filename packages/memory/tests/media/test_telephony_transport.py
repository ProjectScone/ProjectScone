"""A phone call as an audio transport.

The voice session already knows how to run a conversation over an
``AudioTransport``. A carrier speaks JSON with base64 audio at 8 kHz, so
the difference is a translation, not another session: this makes a call
look like every other transport, and the session does not learn it is on
a phone.
"""

from __future__ import annotations

import asyncio
import base64
import json

import pytest

from scone_memory.audio import pcm
from scone_memory.realtime.audio import AudioChunk
from scone_memory.realtime.keypad import Keypress
from scone_memory.telephony import DIALECTS, CarrierTransport, g711


class FakeSocket:
    """A carrier socket: text in, text out, closed once."""

    def __init__(self, incoming=()):
        self.incoming = asyncio.Queue()
        for message in incoming:
            self.incoming.put_nowait(message)
        self.sent: list[str] = []
        self.closed = 0

    async def receive_text(self) -> str:
        message = await self.incoming.get()
        if message is None:
            raise RuntimeError("carrier socket closed")
        return message

    async def send_text(self, text: str) -> None:
        if self.closed:
            raise RuntimeError("carrier socket closed")
        self.sent.append(text)

    async def close(self) -> None:
        self.closed += 1


def start(stream_id="MZ1"):
    return json.dumps({"event": "start", "streamSid": stream_id, "start": {"streamSid": stream_id, "callSid": "CA1"}})


def dtmf(key):
    return json.dumps({"event": "dtmf", "streamSid": "MZ1", "dtmf": {"track": "inbound_track", "digit": key}})


def media(ms=20):
    payload = base64.b64encode(g711.ulaw_encode(b"\x10\x00" * (8 * ms))).decode()
    return json.dumps({"event": "media", "media": {"payload": payload}})


async def test_a_call_arrives_as_audio_at_the_rate_the_session_runs_at():
    socket = FakeSocket([start(), media(), media(), json.dumps({"event": "stop", "streamSid": "MZ1"})])
    transport = CarrierTransport(socket, DIALECTS["twilio"], rate=16000)

    chunks = [chunk async for chunk in transport.receive()]
    assert chunks and all(c.sample_rate == 16000 and c.channels == 1 for c in chunks)
    assert sum(pcm.duration_ms(c.pcm, 16000) for c in chunks) == pytest.approx(40, rel=0.15)
    assert transport.call_id == "CA1" and transport.stream_id == "MZ1"
    assert transport.ended, "the carrier's stop ends the stream rather than hanging on it"


async def test_what_the_session_says_goes_back_in_the_carrier_envelope():
    socket = FakeSocket([start()])
    transport = CarrierTransport(socket, DIALECTS["twilio"], rate=16000)
    await transport.ready()  # the call has to have started before we can answer it

    await transport.send(AudioChunk(pcm=b"\x20\x00" * 320, sample_rate=16000), "turn-1")
    spoken = json.loads(socket.sent[-1])
    assert spoken["event"] == "media" and spoken["streamSid"] == "MZ1"
    assert base64.b64decode(spoken["media"]["payload"]), "the line gets 8 kHz companded audio"

    await transport.clear("turn-1")
    assert json.loads(socket.sent[-1]) == {"event": "clear", "streamSid": "MZ1"}


async def test_a_digit_is_heard_without_interrupting_the_audio():
    socket = FakeSocket([start(), json.dumps({"event": "dtmf", "dtmf": {"digit": "5"}}), media(),
                         json.dumps({"event": "stop", "streamSid": "MZ1"})])
    transport = CarrierTransport(socket, DIALECTS["twilio"], rate=8000)

    chunks = [chunk async for chunk in transport.receive()]
    assert chunks, "the audio kept flowing"
    assert transport.digits == ["5"], "and the keypress was kept, in order"


async def test_speaking_after_the_call_ended_is_refused_rather_than_written_to_a_dead_socket():
    socket = FakeSocket([start(), json.dumps({"event": "stop", "streamSid": "MZ1"})])
    transport = CarrierTransport(socket, DIALECTS["twilio"], rate=8000)
    [chunk async for chunk in transport.receive()]

    with pytest.raises(RuntimeError, match="ended"):
        await transport.send(AudioChunk(pcm=b"\x01\x00" * 160, sample_rate=8000), "turn-1")
    with pytest.raises(RuntimeError, match="ended"):
        await transport.clear("turn-1")


async def test_closing_happens_once_and_a_message_we_ignore_does_not_end_the_call():
    socket = FakeSocket([start(), json.dumps({"event": "connected"}), media(),
                         json.dumps({"event": "stop", "streamSid": "MZ1"})])
    transport = CarrierTransport(socket, DIALECTS["twilio"], rate=8000)
    chunks = [chunk async for chunk in transport.receive()]
    assert chunks, "the unknown event was passed over, not fatal"

    await transport.aclose()
    await transport.aclose()
    assert socket.closed == 1, "closing twice closes once"


async def test_a_carrier_that_disappears_ends_the_call_rather_than_raising_at_the_session():
    """A caller hanging up is ordinary. The session should see its audio
    stream end, not an exception from the far end's socket."""
    socket = FakeSocket([start(), media(), None])  # None: the socket is gone mid-call
    transport = CarrierTransport(socket, DIALECTS["twilio"], rate=8000)

    chunks = [chunk async for chunk in transport.receive()]
    assert chunks, "what did arrive was delivered"
    assert transport.ended, "and the call is over rather than the error being raised upward"


async def test_off_by_default_a_call_yields_only_audio_and_still_notes_the_digits():
    socket = FakeSocket([start(), dtmf("5"), media(), json.dumps({"event": "stop", "streamSid": "MZ1"})])
    transport = CarrierTransport(socket, DIALECTS["twilio"], rate=8000)
    items = [item async for item in transport.receive()]
    assert items and all(isinstance(item, AudioChunk) for item in items)
    assert transport.digits == ["5"]


async def test_with_the_keypad_on_a_carrier_digit_reaches_the_session_as_a_keypress():
    socket = FakeSocket([start(), media(), dtmf("5"), media(), dtmf("#"),
                         json.dumps({"event": "stop", "streamSid": "MZ1"})])
    transport = CarrierTransport(socket, DIALECTS["twilio"], rate=16000, keypad="events")
    items = [item async for item in transport.receive()]
    presses = [item for item in items if isinstance(item, Keypress)]
    assert [(p.key, p.source) for p in presses] == [("5", "event"), ("#", "event")]
    assert presses[0].offset_ms == pytest.approx(20.0), "after one 20 ms media message"
    assert isinstance(items[0], AudioChunk) and items[1] == presses[0], "in the order the carrier sent them"
    with pytest.raises(ValueError, match="keypad"):
        CarrierTransport(FakeSocket(), DIALECTS["twilio"], keypad="loud")


async def test_the_digits_kept_for_reading_say_when_the_bound_bit():
    from scone_memory.telephony import transport as module

    socket = FakeSocket([start(), *[dtmf("1")] * (module.MAX_DIGITS + 3),
                         json.dumps({"event": "stop", "streamSid": "MZ1"})])
    transport = CarrierTransport(socket, DIALECTS["twilio"], rate=8000)
    [item async for item in transport.receive()]
    assert len(transport.digits) == module.MAX_DIGITS
    assert transport.dropped_digits == 3, "a caller leaning on a key is counted past the bound, not forgotten"


async def test_a_key_heard_in_the_audio_reaches_the_session_as_an_inband_keypress():
    import math

    tone = pcm.to_bytes(6000 * math.sin(2 * math.pi * 852 * n / 8000) + 6000 * math.sin(2 * math.pi * 1336 * n / 8000)
                        for n in range(800))  # "8", 100 ms
    quiet = b"\x00\x00" * 480

    def chunked(data):
        return [json.dumps({"event": "media", "media": {"payload": base64.b64encode(g711.ulaw_encode(data[at:at + 320])).decode()}})
                for at in range(0, len(data), 320)]

    socket = FakeSocket([start(), *chunked(quiet + tone + quiet), dtmf("8"),
                         json.dumps({"event": "stop", "streamSid": "MZ1"})])
    transport = CarrierTransport(socket, DIALECTS["twilio"], rate=16000, keypad="inband")
    presses = [item async for item in transport.receive() if isinstance(item, Keypress)]
    assert [(p.key, p.source) for p in presses] == [("8", "inband")], "the carrier's own report is not taken in-band only"
    assert presses[0].tone_ms is not None and presses[0].offset_ms == pytest.approx(60, abs=13)
