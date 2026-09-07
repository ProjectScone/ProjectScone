"""Audio a real call needs: exact 16-bit frames, rate conversion that
does not fold high tones down into the voice, and a gate that says when
someone is speaking. Stdlib only, so installing the memory package does
not pull a numeric stack in."""

from __future__ import annotations

import math

import pytest

from scone_memory.audio import Framer, Resampler, VoiceGate, pcm
from scone_memory.realtime.audio import AudioChunk


def tone(hz: float, ms: int, rate: int, amplitude: float = 0.5) -> bytes:
    samples = [int(amplitude * 32767 * math.sin(2 * math.pi * hz * n / rate)) for n in range(rate * ms // 1000)]
    return pcm.to_bytes(samples)


def crossings(data: bytes) -> int:
    samples = pcm.to_samples(data)
    return sum(1 for a, b in zip(samples, samples[1:]) if (a < 0) != (b < 0))


def test_samples_and_bytes_round_trip_and_refuse_a_broken_frame():
    assert pcm.to_samples(pcm.to_bytes([0, -1, 32767, -32768])) == [0, -1, 32767, -32768]
    with pytest.raises(ValueError):
        pcm.to_samples(b"\x01\x02\x03")
    assert pcm.to_bytes([40000, -40000]) == pcm.to_bytes([32767, -32768]), "values outside the range clip"


def test_stereo_folds_to_mono_and_gain_never_wraps():
    stereo = pcm.to_bytes([100, 200, -100, -200])
    assert pcm.to_samples(pcm.to_mono(stereo, channels=2)) == [150, -150]
    assert pcm.to_samples(pcm.gain(pcm.to_bytes([20000]), 4.0)) == [32767], "loud stays loud, never wraps to quiet"
    assert pcm.rms(pcm.to_bytes([0, 0, 0])) == 0.0
    assert pcm.rms(tone(440, 20, 16000)) == pytest.approx(0.5 / math.sqrt(2), rel=0.02), "a sine sits at its amplitude over root two"
    assert pcm.duration_ms(tone(440, 20, 16000), rate=16000) == pytest.approx(20)


def test_the_framer_cuts_a_ragged_stream_into_exact_frames():
    framer = Framer(rate=16000, ms=20)  # 320 samples, 640 bytes
    assert framer.push(tone(440, 10, 16000)) == [], "half a frame waits for the rest"
    frames = framer.push(tone(440, 35, 16000))
    assert [len(f) for f in frames] == [640, 640], "two whole frames, the remainder held back"
    assert len(framer.flush()) == 640, "the tail is padded once, at the end"
    assert framer.flush() == b"", "and nothing is left after that"


def test_downsampling_keeps_the_voice_and_folds_nothing_down_into_it():
    speech = Resampler(48000, 16000)
    out = speech.feed(tone(1000, 100, 48000))
    assert len(out) == pytest.approx(16000 * 2 * 0.1, rel=0.02), "a third of the samples"
    assert crossings(out) == pytest.approx(200, abs=4), "still a 1 kHz tone"
    assert pcm.rms(out) == pytest.approx(pcm.rms(tone(1000, 100, 48000)), rel=0.15), "and still as loud"

    # 15 kHz cannot exist at 16 kHz; without a filter it would come back as
    # a loud 1 kHz whistle sitting in the middle of the voice.
    folded = Resampler(48000, 16000).feed(tone(15000, 100, 48000))
    assert pcm.rms(folded) < 0.1 * pcm.rms(tone(15000, 100, 48000)), "the tone is rejected, not folded down"


def test_a_resampler_carries_its_state_across_chunk_boundaries():
    whole = tone(1000, 60, 48000)
    once = Resampler(48000, 16000).feed(whole)
    piecewise = Resampler(48000, 16000)
    split = piecewise.feed(whole[:1000]) + piecewise.feed(whole[1000:])
    assert split == once, "a chunk boundary is not a click"


def test_upsampling_gives_a_provider_the_rate_it_asks_for():
    out = Resampler(16000, 48000).feed(tone(500, 100, 16000))
    assert len(out) == pytest.approx(48000 * 2 * 0.1, rel=0.02)
    # The filter starts from silence, so the first few milliseconds ramp up
    # and cross zero on their way; the tone itself is what follows.
    settled = out[48000 * 2 * 5 // 1000:]
    assert crossings(settled) == pytest.approx(95, abs=4), "the tone is unchanged, only the rate is"
    assert Resampler(16000, 16000).feed(b"\x01\x02" * 10) == b"\x01\x02" * 10, "the same rate is a passthrough"


async def test_the_gate_says_when_someone_is_speaking_and_ignores_a_blip():
    gate = VoiceGate(rate=16000, threshold=0.05, start_ms=60, stop_ms=200)
    quiet = AudioChunk(pcm=pcm.to_bytes([0] * 320), sample_rate=16000)
    loud = AudioChunk(pcm=tone(300, 20, 16000), sample_rate=16000)

    assert [await gate.detect(quiet) for _ in range(5)] == [False] * 5
    assert await gate.detect(loud) is False, "one loud frame is a noise, not a turn"
    assert await gate.detect(loud) is False
    assert await gate.detect(loud) is True, "60 ms of it is someone speaking"
    assert await gate.detect(quiet) is True, "a pause inside a sentence is still the sentence"
    assert [await gate.detect(quiet) for _ in range(9)][-1] is False, "and 200 ms of it is the end"

    blip = VoiceGate(rate=16000, threshold=0.05, start_ms=60, stop_ms=200)
    assert await blip.detect(loud) is False
    assert await blip.detect(loud) is False
    assert await blip.detect(quiet) is False, "the count starts again after a quiet frame"
    assert await blip.detect(loud) is False, "so two bursts either side of a gap are not a turn"
    await gate.aclose()


def test_the_gate_refuses_a_rate_it_was_not_built_for():
    gate = VoiceGate(rate=16000)
    with pytest.raises(ValueError, match="16000"):
        import asyncio

        asyncio.run(gate.detect(AudioChunk(pcm=b"\x00\x00" * 160, sample_rate=8000)))
