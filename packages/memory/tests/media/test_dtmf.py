"""Keys heard in the audio: two tones, one from each group, found by Goertzel.

Every signal here is synthesized, so the frequencies, levels and noise
are exactly what the test says they are. The thresholds are ITU-T Q.24's:
accept a tone within 1.5 % of nominal, reject one 3.5 % off, allow the
high group to be up to 4 dB louder than the low (reverse twist) and the
low up to 8 dB louder than the high (normal twist)."""

from __future__ import annotations

import base64
import json
import math
import random

import pytest

from scone_memory.audio import dtmf, pcm
from scone_memory.audio.dtmf import HIGH, LOW, ToneDetector, classify
from scone_memory.realtime.keypad import KEYS
from scone_memory.telephony import DIALECTS, Dtmf, MediaStream, g711
from scone_memory.telephony.stream import DUPLICATE_WINDOW_MS

RATE = 8000


def where(key):
    row = next(i for i, keys in enumerate(dtmf.LAYOUT) if key in keys)
    return LOW[row], HIGH[dtmf.LAYOUT[row].index(key)]


def samples(key, ms, *, rate=RATE, low_dbfs=-10.0, high_dbfs=-10.0, detune=(0.0, 0.0), phase=0.4):
    low, high = where(key)
    a, b = 32768 * 10 ** (low_dbfs / 20), 32768 * 10 ** (high_dbfs / 20)
    fl, fh = low * (1 + detune[0]), high * (1 + detune[1])
    return [a * math.sin(2 * math.pi * fl * n / rate) + b * math.sin(2 * math.pi * fh * n / rate + phase)
            for n in range(rate * ms // 1000)]


def silence(ms, rate=RATE):
    return [0.0] * (rate * ms // 1000)


def noisy(signal, snr_db, rng):
    power = sum(v * v for v in signal) / max(1, sum(1 for v in signal if v))
    sigma = math.sqrt(power / 10 ** (snr_db / 20 * 2))
    return [v + rng.gauss(0, sigma) for v in signal]


def detect(signal, *, chunk_ms=20, rate=RATE, **options):
    detector = ToneDetector(rate, **options)
    data, step, heard = pcm.to_bytes(signal), rate * chunk_ms // 1000 * 2, []
    for at in range(0, len(data), step):
        heard += detector.feed(data[at:at + step])
    return heard


def block(signal):
    return signal[:round(RATE * dtmf.WINDOW_MS / 1000)]


def test_the_layout_is_the_keypad():
    assert sorted("".join(dtmf.LAYOUT)) == sorted(KEYS)
    assert (LOW, HIGH) == ((697, 770, 852, 941), (1209, 1336, 1477, 1633))


def test_every_key_is_heard_once_and_placed_in_the_audio():
    for key in KEYS:
        [tone] = detect(silence(30) + samples(key, 100) + silence(60))
        assert tone.key == key
        assert tone.offset_ms == pytest.approx(30, abs=dtmf.WINDOW_MS / 2), "where the tones began"
        assert 40 <= tone.tone_ms <= 100, "how long they had lasted when accepted"


def test_a_key_is_heard_at_a_pipeline_rate_as_well_as_the_line_rate():
    [tone] = detect(samples("9", 100, rate=16000), rate=16000)
    assert tone.key == "9"
    assert ToneDetector(16000).window == 2 * ToneDetector(8000).window == 410, "a window is a duration"
    for rate in (3000, 8000.0):
        with pytest.raises(ValueError, match="rate"):
            ToneDetector(rate)


def test_one_block_says_why_it_is_not_a_key():
    assert classify(block(samples("5", 40)), RATE).key == "5"
    assert classify(block(silence(40)), RATE).reason == "level"
    assert classify(block(samples("5", 40, low_dbfs=-50, high_dbfs=-50)), RATE).reason == "level"
    rng = random.Random(3)
    assert classify([rng.gauss(0, 3000) for _ in range(205)], RATE).reason == "share", "noise has no tone in it"
    a440 = [10000 * math.sin(2 * math.pi * 440 * n / RATE) for n in range(205)]
    assert classify(a440, RATE).reason == "share"
    assert classify(block(samples("5", 40, high_dbfs=-60)), RATE).reason == "level", "one tone is not a key"
    both = [x + y for x, y in zip(block(samples("1", 40)), block(samples("5", 40, low_dbfs=-16, high_dbfs=-16)))]
    assert classify(both, RATE).reason == "group", "two keys at once are neither, even when one is louder"
    buried = noisy(block(samples("5", 40)), -3, rng)
    assert classify(buried, RATE).reason == "share", "a key under louder noise is not measured as one"


def test_twist_is_allowed_up_to_the_standard_in_each_direction():
    def verdict(low_dbfs, high_dbfs):
        return classify(block(samples("8", 40, low_dbfs=low_dbfs, high_dbfs=high_dbfs)), RATE)

    assert verdict(-10, -13).key == "8", "the high group 3 dB quieter: normal twist within 8 dB"
    assert verdict(-10, -17).key == "8", "7 dB of normal twist"
    assert verdict(-10, -19).reason == "twist", "9 dB of normal twist"
    assert verdict(-13, -10).key == "8", "3 dB of reverse twist"
    assert verdict(-15, -10).reason == "twist", "5 dB of reverse twist"


@pytest.mark.parametrize("group", ["low", "high"])
def test_a_tone_within_one_and_a_half_percent_is_a_key_and_one_three_and_a_half_off_is_not(group):
    for key in KEYS:
        for offset in (0.015, -0.015):
            detune = (offset, 0.0) if group == "low" else (0.0, offset)
            assert classify(block(samples(key, 40, detune=detune)), RATE).key == key, (key, offset)
            # and with the detuned tone as loud as twist allows the far side to be
            louder = dict(low_dbfs=-7.0, high_dbfs=-13.0) if group == "low" else dict(low_dbfs=-12.0, high_dbfs=-9.0)
            assert classify(block(samples(key, 40, detune=detune, **louder)), RATE).key == key, (key, offset, louder)
        for offset in (0.035, -0.035, 0.05, -0.05):
            detune = (offset, 0.0) if group == "low" else (0.0, offset)
            assert classify(block(samples(key, 40, detune=detune)), RATE).key is None, (key, offset)


def test_the_frequency_rule_rejects_a_near_miss_without_the_energy_rules():
    """A short block also measures an off-nominal tone as weak, so the
    share rule often rejects a near miss first. The probes do not depend on
    it: with the share rule off, a low tone 3 or 3.5 % high is still refused,
    and refused for its frequency."""
    for key in KEYS:
        for offset in (0.03, 0.035):
            verdict = classify(block(samples(key, 40, detune=(offset, 0.0))), RATE, min_share=0.0)
            assert (verdict.key, verdict.reason) == (None, "off_nominal"), (key, offset)


def test_a_near_miss_is_not_heard_as_a_digit_over_a_whole_press():
    assert detect(samples("5", 120, detune=(0.035, 0.0))) == []
    assert detect(samples("5", 120, detune=(0.0, -0.04))) == []


def test_a_tone_too_short_to_be_a_press_is_not_a_digit():
    assert detect(silence(20) + samples("3", 20) + silence(60)) == []
    for lead in range(0, 26, 3):  # wherever the press falls against the blocks
        for key in "159D":
            assert [t.key for t in detect(silence(lead) + samples(key, 40) + silence(60))] == [key], (lead, key)
    assert ToneDetector(RATE, min_tone_ms=100).blocks_needed > ToneDetector(RATE).blocks_needed


def test_a_held_key_is_one_digit_and_a_pressed_again_key_is_two():
    assert [t.key for t in detect(samples("7", 600))] == ["7"]
    again = samples("7", 80) + silence(60) + samples("7", 80) + silence(60)
    assert [t.key for t in detect(again)] == ["7", "7"]
    dropout = samples("7", 150) + silence(5) + samples("7", 150, phase=2.0)
    assert [t.key for t in detect(dropout)] == ["7"], "a 5 ms dropout inside a press does not split it"


def test_a_dialled_sequence_survives_noise():
    rng = random.Random(11)
    keys = "4155550123*#9D"
    signal = []
    for key in keys:
        signal += samples(key, 70, phase=rng.random() * 6) + silence(60)
    assert "".join(t.key for t in detect(noisy(signal, 10, rng))) == keys
    assert detect([rng.gauss(0, 4000) for _ in range(RATE * 2)]) == [], "two seconds of noise is no key"


def test_speech_shaped_sound_is_not_a_key():
    """Voiced speech is a comb of harmonics. Whatever its pitch, the energy
    is spread over many of them, so no pair of DTMF bins holds most of it."""
    for f0 in (98.0, 120.0, 175.0, 220.0):
        voice = [sum(3000 / k * math.sin(2 * math.pi * f0 * k * n / RATE) for k in range(1, 30) if f0 * k < 3800)
                 for n in range(RATE // 2)]
        assert detect(voice) == [], f0


def media(signal):
    payload = base64.b64encode(g711.ulaw_encode(pcm.to_bytes(signal))).decode()
    return json.dumps({"event": "media", "media": {"payload": payload}})


def feed(stream, signal, chunk_ms=20):
    frames, step = [], RATE * chunk_ms // 1000
    for at in range(0, len(signal), step):
        frames += stream.inbound(media(signal[at:at + step]))
    return [f for f in frames if isinstance(f, Dtmf)]


def test_a_carrier_stream_hears_keys_in_its_companded_audio():
    stream = MediaStream(DIALECTS["twilio"], rate=16000, digits="inband")
    [digit] = feed(stream, silence(40) + samples("6", 100) + silence(60))
    assert (digit.digit, digit.source) == ("6", "inband")
    assert digit.offset_ms == pytest.approx(40, abs=dtmf.WINDOW_MS / 2)
    assert digit.tone_ms is not None
    assert stream.inbound(json.dumps({"event": "dtmf", "dtmf": {"digit": "1"}})) == [], \
        "in-band only: the carrier's own digit is not reported"


def test_heard_twice_a_key_is_reported_once_and_the_second_hearing_counted():
    """A carrier that reports a key and leaves its tones in the audio would
    otherwise give every press twice."""
    stream = MediaStream(DIALECTS["twilio"], digits="both")
    event = json.dumps({"event": "dtmf", "dtmf": {"digit": "2"}})

    first = feed(stream, samples("2", 100) + silence(60))
    first += stream.inbound(event)  # the carrier's report arrives after the tones
    assert [(d.digit, d.source) for d in first] == [("2", "inband")]

    second = stream.inbound(event) + feed(stream, samples("2", 100) + silence(60))  # this time before them
    assert [(d.digit, d.source) for d in second] == [("2", "event")]
    assert stream.duplicates == 2

    third = feed(stream, samples("3", 100) + silence(60)) + stream.inbound(json.dumps({"event": "dtmf", "dtmf": {"digit": "4"}}))
    assert [d.digit for d in third] == ["3", "4"], "a different key is not a duplicate"
    assert stream.duplicates == 2


def test_pressed_twice_quickly_a_key_heard_both_ways_is_two_digits():
    stream = MediaStream(DIALECTS["twilio"], digits="both")
    event = json.dumps({"event": "dtmf", "dtmf": {"digit": "1"}})
    heard = []
    for _ in range(2):
        heard += feed(stream, samples("1", 80) + silence(60))
        heard += stream.inbound(event)
    assert [d.digit for d in heard] == ["1", "1"]
    assert stream.duplicates == 2
    twice = stream.inbound(event) + stream.inbound(event)
    assert [d.digit for d in twice] == ["1", "1"], "a key reported twice the same way is two presses"


def test_a_key_heard_by_one_way_only_long_after_is_not_a_duplicate():
    stream = MediaStream(DIALECTS["twilio"], digits="both")
    heard = feed(stream, samples("9", 100) + silence(60))
    heard += feed(stream, silence(int(DUPLICATE_WINDOW_MS) + 200))
    heard += stream.inbound(json.dumps({"event": "dtmf", "dtmf": {"digit": "9"}}))
    assert [d.source for d in heard] == ["inband", "event"]
    assert stream.duplicates == 0


@pytest.mark.parametrize("held_ms", [300, 1200, 2000], ids=["short", "past-window", "twice-window"])
def test_a_key_held_down_and_reported_when_released_is_one_press(held_ms):
    """A carrier may report a key when it is let go. The tones began when it
    was pressed, so a key held longer than the window was two keys when the
    window ran from where the tones began; it runs from where they were last heard."""
    stream = MediaStream(DIALECTS["twilio"], digits="both")
    heard = feed(stream, silence(40) + samples("5", held_ms))
    heard += stream.inbound(json.dumps({"event": "dtmf", "dtmf": {"digit": "5"}}))
    heard += feed(stream, silence(100))
    assert [(d.digit, d.source) for d in heard] == [("5", "inband")]
    assert heard[0].offset_ms == pytest.approx(40, abs=dtmf.WINDOW_MS / 2), "still placed where it began"
    assert stream.duplicates == 1


def test_the_window_runs_from_the_end_of_the_tones_and_not_from_the_report():
    stream = MediaStream(DIALECTS["twilio"], digits="both")
    heard = feed(stream, samples("8", 1200) + silence(int(DUPLICATE_WINDOW_MS) + 200))
    heard += stream.inbound(json.dumps({"event": "dtmf", "dtmf": {"digit": "8"}}))
    assert [d.source for d in heard] == ["inband", "event"], "let go more than the window before the report"
    assert stream.duplicates == 0


def test_a_digit_behind_an_older_one_still_leaves_the_window():
    """Digits wait in the order they were reported. One reported later can
    leave the window first, while the oldest's tones are still sounding."""
    stream = MediaStream(DIALECTS["twilio"], digits="both")
    heard = feed(stream, samples("5", 200))  # the oldest: tones still sounding below
    [seven] = stream.inbound(json.dumps({"event": "dtmf", "dtmf": {"digit": "7"}}))
    heard += feed(stream, samples("5", int(DUPLICATE_WINDOW_MS) + 200))
    heard += feed(stream, samples("7", 100))
    assert [(d.digit, d.source) for d in heard] == [("5", "inband"), ("7", "inband")], \
        "the reported 7 was more than the window before these tones"
    assert seven.source == "event" and stream.duplicates == 0


def test_a_key_held_down_moves_on_its_own_press_and_no_earlier_one():
    """Two presses of one key a moment apart, the first heard only in the
    audio. The carrier reports the second when it is let go, and then
    reports a third press whose tones were not in the audio."""
    stream = MediaStream(DIALECTS["twilio"], digits="both")
    event = json.dumps({"event": "dtmf", "dtmf": {"digit": "1"}})
    heard = feed(stream, samples("1", 100) + silence(300) + samples("1", 900))
    heard += stream.inbound(event)
    heard += feed(stream, silence(60))
    heard += stream.inbound(event)
    assert [d.source for d in heard] == ["inband", "inband", "event"]
    assert stream.duplicates == 1


def test_digits_waiting_to_be_paired_are_bounded_and_the_bound_says_when_it_bit(monkeypatch):
    from scone_memory.telephony import stream as module

    monkeypatch.setattr(module, "MAX_UNPAIRED", 3)
    stream = MediaStream(DIALECTS["twilio"], digits="both")
    for key in "12345":  # a burst of carrier digits with no audio between them
        assert len(stream.inbound(json.dumps({"event": "dtmf", "dtmf": {"digit": key}}))) == 1
    assert stream.unpaired_forgotten == 2
    heard = feed(stream, samples("1", 100) + silence(60))
    assert [d.digit for d in heard] == ["1"], "the forgotten event cannot pair, so the tones are reported"
    heard = feed(stream, samples("5", 100) + silence(60))
    assert heard == [] and stream.duplicates == 1, "a remembered one still pairs"

    spaced = MediaStream(DIALECTS["twilio"], digits="both")
    for key in "12345":  # the same burst, a window apart: nothing is waiting long enough to count
        spaced.inbound(json.dumps({"event": "dtmf", "dtmf": {"digit": key}}))
        feed(spaced, silence(int(DUPLICATE_WINDOW_MS) + 100))
    assert spaced.unpaired_forgotten == 0

    single = MediaStream(DIALECTS["twilio"], digits="events")
    for key in "12345":
        single.inbound(json.dumps({"event": "dtmf", "dtmf": {"digit": key}}))
    assert single.unpaired_forgotten == 0, "with one way of hearing there is nothing to pair"
