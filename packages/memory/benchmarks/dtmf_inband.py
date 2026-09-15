"""How well keys are heard in the audio, on synthesized calls.

Run from packages/memory with the tree under test on PYTHONPATH:

    PYTHONPATH=src python benchmarks/dtmf_inband.py --json out.json

Every signal is synthesized from a fixed seed, so the truth is exact and a
run repeats. A dialled key is two sine tones at -10 dBFS each (with up to
2 dB of twist either way and random phases), 70 ms long, 70 ms apart
unless a section says otherwise. White Gaussian noise is added at the
stated signal-to-noise ratio, where the signal is the key's own power
while it sounds and the noise covers the whole 0-4 kHz band. The sum is
companded to mu-law and back, as a carrier's line does, and fed to
``MediaStream(digits="inband")`` as 20 ms media messages, the path a call
takes.

A detection is scored against the key whose tones it falls inside (from
13 ms before a key's start to its end): the right key is a hit, a wrong
key a substitution, none an insertion; a key nobody detected is a miss.

Nothing here is a recording of a real phone or a real voice. The
speech-shaped signals are harmonic combs with vowel formants, steady for
their whole length, which is harder than speech (no pitch or formant
movement) but is still not speech.
"""
from __future__ import annotations

import argparse
import base64
import json
import math
import random
import statistics
import sys
import time
from pathlib import Path

from scone_memory.audio import dtmf, pcm
from scone_memory.telephony import DIALECTS, Dtmf, MediaStream, g711

RATE = 8000
KEYS = "".join(dtmf.LAYOUT)
SNRS = (None, 20, 10, 6, 3, 0)
SEQUENCES = 50
KEYS_PER_SEQUENCE = 10


def pair(key: str) -> tuple[int, int]:
    row = next(i for i, keys in enumerate(dtmf.LAYOUT) if key in keys)
    return dtmf.LOW[row], dtmf.HIGH[dtmf.LAYOUT[row].index(key)]


def key_tone(key, ms, rng, *, detune=(0.0, 0.0), twist_db=None):
    low, high = pair(key)
    twist = rng.uniform(-2, 2) if twist_db is None else twist_db
    a = 32768 * 10 ** (-10 / 20)
    b = a * 10 ** (twist / 20)
    pl, ph = rng.random() * 2 * math.pi, rng.random() * 2 * math.pi
    fl, fh = low * (1 + detune[0]), high * (1 + detune[1])
    return [a * math.sin(2 * math.pi * fl * n / RATE + pl) + b * math.sin(2 * math.pi * fh * n / RATE + ph)
            for n in range(RATE * ms // 1000)]


def add_noise(signal, snr_db, rng, power):
    if snr_db is None:
        return signal
    sigma = math.sqrt(power / 10 ** (snr_db / 10))
    return [v + rng.gauss(0, sigma) for v in signal]


def call(keys, rng, *, tone_ms=70, gap_ms=70, lead_ms=None, snr_db=None, detune=(0.0, 0.0)):
    """A dialled sequence and where each key sounds, in samples."""
    signal, truth = [0.0] * (RATE * (lead_ms if lead_ms is not None else rng.randrange(20, 60)) // 1000), []
    for key in keys:
        tone = key_tone(key, tone_ms, rng, detune=detune)
        truth.append((key, len(signal), len(signal) + len(tone)))
        signal += tone + [0.0] * (RATE * gap_ms // 1000)
    powers = [sum(v * v for v in signal[s:e]) / (e - s) for _, s, e in truth]
    return add_noise(signal, snr_db, rng, statistics.fmean(powers) if powers else 0.0), truth


def heard(signal, digits="inband"):
    stream = MediaStream(DIALECTS["twilio"], digits=digits)
    stream.inbound(json.dumps({"event": "start", "streamSid": "MZ", "start": {"streamSid": "MZ"}}))
    data = g711.ulaw_encode(pcm.to_bytes(signal))
    found = []
    for at in range(0, len(data), 160):
        message = json.dumps({"event": "media", "media": {"payload": base64.b64encode(data[at:at + 160]).decode()}})
        found += [frame for frame in stream.inbound(message) if isinstance(frame, Dtmf)]
    return found


def score(found, truth):
    slack = RATE * 13 // 1000
    hits = subs = inserts = 0
    taken = set()
    for digit in found:
        at = digit.offset_ms * RATE / 1000
        index = next((i for i, (_, s, e) in enumerate(truth) if s - slack <= at < e), None)
        if index is None or index in taken:
            inserts += 1
            continue
        taken.add(index)
        if truth[index][0] == digit.digit:
            hits += 1
        else:
            subs += 1
    return {"keys": len(truth), "hits": hits, "substitutions": subs, "insertions": inserts,
            "misses": len(truth) - len(taken)}


def accuracy(rng):
    rows = []
    for snr in SNRS:
        total = {"keys": 0, "hits": 0, "substitutions": 0, "insertions": 0, "misses": 0}
        exact = 0
        for _ in range(SEQUENCES):
            keys = "".join(rng.choice(KEYS) for _ in range(KEYS_PER_SEQUENCE))
            signal, truth = call(keys, rng, snr_db=snr)
            found = heard(signal)
            exact += "".join(d.digit for d in found) == keys
            for name, value in score(found, truth).items():
                total[name] += value
        rows.append({"snr_db": snr, "sequences": SEQUENCES, "exact_sequences": exact, **total})
    return rows


def durations(rng):
    rows = []
    for snr in (None, 10):
        for ms in (20, 25, 30, 40, 50, 60, 70):
            total = {"keys": 0, "hits": 0, "substitutions": 0, "insertions": 0, "misses": 0}
            for _ in range(20):
                keys = "".join(rng.choice(KEYS) for _ in range(KEYS_PER_SEQUENCE))
                signal, truth = call(keys, rng, tone_ms=ms, gap_ms=60, snr_db=snr)
                for name, value in score(heard(signal), truth).items():
                    total[name] += value
            rows.append({"snr_db": snr, "tone_ms": ms, **total})
    return rows


def gaps(rng):
    """The same key pressed again after a short pause, the case a held-key rule can merge."""
    rows = []
    for gap in (30, 40, 50, 60):
        total = {"keys": 0, "hits": 0, "substitutions": 0, "insertions": 0, "misses": 0}
        for _ in range(20):
            key = rng.choice(KEYS)
            signal, truth = call(key * KEYS_PER_SEQUENCE, rng, tone_ms=70, gap_ms=gap)
            for name, value in score(heard(signal), truth).items():
                total[name] += value
        rows.append({"gap_ms": gap, **total})
    return rows


def near_misses(rng):
    rows = []
    for offset in (0.015, -0.015, 0.025, -0.025, 0.035, -0.035, 0.05, -0.05):
        for group in ("low", "high"):
            detune = (offset, 0.0) if group == "low" else (0.0, offset)
            accepted = presses = 0
            for key in KEYS:
                for _ in range(3):
                    signal, truth = call(key, rng, tone_ms=100, gap_ms=60, detune=detune)
                    found = heard(signal)
                    presses += 1
                    accepted += any(d.digit == key for d in found)
            rows.append({"offset": offset, "group": group, "presses": presses, "accepted": accepted})
    return rows


def vowel(f0, formants, seconds, rng):
    """A steady voiced vowel: harmonics of f0 shaped by resonances at the formants."""
    def gain(f):
        return sum(1 / (1 + ((f - fc) / bw) ** 2) for fc, bw in formants)

    harmonics = [(k * f0, gain(k * f0), rng.random() * 2 * math.pi) for k in range(1, int(3800 / f0))]
    peak = sum(g for _, g, _ in harmonics)
    scale = 16000 / peak
    return [scale * sum(g * math.sin(2 * math.pi * f * n / RATE + p) for f, g, p in harmonics)
            for n in range(int(RATE * seconds))]


def false_keys(rng):
    rows = []
    noise = [rng.gauss(0, 4000) for _ in range(RATE * 60)]
    rows.append({"signal": "white noise, -18 dBFS", "seconds": 60, "keys": len(heard(noise))})
    a440 = [10000 * math.sin(2 * math.pi * 440 * n / RATE) for n in range(RATE * 10)]
    rows.append({"signal": "440 Hz tone", "seconds": 10, "keys": len(heard(a440))})
    formants = {"a": ((730, 90), (1090, 110), (2440, 170)), "i": ((270, 60), (2290, 100), (3010, 170)),
                "u": ((300, 60), (870, 90), (2240, 170)), "a tuned to 697/1209": ((697, 60), (1209, 60), (2440, 170))}
    for name, shape in formants.items():
        count = seconds = 0
        for f0 in (87.0, 99.6, 116.0, 139.0, 174.25, 201.5, 232.0):
            count += len(heard(vowel(f0, shape, 1.0, rng)))
            seconds += 1
        rows.append({"signal": f"vowel /{name}/ at 7 pitches, 87-232 Hz", "seconds": seconds, "keys": count})
    return rows


def cost(rng):
    """CPU seconds to read 20 s of a call, detector on and off, medians of 3 interleaved."""
    keys = "".join(rng.choice(KEYS) for _ in range(40))
    signal, _ = call(keys, rng, tone_ms=70, gap_ms=430, snr_db=20)
    signal = (signal + [0.0] * RATE * 20)[:RATE * 20]
    times: dict[str, list[float]] = {"events": [], "inband": []}
    for _ in range(3):
        for digits in ("events", "inband"):
            start = time.process_time()
            heard(signal, digits)
            times[digits].append(time.process_time() - start)
    return {name: {"runs": runs, "median_s": statistics.median(runs), "per_audio_second_ms": statistics.median(runs) / 20 * 1000}
            for name, runs in times.items()}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--json", type=Path)
    parser.add_argument("--seed", type=int, default=20260915)
    args = parser.parse_args()
    rng = random.Random(args.seed)
    result = {"seed": args.seed, "python": sys.version.split()[0], "accuracy": accuracy(rng),
              "durations": durations(rng), "gaps": gaps(rng), "near_misses": near_misses(rng),
              "false_keys": false_keys(rng), "cost": cost(rng)}
    text = json.dumps(result, indent=1)
    if args.json:
        args.json.write_text(text)
    print(text)


if __name__ == "__main__":
    main()
