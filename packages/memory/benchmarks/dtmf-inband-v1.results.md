# Keys heard in the audio (in-band DTMF) v1

On synthesized calls companded to mu-law and read as 20 ms carrier media
messages, `MediaStream(digits="inband")` heard **500 of 500 keys** at every
signal-to-noise ratio down to **6 dB**, **483 of 500 at 3 dB** and **4 of
500 at 0 dB**, with **no wrong key and no key that was not pressed** at any
ratio. It heard every 40 ms press and no 20 ms one, split no repeated key
separated by 30 ms of silence, accepted every tone 1.5 % off nominal and no
tone 3.5 % or 5 % off, and heard **no key in 60 s of noise, 10 s of a 440 Hz
tone or 28 s of steady synthesized vowels**, including a vowel whose
formants were placed on 697 and 1209 Hz. Reading audio with the detector on
took **33.1 ms of CPU per second of audio**, against **3.3 ms** for the same
stream with it off.

Nothing here is a recording of a phone or a voice. pipecat, the reference
for this feature, reads keys only from carrier event messages and has no
in-band detector to compare against.

## How it was measured

- `benchmarks/dtmf_inband.py`, seed 20260915, Python 3.14.7, run from
  `packages/memory` with `PYTHONPATH=src`. Every number below is from one run.
- A key is two sines at -10 dBFS each, with a random twist of up to 2 dB
  either way and random phases, 70 ms on and 70 ms off unless a row says
  otherwise, after a random 20-60 ms lead.
- Noise is white Gaussian over the whole 0-4 kHz band. The SNR is the key's
  own mean power while it sounds over the noise power. The sum is mu-law
  encoded and decoded (the line), then fed in 160-sample media messages.
- Scoring: a detection counts against the key whose tones it falls inside
  (13 ms of slack before the start). The right key is a hit, a wrong key a
  substitution, a detection inside no key an insertion; an undetected key
  is a miss.
- Accuracy: 50 sequences of 10 random keys from all 16 at each SNR.
  Durations: 20 sequences of 10 keys, 60 ms apart, per duration, clean and
  at 10 dB. Gaps: 20 sequences of one key pressed 10 times, 70 ms each.
  Near misses: each of the 16 keys three times, 100 ms, with the low or the
  high tone moved. False keys: white noise with sigma 4000 (-18 dBFS) for
  60 s; a 440 Hz tone for 10 s; 1 s each of four steady vowels at seven
  pitches from 87 to 232 Hz, harmonics shaped by three formant resonances.
- Cost: `time.process_time` to read 20 s of a call (40 keys at 20 dB SNR,
  500 ms apart) with `digits="events"` and `digits="inband"`, interleaved,
  3 runs each, median. The machine was shared (load average near 50);
  CPU time, not wall time, is reported for that reason.

## Results

Accuracy by SNR (500 keys each):

| SNR | hits | misses | substitutions | insertions | sequences exactly right |
| --- | --- | --- | --- | --- | --- |
| clean | 500 | 0 | 0 | 0 | 50 / 50 |
| 20 dB | 500 | 0 | 0 | 0 | 50 / 50 |
| 10 dB | 500 | 0 | 0 | 0 | 50 / 50 |
| 6 dB | 500 | 0 | 0 | 0 | 50 / 50 |
| 3 dB | 483 | 17 | 0 | 0 | 36 / 50 |
| 0 dB | 4 | 496 | 0 | 0 | 0 / 50 |

Below 6 dB the energy rule (the two tones must hold 60 % of a block's
energy) refuses keys rather than guessing: every error at 3 and 0 dB is a
miss.

Press duration (200 keys each):

| tone | 20 ms | 25 ms | 30 ms | 40 ms | 50 ms | 60 ms | 70 ms |
| --- | --- | --- | --- | --- | --- | --- | --- |
| clean | 0 | 16 | 186 | 200 | 200 | 200 | 200 |
| 10 dB | 0 | 0 | 47 | 200 | 200 | 200 | 200 |

ITU-T Q.24 asks a receiver to accept 40 ms and refuse under 23 ms; 25 and
30 ms are between the two. An earlier version stepped blocks half a window
apart: it heard 143 of 200 clean 40 ms presses (66 at 10 dB) and none at
25 or 30 ms, and with two blocks instead of three it heard 30 of 200 20 ms
presses. Stepping a quarter window apart doubled the blocks measured; that
version cost 11.3 ms of CPU per audio second on a less loaded machine, so
the two cost figures are not comparable with each other.

Repeated key, silence between presses (200 presses each): 30, 40, 50 and
60 ms gaps all 200 hits, no insertions.

Near misses (48 presses each, tone moved from nominal):

| offset | low tone moved | high tone moved |
| --- | --- | --- |
| +1.5 % | 48 accepted | 48 |
| -1.5 % | 48 | 48 |
| +2.5 % | 20 | 7 |
| -2.5 % | 9 | 7 |
| +3.5 % | 0 | 0 |
| -3.5 % | 0 | 0 |
| +5 % | 0 | 0 |
| -5 % | 0 | 0 |

2.5 % is where the frequency rule's boundary falls (probes 5 % either
side), inside Q.24's range where either answer is allowed.

False keys: 0 in each of white noise (60 s), 440 Hz (10 s), /a/, /i/, /u/
and the /a/ with formants on 697 and 1209 Hz (7 s each).

Cost, CPU ms to read 20 s of audio (3 interleaved runs):

| | runs | median | per audio second |
| --- | --- | --- | --- |
| `digits="events"` | 66.2, 66.5, 62.8 | 66.2 | 3.3 ms |
| `digits="inband"` | 665.0, 661.8, 655.2 | 661.8 | 33.1 ms |

## What this does not show

- Real phones and real lines: carrier echo cancellers, codecs other than
  G.711, packet loss and clipped tones were not modelled.
- Real speech: steady vowels have no pitch or formant movement, which
  makes them a harder case for a single block and an easier one for a
  press-length run; talk-off on recorded speech is not measured.
- A key heard both ways (`digits="both"`) is paired by the tests, not
  measured here.
