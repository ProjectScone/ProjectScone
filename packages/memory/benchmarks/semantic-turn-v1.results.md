# Semantic end of turn on scripted utterances — 15 September 2026

A voice session used to answer every final transcript a recognizer
produced, so a speaker who paused mid-clause for longer than the energy
gate's stop was answered with half a question. This compares premature
turn endings with silence alone and with the lexical end-of-turn detector
(`SCONE_SEMANTIC_TURN=1`), and what the detector costs complete turns. No
model, recognizer or recording was used.

## Method

- Fixture: [`semantic-turn-v1.json`](semantic-turn-v1.json), 32 utterances
  authored for this set before anything was measured. 16 are cut
  mid-clause by a pause of 600–2300 ms (one twice); 16 are complete turns
  (one bridged by a 300 ms pause). Transcripts mix punctuated and
  unpunctuated styles. Five cuts were written so the rules should miss
  them: no lexical cue (`like in`, `number is`, `for twenty`), a recognizer
  that punctuated the cut (`a call with.`), and a 2300 ms pause longer than
  the hold. Two complete turns end on a word the rules treat as open
  (`sure I'd love to`, `what you're referring to`).
- Replay (`scone_memory.bench.semantic_turn`): the `VoiceGate` defaults —
  a pause of 400 ms or more ends a transcript, which arrives 400 ms after
  the words; renewed speech is noticed 120 ms after it starts; 300 ms per
  word; transcription instant. Both modes drive the session's own
  `TurnHold` with its defaults (hold 1.5 s, turn bound 10 s). Silence
  alone judges everything `unsure`, which is exactly the session without
  a detector. A deadline that falls at the same moment as an event is
  taken first.
- A premature ending is a turn released before the utterance's last
  words. Added latency is how long after an utterance's last transcript
  its final turn was released.
- Wall clock, same machine (load average 57–65 from other jobs), same
  inputs: `judge_text` over the replay's 49 transcripts × 2000 loops,
  5 repeats; and a real `VoiceSession` (in-memory storage, scripted
  providers) timing each final transcript to the model being asked, 28
  turns per run over the 14 complete turns the rules release at once,
  detector off and on interleaved (order alternating), 5 repeats each.
- Reproduce: `measure(path)`, `judge_cost_ns(path)` and
  `session_latency(texts, semantic=...)`;
  `tests/benchmarks/test_semantic_turn.py` pins the replay counts.

## Results

| | silence alone | lexical detector |
| --- | --- | --- |
| premature endings | 17 | 5 |
| cut utterances answered early | 16 / 16 | 5 / 16 |
| complete utterances answered early | 0 / 16 | 0 / 16 |
| complete turns held | 0 / 16 | 2 / 16 |
| added latency on complete turns | 0 ms | median 0 ms, mean 188 ms, max 1500 ms |
| added latency on a rescued cut's final turn | — | 0 ms (all 16) |
| release reasons | silence 49 | semantic_complete 24, silence 10, semantic_incomplete_timeout 3 |

The five still answered early are the five written to be missed: cut-04,
cut-09 and cut-16 (no cue, released as `silence`), cut-15 (punctuated by
the recognizer, `semantic_complete`) and cut-11 (hold ran out,
`semantic_incomplete_timeout`). An earlier version of these rules called
any unpunctuated transcript that opened with a question word complete, so
cut-04 ("What's the weather going to be like in") was released as
`semantic_complete` although this record said `silence`. That rule changed
no release time (complete and unsure both end the turn at the pause), only
the reason, and it wrongly declared mid-clause cuts complete ("what I really
need is", "how do I get from"); it was removed, and `when`, `why` and a bare
`how` no longer excuse a trailing preposition. Premature endings and held
complete turns are unchanged; four releases moved from `semantic_complete`
to `silence` (cut-04's first fragment, done-06, done-08, done-14), and
`test_the_fixture_measurement` now pins the reasons as well as the counts. The two held complete turns are the two
written with a trailing `to`; each waited the full 1.5 s hold.

Wall clock:

- `judge_text`: 6641, 5338, 7953, 25095, 21108 ns per call; median
  7953 ns (8 µs). After the review fixes (a word pattern that reads digits
  and every script, the question-word rule removed), the rules before and
  after, interleaved with alternating order, 5 repeats of 2000 loops over
  the 49 transcripts at load average 48: before 6298, 5819, 5655, 7739,
  5037 ns (median 5819); after 5744, 7124, 7755, 7131, 6904 ns (median
  7124). About 1.3 µs more per transcript, within the spread of either.
- Transcript to model, run medians over 28 turns — off: 7.872, 5.034,
  24.667, 12.474, 7.502 ms (median 7.872); on: 3.372, 23.990, 15.075,
  6.123, 3.908 ms (median 6.123). On minus off per repeat: −4.5, +19.0,
  −9.6, −6.4, −3.6 ms (median −4.5). The run-to-run spread on this loaded
  machine (3–25 ms) is three orders of magnitude above the 8 µs the rules
  cost, so no difference between the modes is measurable here; the
  sign of the median is noise, not a speed-up.

## What this does not show

The fixture spells every number out, so a transcript ending on a digit
("set a timer for 20") is covered only by unit tests, which the replay
does not count. The fixture is scripted text with modelled timing, not recorded speech:
real pause lengths, recognizer punctuation and transcription latency were
not measured, and the premature-ending counts depend on how the cuts were
written. A model behind `ChatEndOfTurn` was not measured; its latency
would come out of every turn.
