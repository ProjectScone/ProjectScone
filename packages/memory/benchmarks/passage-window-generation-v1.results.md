# Passage continuity generation results — 8 September 2026

The [frozen follow-up protocol](passage-window-generation-v1.protocol.md) produced
**200 completed Gemma E4B responses, with zero failures**. Adding neighboring
passages yielded one additional exact match overall: three gains and two losses.
This small, mixed development result does **not** justify enabling the option
by default. `neighbor_chunks` remains 0 unless explicitly configured.

| Metric | Original baseline | Neighboring passages |
|---|---:|---:|
| Overall exact match | 64.0% | 64.5% |
| Overall token F1 | 72.95% | 73.78% |
| HotpotQA exact match / F1 | 46.0% / 57.34% | 48.0% / 59.34% |
| SQuAD exact match / F1 | 82.0% / 88.56% | 81.0% / 88.22% |
| Abstentions | 20/200 | 20/200 |
| Runtime failures | 0/200 | 0/200 |
| Inference p50 / p95 | 2.29 / 5.88 s | 6.60 / 23.49 s |

Each dataset uses 100 unchanged questions. Every response remains in its
denominator; abstentions score zero. No model fine-tuning, question edits,
answer repairs or replacement retries were used. The original baseline
responses and scores remain unchanged.

## What changed

Two exact-match gains supplied a more specific or previously missing answer;
another removed extra wording from an already informative answer. One loss
added wording that the official exact-match metric penalizes. The other was
a substantive error: asked to compare two named music groups, the model answered
with a third group present in its context.

The [context-only experiment](passage-window-v1.results.md) recovered complete
annotations for three questions. None became a new exact match here: the two
Hotpot answers were already correct. The SQuAD passage-boundary case improved
from a cut-off statement to the substantive answer, raising token F1 from
14.81% to 87.50%, but extra wording still prevented exact match. Better source
coverage and better generation are distinct measurements.

The guitar answer also recovered relevant support that the frozen literal
coverage metric misses: the gold sentence begins with whitespace absent at
the stored chunk boundary. The sentence's information is present. We retain
the original metric and scores, with this limitation explicitly recorded.

An exploratory repeat diagnostic found that **64 complete requests were
byte-for-byte identical** to their baseline inputs (52 Hotpot, 12 SQuAD).
All 64 produced identical raw answers. This subset was selected using request
hashes during inference, before inspecting new answers. It is useful evidence
about these repeats, not a guarantee of determinism or a contemporaneous
control for all 200 questions.

## Validation and limits

All 100 new Hotpot answer EM/F1 scores match the unchanged
[official evaluator](https://github.com/hotpotqa/hotpot/blob/master/hotpot_evaluate_v1.py)
within `1e-12`. Supporting-fact and joint scores remain unmeasured because
the frozen output format requested no supporting-sentence IDs.

The model digest, original questions, response instruction, 8192-token model
context, temperature 0, `reasoning_effort=none`, 256-output-token limit and
120-second deadline were unchanged. All 200 expanded requests were frozen
before inference, with five ranked anchors and an 8,000-byte source budget.
Generation read no gold labels; the separate scorer checked completion and
artifact/source hashes before scoring.

This is a historical-control comparison on already-inspected development
questions. Timing was not isolated: other applications were active, system
swap was in use, and Ollama reported a different model memory layout than in
the original run. The observed latency increase cannot be assigned solely to
passage expansion. The 200 reserved questions remain unrun; no broad
generalization or full semantic-accuracy claim follows from this result.

Requests, public responses, per-question scores, protocol/model/source
manifests and parity checks are retained separately in the ignored
`bench-runs/passage-window-dev-2026-09-08/` directory. The original baseline
is documented [here](public-qa-v1.results.md).

Final response artifact SHA256:
`1ca9cfbfbae38f6b694cc36a047de4086c9a00af9428b1cababcb209ad4fbb53`.
Generation manifest SHA256:
`54186ad3b6ca64aab573e352b762a60c5724ba6bbde0dc673b137ae81d6c1118`.
