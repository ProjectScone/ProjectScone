# Retained-source generation comparison v1

**Decision: keep five sources as the default.** Increasing the retained evidence
limit improved literal retrieval coverage but reduced Gemma answer accuracy on
this development sample. All 600 new responses completed without inference or
cleanup failures. The process exited zero and frozen artifact checks passed.

## Frozen setup

The [protocol](source-budget-generation-v1.protocol.md) was committed and pushed
as `97b49aa` before inference. Each of the 200 original questions ran once with
five, eight and ten sources, rotating arm order per question. This includes a
contemporaneous five-source control. All requests were the saved production
contexts from the [source-budget comparison](source-budget-v1.results.md), with
the same 8,000-byte limit and original source text. No replacement contexts,
question rewrites, fine-tuning, answer repairs or retries were used. The 200
reserved questions remain unrun.

Inference used self-managed Ollama and `gemma4-e4b-ctx8k:latest`, with the original
model digest, 8192 context, temperature 0, `think=False`, a 256-token output cap
and a 120-second deadline. Model residency checks allowed only that model or an
empty cold start. Labels were read only after all responses were terminal.

## Answer quality

| Dataset | Metric | Five sources | Eight sources | Ten sources |
|---|---|---:|---:|---:|
| All 200 | Exact match | 64% | 63% | 62% |
| All 200 | F1 | 0.72949 | 0.72069 | 0.71264 |
| Hotpot, 100 | Exact match | 46% | 45% | 44% |
| Hotpot, 100 | F1 | 0.57335 | 0.56646 | 0.55620 |
| SQuAD, 100 | Exact match | 82% | 81% | 80% |
| SQuAD, 100 | F1 | 0.88563 | 0.87492 | 0.86908 |
| All 200 | Abstentions | 20 | 17 | 17 |
| All 200 | Failures | 0 | 0 | 0 |

Each arm contains 200 completed responses, with no truncations or cleanup
errors. All 300 Hotpot answer scores match the unchanged official evaluator.
Supporting-fact and joint generation scores remain unmeasured.

An independent audit recomputed all 600 answer scores, all nine arm summaries,
paired comparisons and coverage-gain subsets, and verified the frozen hashes
and official evaluator parity against the raw responses.

| Paired comparison, all 200 | EM gains / losses / ties | F1 gains / losses / ties | Raw answer changes |
|---|---:|---:|---:|
| Five to eight | 3 / 5 / 192 | 6 / 9 / 185 | 29 |
| Five to ten | 4 / 8 / 188 | 7 / 14 / 179 | 35 |
| Eight to ten | 2 / 4 / 194 | 3 / 8 / 189 | 19 |

All exact-match gains occur on Hotpot. Eight sources lose four Hotpot answers
and one SQuAD answer; ten lose six Hotpot and two SQuAD answers against five.
Against eight, ten sources gain two Hotpot answers and lose three Hotpot plus
one SQuAD answer.

Every fresh five-source request **and raw answer** matches the historical
baseline (200/200). The original baseline is preserved separately. This repeat
agreement strengthens this comparison but does not establish universal
determinism at temperature zero.

## Coverage gains do not guarantee answer gains

Complete Hotpot annotation coverage was 72%, 86% and 92% at five, eight and ten
sources, respectively. The larger contexts preserve the shorter contexts'
source sequences as exact prefixes. This experiment changes evidence delivery,
not which text was originally indexed.

A post-run diagnostic of the questions gaining complete coverage finds:

- Eight sources: 14 questions gain coverage; exact matches within those questions
  rise from 4 to 6.
- Ten sources: 20 questions gain coverage; exact matches within those questions
  rise from 5 to 8.

Those gains are real, but they are outweighed by regressions elsewhere. Literal
annotation coverage is not semantic understanding or a probability of a correct
answer. Its known chunk-boundary limitations are unchanged.

All 13 questions whose exact-match outcome changes between any two arms are
shown below. `Abstain` denotes the literal `INSUFFICIENT_EVIDENCE` response.
Descriptions abbreviate the original questions; inference used them unchanged.

| Question ID / topic | Five sources | Eight sources | Ten sources |
|---|---|---|---|
| Hotpot `5ae791ef55429952e35ea979`, officer's birth month | February | February | Abstain |
| SQuAD `570d30fdfed7b91900d45ce3`, Mallee weather | Warmest regions | Longer phrase about hot winds and semi-deserts | Same longer phrase |
| Hotpot `5a8078e95542992bc0c4a72c`, presenter's network | Sky News | Abstain | Abstain |
| Hotpot `5ae0c7e755429945ae95944c`, attraction's county | Bay Lake | Bay Lake | Orange County |
| Hotpot `5ae0ff95554299422ee995b0`, Larry Johnson's son | Larry Alphonso Johnson Jr. | Tony Johnson | Tony Johnson |
| Hotpot `5a733d5d5542991f29ee2d71`, poet's magazine | PEN America | Abstain | Abstain |
| Hotpot `5ab7f1b65542991d322237d3`, Soviet officer | None | Francis Gary Powers | Rudolf Abel |
| SQuAD `572ff12e04bcaa1900d76eff`, Bingen–Bonn river section | The Middle Rhine | Middle Rhine | Rhine |
| Hotpot `5abb8e2d554299642a094aa4`, film co-star | John Candy | Eugene Levy | Eugene Levy |
| Hotpot `5a888c9d5542997e5c09a612`, band farther west | Audioslave | Audioslave | Abstain |
| Hotpot `5ae03cd855429924de1b7072`, writers' country comparison | Abstain | Yes | Yes |
| Hotpot `5a7331705542991f9a20c67a`, Iraq goalkeeper | Abstain | Emad Hashim | Emad Hashim |
| Hotpot `5ae47ae05542995ad6573d4f`, namesake song | Billie Holiday's “Strange Fruit” | “Strange Fruit” | Billie Holiday's “Strange Fruit” |

The county, country-comparison and goalkeeper gains coincide with newly complete
evidence. Other changes occur despite complete annotations already being
present: some are answer-format changes, others select a wrong entity or abstain.
These observations do not identify a single universal failure mechanism.

## Observed inference cost

Across 200 responses per arm:

| Measurement | Five sources | Eight sources | Ten sources |
|---|---:|---:|---:|
| Total latency p50 | 1,883.93 ms | 2,430.52 ms | 1,786.46 ms |
| Total latency p95 | 5,747.15 ms | 6,465.99 ms | 8,621.08 ms |
| First-token latency p50 | 1,617.47 ms | 2,173.24 ms | 1,539.71 ms |
| First-token latency p95 | 5,671.23 ms | 6,247.59 ms | 7,833.57 ms |

The first request includes cold-start cost (8,090.62 ms). Model snapshots report
9,636,843,355 loaded bytes and an 8192-token context; this is not peak host RSS.
Short type/unit checks, a cached-BGE smoke test and real Qdrant contract tests
ran on the shared host during inference. Related prompts may also benefit from
prefix caching. Arm rotation reduces ordering bias but cannot remove these
effects. In particular, ten sources' lower median does not prove lower intrinsic
cost; its tail latency is higher. These are observed timings, not production SLOs.

## Decision and provenance

Increasing the source limit alone is not a supported default improvement. Keep
all three outcomes and investigate evidence selection, relation handling and
answer verification before another frozen comparison. These are development-set
results from one model and a 2,176-paragraph corpus, not full-Wikipedia or
held-out performance claims.

The prior context-preparation process's native shutdown failure remains recorded
in its separate report. This inference run consumed its complete audited output
and itself exited cleanly. The telemetry and Qdrant startup fixes were developed
in separate checkouts; this run's package source stayed unchanged.

Raw artifacts remain outside git in
`bench-runs/source-budget-generation-dev-2026-09-08/`.

- Model digest: `858470a535014f5c6847e388069aa64cb4f892e99309c851b59be288eec784cf`
- Manifest SHA256: `3b3148f8cd081b05fe8ca67aae337df40a9bcd1b38e085be4cacc0850e55e4a4`
- Observations SHA256: `9db75816e35fa05344af16ca8a29dc17b3cffd1bbad7864fbea79ca5c37efff2`
- Scores SHA256: `d887d2f756f4933c5cd2a89f3ae8c87c0ab093693e127f0f19e86a6c2ba56069`
- Scorer SHA256: `caa3d70d97a97828be3f5abcb569ba31bd56c5d8e3d6c0d6043907f4f24decf5`
