# Jev on downloaded public QA — 2026-09-19

Jev improved retrieval on this frozen Scone evaluation. It did not improve every
metric on every question, and this run does not establish answer quality or
performance against a strong neural embedding model.

## Method and provenance

Executed the [pre-run protocol](jev-public-qa-v1.protocol.md) using Scone's
`MemoryEngine`, SQLite document/vector stores and native ingestion/recall. All
2,176 downloaded source paragraphs were indexed; all 200 reserved questions
(100 HotpotQA, 100 SQuAD) ran through each of four arms. All 800 observations
completed, with no retrieval errors. Jev applied on all 400 requested reranks,
with no fallback. The alias `~typesafe/jev-latest` resolved consistently to
`typesafe/jev-1.13-20260917`.

The runner did not read gold annotations; scoring occurred after the complete
run. Code and corpus/question hashes remained unchanged during execution.
The evaluation harness was committed as `2240fd8b`; the Jev adapter was
committed as `8c54ae84`. This reserved split is now consumed, not an untouched
holdout for future tuning.

Local evidence, excluded from git:

- Inputs: `bench-runs/public-qa-2026-09-08/` in the primary checkout.
- Results: `bench-runs/jev-public-qa-20260919/run-1/`, including manifests,
  completion hashes, all ranked observations, per-question scores and SQLite
  source index.
- Corpus SHA256: `075d39e15f6e349ff1bc70c9d7c5a04a28cfb1ceb74b747e1f9244097497dc4b`.
- Reserved queries SHA256: `670fccb87f03967c3582bffe0bde1db127dfea576101987362f2923fc0fcbd08`.
- Gold SHA256: `094334ef8713d873018be3fb1144b6e91980dc2dd485a90446c0a905b1e0fa9f`.

## Overall results

Recall is the mean fraction of annotated supporting documents represented in
the first k passages. Complete coverage means every annotated supporting
document is present. Duplicate chunks never count as extra supporting documents.
MRR measures the reciprocal rank of the first annotated supporting passage.

| Scone retrieval | Recall@5 | Complete@5 | Recall@10 | Complete@10 | MRR@10 | p50 ms | p95 ms |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Hybrid | 85.75% | 75.0% | 92.25% | 85.5% | 0.9012 | 47 | 129 |
| Hybrid + Jev | 97.00% | 94.5% | 97.50% | 95.5% | 0.9817 | 543 | 764 |
| Text only | 85.75% | 75.0% | 92.25% | 85.5% | 0.9012 | 8 | 22 |
| Text only + Jev | 97.00% | 94.5% | 97.50% | 95.5% | 0.9817 | 506 | 647 |

Jev added approximately 496 ms to hybrid median latency and 498 ms to text-only
median latency. These are local sequential-run observations, not a throughput
or concurrent-load benchmark.

Paired outcomes are identical for hybrid and text-only comparisons:

| Metric | Improved questions | Worse questions | Ties | Mean change |
| --- | ---: | ---: | ---: | ---: |
| Recall@5 | 41 | 0 | 159 | +11.25 percentage points |
| Complete@5 | 39 | 0 | 161 | +19.50 percentage points |
| Recall@10 | 20 | 0 | 180 | +5.25 percentage points |
| Complete@10 | 20 | 0 | 180 | +10.00 percentage points |
| MRR@10 | 27 | 1 | 172 | +0.0805 |

An exploratory paired bootstrap after scoring (10,000 question resamples,
Python `Random(20260919)`, sorted question IDs) gave a 95% percentile interval
of +8.25 to +14.50 percentage points for Recall@5 improvement. Complete@5's
interval was +14.00 to +25.00 points; MRR's was +0.0512 to +0.1125. These
describe sample uncertainty, not repeatability across model calls or datasets.
The artifact is `paired-bootstrap.json`; bootstrap analysis was not part of
the frozen primary protocol.

## Where the improvement occurs

Quality aggregates below are identical between the hybrid and text-only arms.

| Dataset | Ranking | Recall@5 | Complete@5 | Recall@10 | Complete@10 | MRR@10 |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| HotpotQA, 100 | Baseline | 72.5% | 51.0% | 85.5% | 72.0% | 0.8573 |
| HotpotQA, 100 | Jev | 95.0% | 90.0% | 96.0% | 92.0% | 0.9783 |
| SQuAD, 100 | Baseline | 99.0% | 99.0% | 99.0% | 99.0% | 0.9450 |
| SQuAD, 100 | Jev | 99.0% | 99.0% | 99.0% | 99.0% | 0.9850 |

Multi-document HotpotQA accounts for the coverage improvement. SQuAD coverage
was already near its ceiling; Jev improved ordering without recovering its one
missing source.

| Dataset | Arm | p50 / p95 ms | Retrieval failures | Jev applied / fallback |
| --- | --- | ---: | ---: | ---: |
| HotpotQA | Hybrid | 49 / 133 | 0 | 0 / 0 |
| HotpotQA | Hybrid + Jev | 543 / 792 | 0 | 100 / 0 |
| HotpotQA | Text | 9 / 29 | 0 | 0 / 0 |
| HotpotQA | Text + Jev | 503 / 615 | 0 | 100 / 0 |
| SQuAD | Hybrid | 45 / 105 | 0 | 0 / 0 |
| SQuAD | Hybrid + Jev | 543 / 743 | 0 | 100 / 0 |
| SQuAD | Text | 7 / 15 | 0 | 0 / 0 |
| SQuAD | Text + Jev | 512 / 674 | 0 | 100 / 0 |

## Regressions and unresolved cases

The single MRR regression is `hotpotqa:5abff0785542997d64295957`, asking which
Australian actress starred in Heaven and portrayed Katharine Hepburn. The
baseline ranked a Cate Blanchett source first but missed the Heaven source in
its top ten. Jev ranked an unannotated Aviator article first, Heaven second,
and Cate Blanchett third/fourth. MRR fell from 1.0 to 0.5 while supporting-document
recall rose from 0.5 to 1.0. Both outcomes remain in the reported metrics.

Nine questions still lack complete support at ten passages: eight HotpotQA
questions each miss one of two required documents, and one SQuAD question
misses its sole source. The SQuAD question is the context-dependent “How was
this possible”; the benchmark supplies no preceding dialogue. Nothing was
rewritten or excluded to improve the score.

The unresolved HotpotQA IDs are `5abd6db755429933744ab7d0`,
`5a7f341655429930675136a0`, `5a8408a9554299123d8c21d3`,
`5ae0ba9055429924de1b715c`, `5ac1b3a75542994ab5c67dd2`,
`5ab56f7a554299637185c59a`, `5ac275e755429921a00aaf81`, and
`5abf8ae85542990832d3a14b`. The SQuAD ID is `57263dcd89a1e219009ac5a4`.
These observations do not distinguish missing candidates from reranking
mistakes outside the retained top ten.

## Conversation integration check

The separate `benchmarks/jev_conversation.py` check copies the evaluated SQLite
index and exercises Scone's actual conversation API, tool runtime, Jev and
paid `google/gemma-4-31b-it`, then reconstructs the app and engine. It uses the
first reserved question unchanged, not a hand-picked successful question.

This check found that `ScopedMemoryTools` bypassed the configured reranker.
The fix ranks candidates after scope/session filtering, then revalidates
source retention after the asynchronous scorer. Tests cover excluded evidence,
provider failure, deletion during ranking and the existing policy to skip
long-running listwise chat in tool search.

The final check (`bench-runs/jev-conversation-20260919/run-3/`) applied Jev and
answered “BSkyB has an operating license from Ofcom.” The answer and episode
identity survived restart without reranking. Full receipt equality did **not**
hold: live `history`, `memory_context`, `provider_completion`, `turn_id` and
`user_episode_id` fields were absent after restart. This is an unresolved
diagnostic persistence limitation, not evidence of full receipt durability.
Failed run-1 and run-2 artifacts remain available.

## What this establishes, and what remains

- Measured improvement over the tested Scone baseline, especially multi-document
  coverage, with a measurable latency cost. Jev remains opt-in.
- This uses `HashEmbedder`, whose default vector fusion weight is 0.01. Hybrid
  and text-only quality aggregates matching does **not** prove embeddings are
  unnecessary. The paths were different: baseline top-ten passage lists differ
  on 4 questions, and Jev top-ten lists differ on 178 questions.
- A comparison with a strong, permitted neural embedder remains necessary.
- Document-level support metrics do not establish full supporting-sentence
  coverage, factual answer accuracy, abstention quality or hallucination rates.
- One live answer/restart check establishes integration, not an answer-quality
  benchmark. UI acceptance, agent-created tools, memory conflict decisions and
  graph quality are not evaluated by these scores.
- Public datasets may have appeared in model training; this is not a
  contamination-controlled or private-domain benchmark. Future tuning needs a
  new evaluation split and pinned model version.
