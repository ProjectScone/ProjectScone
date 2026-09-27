# QASPER development: faster routing, stronger evidence, similar answer accuracy

Vector-guided section routing reduced median retrieval time from **964.4 to
607.1 ms (37.1%)** versus hierarchical routing. Median retrieval + rerank +
generation time fell from **2,457.6 to 2,097.3 ms (14.7%)**. It retrieved more
complete evidence than the matched LlamaIndex hybrid configuration, but answer
F1 was effectively tied and stage-sum latency remained **43.5% higher** than
LlamaIndex. Keep vector-guided routing opt-in; this does not establish overall
superiority or justify changing the default policy.

The complete official development split ran on September 26, 2026: **281 papers,
1,005 questions, 4,020 scheduled observations, zero missing**. Frozen source
commit: `2f76d176`. All development paper/question IDs are disjoint from the
previous test run. The earlier [test results](RESULTS.md) remain unchanged;
scores across these different splits are not before/after measurements.

All four arms matched the official QASPER evaluator's answer F1 and both evidence
F1 variants within 1e-12. The audit verified input/source/artifact hashes,
the full attempt/observation schedule, and every persisted generation context.
All 15,804 exported paragraph strings and byte spans match the original data.
Inference used the annotation-free export; gold was used only for offline scoring.

Resources and limits are frozen in [DEV_PROTOCOL.md](DEV_PROTOCOL.md): shared
700-character chunks, Nemotron 3 Embed 1B vectors, Jev relevance judgments, Gemma
4 31B answers, and five-item/8,000-byte final contexts. LlamaIndex uses vector +
BM25 reciprocal-rank fusion. This is known-paper text-and-caption QA, not pooled
corpus discovery. All documents, indexes, and generated artifacts stayed local;
the user explicitly authorized the existing hosted model endpoints.

## Quality

Every row includes all 1,005 questions. The three output-token-limit failures
(two hierarchy, one vector-guided) score zero on every quality metric.

| Arm | Answer F1 | Normalized EM | Evidence F1 | Evidence recall | Failed generations |
| --- | ---: | ---: | ---: | ---: | ---: |
| Scone flat vectors | 44.5229% | 17.4129% | 20.1442% | 40.3696% | 0 |
| Scone hierarchy | 43.4252% | 16.9154% | 21.7101% | 41.8469% | 2 |
| Scone vector-guided | 44.4331% | 17.2139% | 22.1988% | 42.2790% | 1 |
| LlamaIndex hybrid | 44.4109% | 17.9104% | 19.9132% | 38.9610% | 0 |

Paired paper-cluster bootstrap, 2,000 resamples, seed 20260924. Deltas and 95%
percentile intervals below are **percentage points**. The primary prespecified
comparison is vector-guided minus hierarchy; all scheduled contrasts are shown.

| Contrast | Answer F1 delta [95% CI] | Evidence F1 delta [95% CI] | Evidence recall delta [95% CI] |
| --- | ---: | ---: | ---: |
| Vector-guided minus hierarchy | +1.0079 [+0.0870, +1.9497] | +0.4888 [-0.2734, +1.2747] | +0.4321 [-0.5821, +1.4620] |
| Vector-guided minus flat | -0.0898 [-1.0029, +0.8635] | +2.0547 [+1.0837, +3.2603] | +1.9094 [+0.7907, +3.1822] |
| Vector-guided minus LlamaIndex | +0.0222 [-1.1318, +1.2084] | +2.2857 [+1.2366, +3.4780] | +3.3180 [+1.9585, +4.7425] |
| Hierarchy minus flat | -1.0977 [-2.1368, -0.1077] | +1.5659 [+0.5428, +2.6382] | +1.4773 [+0.1951, +2.8113] |
| Hierarchy minus LlamaIndex | -0.9857 [-2.2656, +0.2796] | +1.7969 [+0.6353, +3.1122] | +2.8859 [+1.2906, +4.5705] |

The vector-guided answer advantage over hierarchy excludes zero in this single
run, while its answer differences against both controls do not. Its evidence
advantage over both controls excludes zero; the evidence difference against
hierarchy does not. Intervals are unadjusted for multiple metrics and describe
paper-sampling uncertainty, not repeated-generation variance or model training
overlap. They do not establish equivalence between methods.

Complete original paragraphs must appear verbatim in packed context to receive
evidence credit. Split/clipped paragraphs do not count. The primary metrics retain
figure/table evidence that text retrieval cannot provide. Text-only variants:

| Arm | Text evidence F1 | Text evidence recall |
| --- | ---: | ---: |
| Scone flat vectors | 20.9015% | 41.8887% |
| Scone hierarchy | 22.3773% | 43.3112% |
| Scone vector-guided | 22.8701% | 43.6687% |
| LlamaIndex hybrid | 20.5868% | 40.5050% |

Post-hoc context check: vector-guided and hierarchy contexts were identical on
750/1,005 questions (74.6%); 209 of those still produced different answer strings.
Those identical-context questions contributed +0.1730 points of the +1.0079
answer-F1 difference; changed contexts contributed +0.8349 points. This accounting
does not isolate causation. Vector-guided contexts also matched flat retrieval on
772 questions and LlamaIndex on 307. Do not attribute every answer difference to
retrieval when independent generation remains stochastic at temperature zero.

## Latency and consumption

| Arm | Retrieval median / p95 | Generation median / p95 | Stage-sum median / p95 |
| --- | ---: | ---: | ---: |
| Scone flat vectors | 1.7 / 2.8 ms | 875.9 / 4,149.2 ms | 1,413.5 / 4,619.7 ms |
| Scone hierarchy | 964.4 / 1,832.5 ms | 887.0 / 3,730.4 ms | 2,457.6 / 5,454.0 ms |
| Scone vector-guided | 607.1 / 1,011.9 ms | 922.6 / 3,607.3 ms | 2,097.3 / 4,738.5 ms |
| LlamaIndex hybrid | 12.5 / 23.6 ms | 943.9 / 3,869.3 ms | 1,461.7 / 4,286.7 ms |

Address-route attempts fell from **1,809 to 1,005 (44.4%)** in persisted routing
receipts. Median routing time fell from 674.1 to 421.6 ms. Shared reranking took
456.3 ms median / 681.9 ms p95 and is counted once physically. Routing was
uncached; routing-policy order alternated and generation order rotated by global
question ordinal. Medians of individual stages need not sum to the stage-sum
median.

The stage sums exclude index/vector preparation, scheduling waits, and discarded
work before the rerank timeout described below. They are not production chat
latency or complete run wall time. Initial successful vector preparation took
452.4 seconds for 14,564 distinct passage strings and 1,005 question records;
resume preparation took 4.5 seconds entirely from cache. There were 122 recorded
embedding requests, 1,005 successful shared-rerank receipts, and 4,020 generation
receipts. Mean final context bytes: flat 2,755.3; hierarchy 2,755.8; vector-guided
2,728.7; LlamaIndex 2,770.9. Every persisted context stayed within the byte cap.

Provider-reported charges total **$0.534359362** for embedding, shared reranking,
and generation receipts. This is not the total bill: routing/fetch monetary costs
are absent, and failed/unpersisted preparation may have incurred unrecorded usage.
Routing receipts record 2,244,296 input / 278,157 output tokens; 968 successful
fetch decisions record 484,743 input / 39,579 output tokens. Request counts and
token totals describe retained receipts, not a complete provider billing ledger.

## Failure handling and routing behavior

The first sandbox attempt failed DNS before any model results. The unchanged
manifest resumed with network access. A later shared-rerank request timed out
after 30 seconds at 880 saved answers. The resume audit confirmed zero unfinished
generation attempts; saved answers and prepared contexts were reused. Unprepared
retrieval/rerank work was rerun. No saved answer was replaced or regenerated.

Clarification to the protocol's broad stop-on-provider-failure wording: native
route/fetch errors are caught by the existing retrieval policy and can fall back
to flat retrieval. Unhandled rerank/generation errors stop the bounded run batch.
The native fallback behavior was unchanged during this run and is counted below.

| Effective mode | Hierarchy | Vector-guided |
| --- | ---: | ---: |
| Broad vectors | 515 | 530 |
| Scoped vectors | 437 | 427 |
| Original sections | 53 | 48 |

| Final reason | Hierarchy | Vector-guided |
| --- | ---: | ---: |
| Jev selected a retrieval mode | 483 | 477 |
| No matching address | 433 | 421 |
| Invalid routing response | 74 | 103 |
| Routing provider failure | 2 | 0 |
| Routing timeout | 2 | 2 |
| Fetch decision failed | 3 | 2 |
| Original selection exceeded budget | 8 | 0 |

Vector-guided routing reduces address work but does not fix invalid responses:
they increased from 74 to 103. These receipts do not retain the raw rejection
details, so their cause is not established. Keep validation strict. Any further
policy changes need a separately frozen evaluation, rather than tuning this run
and presenting it as untouched evidence.

## Audit artifacts

Generated artifacts remain local under
`bench-runs/qasper-vector-development-2026-09-25/full-v1/`: `manifest.json`,
`completion.json`, `observations.jsonl`, `prepared.jsonl`, `attempts.jsonl`,
`judgments.jsonl`, `report.json`, `official-parity.json`, and `resume-audit-1.json`.
The original development JSON is in the sibling `raw/` directory; the
annotation-free inference bundle is in `dataset-full/`.

| Artifact | SHA-256 |
| --- | --- |
| Original development JSON | `2ae7ee62a65b1c4225791c70de80c2aad4e8998cf1fd4f09a53103db4f21af93` |
| Run manifest | `99e3c5d986e80af7eb033d1256dd9b3c8600b088e3025f52573fe126736a8ee0` |
| Observation journal | `ef238346c7cab6c7e0b5836c3cee56e24ebd994374510e500716037cf961d658` |
| Completion audit | `2daf223fe615863556a90f8d0e949487d67692f7791d9a02605b6a0babb8f6aa` |
| Official evaluator | `781aba7cd8e524bef4f0a1b4bf3504e5b02cb1d8d5bf32a8f0a89dfa83e86bfe` |
| Local audit/scoring helper | `245161bb7c884ab8fb120dc346d8bd07dbfcbf81bce26555e7de987993c73044` |
