# Local structure retrieval: QASPER development results

Measured September 27, 2026, at commit `546a6062244d27bd674d1320af31aa730cc146b3`.
The frozen [protocol](PROTOCOL.md) covered all 1,005 questions from 281 papers.
All questions completed; no rerank failures or resumed work occurred.

Local expansion removes hosted routing/fetch decisions and reduces median warm
retrieval from **622.09 ms to 2.01 ms**. Including the shared evidence reranker,
the median stage sum falls from **1,006.26 ms to 380.92 ms** (62.1%). Evidence
recall increases, but evidence F1 decreases against vector-guided routing. Keep
this mode opt-in: the experiment supports a latency/recall tradeoff, not a
general quality improvement.

## Matched results

Evidence metrics are percentages; latency values are milliseconds. All four
arms use the same questions, cached vectors, candidate budgets, and shared Jev
rerank judgments. LlamaIndex is the installed vector + BM25 fusion reference.

| Arm | Evidence F1 | Evidence recall | Retrieval p50 / p95 | Retrieval + rerank p50 / p95 | Mean context bytes |
| --- | ---: | ---: | ---: | ---: | ---: |
| Flat vectors | 20.50 | 40.55 | 1.78 / 4.78 | 379.69 / 602.23 | 2,752.79 |
| Vector-guided routing | 22.72 | 42.55 | 622.09 / 937.80 | 1,006.26 / 1,520.35 | 2,726.81 |
| Local structure | 20.10 | 64.71 | 2.01 / 8.68 | 380.92 / 603.80 | 4,611.05 |
| LlamaIndex | 20.19 | 39.04 | 3.77 / 12.93 | 382.67 / 605.80 | 2,769.25 |

| Arm | Text-only evidence F1 | Text-only evidence recall |
| --- | ---: | ---: |
| Flat vectors | 21.27 | 42.12 |
| Vector-guided routing | 23.40 | 44.04 |
| Local structure | 20.28 | 66.93 |
| LlamaIndex | 20.88 | 40.60 |

Local expansion retains 69.1% more context bytes than vector-guided routing.
Complete source paragraphs improve evidence coverage while adding irrelevant
text, consistent with the recall gain and F1 loss. This benchmark counts complete
reference paragraphs present in the final context; it does not credit fragments.
All-evidence and text-only F1 matched the official QASPER evaluator for every arm.

## Paired uncertainty

Deltas below are local structure minus the named baseline, in percentage points,
with 95% paired paper-cluster bootstrap intervals (2,000 draws, seed 20260924).
Intervals are unadjusted and reflect sampled papers, not repeated model runs.

| Baseline | Evidence F1 delta [95% CI] | Evidence recall delta [95% CI] |
| --- | ---: | ---: |
| Vector-guided routing | -2.61 [-4.41, -1.01] | +22.16 [+18.75, +25.32] |
| Flat vectors | -0.40 [-2.03, +1.10] | +24.16 [+20.91, +27.29] |
| LlamaIndex | -0.08 [-1.87, +1.50] | +25.68 [+22.33, +28.71] |

The F1 regression against vector-guided routing is supported by this interval.
The flat and LlamaIndex F1 intervals include zero; that does not establish
equivalence. Text-only deltas versus vector-guided routing were -3.11 points
F1 [-4.95, -1.44] and +22.89 points recall [+19.41, +26.18].

## Runtime and scope

- Local structure made zero routing/fetch calls. Vector-guided routing made
  1,005 routing requests and 485 fetch-decision attempts. Route reasons were
  480 `jev_selected`, 421 `no_match`, 99 `invalid_response`, and five
  `fetch_decision_failed`; existing vector fallbacks remained in the measured arm.
- Shared reranking took 376.94 ms median / 600.53 ms p95. Each arm is charged
  this same observed time. A separately deployed arm could send a smaller batch;
  these are matched stage sums, not independent deployment wall-clock measures.
- Source/index preparation totaled 13.67 seconds across 281 papers and is
  excluded from query timing. All 33,250 embedding lookups hit the local cache;
  no embedding requests were made. Route caching was disabled.
- Eight questions ran concurrently within each paper. Regression tests also
  ran on the same host during the experiment. Absolute timing depends on this
  load and the hosted provider; within-run arms shared these conditions.
- The 1,005 retained rerank receipts report $0.264496974. Routing/fetch charges
  were not included in these receipts, so this is not the total benchmark bill.
- This previously inspected development split is not a fresh holdout. Retrieval
  is per known paper, not corpus-wide document discovery. No answer generation
  ran: answer accuracy, query-embedding latency, request scheduling, and full chat
  latency are unmeasured. Larger context may increase later generation latency.

The remaining measured latency is dominated by the shared reranker. Removing
routing calls addresses the serial routing cost; it does not make the complete
retrieval-and-rerank path a two-millisecond operation.

## Local artifacts and audit

Artifacts remain outside Git under
`bench-runs/local-structure-2026-09-27/full-v1/`: `manifest.json`,
`observations.jsonl`, `completion.json`, `report.json`, `scoring.log`, and
`audit_scoring.py`. The manifest pins source/input/cache hashes, package versions,
models, budgets, and the full schedule. The scorer checked all 1,005 identities,
4,020 arm contexts, final context byte caps, finite timings, frozen inputs and
source hashes, and official evidence-F1 parity before writing the report.
Raw gold was used only in offline scoring after inference completed.

| Artifact | SHA-256 |
| --- | --- |
| Manifest | `898459437b17c582379d15972711daf1bb8b76575062c1cbed102593fa891346` |
| Observations | `909bf0a75cbd7680aa2e7df24de63c51a830ab32b23278fec1aaee63875ca298` |
| Completion | `8577594d58349229172ddb893b237940e08df494e4c6fb6b00fc4df037cb9c15` |
| Raw development data | `2ae7ee62a65b1c4225791c70de80c2aad4e8998cf1fd4f09a53103db4f21af93` |
| Official evaluator | `781aba7cd8e524bef4f0a1b4bf3504e5b02cb1d8d5bf32a8f0a89dfa83e86bfe` |
| Audit scorer | `8f87bf9fba1e2e8ccb583ae529f1171b5abd2109ef6d6bb0947471a106020c66` |
