# Public cross-encoder retrieval comparison v1

**Decision: keep this reranker optional.** It slightly improved prepared Hotpot
annotation coverage, regressed top-10 supporting-document coverage, and added
roughly 1.5 seconds to median preparation on this machine. A larger candidate
pool alone did not improve the measured coverage. No generation was run here.

## Frozen comparison

The [protocol](public-reranking-v1.protocol.md) was committed as `fe61827` before
evaluation. All 200 unchanged development questions from
[public-qa-v1](public-qa-v1.results.md) ran through three configurations, producing
600 observations. Each observation includes one production context preparation
and one separate native recall. All completed; no replacement retries, question
rewrites, gold filters, seeded facts or fine-tuning were used. The reserved 200
questions remain unrun.

The run used copied SQLite and Qdrant 1.19.1 stores containing the same 2,176
original paragraphs, the cached BGE-small-en-v1.5 embedder, and Python 3.14. The
reranker was the already provisioned `Xenova/ms-marco-MiniLM-L-6-v2`, revision
`a09144355adeed5f58c8ed011d209bf8ee5a1fec`, running in process on CPU with two
threads, batches of eight, and a 512-token full-pair cap.

The configurations were:

- **Legacy:** existing fusion. Context recall requests 20 items, giving an
  implicit candidate depth of 80 per lane. The separate top-10 ranking call has
  an implicit depth of 40.
- **Pool 128:** candidate depth 128 per lane, without reranking.
- **Rerank 64:** candidate depth 128, rerank up to 64 candidates within 128,000
  payload bytes and a 10-second deadline.

Every configuration used five final source anchors, an 8,000-byte context budget,
and no neighboring-passage expansion. Arm order rotated by question. Actual
store copies, embedding and reranker artifacts, original baseline requests,
questions, corpus and package source were fingerprinted and checked unchanged.

## Retrieval results

HotpotQA, 100 questions:

| Metric | Legacy | Pool 128 | Rerank 64 |
|---|---:|---:|---:|
| Supporting-document recall @5 | 87.50% | 87.50% | 88.00% |
| All supporting documents @5 | 75% | 75% | 76% |
| Supporting-document recall @10 | 97.00% | 97.00% | 96.50% |
| All supporting documents @10 | 94% | 94% | 93% |
| Prepared annotation coverage, macro | 86.80% | 86.80% | 87.23% |
| All annotations in prepared context | 72% | 72% | 74% |

The net gain of two fully covered Hotpot questions contains **nine gains and
seven losses**. At top 5, complete-document coverage has nine gains and eight
losses; at top 10, two gains and three losses. The small net changes conceal
real regressions and do not establish a general ranking improvement.

For all 100 SQuAD questions, document recall and complete-document coverage at
both cutoffs stayed at 100%; prepared answer-presence stayed at 99%. Pool 128
produced no per-question coverage changes on either dataset.

All 200 repeated legacy requests, retained source sequences, ranked document
sequences and context byte counts match the original baseline. Pool 128 matches
195 original requests/source sequences and 103 ranked document sequences;
reranking matches none of those complete request or ranked sequences. Thus
unchanged coverage does not mean that the evidence order stayed unchanged.

## Runtime and fallbacks

All 600 contexts were prepared and all 600 native recalls completed. All 400
reranker invocations were applied: 200 in context preparation and 200 in native
recall. There were no overlong pairs, ranking failures, deadline fallbacks or
degraded retrieval lanes in this run.
Each reranker call received 64 candidates. The longest observed query/passage
pair was 249 tokens, well within the configured 512-token cap.

Across 200 questions per configuration:

| Measurement | Legacy | Pool 128 | Rerank 64 |
|---|---:|---:|---:|
| Context preparation p50 | 26.81 ms | 27.08 ms | 1,571.11 ms |
| Context preparation p95 | 40.16 ms | 40.43 ms | 2,189.20 ms |
| Separate native recall p50 | 23.62 ms | 24.95 ms | 1,595.46 ms |
| Separate native recall p95 | 34.67 ms | 36.49 ms | 2,190.36 ms |
| Context bytes, mean | 3,578.72 | 3,576.98 | 3,552.04 |
| Context bytes, median | 3,603.5 | 3,603.5 | 3,598 |
| Context bytes, p95 | 4,230 | 4,225 | 4,230 |
| Context bytes, maximum | 4,473 | 4,473 | 4,576 |

Timings exclude model construction. Pair-length diagnostics run outside both
request timers and production deadlines. Context preparation and native recall
are separate calls, not additive measurements of a single request. The host was
not dedicated: editing and short type/test checks occurred in a separate
checkout while inference ran. Broader refactor tests ran after measurement.
Rotating order reduces systematic ordering effects but does not eliminate cache
or host-activity effects. These are observed CPU timings, not production SLOs.

## Limits and follow-up

These are retrieval annotation metrics, not answer accuracy, semantic confidence
or supporting-fact/joint generation scores. Literal quote coverage retains the
baseline scorer's known chunk-boundary limitations. One model, one candidate
configuration and 200 development questions cannot establish what every
cross-encoder or deployment will do.

The wider pool did not solve the remaining failures, and this reranker trades
gains against losses. Further work should inspect missing evidence and multi-hop
selection rather than assume more candidates or a larger model guarantees a
better answer. Any new configuration needs a separately frozen comparison;
generation effects also remain unmeasured for these reranked contexts.

## Audit artifacts

Raw sources, requests and results remain outside git under
`bench-runs/public-reranking-dev-2026-09-08/`. The initial post-run report had a
reporting-only aggregation error in request/byte equality fields and omitted
byte/degradation aggregates; it is preserved as
`scores-initial-reporting-bug.json`. Those fields were corrected and the same
untouched observations rescored. Retrieval metric definitions and values did
not change.
An independent audit rebuilt all 600 row metrics and all nine aggregate groups
from the raw observations and labels and reproduced the final report exactly.

- Manifest SHA256: `2775ed55840225bf2fe53c5c12b04f1a829c81c64ebe7a28e324cf2aab88b822`
- Observations SHA256: `7649ceb7d32ebb32ea0231620c7d8bb73ddb557ace3f54698d3464b7eb1ccfac`
- Final scores SHA256: `84811d21ce8a8787eb1492ed4b287ac42433223a06c806584e56143481796c5f`
- Scorer SHA256: `c136a8153d41b051b5e32816ebec71c906108a02aafd0d5b42e16762a754f603`
