# Local paragraph ranking: development results

The frozen [protocol](HYBRID_PROTOCOL.md) ran September 27, 2026 at commit
`90d3cdf6`, on all 1,005 questions / 281 papers from the previously inspected
QASPER development export. All questions completed. No hosted requests were made;
all 17,630 embedding lookups reused the local cache.

`local_hybrid` combines vector-projected paragraphs and native BM25 using
reciprocal rank fusion. It takes **1.77 ms median / 2.71 ms p95**, including final
context packing, and improves evidence F1 over flat retrieval without reranking.
It loses evidence recall compared with the earlier hosted-reranked section
expansion policy. It is an explicit latency/quality choice, not a replacement for
every use of a reranker.

## Current run: no hosted reranking

Evidence metrics are percentages; times are milliseconds. All arms share cached
query/source vectors, final five-item / 8,000-byte bounds, serial execution, and
rotating arm order. Paragraph indexing adds a local lexical lane and changes the
retrieval unit; candidate sets are intentionally different.

| Arm | Evidence F1 | Evidence recall | Retrieval + packing p50 / p95 | Mean context bytes |
| --- | ---: | ---: | ---: | ---: |
| Flat vectors | 16.40 | 30.59 | 1.60 / 2.57 | 2,456.79 |
| Section expansion | 17.11 | 55.60 | 1.62 / 2.66 | 4,382.42 |
| Paragraph hybrid | 19.07 | 47.39 | 1.77 / 2.71 | 3,395.69 |

Text-only evidence F1 / recall were 16.98 / 32.07 for flat vectors, 17.26 / 57.54
for section expansion, and 19.29 / 49.10 for paragraph hybrid.

Hybrid minus flat F1 is **+2.68 percentage points**, with paired paper-bootstrap
95% interval **[+1.04, +4.30]**. Recall improves +16.80 points [+13.28, +20.18].
Versus section expansion without reranking, F1 improves +1.96 points
[+1.02, +2.86], while recall falls -8.21 points [-10.39, -5.96].

## Historical hosted comparison

The exact same question identities were paired with the unchanged prior
[four-arm journal](RESULTS.md), which used shared Jev reranking. This supports a
paired evidence comparison, but its latency is from a different run and shared
batch workload. Do not calculate a causal latency speedup from these numbers.

| Historical arm with shared Jev rerank | Evidence F1 | Recall | Retrieval + rerank p50 |
| --- | ---: | ---: | ---: |
| Flat vectors | 20.50 | 40.55 | 379.69 ms |
| Vector-guided routing | 22.72 | 42.55 | 1,006.26 ms |
| Section expansion | 20.10 | 64.71 | 380.92 ms |
| LlamaIndex | 20.19 | 39.04 | 382.67 ms |

Local hybrid minus historical section expansion F1 is -1.03 points
[-2.11, +0.08]; the interval includes zero and does not establish equivalence.
Recall falls -17.33 points [-19.73, -14.77]. Against historical vector-guided
routing, F1 falls -3.64 points [-5.58, -1.79], while recall improves +4.84 points
[+1.00, +8.82]. Removing hosted ranking is therefore not a quality-neutral change.

## Scope and audit

Index/source preparation totaled 4.97 seconds and is excluded from query times.
Query embeddings are cached. No answers were generated; neither answer accuracy
nor full chat latency was measured. This is per-known-paper retrieval on an
inspected development split, not corpus-wide discovery or a fresh holdout.
Intervals use 2,000 paired paper-cluster draws, seed 20260924, are unadjusted, and
do not capture repeated-model variance. Complete source paragraphs define evidence
credit consistently for every arm; fragments receive none.

Local artifacts are under `bench-runs/local-paragraph-hybrid-2026-09-27/full-v1/`.
The retained `audit_scoring.py` verified source/input/prior hashes, all 1,005
identities, 3,015 local contexts, byte caps, zero local model-decision timings,
and official all-evidence/text-only F1 parity for all seven local/historical arms.
Gold was read only after inference completed. Previous artifacts are unchanged.

| Artifact | SHA-256 |
| --- | --- |
| Manifest | `b01ec804f4900b5b1a41e1e6e8b45d7730232bd869567eca7bb7eb61ed475742` |
| Observations | `a81d9aa029b24ee00ed35f67a5ce685d751ce11d6843efbf8d912044731be82d` |
| Completion | `3e682bc0b27e0af4facdd5fa2a4dba9cf684238505f1ebfca152ddfc74f5588a` |
| Audit scorer | `f8266e3c87d9c9e76b422cf11866a6d181481186755bae6ab42749b537d7b58b` |
