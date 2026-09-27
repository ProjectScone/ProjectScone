# Compact paragraph candidates: standalone Jev results

The frozen [protocol](COMPACT_PROTOCOL.md) ran September 27, 2026 at commit
`5075d903`, retaining native retrieval from `90d3cdf6`. All 1,005 development
questions / 281 papers completed, with zero failed pairs. This follow-up was
motivated by the [local-only result](HYBRID_RESULTS.md); neither experiment was
tuned after its inference began.

Paragraph hybrid candidates followed by Jev improve evidence F1 from **20.13%
to 25.13%**, while reducing final context by **23.2%**. Recall falls from 64.66%
to 60.12%. Hosted reranking is **not faster**: median retrieval plus reranking is
346.26 ms versus 344.49 ms. Narrower evidence improves precision here; it does not
remove the hosted request's latency.

## Matched current run

Each arm independently retrieves at most 32 candidates / 8,000 source bytes and
sends its own Jev batch. Final contexts contain at most five items / 8,000 bytes
including headers. Query vectors are cached; no address/fetch decisions occur.
Arm order rotates per question, with eight concurrent questions within each paper.

| Measure | Section expansion + Jev | Paragraph hybrid + Jev |
| --- | ---: | ---: |
| Evidence F1 | 20.13% | 25.13% |
| Evidence recall | 64.66% | 60.12% |
| Text-only evidence F1 | 20.30% | 25.41% |
| Text-only evidence recall | 66.88% | 62.11% |
| Retrieval p50 / p95 | 2.00 / 5.88 ms | 2.16 / 7.31 ms |
| Reranking p50 / p95 | 342.47 / 447.56 ms | 343.74 / 479.95 ms |
| Retrieval + rerank p50 / p95 | 344.49 / 449.61 ms | 346.26 / 482.14 ms |
| Mean final context bytes | 4,619.34 | 3,546.41 |
| Mean reranked candidates | 10.53 | 15.23 |
| Rerank requests / receipts | 1,005 | 1,005 |
| Reported rerank cost | $0.126231 | $0.142998828 |

Paragraphs are narrower but fit more candidates into the same source-byte budget,
so the model makes more relevance judgments. This run supports improved evidence
F1, not a claim that compact candidates lower hosted reranking latency or cost.

The paired evidence F1 difference is **+5.01 percentage points [95% CI +3.95,
+6.13]**. Recall is **-4.54 points [-6.57, -2.45]**. Intervals use 2,000 paired
paper-cluster bootstrap draws, seed 20260924, are unadjusted, and do not capture
repeated-model variance. Quality uses complete source paragraphs consistently;
partial fragments receive no credit.

## Historical reference and runtime choice

The unchanged prior journal covers the same question identities. Paragraph hybrid
followed by Jev exceeds historical vector-guided routing in evidence F1 by +2.42 points
[+0.58, +4.21] and recall by +17.56 points [+13.97, +21.09]. Against the historical
LlamaIndex reference, F1 improves +4.95 points [+3.15, +6.69] and recall improves
+21.08 points [+17.53, +24.50]. These are development evidence comparisons with
the recorded runs, not general superiority claims about either framework.

Prior runs charged each arm a shared union-batch rerank time. This run times
independent batches, so the historical 381 ms versus current 346 ms does **not**
establish a speedup caused by the new retrieval policy. The current matched
section control at 344 ms makes that distinction explicit.

For a latency-sensitive caller, `mode='local_hybrid'` can supply locally ranked
evidence directly: the separate no-hosted-rerank experiment measured 1.77 ms median
with 19.07% evidence F1 / 47.39% recall. For higher evidence F1, retrieve up to 32
hybrid candidates and use the existing evidence reranker as this runner does.
Neither choice changes the default API policy or the running webapp pipeline.

## Scope and audit

All source/query embeddings were cached. Source/index preparation totaled 6.91
seconds and is excluded from query timing. Regression tests shared the machine
during this run; hosted provider timing and concurrent load limit generalization.
The corpus is a previously inspected development split and retrieval assumes the
paper is known. No generated answers were evaluated, so neither answer accuracy
nor full chat latency is established. Smaller context may affect later generation,
but that effect is unmeasured.

A subsequent [evidence representation audit](../answer_target/RESULTS.md)
identified caption-marker, heading, substring, and whitespace alignment limits in
the frozen evidence extraction policy. The scores above remain unchanged; they
must not be read as a complete audit of whether every relevant source fact was
available. Official evaluator parity verifies the scores for the extracted
predictions, not the completeness of that extraction policy.

Local artifacts remain under `bench-runs/compact-paragraph-jev-2026-09-27/full-v1/`.
The retained audit scorer verified all scheduled identities, 2,010 current arm
contexts, final/candidate byte bounds, pinned model receipts, frozen source/input
and prior hashes, and official evidence-F1 parity for both current and four
historical arms. Gold was read only after inference completed. Both experiments
and all prior artifacts are preserved, including the unfavorable recall results.

| Artifact | SHA-256 |
| --- | --- |
| Manifest | `5ec531b5432529a9b9affa9c0d46d0ad360bd3b6aae6c886e3b06a784ad32401` |
| Observations | `81b0b5a6b1e5a1858b501ab021f874c02cd4eb9fa35ba8b334e1b3465413fde1` |
| Completion | `f503508f0b9e7aabda220fe991668a046ec07c54a80377518cf43911558a4e00` |
| Audit scorer | `5fcc642841079299ff99fe5fc648f4c5395b2a446003b69395997d560a8070fd` |
