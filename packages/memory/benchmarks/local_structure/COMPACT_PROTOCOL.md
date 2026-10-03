# Compact paragraph candidates with standalone Jev: development experiment v1

The completed local-only experiment motivates this second, separate comparison.
It showed improved evidence F1 over flat vectors without reranking, but lower
recall than the prior Jev-reranked expansion arm. Preserve those results unchanged.
This is another development experiment on the previously inspected 281 papers /
1,005 QASPER questions, not a holdout. Freeze before inference; no tuning afterward.

Compare `structure_jev` (local section expansion) and `hybrid_jev` (local paragraph
vector/BM25 fusion), using the frozen native implementation at commit 90d3cdf6.
Each independently retrieves at most 32 candidates / 8,000 source bytes, deduplicates
identical text, and sends its own batch to the existing pinned
`typesafe/jev-1.13-20260917` through the previously authorized OpenRouter endpoint.
Keep the existing evidence relevance prompt and final five-item / 8,000-byte
context packing. No routing/fetch calls or answer generation. No confidence gate.

The two arms make independent requests: report their actual separate retrieval
and reranking timings, not a shared union-batch time. Rotate arm order by question
ordinal. Eight concurrent questions per paper; papers sequential; one reused HTTP
client. Report provider noise and concurrent host load. Both arms share the same
cached Nemotron vectors; abort any cache miss, make no new embedding requests.
Source/index preparation is outside warm query timing. Larger final contexts can
affect unmeasured generation latency. Old shared-batch latency remains historical.

Persist both arm contexts, separate model receipts, timings (including failures),
and source/input/cache/model/schedule hashes locally. Refuse nonempty output and
concurrent writers. No resume. Recheck source, input and prior artifact hashes at
completion. A rerank failure marks the paired question failed: count it in the
denominator and assign zero evidence quality to both arms, retaining elapsed time.

Only after all inference completes, score complete-paragraph all-evidence and
text-only F1/recall. Audit all scheduled identities and final byte bounds; verify
official evaluator parity. Compare paired question quality and paper-cluster
bootstrap intervals (2,000 draws, seed 20260924), unadjusted and without repeated
model variance. Report both arms even if the new one loses. No automatic default
change or claim about answer accuracy or full chat latency.
