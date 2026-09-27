# Complete answer results: Scone versus LlamaIndex

All **17,975 questions now have answers from both systems**: 35,950 successful
answers, zero missing answers. The OpenRouter recovery supplied the 6,966
missing answers and preserved all 28,984 original successful answers exactly
as JSON values. Full-schedule scoring and artifact/input integrity checks passed.

| Evaluation | Scone EM | LlamaIndex EM | Scone F1 | LlamaIndex F1 | Missing answers per arm |
| --- | ---: | ---: | ---: | ---: | ---: |
| HotpotQA, all 7,405 questions | 49.60% | 50.05% | 61.53% | 61.83% | 0 |
| SQuAD, all 10,570 questions | 75.77% | 75.99% | 84.75% | 84.77% | 0 |
| Combined, all 17,975 questions | 64.99% | 65.30% | 75.19% | 75.32% | 0 |

**Scone remains slightly behind in answer quality for this configuration.**
The combined exact-match difference is -0.312 percentage points: 269 paired
Scone wins, 325 losses and 17,381 ties. The descriptive paired-question
bootstrap 95% interval is [-0.562, -0.056] percentage points. Topic dependence
and generation variability limit interpretation; this is not a universal
ranking of the frameworks.

Scone's combined candidate document recall at 32 is 96.02% versus 95.72%.
Context document recall at five is 90.57% versus 90.34%. The small coverage
advantage did not produce an answer-quality win. Among 10,823 pairs with
identical generation requests, raw answers differed on 418; exact match
favored Scone on 74 and LlamaIndex on 108. Some observed answer differences
therefore reflect provider generation variability rather than retrieval.

### Recovery provenance

This is a **composite evaluation with an explicit provider switch**. Original
successful questions used direct Jev `jev-1.13.0`; the 3,483 previously blocked
question pairs used OpenRouter `typesafe/jev-1.13-20260917`, as authorized by
the user. Both arms shared each reranking batch. Saved candidates, their order,
relevance rubric, context limits and paid Gemma generation parameters were
preserved. This does not establish identical behavior across provider deployments.

OpenRouter returned one HTTP 529 during recovery. The runner stopped, and an
explicit checkpoint resume finished the remaining work without repeating any
completed answer. The rejection remains in the provider error journal; zero
missing answers does not mean zero historical request errors.

Recovery Jev receipts report 29,353,323 input tokens, 2,560,347 output tokens
and $1.232840. Recovery generation reports 4,842,596 prompt tokens, 46,626
completion tokens and $0.549732. These amounts supplement the original usage
below. Reused retrieval timings and later API timings must not be presented
as a new end-to-end wall-clock measurement.

Verified local evidence: `bench-runs/llamaindex-full-2026-09-24/recovery-openrouter-1/`.

- Manifest SHA-256: `0e10bb2106689f5b1b9a710a3e928d9180b7ae1d08b4e7c66b76692baa1b5ace`
- Completion SHA-256: `134c7a5b033f29d82e16e0118b4e57214d438150fc596d1d2e373efb59667c05`

See [RECOVERY.md](RECOVERY.md) for the frozen recovery protocol. The original
attempt-level findings below remain preserved for availability and cost auditing;
their zero-filled failed answers are superseded by the recovered quality table above.

## Original run before recovery

The original September 24, 2026 run did **not** demonstrate a Scone win. Every
scheduled question was attempted, but a provider failure prevented a complete
answer-quality evaluation. This report preserves those failures rather than
presenting the successful subset as the full dataset.

## Coverage and integrity

- All 7,405 HotpotQA development and 10,570 SQuAD 1.1 development questions:
  17,975 questions, 35,950 terminal arm outcomes, no sampling.
- 68,702 pooled paragraphs, including distractors; 86,369 indexed chunks
  with 86,325 distinct chunk texts shared between frameworks.
- 14,492 questions produced answers in both arms (28,984 generated answers).
- The final 3,483 SQuAD questions failed shared direct-Jev reranking with
  HTTP 402 in both arms. They produced no generated answers. The saved failure
  receipt contains the HTTP code, not a provider billing explanation; the
  underlying account condition has not been verified.
- All 3,260 embedding requests returned HTTP 200. All generated answers used
  paid `google/gemma-4-31b-it`; successful Jev responses identify `jev-1.13.0`.
- Strict scoring verified every planned question/arm exactly once and all
  recorded artifact/input hashes. The runner verified unchanged source/input
  hashes. Its `completed` flag means the schedule ended, not that every request
  succeeded. The progress file's last periodic count, 35,940, is stale; the
  validated observation count is 35,950.

## Answer and retrieval results

Percentages below retain failed answers as zero, as specified before inference.
The SQuAD and combined answer scores are therefore affected by provider
availability and must not be described as a successful full-dataset evaluation.

| Evaluation | Scone EM | LlamaIndex EM | Scone F1 | LlamaIndex F1 | Failures per arm |
| --- | ---: | ---: | ---: | ---: | ---: |
| HotpotQA, all 7,405 questions | 49.60% | 50.05% | 61.53% | 61.83% | 0 |
| SQuAD, all 10,570 scheduled questions | 51.11% | 51.22% | 56.82% | 56.80% | 3,483 |
| Combined, all 17,975 scheduled questions | 50.49% | 50.74% | 58.76% | 58.87% | 3,483 |

On HotpotQA, Scone won 89 exact-match pairs and lost 122; 7,194 tied.
The Scone-minus-LlamaIndex EM difference was -0.446 percentage points, with a
descriptive paired-question bootstrap 95% interval of [-0.810, -0.054] points.
Topic dependence and provider generation variation limit this interval's
interpretation. This is a small observed loss for this configuration.

Across all scheduled questions, candidate document recall at 32 was 96.02%
for Scone versus 95.72% for LlamaIndex. On fully answered HotpotQA, context
document recall at five was 83.36% versus 83.32%. Slightly better candidate
coverage did not translate into better answer accuracy.

Of 14,492 successfully answered pairs, 8,828 had identical generation requests.
Their raw answers differed on 327 pairs; exact-match scoring favored Scone on
61 and LlamaIndex on 85. Separate temperature-zero calls were not deterministic.
Not every answer difference can be attributed to retrieval.

## Timing and recorded usage

HotpotQA has no failed requests and supplies the cleanest timing comparison:

| HotpotQA stage | Scone median / p95 | LlamaIndex median / p95 |
| --- | ---: | ---: |
| Retrieval | 331 / 566 ms | 153 / 342 ms |
| Shared Jev batch | 320 / 646 ms | 320 / 646 ms |
| Generation | 725 / 2,981 ms | 735 / 3,091 ms |
| Sum of measured stages | 1,468 / 3,725 ms | 1,287 / 3,632 ms |

Scone's median retrieval was approximately 2.16 times LlamaIndex's. These
measurements use prewarmed query vectors and four concurrent jobs. The shared
Jev batch is charged in full to each arm's stage sum; it is not independent
per-arm work. These are neither cold-query nor webapp time-to-first-token measurements.

Preparation took 6,691 seconds in this process. Manifest creation to completion
was approximately 6 hours 40 minutes. Provider receipts recorded:

- Embeddings: 9,967,931 tokens; reported cost $0.111014.
- Generation, both arms combined: 20,005,973 prompt tokens and 170,060
  completion tokens; reported cost $2.208721.
- Successful shared Jev batches, counted once: 121,879,847 input tokens and
  10,381,459 output tokens. No monetary cost was included in those receipts.

The OpenRouter amounts exclude Jev and are not the total experiment cost.

## Evidence and next work

Local evidence is retained under
`bench-runs/llamaindex-full-2026-09-24/run-1/`, including exact requests,
observations, judgments, provider usage, manifest, completion receipt and
offline `scores.json`. Generated artifacts are not committed.

- Manifest SHA-256:
  `7043f14b69ccdd8e31313e6e8b1a62ed1bbb1d34ba3277c05a6c577d9161a116`
- Completion receipt SHA-256:
  `8a22279f57519e5ea267e3899a0c49af50980207faf1fdaeec82251a879b6346`

The separate recovery above resolved missing-answer coverage and added a
provider-failure stop while preserving this original run. Next, profile
Scone's retrieval path, then investigate evidence selection and answer
errors under a separately frozen protocol. Do not tune on these results and
call the same data an untouched holdout.

This tests hybrid retrievers with shared chunks, vectors, reranking and answer
generation. It does not compare every framework configuration, native ingestion,
full Wikipedia, or official leaderboard task settings. See [PROTOCOL.md](PROTOCOL.md).
