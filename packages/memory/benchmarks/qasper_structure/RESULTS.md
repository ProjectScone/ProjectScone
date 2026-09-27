# Full QASPER evaluation: better evidence, no clear answer-accuracy win

The structure-address layer improves evidence retrieval over the matched
LlamaIndex hybrid configuration and Scone's native vector control. Its answer F1
lead is small and its paired confidence interval crosses zero. It also adds about
1.08 seconds to median query stage time. This supports keeping the feature
experimental and opt-in; it does not establish that Scone beats LlamaIndex overall.

## Complete test run

Executed the entire official QASPER v0.3 test split: **416 papers, 1,451 questions,
4,353 scheduled observations, zero missing**. No question was sampled out, no
successful answer was regenerated, and the run required no resume. Two structure
arm answers hit the 256-output-token limit; they count as zero on every quality
metric. The other 4,351 generations completed successfully.

Frozen implementation: `37a97771` on `feature/qasper-structure-evaluation`.
Run completed September 25, 2026. Original split, exported inputs, native source,
benchmark source, protocol and completion artifact hashes were verified. All three
arms' answer F1 and both evidence F1 variants match the **official QASPER evaluator**
to numerical tolerance below 1e-12. Failed generations are omitted from official
prediction exports to reproduce the explicit zero-failure penalty; they remain
present in the complete observation journal and every reported denominator.

All arms used the same source-aligned chunks (700-character target), shared Nemotron 3
Embed 1B API vectors, shared OpenRouter Jev relevance judgments, paid Gemma 4 31B
answer model and context limits. The reference is installed LlamaIndex vector +
BM25 reciprocal-rank fusion. This is **known-paper, text-and-caption QA**, not
cross-corpus discovery. No local model inference was used. See
[PROTOCOL.md](PROTOCOL.md) for the frozen task and resource limits.

## Quality

Percentages below include all 1,451 questions in each arm.

| Arm | Answer F1 | Normalized EM | Evidence F1 | Evidence recall | Failed generations |
| --- | ---: | ---: | ---: | ---: | ---: |
| Scone vector control | 53.5295% | 26.3267% | 24.4826% | 47.9668% | 0 |
| Scone structure addresses | 53.9327% | 26.5334% | 25.4865% | 49.4141% | 2 |
| LlamaIndex hybrid | 53.7454% | 26.7402% | 24.1131% | 47.1294% | 0 |

Paired paper-cluster bootstrap, 2,000 resamples, seed 20260924. Deltas and
95% percentile intervals are **percentage points**, not relative percentages.

| Structure minus baseline | Answer F1 delta [95% CI] | Evidence F1 delta [95% CI] | Evidence recall delta [95% CI] |
| --- | ---: | ---: | ---: |
| Scone vector control | +0.4033 [-0.3365, +1.1371] | +1.0040 [+0.3045, +1.6717] | +1.4473 [+0.5659, +2.3566] |
| LlamaIndex hybrid | +0.1873 [-0.6889, +1.0312] | +1.3734 [+0.5481, +2.2312] | +2.2847 [+1.0123, +3.5278] |

Structure answer F1 beats LlamaIndex on 258 questions, loses on 251, and ties on
942. Its normalized EM is 0.2068 points lower. The evidence intervals exclude zero;
the answer F1 and EM intervals do not. Intervals are unadjusted for multiple
metrics and reflect this one configuration and stochastic generation run. They
measure paper-sampling uncertainty, not variance across repeated model generations.

Evidence is measured as complete original paragraphs present in the final packed
context, rather than citations selected by the answer model. Split or clipped
paragraphs receive no credit. Primary evidence metrics retain figure/table gold
entries that text retrieval cannot return. In the official text-only variant:

| Arm | Text evidence F1 | Text evidence recall |
| --- | ---: | ---: |
| Scone vector control | 25.0147% | 49.3989% |
| Scone structure addresses | 26.1483% | 50.9784% |
| LlamaIndex hybrid | 24.5429% | 48.3491% |

## Latency and resources

Query vectors were precomputed and shared. Stage sums include retrieval, the
shared Jev rerank call and generation; they exclude index/query-vector preparation,
benchmark scheduling waits, and UI/network transport to an application client.
They are not measured production chat latency. Routing was uncached per question.

| Arm | Retrieval median / p95 | Generation median / p95 | Query stage-sum median / p95 |
| --- | ---: | ---: | ---: |
| Scone vector control | 1.8 / 5.2 ms | 932.0 / 3,686.0 ms | 1,385.4 / 4,119.1 ms |
| Scone structure addresses | 998.7 / 1,866.4 ms | 926.0 / 3,656.5 ms | 2,469.1 / 5,429.3 ms |
| LlamaIndex hybrid | 6.7 / 12.5 ms | 920.5 / 3,488.9 ms | 1,383.8 / 4,015.1 ms |

Shared reranking: 394.6 ms median, 681.5 ms p95. Structure routing: 724.3 ms
median; fetch selection: 253.1 ms median, including zeros when selection is skipped.
Medians of individual stages need not sum to the median of their sums.

Shared vector preparation took **620.5 seconds** across 176 API requests for
20,963 distinct passage texts and 1,451 question records. Complete run wall time
was approximately **75.3 minutes**, including preparation and scheduling.

Mean final context bytes were 2,727.7 (native control), 2,755.0 (structure), and
2,735.3 (LlamaIndex); every arm stayed within five items and 8,000 bytes.

Physical consumption: 1,451 shared rerank requests, 4,353 answer requests,
2,661 route attempts, and 731 successful fetch decisions. Provider-reported
charges total **$0.63750768 for embedding, shared reranking and generation only**.
Routing/fetch receipts retain token counts, but not monetary charges, so this is
**not the total bill**. Routing receipts recorded 1,835,690 input / 203,428 output
tokens; fetch receipts recorded 371,829 input / 29,891 output tokens. Invalid or
timed-out routing responses may have incurred additional unrecorded usage.
Shared rerank cost was counted once physically, not once per arm.

## Routing behavior and next work

The structure arm returned original sections on 74 questions (5.10%), scoped
vectors on 655 (45.14%), and broad vectors on 722 (49.76%). Final reasons:

| Reason | Questions |
| --- | ---: |
| Jev selected a retrieval mode | 725 |
| No matching address | 614 |
| Invalid routing response | 101 |
| Original selection exceeded context budget | 6 |
| Routing timeout | 3 |
| Fetch decision failed | 2 |

The broad-vector fallback worked, but the current router frequently adds work
without narrowing retrieval. Next experiments should diagnose the 101 invalid
responses, test cheaper routing gates, and improve selection using vector-hit
section addresses. Develop those policies on a separate development split and
freeze them before another full evaluation. Do not tune against these test answers
and present a later rerun as an untouched test.

## Local audit artifacts

Artifacts remain local and are not committed:
`bench-runs/qasper-structure-2026-09-24/full-v1/`.
Key files: `manifest.json`, `completion.json`, `observations.jsonl`,
`prepared.jsonl`, `judgments.jsonl`, `embedding-calls.jsonl`, `report.json`, and
`official-parity.json`. The original gold file is in the sibling `raw/` directory;
model inference used only the separate gold-free `dataset-full/` export.

| Artifact | SHA-256 |
| --- | --- |
| Original test JSON | `6e29ad410e6e39aa1936017fb965b30a20eb2e7751997f55b97c9d281aa884e5` |
| Run manifest | `6709c0cc4cfe0377a413f9c0a78b412cbd05e8bf89b3c22f2bf993414809a476` |
| Observation journal | `d7d35f2bf52aa6fd8844cc8ce51ce007c069b4c73f1cd4eaf510f5ab4eb1a793` |
| Completion audit | `2aaef0bde5949115bfcf0eddf9caf9a439be4ccc1020616a122c66a267961726` |
