# LongMemEval-M, 100 items, same reader and judge: Scone 85% vs LlamaIndex 76%

## Run

- **Date and sample:** 28 September 2026. `stratified_sample(n=100, seed=42)` of LongMemEval-M (cleaned), with a
  median of 476 sessions per item. A full history is about 5 million characters, far past the reader's
  262k-token window, so there is no full-context arm. The retrieval system decides the answer.
- **Reader, judge and prompts:** the same as the [S run](RESULTS-s100.md): `google/gemma-4-31b-it` with upstream's
  step-by-step prompt, and `gpt-4o-2024-08-06` with upstream's judge prompts. Settings are in [PROTOCOL.md](PROTOCOL.md).
- **Embedder:** both systems ranked with `bge-base-en-v1.5`.
  - Vectors came from OpenRouter's hosted copy through the prefill stage (`prefill.py`), under the local model's id,
    window and tokenizer.
  - All 301,329+ ranking lookups were served from the cache, and none went to the model.
  - On a six-item check, rankings matched a local run exactly for both systems.
  - Texts past the 512-token window were clipped where the local model stops reading.
  - The hosted copy refused 2 of the roughly 950k embedded texts at that length (Korean text, which it tokenizes
    longer). These were shortened further, so those 2 vectors differ from the local ones.
- **Raw files:** `bench-runs/longmemeval-matched-2026-09-27/m100-bge-base-cot/`.

## Answers

| Arm | Accuracy | Wilson 95% | All evidence in top k | Median prompt tokens |
| --- | ---: | --- | ---: | ---: |
| oracle (evidence sessions only) | 91.0% | 84–95 | by construction | 6,287 |
| **scone@10** | **85.0%** | 77–91 | **89.1%** | 29,508 |
| llamaindex@10 (vector + BM25 fusion) | 76.0% | 67–83 | 81.5% | 29,992 |
| **scone@5** | **78.0%** | 69–85 | **78.3%** | 15,103 |
| llamaindex@5 | 70.0% | 60–78 | 67.4% | 14,897 |

Paired on the same items:

| Comparison | Scone wins | LlamaIndex wins | Two-sided sign test |
| --- | ---: | ---: | --- |
| scone@10 vs llamaindex@10 | 12 | 3 | p = 0.035 |
| scone@5 vs llamaindex@5 | 14 | 6 | p = 0.115 |
| scone@10 vs oracle | 2 | 8 | — |

### By question type

| Type | n | oracle | scone@5 | scone@10 | llamaindex@5 | llamaindex@10 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| multi-session | 22 | 95% | 55% | **73%** | 41% | 59% |
| temporal-reasoning | 24 | 92% | 75% | **92%** | 62% | 71% |
| knowledge-update | 15 | 93% | 93% | 93% | 93% | 93% |
| single-session-user | 14 | 100% | 93% | 100% | 86% | 93% |
| single-session-assistant | 11 | 100% | 100% | 100% | 100% | 100% |
| single-session-preference | 6 | 67% | 50% | 50% | 50% | 33% |
| abstention | 8 | 62% | 88% | 62% | 75% | 75% |

**Failed generations:** 1 oracle, 1 scone@5, 2 scone@10, 2 llamaindex@5, 3 llamaindex@10. All of them hit the
reader's 800-token cap and are scored as incorrect.

## What it shows

- **At about 480 sessions per history, retrieval decides the answer, and Scone's defaults retrieve better than
  LlamaIndex's best configuration.** Scone puts all the evidence in the top 10 for 89.1% of answerable items,
  against 81.5%. At top 5 it is 78.3% against 67.4%. At top 10 the evidence coverage is 9 wins to 2.
- **The retrieval gap carries through to answers.** Accuracy is +9 points at top 10 and +8 points at top 5.
  - The gain is concentrated where evidence is spread across sessions: +14 points on multi-session and +21 points on
    temporal reasoning at top 10.
  - With the same reader, scone@10 comes within 6 points of the oracle.
- **The two systems use the same prompt budget.** Median prompt tokens are within 2% of each other at each k.
- **Speed:** Scone's recall takes a median of 98 ms per question over about 11k chunks. The LlamaIndex figure
  (4.7 s median) is not comparable, because it includes building its index for every item.
- **Contrast with LongMemEval-S.** On S, the whole history fits in the reader's window and the two systems tied. The
  difference only shows once the memory is too large to read.

## Limits

- **Sample size.** This is one sample of 100 items. At top 5 the interval still admits no difference (p = 0.115).
  Claiming a ranking needs all 500 items, a second judge from another model family, and repeated runs.
- **Configurations compared.** Scone runs its engine defaults. LlamaIndex runs its own defaults (512-token
  `SentenceSplitter` chunks) with its BM25 retriever fused in. Neither was tuned for this benchmark.
- **One reader and one embedder.** The embedder is bge-base. Stronger embedders or small local readers may change the
  gap.
- **Other systems.** Mem0, Graphiti and Mastra have not been run in this harness yet.
