# LongMemEval-M, all 500 items: Scone 80.6% vs LlamaIndex 77.2% with the same reader and judge

## Run

- **Date and data:** 28 September 2026, every item of LongMemEval-M (cleaned). The dataset is read one item at a time
  from JSONL.
- **Embedder:** both systems ranked with `bge-base-en-v1.5`, filled through the hosted-copy prefill.
- **Clock:** every item ranked with the engine clock at its question date.
- **Settings:** in [PROTOCOL.md](PROTOCOL.md).
- **Raw files:** `bench-runs/longmemeval-matched-2026-09-27/m500-bge-base-cot/`.

## Answers (completed 2 October 2026)

- **Spending cap and resume:** the OpenRouter key's monthly cap had refused every judge request and 50 answers. After
  the cap reset, the resume asked those requests again and kept everything already answered.
- **Reader and judge:** every arm used the same reader (`google/gemma-4-31b-it`, upstream's step-by-step prompt) and
  the same judge (`gpt-4o-2024-08-06`, upstream's judge prompts).
- **Completeness:** all 2,500 answers were judged, with 0 judge errors.
- **Failed generations:** answers that hit the reader's 800-token cap are scored incorrect. There were 5 for oracle,
  5 for scone@5, 11 for scone@10, 9 for llamaindex@5 and 8 for llamaindex@10.
- **Cost:** the completion cost $2.46 in provider charges.

| Arm | Accuracy | Wilson 95% | All evidence in top k |
| --- | ---: | --- | ---: |
| oracle (evidence sessions only) | 93.6% | 91.1–95.4 | by construction |
| **scone@10** | **80.6%** | 76.9–83.8 | 84.7% |
| llamaindex@10 (vector + BM25 fusion) | 77.2% | 73.3–80.7 | 81.9% |
| **scone@5** | **76.2%** | 72.3–79.7 | 75.5% |
| llamaindex@5 | 72.0% | 67.9–75.8 | 71.1% |

Paired on the same 500 items:

| Comparison | Scone only | LlamaIndex only | Sign test |
| --- | ---: | ---: | --- |
| scone@5 vs llamaindex@5 | 43 | 22 | p = 0.013 |
| scone@10 vs llamaindex@10 | 37 | 20 | p = 0.033 |

### By question type

| Type | n | oracle | scone@5 | scone@10 | llamaindex@5 | llamaindex@10 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| temporal-reasoning | 127 | 94.5% | 71.7% | **82.7%** | 70.1% | 74.8% |
| multi-session | 121 | 93.4% | **57.9%** | **63.6%** | 48.8% | 61.2% |
| knowledge-update | 72 | 94.4% | 87.5% | 90.3% | 86.1% | 87.5% |
| single-session-user | 64 | 100.0% | **90.6%** | **93.8%** | 81.2% | 85.9% |
| single-session-assistant | 56 | 98.2% | 100.0% | 100.0% | 100.0% | 100.0% |
| single-session-preference | 30 | 73.3% | 60.0% | 46.7% | 56.7% | 60.0% |
| abstention | 30 | 86.7% | 83.3% | 86.7% | 83.3% | 83.3% |

### Reading

- **Scone answers more questions correctly than LlamaIndex at both depths, and the paired differences are
  significant.** It also puts every evidence session in context more often: 75.5% vs 71.1% at top 5, and 84.7% vs
  81.9% at top 10.
- **Where it gains:** single-session-user, temporal reasoning (+7.9 at top 10) and multi-session (+9.1 at top 5).
- **Where it does not:** LlamaIndex is ahead on preference questions at top 10, but there are only 30 of them.
- **Headroom:** the oracle shows 13 points left at top 10. scone@10 answered 13 items the oracle missed and missed
  78 it answered. Most of the gap is in multi-session and temporal questions, where evidence is spread over several
  sessions.
- **Against the 100-item sample:** the sample showed +9 points at top 10 (85% vs 76%). The full set gives +3.4.
  Samples are not results.

## Retrieval: all evidence sessions within the top k

These are the 470 items that have evidence (the 30 abstention items are excluded).

| k | Scone | LlamaIndex (vector + BM25 fusion) | Paired (Scone–LlamaIndex) | Two-sided sign test |
| --- | ---: | ---: | --- | --- |
| 5 | **75.5%** | 71.1% | 38–17 | p = 0.0065 |
| 10 | **84.7%** | 81.9% | 27–14 | p = 0.06 |

By question type:

| Type | n | Scone@5 | LlamaIndex@5 | Scone@10 | LlamaIndex@10 |
| --- | ---: | ---: | ---: | ---: | ---: |
| multi-session | 121 | 59.5% | 51.2% | 75.2% | 70.2% |
| temporal-reasoning | 127 | 65.4% | 65.4% | 76.4% | 74.8% |
| knowledge-update | 72 | 93.1% | 88.9% | 95.8% | 94.4% |
| single-session-user | 64 | 90.6% | 81.2% | 96.9% | 89.1% |
| single-session-assistant | 56 | 100.0% | 100.0% | 100.0% | 100.0% |
| single-session-preference | 30 | 63.3% | 56.7% | 76.7% | 80.0% |

Scone's recall latency over each item's roughly 11k chunks: median 104 ms, p95 210 ms.

## What changed from the 100-item sample

**The full set confirms Scone's retrieval lead, but at less than half the size the sample showed:**

| k | 100-item sample | All 500 items |
| --- | ---: | ---: |
| 5 | +10.9 | +4.4 |
| 10 | +7.6 | +2.8 |

- **Where Scone leads:** the lead at k=5 is significant, and it is largest on multi-session and single-session-user
  questions.
- **Where it does not:** temporal reasoning is level at k=5, and LlamaIndex is ahead on preference questions at k=10.
- **Difference in setup:** the sample ranked at the wall clock and this run ranks at each question's date, so the two
  are not the same configuration.
- **Consequence:** the sample's +9-point answer gap should be expected to shrink as well. The answer comparison waits
  on the judge.
