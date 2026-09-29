# LongMemEval-M, all 500 items: retrieval complete, answers blocked by the API spending cap

## Run

- **Date and data:** 28 September 2026, every item of LongMemEval-M (cleaned). The dataset is read one item at a time
  from JSONL.
- **Embedder:** both systems ranked with `bge-base-en-v1.5`, filled through the hosted-copy prefill.
- **Clock:** every item ranked with the engine clock at its question date.
- **Settings:** in [PROTOCOL.md](PROTOCOL.md).
- **Raw files:** `bench-runs/longmemeval-matched-2026-09-27/m500-bge-base-cot/`.

## Status

- **Ranking:** complete for all 500 items.
- **Answering:** the answer stage wrote 2,500 rows. Then the OpenRouter key reached its monthly $20 cap:
  - every judge request (2,412) was refused with HTTP 403;
  - 50 llamaindex@10 answers were refused the same way.
- **No accuracy yet.** None is reported until those requests run.
- **Resume:** now asks refused requests again. Answers the reader cut off itself (`incomplete_generation`) are kept
  and scored incorrect, as before.

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
