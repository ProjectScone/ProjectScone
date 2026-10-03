# LongMemEval-S, 100 items, same reader and judge: Scone ties LlamaIndex, and the benchmark is saturated

Run: 27 September 2026, `stratified_sample(n=100, seed=42)` of LongMemEval-S. Every arm used the same reader
(`google/gemma-4-31b-it`, upstream's step-by-step prompt) and the same judge (`gpt-4o-2024-08-06`, upstream's judge
prompts). Scone and LlamaIndex both ranked with `bge-small-en-v1.5`. Settings are in [PROTOCOL.md](PROTOCOL.md).
Raw files are in `bench-runs/longmemeval-matched-2026-09-27/s100-bge-small-cot/`.

| Arm | Accuracy | Wilson 95% | All evidence in context | Median prompt tokens |
| --- | ---: | --- | ---: | ---: |
| oracle (evidence sessions only) | 93.0% | 86–97 | by construction | 5,979 |
| full context | 90.0% | 83–94 | by construction | 109,805 |
| scone@10 | 88.0% | 80–93 | 96.8% | 28,593 |
| llamaindex@10 (vector + BM25 fusion) | 88.0% | 80–93 | 96.8% | 28,750 |
| scone@5 | 86.0% | 78–91 | 92.6% | 15,059 |
| llamaindex@5 | 86.0% | 78–91 | 88.3% | 15,346 |

## Paired comparisons

Counted on the same 100 items:

| Comparison | Wins | Losses |
| --- | ---: | ---: |
| scone@5 vs llamaindex@5 | 6 | 6 |
| scone@10 vs llamaindex@10 | 3 | 3 |
| scone@10 vs full context | 6 | 8 |

No difference here is distinguishable from noise.

## Failed generations

All are the reader hitting upstream's 800-token cap on step-by-step output, and all count as incorrect.

| Arm | Failed generations |
| --- | ---: |
| full | 3 |
| scone@5 | 1 |
| scone@10 | 1 |
| llamaindex@5 | 1 |
| llamaindex@10 | 2 |

One item fails this way in every arm.

## What it shows

- **With a 31B reader, LongMemEval-S does not separate retrieval systems.** Given only the evidence, the reader tops
  out at 93%. The full history (110k tokens) fits in its window and reaches 90%.
- **Scone@10's errors are mostly reading errors, not retrieval errors.** Of its 12 errors, 2 had evidence missing from
  context. The other 10 had all the evidence in context: counting across sessions, date arithmetic, and abstention.
  Several of those also fail under full context or oracle.
- **Scone's measured advantage is cost.** It stays within 2 points of full context using 26% of the prompt tokens,
  with about 10 ms query-time recall.
- **Date parsing gives little headroom.** Only 59 of the 500 LongMemEval-S questions contain a date phrase that
  `retrieval/dates.py` parses. For 17 of them, the evidence session falls outside the parsed window, because events are
  reported after they happen. So a date filter would lose questions, and a date boost has little room to gain.

## Consequence

Beating other frameworks on memory has to be shown where retrieval decides the answer:

- LongMemEval-M, where about 500 sessions per item cannot fit in any reader's window;
- small local readers, where published results show context selection matters far more;
- harder suites (BEAM).

This run is the S baseline those comparisons build on. It is not a claim of superiority.
