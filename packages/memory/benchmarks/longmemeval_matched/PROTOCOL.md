# Matched-reader LongMemEval

Published LongMemEval scores are mostly self-reported. Each vendor uses its own harness, reader model and judge. On
LoCoMo, changing only the reader has moved a score by 7 points, and changing only the judge by 3. This harness
removes those variables. Every arm hands its sessions to **the same reader**. **The same judge** scores every answer
with LongMemEval's own prompts. So the differences between arms come from what each system put in the context, and
nothing else.

## Fixed across arms

- **Prompts:** LongMemEval's reader template (`--cot` selects upstream's step-by-step variant) and its judge prompts,
  copied verbatim from upstream commit `9e0b455f4ef0e2ab8f2e582289761153549043fc`. A check against the upstream
  source proved the templates byte-identical: 12 judge cases and the reader template.
- **Context format:** sessions go into the prompt sorted by date, formatted as upstream does.
- **Reader:** `google/gemma-4-31b-it` through OpenRouter, temperature 0, reasoning off. Output is capped at 800 tokens
  with `--cot` and 500 without, matching upstream.
- **Judge:** `openai/gpt-4o-2024-08-06`, the model upstream names. Temperature 0, 10 output tokens, and upstream's rule
  that any "yes" in the reply counts as correct.
- **Sample:** LongMemEval-S, selected by `scone_memory.bench.runner.stratified_sample` (seed 42). `--sample 0` runs all
  500 items.
- **Embedder:** Scone and LlamaIndex both rank with the same local embedder (`--embed-model`, run in-process through
  fastembed). Retrieval is the only variable.

## Arms

| Arm | Context |
| --- | --- |
| `full` | Every session in the haystack: the full-context baseline. |
| `oracle` | Only the evidence sessions: the reader's upper bound. |
| `scone@k` | The top k distinct sessions from Scone's engine defaults, with passages folded to sessions. |
| `llamaindex@k` | The top k distinct sessions from LlamaIndex's `VectorStoreIndex` + `BM25Retriever`, fused by reciprocal rank: its best configuration, not its default. |

## Scoring rules

- A failed or missing generation scores as incorrect. So does a judge request that errors.
- Every run reports its counts of failed, missing and unjudged items, so nothing drops silently.
- Accuracy is reported per arm, with a Wilson 95% interval, and per question type. Abstention (`_abs`) items are
  their own type.
- `all_evidence_in_context` is the share of non-abstention items whose evidence sessions all fall within the arm's top
  k.
- Reader prompt tokens come from the provider's usage records.

## Running

```
PYTHONPATH=src:benchmarks python -m longmemeval_matched.run all \
  --dataset longmemeval_s.json --run-dir RUN --sample 100 --depth 10 \
  --embed-model bge-small-en-v1.5 --cot --arms full,oracle,scone@5,scone@10,llamaindex@5,llamaindex@10
```

Each stage (`rank`, `answer`, `judge`) appends to its own JSONL file and resumes from it. `report` writes
`report.json`.

## Limits

- One reader and one judge. A second judge from another model family is still needed before claiming a ranking.
- 100 items gives roughly ±10-point intervals, which is enough to find large gaps but not small ones. A headline claim
  needs all 500 items.
- Competitors beyond LlamaIndex (Mem0 OSS, Graphiti, Mastra) are not wired in yet.

## Is LlamaIndex handicapped by the chunk settings? ([`llama_sensitivity.py`](llama_sensitivity.py))

The matched runs split LlamaIndex's documents at 512 model tokens with no overlap, so that no node exceeds bge's
512-token window. LlamaIndex's own default is 1,024 tokens with 200 of overlap, and zero overlap can cut evidence
across long chat sessions. The check re-ranked the saved LongMemEval-S 100-item run (bge-small; 94 items have
evidence) under LlamaIndex's alternatives:

| Configuration | all@5 | all@10 | Scone-only / this-only at all@5 |
| --- | ---: | ---: | --- |
| Scone (saved) | **92.6%** | **96.8%** | |
| LlamaIndex 512/0, vector + BM25 (saved, as benchmarked) | 88.3% | 96.8% | 6 / 2 |
| The same, re-run | 88.3% | 96.8% | identical on 94 of 94 |
| 512 tokens, 200 overlap, vector + BM25 | 87.2% | 96.8% | 6 / 1 |
| Its defaults (1,024 / 200), vector + BM25 | 88.3% | 96.8% | 6 / 2 |
| Its defaults, vector only (out of the box) | 89.4% | 94.7% | 6 / 3 |

**None of LlamaIndex's own settings does better than the configuration benchmarked.** Overlap slightly lowers its
all@5, and its defaults change nothing at all@5 or all@10. Scone leads every variant at all@5, but at 94 items none
of these differences is significant (sign test p from 0.13 to 0.51).

LongMemEval-M was not re-ranked under these variants. Its long sessions would need about a million new embeddings,
which waits on the hosted prefill.
