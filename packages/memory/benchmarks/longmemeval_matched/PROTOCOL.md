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
