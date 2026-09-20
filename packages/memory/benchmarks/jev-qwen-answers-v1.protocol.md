# Qwen, Qdrant and Jev answer evaluation v1

This freezes the neural-embedding comparison before inference. Use the same
2,176 downloaded paragraphs and all 200 original questions (100 HotpotQA,
100 SQuAD) as `jev-answers-v1`, through Scone's own production framework.
This is a reused evaluation set, not an untouched holdout.

The two paired arms are hybrid retrieval without Jev and hybrid retrieval with
Jev. Both use `qwen/qwen3-embedding-8b` over OpenRouter, native 4,096-dimensional
normalized vectors, and a fresh uniquely named collection on local Qdrant.
SQLite retains the documents and lexical index. The query prefix is exactly:

```
Instruct: Find passages that provide evidence to answer the question.
Query:
```

Append the query immediately after `Query:`. Documents have no instruction.
Scone's default neural vector/text fusion weights are 1.0/1.0, whereas the
historical HashEmbedder experiment used 0.01/1.0. The across-experiment comparison
therefore measures the changed retrieval stack, not an isolated causal effect
of model parameters or Qdrant. The within-run Jev comparison holds it fixed.

Keep the previous experiment's chunk target (700 characters), 64 candidates
per lane, at most 32 reranked passages/64,000 bytes, pinned
`typesafe/jev-1.13-20260917`, ten-second rerank timeout, five delivered sources,
8,000 context bytes, default structured retrieval, and 30-second context deadline.
Use the unchanged benchmark prompt and `google/gemma-4-31b-it` with temperature
zero, thinking disabled and 256 maximum output tokens. Generate sequentially
and alternate arm order by question. No query rewriting, extra tools, reviewer,
gold-derived routing, prompt tuning, retry or alternate models. Record fallbacks.

Run the existing `scone_memory.testing.jev_answers` harness with
`--qdrant-url http://127.0.0.1:64076`. Preserve all 400 prepared requests and
answers, source IDs/text, receipts, timing, failure statuses, resolved Jev model,
configuration and input/code/protocol hashes. Do not read gold during inference.
Keep the new evaluation collection for inspection; never reuse or replace the
existing Qdrant collection. Interrupted runs remain unscored.

Score only after all cases finish and integrity checks pass: normalized exact
match, token F1, abstentions, failures, paired gains/regressions, source document
coverage, and context/generation latency. Compare with the historical Hash run
without suppressing losses. Exact match is not semantic correctness or evidence
faithfulness; inspect aliases and counterexamples separately. A single hosted
generation per arm and public-data contamination limit conclusions.

This measures native RAG context and generation, not UI acceptance or full agent
workflows. Artifacts go to `bench-runs/jev-qwen-answers-20260919/run-1/`, outside
git. Commit the tested harness and this protocol before inference.
