# OpenRouter recovery of the complete QA schedule

The original full run attempted 17,975 question pairs. Its final 3,483 pairs
received direct TypeSafe HTTP 402 before generation. A separate diagnostic
confirmed that the organization had no available TypeSafe API credits. The
user authorized OpenRouter's Jev for these missing answers.

This recovery preserves the original run and every successful answer. It is
a new, explicitly labeled composite evaluation, not a replacement first run
or proof that two provider deployments are identical.

Freeze before recovery inference:

- Verify original completion, artifact and dataset hashes. Accept only pairs
  where both arms failed with `rerank_http_402`; never select cases by quality.
- Reuse each failed question's saved candidate texts, source IDs and original
  per-arm ordering. Do not redo retrieval, embeddings, chunking or indexing.
- Keep the same relevance instructions, criteria, shared candidate union,
  evidence limits, answer prompt, paid Gemma model and generation parameters.
- Route shared Jev judgments to OpenRouter's Decisions API with pinned model
  `typesafe/jev-1.13-20260917`. Require that model in responses and retain raw
  usage/cost receipts. Credentials come from the existing environment loader.
- Four concurrent question jobs. Preserve original alternating arm order.
  Stop scheduling further batches on a provider or generation failure.
  Already-started pairs may finish. There are no automatic HTTP retries.
- Persist successful reranking before generation. Journal each generation
  attempt before sending it. On explicit resume, reuse saved judgments and
  completed answers. An interrupted generation becomes a recorded failure,
  never a silently repeated answer. A rejected reranking request can be
  retried by an explicit resume; every rejection remains in the error journal.
- Write a separate full-schedule artifact combining byte-equivalent JSON
  values for original successes with newly recovered answers. Record parent
  completion hash, recovery source/protocol hashes and the exact recovery IDs.
- Preserve failures in the original report. Publish recovered full-dataset
  quality only when all 17,975 pairs have successful answers. If recovery fails,
  disclose remaining missing answers instead of dividing them away.
- Report the provider switch and reused retrieval timings. Combined stage
  sums for recovered rows are assembled from different execution periods;
  they are not a new end-to-end wall-clock measurement.

Run from the isolated benchmark environment using the existing `.env.local`
loader and `PYTHONPATH=packages/memory/src:packages/memory/benchmarks`:

```sh
python -m matched_qa.recovery \
  --dataset /absolute/path/to/dataset-full \
  --parent /absolute/path/to/run-1 \
  --output /absolute/path/to/recovery-openrouter-1 --concurrency 4
```

Use `--resume` only with the same inputs, source and configuration. The usual
offline scorer verifies the full merged schedule. `answers_complete` in the
recovery completion receipt separately confirms successful answer coverage.

API reference: [OpenRouter Jev tutorial](https://openrouter.ai/blog/tutorials/how-to-use-jev/).
