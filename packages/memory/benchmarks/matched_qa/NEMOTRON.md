# Nemotron embedding configuration for the matched full-data benchmark

Use `--embedding-profile nemotron`, or set
`SCONE_BENCH_EMBEDDING_PROFILE=nemotron` in the private environment loaded by
`scripts/local_env.py`. This selects the same configuration for both Scone
and LlamaIndex. It does not migrate the running application's index.

| Setting | Value |
| --- | --- |
| Embedding API | OpenRouter `/api/v1/embeddings` |
| Requested model | `nvidia/nemotron-3-embed-1b:free` |
| Vector dimensions | 2,048, full width, L2 normalized |
| Query prefix | `query: `, including trailing space |
| Document prefix | `passage: `, including trailing space |
| Embedding batch size | 128 texts |
| Minimum interval between embedding requests | 3.2 seconds |
| Native collection | `matched_scone_nemotron_3_embed_1b` |
| Reference collection | `matched_scone_nemotron_3_embed_1b_llamaindex` |
| Shared reranker | OpenRouter `typesafe/jev-1.13-20260917` |
| Answer model | Paid `google/gemma-4-31b-it`, unchanged |

The remainder of [PROTOCOL.md](PROTOCOL.md) applies: all 17,975 development
questions and 68,702 pooled paragraphs, identical 700-character target chunks,
64 candidates per lane, 32 fused candidates, five evidence chunks, 8,000-byte
context limit, matching prompts and generation budgets. There is no sampling.
The run manifest records the profile, model, dimensions, both prefixes, physical
batch size, request interval, collection and this document's hash. Embedding
receipts record the provider's resolved model and usage.

Use a **fresh output directory and the profile's separate collections**. Both
frameworks share newly computed Nemotron vectors. Existing Qwen vectors and
completed results must remain intact; dimensions and vector spaces differ.
The document/query prefixes are explicit native configuration, not an external
embedding framework. Their identity participates in cache/index compatibility.
Queries never receive the document prefix or reuse an incompatible document vector.

Example, with existing credentials loaded through `scripts/local_env.py`:

```sh
python -m matched_qa.run \
  --dataset /absolute/path/to/dataset-full \
  --output /absolute/path/to/nemotron-full/run-1 \
  --qdrant-url http://127.0.0.1:16437 \
  --embedding-profile nemotron --concurrency 4
```

Select `--embedding-profile qwen` explicitly to reproduce the earlier model
configuration. Historical frozen-run resume still requires its original checkout;
changing source or profile is a new run, never an in-place resume.

## Verified API behavior and limits

On September 24, 2026, the OpenRouter embedding catalog listed only the free
route for Nemotron 3 Embed 1B. Live requests using the existing embedding key
returned 2,048 dimensions, including one 128-passage batch. The response model
was `private/openrouter/nvidia/nemotron-3-embed-1b`. The public route does not
provide a pinned dated revision; record this limitation when comparing runs.

The account reported a 1,000 free-request daily allowance. Preparing this
corpus and all queries needs at most 816 batches of 128 before caching/deduplication,
excluding probes, errors and any other account usage. The 3.2-second pacing
stays below the documented 20-request/minute free-model ceiling. Quotas are
shared and provider availability may still interrupt preparation. Persisted
vectors allow preparation to resume without starting over.

NVIDIA lists HotpotQA and SQuAD among training-data sources; exact split overlap
is not established here. These development questions cannot be called an
untouched model holdout. Any change from the Qwen results also spans a provider
change for some Jev judgments, so it cannot be attributed solely to embeddings.

References: [NVIDIA model instructions](https://huggingface.co/nvidia/Nemotron-3-Embed-1B-BF16),
[OpenRouter model](https://openrouter.ai/nvidia/nemotron-3-embed-1b:free),
[OpenRouter limits](https://openrouter.ai/docs/api/reference/limits).
