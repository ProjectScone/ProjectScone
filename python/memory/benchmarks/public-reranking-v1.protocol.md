# Public retrieval reranking development comparison v1

Freeze this protocol before evaluating. This is a retrieval experiment, not an
LLM generation score or a claim of general accuracy.

- Reuse the original public-qa-v1 corpus of 2,176 paragraphs and all 200 unchanged
  development questions (100 HotpotQA, 100 SQuAD). Keep the reserved 200 unrun.
  No question rewriting, gold filtering, seeded facts, training or answer repair.
- Use an isolated copy of the original SQLite ledger and Qdrant 1.19.1 storage,
  with the same cached BGE-small-en-v1.5 embedding model. Preserve the originals.
- Compare three arms: legacy fusion (implicit lane depth 80 for recall limit 20),
  fusion with candidate_limit=128, and candidate_limit=128 with the existing
  OfflineCrossEncoderReranker. Rerank at most 64 candidates / 128,000 payload bytes,
  with a 10-second ranking deadline; all arms retain a 30-second recall deadline.
- Model: already provisioned Xenova/ms-marco-MiniLM-L-6-v2, revision
  a09144355adeed5f58c8ed011d209bf8ee5a1fec. ONNX model SHA256:
  c623d0bcb99f4622beb413eaef00cfbe5db20df9f1dd982da4b4f26022881870.
  CPU inference, two threads, batch size eight, maximum 512 tokens per full pair.
  Record all artifact hashes and runtime versions. No network model inference.
- Use production MemoryContext(limit=5, max_context_bytes=8000,
  structured_paths=True, neighbor_chunks=0). Separately call native recall with
  limit=10 for document ranking metrics, as in the original baseline (legacy lane
  depth 40 for that call). Record both
  calls; their timings are separate measurements, not a single request latency.
- Rotate arm order by question index modulo three. Each arm gets one context
  preparation and one native recall per question, with no replacement retries.
  Retain failures in denominators. Save raw requests, receipts, ranked source IDs,
  ranking traces, degraded statuses, timings and per-call token-length diagnostics.
- The observer measures untruncated pair lengths without modifying candidates.
  Overlong pairs retain the production adapter's whole-call refusal and baseline
  fallback; do not truncate, exclude troublesome questions or silently substitute
  a different model. Record refusals and deadline failures separately.
- The runner must not load gold labels. Score only after all 600 arm/question
  observations are terminal and source/code/model hashes remain unchanged.
  Report supporting-document recall and all-documents coverage at 5 and 10,
  literal prepared-annotation coverage, complete-annotation coverage, context
  bytes, p50/p95 latency, failures and ranking application/fallback counts.
- Validate legacy contexts/rankings against the original prepared baseline and
  report differences. The repeated baseline is contemporaneous, but cache and
  host activity can still affect timing. Literal quote matching has known chunk
  boundary limitations; preserve the scorer and disclose those limits.
- Preserve raw artifacts outside git. Publish the protocol and an honest aggregate
  report including regressions. Do not enable reranking by default from this
  development sample alone; generation effects require a separate frozen run.
