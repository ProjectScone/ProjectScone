# Jev answer evaluation v1 — frozen before generation

Compare final answers through Scone's native `MemoryContext` and
`OpenAICompatibleTextModel`, with and without Jev. This follows the retrieval
experiment; the reserved split has already been evaluated for retrieval, and
must not be described as an untouched holdout. No prompts or rubrics are tuned.

- Ingest all 2,176 downloaded paragraphs into a fresh local Scone SQLite index,
  with HashEmbedder, 700-character chunks and 64 candidates per lane.
- Use all 200 original reserved questions (100 HotpotQA, 100 SQuAD). Two arms,
  `hybrid` and `hybrid_jev`, alternate order per question. Generate all 400
  answers sequentially. No selection by earlier outcomes.
- Baseline has no reranker. Jev uses the shipped adapter/rubric, pinned
  `typesafe/jev-1.13-20260917`, at most 32 passages/64,000 serialized bytes,
  with a ten-second rerank deadline. Failure retains production baseline
  fallback; record resolved Jev models and context receipts.
- Both arms use production memory-context preparation with kind=file,
  five sources, 8,000 context bytes, 30-second preparation timeout and default
  structured paths. No seeded claims, question rewrites, neighboring chunks,
  reviewer, answer correction or extra retrieval tools are enabled. Preserve
  native retrieval planning, scope/source checks and actual delivered evidence.
- Generate using paid `google/gemma-4-31b-it` through the existing Scone text
  adapter, temperature zero, thinking disabled, 256 output tokens, 60-second
  provider timeout, 65-second overall capture timeout. Use the existing
  `benchmark_messages` short-answer/INSUFFICIENT_EVIDENCE prompt unchanged.
  No retries or alternate models. No Chinese model is used.
- Save exact prepared messages, source text/document IDs, receipts, request
  hashes, raw public answer strings, completion/failure statuses and timings
  before proceeding. Credentials/headers are not persisted. No gold labels
  are loaded by the inference command. Failed preparation scores as failure;
  failed/incomplete generation retains its raw prefix but scores zero.
- Freeze input, protocol and package source hashes before requests. Scoring
  requires a complete 400-observation schedule, verified request/artifact/input
  hashes, and unchanged code/protocol throughout execution. Interrupted runs
  remain unscored; they are not replaced with successful-only prefixes.
- After all cases finish, score normalized exact match and token F1 using the
  existing Scone public QA scorer. Report all-question denominators, per-dataset
  outcomes, paired wins/losses/ties, abstentions, failures, Jev application and
  latency. Preserve and inspect regressions without changing the experiment.

This tests a RAG answer path, not UI acceptance or all agent workflows. Exact
match/F1 are reference-answer metrics, not semantic faithfulness or proof that
an answer came from evidence. HashEmbedder remains a weak baseline; a strong
permitted neural-embedding comparison is separate. Public-data contamination,
one generation per arm, backend variability and aliases in the chat provider
limit generalization. Publish results regardless of direction.

Artifacts: `bench-runs/jev-answers-20260919/run-1/` in the primary checkout,
outside git. Commit this protocol and tested harness before inference.
