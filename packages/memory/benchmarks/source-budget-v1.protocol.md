# Retained-source budget development comparison v1

Freeze before running. This experiment varies the number of source anchors
retained in the prepared context, not the questions or retrieval candidate pool.

- Use all 200 original public-qa-v1 development questions, the same 2,176 original
  paragraphs, SQLite ledger, Qdrant 1.19.1 vectors and cached BGE-small-en-v1.5
  embedding artifacts. Copy stores and verify their content hashes against the
  originals. Keep the 200 reserved questions unrun.
- Three production MemoryContext configurations: limit=5, limit=8 and limit=10.
  All use max_context_bytes=8000, recall_timeout=30, structured_paths=True,
  neighbor_chunks=0 and no reranker. For all three limits, the current context
  implementation calls native recall with limit=20, retaining the same implicit
  lane candidate depth of 80. Verify this in the code and record the unchanged
  package hashes.
- Rotate configuration order by question index modulo three. Make one preparation
  attempt for each arm/question pair, 600 in total. No answer generation, question
  rewriting, gold filters, seeded facts, training, answer repairs or retries.
- Save complete requests, receipts, retained original source identities/text,
  context bytes, source count, status and preparation latency. The runner must
  not read gold. Preserve failed preparations in the 200-question arm denominator;
  do not publish an incomplete run as complete.
- Fingerprint code, protocol, runner, source files, baseline requests, actual
  copied stores and embedding artifacts before and after preparation. Save the
  frozen manifest and raw observation digest. Retain all artifacts outside git.
- Only after all 600 observations are terminal, score literal prepared annotation
  coverage and complete annotation coverage with the original scoring rules:
  Hotpot supporting-sentence substrings in their annotated source document, and
  SQuAD answer substrings in an annotated supporting document. Preserve known
  literal-match/chunk-boundary limitations. This is not semantic correctness.
- Report all three configurations by dataset, failures, paired gains/losses,
  retained source counts, context-byte distribution and preparation p50/p95.
  Verify the five-source arm against the original prepared requests and retained
  source sequences. Native top-k ranking is not rescored by this experiment.
- No default change follows from annotation coverage alone. More evidence can
  also distract generation. Any generation comparison needs a separately frozen
  protocol with unchanged questions, original raw responses and all failures.
