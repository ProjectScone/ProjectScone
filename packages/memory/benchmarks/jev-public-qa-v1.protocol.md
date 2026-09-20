# Jev public QA evaluation v1 — fixed before running

- Use the downloaded `public-qa-2026-09-08` corpus: all 2,176 source paragraphs
  and all 200 reserved questions, 100 each from HotpotQA and SQuAD. Verify
  existing dataset SHA256 values. No question selection, rewriting, seeded facts
  or gold labels in inference. This consumes the reserved split for evaluation;
  it must not subsequently be described as an untouched holdout.
- Run original Scone `MemoryEngine` ingestion and recall with SQLite stores,
  HashEmbedder, 700-character chunk target, 64 candidates per retrieval lane,
  32 candidates / 64,000 serialized bytes for reranking, ten returned passages,
  ten-second rerank deadline and thirty-second whole-recall deadline.
- Four arms: vector-plus-text baseline, vector-plus-text with Jev, text-only
  baseline, text-only with Jev. Rotate arm order per question. All arms use the
  same source index. HashEmbedder is the framework's hashing baseline, not a
  neural embedding model; this experiment cannot establish performance against
  a strong neural embedder. No Chinese model is used.
- Jev uses the shipped relevance rubric and `~typesafe/jev-latest`; record
  resolved versions and whether ranking actually applied. Do not tune the
  rubric after seeing this split. API failures retain Scone's baseline fallback;
  log those outcomes, keep all questions in denominators, and do not retry.
- Preserve original ranked passage IDs, texts, source-document IDs, scores,
  traces, timings, input hashes and source-code hashes outside git. The runner
  does not load gold labels. Score only a complete 800-observation run with
  unchanged code and inputs. No generated answers in this retrieval experiment.
- Report mean supporting-document recall, all-supporting-document coverage at
  five and ten returned passages, first-supporting-passage MRR@10, paired wins
  and losses, p50/p95 latency, failures and rerank application/fallback counts,
  both overall and per dataset. Duplicated chunks do not increase the number
  of gold documents recovered. Preserve regressions and per-question scores.
- Do not enable Jev by default based only on this run. Answer-generation quality
  and a neural-embedding comparison remain separate evaluations.
