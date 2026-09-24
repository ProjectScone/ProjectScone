# Matched Scone–LlamaIndex complete development-set evaluation

Goal: measure the answer-quality gap against an actual open-source LlamaIndex
pipeline under matched models and evidence budgets. Run every question in both
development files, not a sample. This compares specified configurations, not
every possible configuration or a private-test leaderboard submission.

Freeze before inference:

- All 7405 HotpotQA and 10570 SQuAD 1.1 development questions: 17975 questions,
  35950 scheduled answer attempts. Previously used questions remain included;
  prior exposure is audited. All supplied contexts enter one shared corpus,
  including distractors. No gold-source restriction during retrieval.
- Original source-file hashes verified against the earlier export. Model-visible
  files contain only corpus and questions; gold labels are read only by export
  and offline scoring. Public data and prior development exposure prevent an
  untouched-holdout claim. The 68702 distinct source paragraphs include every
  supplied distractor. One original HotpotQA sentence index is invalid; preserve
  its question, answer and document support, and record the annotation defect.
- Native Scone MemoryEngine hybrid retrieval versus LlamaIndex 0.14.24
  VectorStoreIndex + BM25Retriever + reciprocal-rank QueryFusionRetriever.
  No query expansion, hosted storage, agent framework or local model.
- Both use Qwen3 Embedding 8B through OpenRouter, the same 4096-dimensional
  vectors for identical text, and shared cached query vectors. Both index the
  same Scone-generated 700-character target chunks for this controlled retrieval
  comparison; this does not test LlamaIndex's native chunking or ingestion.
  Both use local Qdrant 1.19.1 dense indices; Scone uses native SQLite lexical
  search and LlamaIndex uses its BM25 retriever. No cloud infrastructure.
- Each lane returns up to 64 candidates; fusion retains 32. The same direct Jev
  relevance question reranks each shortlist. Identical query/passage pairs
  across arms share one judgment to avoid stochastic reranking asymmetry.
  The joint batch has up to 64 questions; record each arm's conceptual budget
  and shared physical usage separately. Batch timing is not a causal per-arm
  latency comparison.
- Both deliver up to five chunks within an 8000-byte evidence budget through
  identical formatting and instructions. Paid `google/gemma-4-31b-it`, temperature
  zero, 256 output tokens, 60-second provider deadline, no reasoning requested.
  Alternate generation arm order by question. Retain failures; no answer retries.
- Record exact requests, retrieved/source IDs, replies, model/usage metadata,
  preparation/reranking/generation times and artifact hashes. No credentials.
  Query embeddings are prewarmed; report this rather than claiming cold latency.
- Report EM/F1 by dataset, paired wins/losses/ties and bootstrap intervals,
  source coverage, failures and p50/p95 stage times. Keep failures in answer
  denominators; generation failure does not erase measured retrieval coverage.
  Full development splits over a pooled paragraph corpus, not full Wikipedia;
  SQuAD receives the pooled corpus rather than its original oracle paragraph.
- Four concurrent question jobs; each alternates arm order. Timings reflect
  shared local/API load. Durable attempt journals mark interrupted attempts as
  failures on resume; completed answers are never regenerated. Resume requires
  identical code, inputs, dependency versions and configuration. Index/vector
  preparation can resume from persistent local caches without answer attempts.

## Implementation and verification

1. `data.py`: full-split export and separate gold; validate collisions,
   prior exposure and source hashes. `scoring.py`: strict paired schedule and integrity
   checks, failed answers score zero, paired uncertainty retained.
2. `run.py`: native engine and isolated LlamaIndex dependency, shared vectors,
   common reranking/prompt/generator, incremental observations and error records.
   No production imports of competitor code or runtime changes.
3. Behavior tests for context byte budgets, shared ranking, score denominators,
   duplicate/missing observations and input integrity. Strict type checking.
4. Freeze protocol/source hashes, execute all 17975 paired questions, score only
   complete unchanged artifacts. Preserve raw evidence locally, document losses
   and gaps, commit/push and open a draft MR.

## What constitutes progress toward the target

A positive comparison identifies a candidate; it cannot establish universal superiority.
Next compare independently tuned configurations (native chunking, fusion and
reranking) with equal tuning budgets and independent evaluation. Then repeat on
multi-hop and changing-memory workloads with latency/cost constraints. The target
is a repeatable answer-quality improvement at comparable resources. Claims must
name configuration, corpus, split, metrics and uncertainty.

References: [LlamaIndex BM25 and hybrid retrieval](https://developers.llamaindex.ai/python/framework/integrations/retrievers/bm25_retriever/),
[TypeSafe reranking](https://docs.typesafe.ai/cookbooks/rerank_typesafe),
[HotpotQA](https://hotpotqa.github.io/),
[SQuAD](https://rajpurkar.github.io/SQuAD-explorer/).
