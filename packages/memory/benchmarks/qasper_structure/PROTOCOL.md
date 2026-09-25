# Full QASPER structure-address comparison, v1

Freeze this protocol before generating test answers. Run every question in the
original QASPER v0.3 official test archive: 416 papers, 1,451 questions. No sampling,
question filtering, answer-type exclusions or post-result policy tuning.

Source: https://qasper-dataset.s3.us-west-2.amazonaws.com/qasper-test-and-evaluator-v0.3.tgz
Dataset documentation: https://huggingface.co/datasets/allenai/qasper

This is QASPER's **known-paper** QA task. Every arm gets the same specified paper,
not an oracle advantage available only to Scone. This does not measure discovery
across a pooled corpus. Original section paths become Markdown headings; text
paragraphs and captions remain unchanged. No image content is invented. All
questions, including figure-dependent and unanswerable ones, remain scheduled.

## Arms

- `scone_flat`: native per-document vector retrieval control.
- `scone_structure`: the same vectors plus the shipped Jev section-address and
  automatic original/scoped-vector/broad-vector selection policy.
- `llamaindex`: installed LlamaIndex vector + BM25 reciprocal-rank fusion,
  num_queries=1 (no query rewriting), no model-generated retrieval queries.

The native vector control isolates structure's contribution; it is not a claim
that all Scone retrieval options were enabled. The reference has a lexical lane,
so a win over it would be stronger than a vector-only comparison. This single
configuration cannot establish superiority over every LlamaIndex configuration.

## Matched resources

Same Scone source-aligned chunks (700-character target), same API Nemotron 3 Embed
1B vectors (2,048 dimensions, `query: ` / `passage: ` prefixes), same original
paper access. All vectors are prepared once and reused; no local inference.
Nemotron uses OpenRouter's unpinned `nvidia/nemotron-3-embed-1b:free` route.
API embedding requests are spaced at least 3.2 seconds apart; usage is recorded.

Each arm produces up to 32 candidates within an 8,000-byte candidate-text cap.
Original sections compete within that cap. A shared OpenRouter Jev
`typesafe/jev-1.13-20260917` scores the union of unique candidate texts; identical
texts share exactly the same relevance decision. Batches contain at most 64
independent judgments. Each arm orders by these scores, stable retrieval ties.
Final context contains at most five items and 8,000 bytes including source headers.

All arms use the same paid `google/gemma-4-31b-it`, temperature 0, no reasoning,
256 output-token limit and identical answer instructions. Output `Unanswerable`
when evidence is insufficient, as required by the official scoring semantics.
Generation is performed independently per arm and remains stochastic even at
zero temperature. Arm generation order rotates by question; no successful answer
is regenerated during resume. Provider failures stop further scheduling after
the current bounded batch; completed answers remain durable. Incomplete/failed
questions stay visible and do not vanish from denominators. Output-token-limit
failures are recorded and scheduling continues. All arm contexts and retrieval
timings are persisted before generation, so a restart reuses exactly those contexts.

## Metrics and provenance

Official answer token F1 (maximum over annotations), supplemental normalized EM,
and full-original-paragraph evidence F1/recall, with all-evidence and text-only
variants. A paragraph counts as retrieved only when its entire original string
appears in packed context; clipped/split paragraphs are conservatively absent.
Report missing/failed counts, source bytes, query preparation separately from warm
retrieval, Jev routing/fetch/rerank, generation and end-to-end stage sums. Shared
rerank cost/latency is incurred once physically; per-arm attribution must not be
summed as physical run consumption. Paper-cluster bootstrap retains within-paper
question dependence. Record provider usage/cost where supplied; do not infer
unreported monetary costs.

Inference reads only corpus/questions exports, never gold annotations. Offline
scoring reads original gold only after answers are persisted. Export, code,
protocol, environment versions and complete question/arm schedule are frozen in
the run manifest. Prepared vectors persist locally for resume. A resumed run
requires identical manifest. Results from development fixture probes are not
mixed with this test set. Test model training overlap is unknown; no untouched
model-holdout claim is made.
