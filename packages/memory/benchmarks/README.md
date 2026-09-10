# Public QA experiments

The completed [8 September 2026 baseline](public-qa-v1.results.md) records
600 responses, retrieval coverage, answer scores, latency, and loaded-model sizes.
The separate [passage continuity comparison](passage-window-v1.results.md)
measures context coverage with optional neighboring chunks; it does not rescore
or replace the original model responses.
The [generation follow-up](passage-window-generation-v1.results.md) records
200 additional Gemma responses, including exact-match gains and regressions.
The [cross-encoder retrieval comparison](public-reranking-v1.results.md) measures
wider candidate pools and CPU reranking, including coverage regressions and cost.
The [retained-source comparison](source-budget-v1.results.md) measures five,
eight and ten sources within the same context budget, with a separately recorded
native shutdown failure after all observations were saved.
Its [paired generation follow-up](source-budget-generation-v1.results.md)
records 600 new Gemma responses: better evidence coverage did not improve answer
accuracy, so the source-count default remains unchanged.
The [public answer-review comparison](public-answer-review-v1.results.md) tests
the existing revision gate on all 200 original drafts. Exact match falls from
64% to 62.5%; the report preserves all adopted and unaccepted proposals.
The [format-aware review follow-up](public-format-review-v1.results.md) supplies
the original answer instructions to that reviewer. Exact match falls to 62%:
preserving output instructions does not fix comparison and entity-role errors.
The report retains every changed answer and distinguishes fallback preservation
from successful repair. Review remains optional.
The [computation-tool comparison](public-compute-v1.results.md) records 400
structured tool-loop turns: exact match was 57.5% with computation disabled
and 56.5% enabled. The model never selected the calculator, despite receiving
its schema, so this measures tool exposure rather than calculation accuracy.
Both failures, all changed answers, and retrieval-packet differences are retained.

The [repeated-search compaction comparison](public-search-compaction-v1.results.md)
records 400 paired tool-loop turns. Exact match stays at 56.5% while offered
tool-result bytes fall 28.11%; dataset-level gains and regressions keep this
presentation option disabled by default.

The [v1 protocol](public-qa-v1.protocol.md) fixes the sample and settings before
inference. These experiments exercise Scone's native memory retrieval and model
adapter using original public questions and source paragraphs. They do not
establish full-Wikipedia retrieval performance or universal answer accuracy.

Use Python 3.14 with Scone's `qdrant`, `local-embed`, and `remote-embed` extras,
a self-managed Qdrant 1.19.1 server, cached `BAAI/bge-small-en-v1.5` embeddings,
and the three installed Ollama model aliases specified by the protocol. Each
alias must have `num_ctx 8192`. The runner verifies their settings and records
their digests. It does not download models or send inference to a hosted service.

From the repository root, put the original dataset files in an ignored run folder:

```text
bench-runs/public-qa-2026-09-08/raw/hotpot_dev_distractor_v1.json
bench-runs/public-qa-2026-09-08/raw/squad_dev_v1.1.json
```

Source URLs and Hotpot mirror provenance are in the protocol. The exporter checks
both files against the frozen v1 SHA256 values. Preserve the datasets' original
attribution and terms; downloaded files and experiment outputs are not committed.

```sh
export PYTHONPATH=packages/memory/src
python -m scone_memory.testing.public_qa_run export bench-runs/public-qa-2026-09-08
python -m scone_memory.testing.public_qa_run prepare bench-runs/public-qa-2026-09-08 \
  --qdrant-url http://127.0.0.1:53410
python -m scone_memory.testing.public_qa_run generate bench-runs/public-qa-2026-09-08 \
  --endpoint http://127.0.0.1:11434
python -m scone_memory.testing.public_qa_score bench-runs/public-qa-2026-09-08
```

`export` separates corpus, questions, reserved questions, and gold labels.
`prepare` indexes all source paragraphs, saves native retrieval receipts, and
freezes every model request. `generate` never reads gold labels. `score` is the
separate offline label consumer. Do not change source code or frozen inputs
between preparation and the end of inference.

Resume an interrupted generation with the same command. An already started
attempt becomes an interrupted failure; it is never retried. Only unattempted
observations run. Full scoring requires all 600 planned observations to be
terminal, with failures retained in the denominator. Preparation intentionally
requires a fresh directory: preserve a failed preparation before starting a new
one, rather than silently overwriting it.

Inspect `scores.json` for original raw answers, status, EM/F1, retrieval annotation
coverage, and latency counts. `model-memory.jsonl` contains Ollama's loaded-model
sizes before and after each block; these are not process RSS or peak system RAM.
Treat dataset-specific metrics separately and keep the 200 reserved questions
unrun until a later experiment's settings are frozen.
