# Public QA experiments

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
export PYTHONPATH=python/memory/src
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
