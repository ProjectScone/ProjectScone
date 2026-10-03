# Full-dataset comparison with LlamaIndex

For the Nemotron embedding configuration, see [NEMOTRON.md](NEMOTRON.md).
Select it with `--embedding-profile nemotron` or
`SCONE_BENCH_EMBEDDING_PROFILE=nemotron`; both arms use the same profile and
separate vector collections. The Qwen results below remain the completed baseline.

The [completed results](RESULTS.md) include all 35,950 successful answers after
an explicitly recorded OpenRouter Jev recovery of 3,483 question pairs blocked
by direct-provider billing. Combined EM is 64.99% for Scone versus 65.30% for
LlamaIndex; F1 is 75.19% versus 75.32%. The [recovery protocol](RECOVERY.md)
preserves original outcomes and successful answers and identifies the provider switch.

This benchmark runs all **17,975 development questions** from HotpotQA
(7,405) and SQuAD 1.1 (10,570), with **35,950 answer attempts** across Scone
and LlamaIndex. It does not sample questions. The pooled corpus contains
68,702 distinct original paragraphs, including all supplied distractors.

Read the frozen [protocol](PROTOCOL.md) before interpreting scores. Both arms
share Qwen embeddings, chunk text, direct Jev relevance judgments, evidence
budgets, and paid Gemma generation. The comparison isolates the two hybrid
retrievers through a common answer pipeline. LlamaIndex is a benchmark-only
dependency, never an implementation dependency of native Scone retrieval.

This differs from the original task settings: HotpotQA receives the pooled
corpus rather than ten per-question paragraphs, and SQuAD receives that corpus
rather than its supplied answer paragraph. Complete question coverage does not
make these official leaderboard scores. The original 400 development questions
remain included and are flagged as previously seen. Questions are not necessarily
independent by topic, and public data may overlap model training.

## Isolated environment and local storage

Create an external Python 3.12 virtual environment and install `requirements.txt` there.
Do not add competitor dependencies to Scone's runtime environment or package.
Use a local Qdrant 1.19.1 server dedicated to this benchmark. The configured
endpoint is validated as loopback-only. Keep its storage and all generated
artifacts under an ignored `bench-runs/` directory.

The runner reads `SCONE_EMBED_API_KEY` and `SCONE_CHAT_API_KEY`, falling back
to `OPENROUTER_API_KEY`, plus `TYPESAFE_API_KEY` and optional
`TYPESAFE_DEFAULT_MODEL`. The protocol fixes Qwen and paid Gemma model IDs.
Use the existing `scripts/local_env.py` loader; do not copy credentials into
commands, source files, manifests or result reports.

## Export every question

Set `PYTHONPATH=packages/memory/src:packages/memory/benchmarks` when invoking
the external environment's Python from the repository root:

```python
from pathlib import Path
from matched_qa.data import export

previous = Path("bench-runs/public-qa-2026-09-08")
export(previous / "raw", previous, Path("bench-runs/llamaindex-full/dataset"))
```

The exporter validates both original file hashes against the historical
dataset manifest. It retains every question and original answer. One HotpotQA
supporting-sentence index is outside its paragraph; the exporter records that
defect while retaining its valid document support. No question is excluded.

## Run and resume

```sh
python scripts/local_env.py --env-file /absolute/path/to/.env.local -- \
  /absolute/path/to/benchmark-env/bin/python -u -m matched_qa.run \
  --dataset bench-runs/llamaindex-full/dataset \
  --output bench-runs/llamaindex-full/run-1 \
  --qdrant-url http://127.0.0.1:16437 --concurrency 4
```

Use the same command with `--resume` after interruption. Resume checks source,
protocol, dataset, dependency versions and configuration. SQLite stores shared
vectors; each framework's local vector index has its own collection. Atomic
offsets allow index preparation to continue. A process lock prevents duplicate
workers on one output directory. Do not point multiple runs at the same vector
collections; use a separate dedicated server/storage directory for another run.

Before each question starts, both scheduled attempts are journaled and fsynced.
Completed rows are also fsynced. A previously started attempt without a completed
row becomes an `interrupted_attempt` failure on resume, not an invisible retry.
Torn final journal bytes are preserved separately before recovery. These failures
stay in the full answer denominator. Provider errors retain status codes without
credential headers. Requests are not retried by the generation helper.

`progress.json` reports the phase and terminal answer count. `embedding-calls.jsonl`
records physical embedding calls, tokens/cost when reported by the provider, and
timing. `judgments.jsonl` retains shared physical Jev requests/responses;
`observations.jsonl` retains each arm's candidates, context IDs, exact generation
request, answer, model usage and stage timings. Shared reranking batch times are
reported for both arms but must not be summed as independent physical work.
Interrupted attempts with no timing receipt have zero recorded duration, which
does not mean no time was spent; report their count with latency summaries.

## Score without hiding unfinished work

```python
from pathlib import Path
import json
from matched_qa.scoring import score

report = score(Path("bench-runs/llamaindex-full/dataset"),
               Path("bench-runs/llamaindex-full/run-1"))
Path("bench-runs/llamaindex-full/run-1/scores.json").write_text(
    json.dumps(report, indent=2) + "\n")
```

Normal scoring requires every scheduled pair and matching artifact hashes.
`allow_partial=True` is for monitoring only: it reports completion fraction,
observed denominators and comparisons only for fully observed pairs. Partial
scores are not full-dataset results. All failures score zero on answer EM/F1;
generation failure does not erase measured retrieval coverage.

Compare identical-request pairs when interpreting answer differences: even at
temperature zero, separate provider calls can disagree on identical evidence.
Shared Jev judgments remove one source of that variation; they do not make
generation deterministic or prove that every answer gain came from retrieval.

```sh
python -m pytest packages/memory/benchmarks/matched_qa -q
mypy --strict --follow-imports=silent packages/memory/benchmarks/matched_qa
```

The benchmark is evidence toward a specific competitive target. A win here
must name this configuration and uncertainty. Stronger claims require independently
tuned LlamaIndex variants, native ingestion comparisons, other datasets and
changing-memory evaluations under matched total resource budgets.
