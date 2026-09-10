# Reproduce the bounded Qdrant storage experiments

These scripts reproduce the deterministic data and query schedules used by the
10 September 2026 storage experiments. They require an existing, locally managed
Qdrant server and a Python environment with Scone's `qdrant` extra and NumPy.
They do not start services, download dependencies or models, or access cloud
endpoints. `--help` works without importing the optional database dependencies.

From the repository root, supply the server's actual loopback port and new output
paths. Existing result files are refused so another run cannot overwrite them.

```sh
PYTHONPATH=packages/memory/src python packages/memory/benchmarks/qdrant_payload_indexes.py \
  --url http://127.0.0.1:6333 \
  --output bench-runs/qdrant-storage-repeat/payload-indexes.json

PYTHONPATH=packages/memory/src python packages/memory/benchmarks/qdrant_hnsw_effort.py \
  --url http://127.0.0.1:6333 \
  --output bench-runs/qdrant-storage-repeat/hnsw-effort.json
```

Only loopback HTTP(S) URLs without credentials, query parameters, or fragments
are accepted. Each runner creates its own random UUID collection, never uses an
existing collection, and attempts deletion in `finally`. The JSON includes the
cleanup result or failure alongside any partial measurements. Hard termination
such as `SIGKILL` cannot execute cleanup; the printed collection name identifies
the runner's resource. Generated JSON belongs in ignored `bench-runs/`, not Git.

Both experiments fix the corpus at 20,000 normalized 64-dimensional float32
vectors from NumPy's `default_rng(413)`. Query vectors are drawn from the same
generator after the corpus. Payloads assign format `image` to every tenth point,
and cycle 100 entity IDs in groups of ten. Format, entity, and combined filters
therefore match 10%, 1%, and 0.1% of the corpus. Every query asks for ten results.

## Payload index comparison

`qdrant_payload_indexes.py` uses eight distinct query vectors per filter. It
collects exact references, warms each query once, and runs five repetitions
first without metadata indexes, then with keyword indexes on
`meta.document_format` and `meta.entity_id`. The stages remain in that fixed
order. It records all 240 timed requests, returned IDs/scores, exact references,
median/p95 latency, index status, and correctness differences. The timed path is
the Scone Qdrant adapter.

This experiment leaves the server's vector-index creation defaults intact. In
the original run, `indexed_vectors` was zero throughout: those results measure
payload filtering, not HNSW scaling. Another server configuration can build
vector indexes for this corpus, so always inspect the recorded index counts.

## HNSW search effort comparison

`qdrant_hnsw_effort.py` explicitly precreates only its own collection with one
shard, `m=16`, `ef_construct=100`, `full_scan_threshold=16` KB, and one indexing
thread. It disables indexing during upload, then enables an
`indexing_threshold=64` KB. Before timing queries, it waits for all 20,000 vectors
to be indexed and for green/OK optimizer state, with a 180-second deadline.
The reported build time is observed readiness after enabling indexing; polling
every two seconds makes it an upper observation, not precise CPU build time.

It collects exact references for 24 distinct queries per scope, including an
unfiltered scope. Search effort runs in the fixed order 32, 64, 128, 256. Each
scope/effort combination warms each query once and runs three repetitions,
producing 1,152 timed requests. The JSON records every returned/reference pair,
recall@10, median/p95 latency, server/client versions, collection settings, upload
time, and observed build time. The timed path calls the Qdrant client directly.

## Interpretation limits

These are small synthetic experiments on a shared machine with fixed execution
order. Repetitions reuse the same query vectors; they are not independent test
questions. Percentiles include client and loopback transport overhead. Exact
reference collection is separate from the warmed timing schedule.

A built HNSW index does not prove that every query traversed the graph. Qdrant
can choose full scans for selective filters, including per-segment decisions.
This protocol does not establish ten-million-vector capacity, production
latency, ingestion throughput under concurrency, memory requirements, semantic
retrieval quality, or a universal recommended `hnsw_ef` value.
