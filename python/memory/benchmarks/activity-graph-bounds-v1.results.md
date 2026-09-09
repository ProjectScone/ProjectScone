# Activity graph fact-read measurements

This is a synthetic storage-work experiment, not a retrieval or answer-accuracy benchmark.

## Fixture and method

- Baseline: `f6f19d6` (activity assembly extracted without changing reads).
- Candidate: bounded activity fact reads, default budget 400, on `feature/bounded-activity-graph`.
- Python 3.14.7; SQLite in memory; 5,000 synthetic facts in one space.
- The final five facts cite the focused episode; the others have no source.
- Empty in-memory event/vector stores, a hash embedder, and no model calls.
- Five sequential graph calls per case; setup excluded. Reported timings and traced peak allocations are medians. Tracing runs during timing; these are not RSS measurements.
- The same fixture script ran before and after. This shared-host result is descriptive, not a portable latency guarantee.

## Results

| Case | Fact rows read, before → after | Claims shown, before → after | Median time with tracing, before → after | Median traced peak, before → after |
|---|---:|---:|---:|---:|
| Focused | 5,000 → 5 | 5 → 5 | 203.040 → 0.302 ms | 8,087,088 → 18,451 bytes |
| Unfocused | 5,000 → 401 | 5,000 → 400 | 730.775 → 14.844 ms | 10,833,030 → 865,751 bytes |

Every fact-read count was identical across the five calls for its case. All preexisting focused graph payload fields matched exactly, including nodes, edges, ordering and provenance. The candidate adds fact coverage fields.

The unfocused result intentionally changes: 400 claims are returned with one extra row read to detect truncation. Both `truncated` and `facts_truncated` are true. Its faster time therefore includes reduced output assembly, not just a faster equivalent-output query. Incident-link reads and total graph size remain outside this fact budget.

## Artifact integrity

Raw output remains in the ignored `bench-runs/bounded-activity-graph-2026-09-09/` directory. JSON stores per-call row counts, median measurements and graph payloads; it does not retain individual timing samples.

- `measure.py` SHA-256: `0831e2e444bbf432f6375a4c59d5948be0c9b80145314514dd628db3525fac8a`
- `before.json` SHA-256: `8fc6f9d2bd5c3a9b3e60b02a3deebecd92ecc8970c310da26221ead5def989e3`
- `after.json` SHA-256: `6fcaf600d7be349ba61afe26864fe1dc3abdb11d6e5f94370abe10679652c673`
