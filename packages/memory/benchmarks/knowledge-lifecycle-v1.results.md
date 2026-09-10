# Knowledge lifecycle: a running framework server

This operational experiment launches the real `scone-memory serve` CLI and
uses HTTP over TCP, SQLite, file attachments, and an existing self-managed
Qdrant instance. It uses no pytest runner, ASGI test transport, fake database
client, hosted service, or model download.

## Fixed inputs and scope

- One two-page generated PDF: “Juniper calibration uses Polaris.” and
  “The deployment window opens on Friday.”
- One 32×32 yellow PNG with supplied caption “A yellow calibration tile.” and
  an explicit `fixture:tile` entity-to-caption reference.
- `HashEmbedder`, 256 dimensions. This is a deterministic lexical fixture,
  **not semantic retrieval or generation quality evaluation**.
- A fresh SQLite database/blob directory and UUID-named Qdrant collection;
  two independently authorized spaces. The collection is deleted and its
  absence verified after the run. Existing collections are not altered.

## Recorded September 10, 2026

The first run completed 26 workflow checks before shutdown (28 including
collection cleanup) and failed. SIGTERM terminated the process before the
launcher reached its owned-engine cleanup. A separate subprocess regression
confirmed the cleanup boundary was skipped. Uvicorn replays termination after
ASGI shutdown; the launcher now converts that replay to Python unwinding so
the final backend cleanup runs before exit status 143.

After the fix, **54 checks passed across three server launches**:

1. Upload PDF/image originals, parse/index their context, deduplicate a PDF
   retry, search, resolve a PDF chunk to page one, and retrieve unchanged bytes.
2. Reject unauthenticated access and cross-space ingestion/downloads; return
   no cross-space image matches.
3. Stop and restart on the same storage; repeat retrieval and byte checks.
4. Forget both episodes; verify source revocation and empty recall/image search.
5. Restart again; verify deletion and revoked PDF provenance persist.
6. Delete only the run's Qdrant collection and verify cleanup.

These are explicit assertions, including HTTP status assertions, not 54 unique
product capabilities or QA questions. The source inventory contains two fixtures,
so this does not establish production capacity, ANN recall, OCR accuracy,
concurrency performance, or full-product readiness. No LLM was invoked.

Illustrative timings from the first successful run on Python 3.14.7:

| Request | Observed wall time |
|---|---:|
| PDF parse and index | 172.82 ms |
| Identical PDF retry | 156.78 ms |
| Image context index | 140.79 ms |
| PDF recall, before / after restart | 8.08 / 7.99 ms |
| Entity image search, before / after restart | 5.07 / 9.03 ms |

These are individual observations, not percentile estimates or speedup claims.
PDF retries still parse before recognizing the same extraction identity.

## Reproduce

Install the framework with `api,pdf,images,qdrant` extras. Supply an existing
loopback Qdrant endpoint and a **new** output directory, from the repository root:

```sh
PYTHONPATH=packages/memory/src python packages/memory/benchmarks/knowledge_lifecycle.py \
  --qdrant-url http://127.0.0.1:6333 \
  --output ./bench-runs/knowledge-lifecycle-new
```

The runner refuses non-loopback Qdrant URLs and existing output directories.
It generates temporary API keys in process memory, starts servers on available
loopback ports, and preserves fixtures, server logs, SQLite/blob state and
`results.json`. Results include input hashes, source hash, dependency versions,
individual request timings, returned rankings/sources, and cleanup status.
No generated artifacts or keys belong in version control. Failed runs retain
their own output directories rather than overwriting successful evidence.

The original private raw runs are retained as
`bench-runs/knowledge-lifecycle-2026-09-10/run-1` and `run-2` in the development
workspace. This report preserves their failure and successful measurements.
The final reviewed runner passed the same 54 checks in `run-3`, recording
framework source SHA-256
`3b9f4eb652a0add39d3e06f5e4678ceca3b5e8e6be7d8ecdeda537d8bb51eb4c`.
Its dependencies were pypdf 6.18.0, Pillow 12.3.0, httpx 0.28.1,
qdrant-client 1.19.0 and Uvicorn 0.52.4. The timings above remain the original
`run-2` observations; they were not replaced with the fastest later measurements.

HTTP currently exposes PDF text-layer ingestion. OCR remains an explicitly
configured native-Python pipeline; semantic retrieval, model answering,
OpenSearch and ElastiCache live-service validation need separate experiments.
