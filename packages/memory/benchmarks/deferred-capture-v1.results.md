# Deferred capture scheduling — October 3, 2026

Controlled local comparison of the default inline path and opt-in deferred text
capture. Five samples per mode, alternating inline then deferred. Each sample
uses its own SQLite document/vector database, the same retained calibration note,
question and scripted public reply. Hash vectors have a deliberate 250 ms delay
per document embedding call; query embeddings have no added delay. No external
model, service, private dataset or live memory store is used.

Command, from `packages/memory`:

```sh
PYTHONPATH=src python benchmarks/deferred_capture.py --runs 5 --delay 0.25
```

| Median | Inline | Deferred |
| --- | ---: | ---: |
| First public text | 254.019 ms | 1.661 ms |
| Completed reply with retained user/assistant sources | 507.508 ms | 2.595 ms |
| Both sources fully indexed | 507.514 ms | 507.246 ms |

Deferred mode moves the two document embedding waits out of response delivery;
it does not remove indexing work. All samples drain admitted jobs and verify
retained source IDs, an empty ingestion journal, and vector recall before
cleanup. Two document embedding calls occur in each mode. Query embedding and
generation latency are not improved or simulated here. The approximately 99%
delivery reduction is specific to this controlled delay, not a product-wide
speed or real-model benchmark.

The test suites separately cover blocked embeddings, bounded retry/overflow,
joined shutdown, pending-source deletion, identity drift, uncertain write
acknowledgment and native restart recovery. Synthetic source-backed answers
verify provenance and lifecycle, not generated-answer correctness. Current
limits: exhausted/overflow jobs need explicit recovery or reopening; startup
recovery may still wait on the embedder; capture statuses are acknowledgment
snapshots; ordinary inline capture remains the default.

A separate local smoke check used the installed `nomic-embed-text:latest`
(768 dimensions) through Ollama's loopback `/v1/embeddings` endpoint, native
`RemoteEmbedder`, temporary SQLite and `FileBlobStore`. Deferred capture returned
`pending`; after the owned worker drained, vector-only recall returned the
captured source and no ingestion intent remained. Reopening storage returned
the same episode ID and exact text. No model downloads, generation calls or
live-store changes were involved. This one-source check verifies the real
embedding lifecycle, not retrieval accuracy or comparative model latency.

Verification on the final code: conversation/configuration suite 845 passed,
45 skipped; benchmark suite 489 passed, 7 skipped; framework mypy checked
514 source files. The broader framework run recorded 16,115 passed, 601 skipped
and one source-snapshot mismatch: the evaluator captured `realtime/text.py`
before the final reply-size guard was added, then compared its hash to the
edited file. The unchanged provenance test passed in the fresh benchmark run.
