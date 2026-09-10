# Storage scaling: measurements and validation gates

Retrieval correctness comes first. A faster configuration must preserve tenant,
entity, metadata and temporal filters, while its approximate-search recall is
measured against an exact reference. Small-fixture timing does not establish
capacity or latency at ten million vectors.

## Measurements on 2026-09-10

The [Qdrant experiment protocol](../benchmarks/qdrant-storage-v1.protocol.md)
provides reproducible runners, fixed seeds, query schedules, server settings and
raw-output fields. Each runner saves per-query timings, returned IDs, exact
references and cleanup status. Raw run files remain separate from source control;
reported medians and recall are derived from those retained observations.

### Image context parsing

A fixed supplied HTML document with 300 image occurrences and 412,387 bytes was
parsed seven times before and after removing repeated hashing of the same source.
Median latency was 44.45 ms before and 2.90 ms after. Output provenance still
retains the same source digest for every occurrence. This is a local parser
microbenchmark, not generation or end-to-end retrieval performance.

### Qdrant payload indexes

A real, self-managed Qdrant server was tested with 20,000 deterministic 64-D
vectors in a disposable collection. The final recorded run used 240 timed queries,
fixed exact reference results, and identical requests before/after payload indexes.
All returned IDs and scores matched the references.

| Filter selectivity | Median before / indexed | p95 before / indexed |
|---|---:|---:|
| document format, 10% | 5.261 / 1.876 ms | 6.540 / 2.659 ms |
| entity ID, 1% | 5.180 / 2.082 ms | 5.879 / 2.968 ms |
| combined, 0.1% | 5.729 / 1.941 ms | 6.603 / 2.900 ms |

This fixture had zero HNSW-indexed vectors in both configurations. It demonstrates
payload-filter benefits only. The inspected server defaults were `m=16`,
`ef_construct=100`, indexing threshold 10,000 KB and full-scan threshold 10,000 KB.
The 64-D float32 vectors totaled 5.12 MB, below that indexing threshold.
Qdrant's optimizer builds indexes by segment size; the query planner can still
choose exact scans for small filtered subsets even when HNSW is available.

Configure only the scalar metadata fields the workload actually filters:

```python
vectors = QdrantVectorIndex(
    "http://127.0.0.1:6333",
    metadata_indexes=("document_format", "entity_id"),
)
```

The equivalent environment variable is
`SCONE_QDRANT_METADATA_INDEXES=document_format,entity_id`. The default is empty,
which preserves existing collection configuration. Existing field-type conflicts
fail explicitly and interrupted payload-index creation can resume. Image entity
identity filters use the existing `tags` index; an `entity_id` metadata index is
useful only when an ingestion source actually supplies that scalar metadata field.

### Qdrant HNSW search effort

A separate disposable collection explicitly built HNSW for all 20,000 64-D
vectors. Qdrant 1.19.1 reported seven segments, 20,000 indexed vectors and a green
optimizer state. Readiness was observed 2.015 seconds after enabling indexing;
the two-second polling interval means this is not exact index-build CPU time.
The run retained 1,152 timed trials: 24 queries per scope, three repetitions,
four search-effort settings and four filter scopes, with exact reference results.

| `hnsw_ef` | Unfiltered recall@10 | Median | p95 |
|---:|---:|---:|---:|
| 32 | 94.17% | 1.862 ms | 2.837 ms |
| 64 | 97.92% | 1.904 ms | 2.940 ms |
| 128 | 100% | 1.975 ms | 3.141 ms |
| 256 | 100% | 2.101 ms | 2.944 ms |

Filtered scopes all achieved 100% in this fixture, but selective queries may
still use exact scans. These are synthetic-vector nearest-neighbor measurements,
not semantic retrieval or answer accuracy, and not ten-million-vector results.
Use `QdrantVectorIndex(hnsw_ef=128)` or `SCONE_QDRANT_HNSW_EF=128` to explicitly
choose query effort. Omission preserves server policy. This option changes neither
collection build settings nor the model's embeddings. Existing collections must
have matching dimensions and cosine distance before Scone adds payload indexes.

### S3 attachment links

With a 1 MiB object in the Moto emulator, each new or repeated link previously
issued a GET and downloaded 1 MiB. Version-specific full-object SHA-256 HEAD
verification now uses one HEAD and downloads zero body bytes. The checksum-only
change retained six strongly consistent metadata reads. Remeasuring after the
indexed catalog change records seven, including the catalog readiness check;
new links also use one metadata transaction, repeated links use none. Actual
image downloads still GET and hash the bytes. Legacy/composite/unsupported
checksums use the original GET+hash path.

These are request/byte counts, not measured AWS latency or throughput.

### Indexed S3 attachment catalog

Previously, `for_episode()` queried every hold in a space: 1,024 holds required
11 partition queries plus two control reads, and more holds were refused. The
versioned catalog now queries an episode partition and validates only its linked
holds. It removes that per-space count cap; the separate 1,024 episode-reference
limit on each hold remains.

The Moto fixture below keeps exactly one attachment in the requested episode.
Migration is measured separately from the subsequent lookup:

| Total space holds | Migration Query / GetItem / transaction calls | Subsequent lookup Query / GetItem / transaction calls |
|---:|---:|---:|
| 1 | 2 / 4 / 3 | 1 / 4 / 0 |
| 101 | 3 / 5 / 4 | 1 / 4 / 0 |
| 1,030 | 12 / 14 / 13 | 1 / 4 / 0 |

Lookup request counts now depend on the requested episode's attachments, not
unrelated space holds. Ownership validation adds point reads, so a tiny catalog
can use more total requests than the old scan. This is an access-pattern result,
not a universal latency gain. Legacy migration visits existing holds once and
checkpoints progress; it is not included in the steady-state result.

Read the [catalog migration and recovery guide](s3-catalog.md) before upgrading.
List-returning APIs still materialize their outputs, and engine-wide deletion
still performs an uncheckpointed preview before starting blob release. Those
paths, per-space write contention and provisioned-service throughput need more
work and measurement before claiming a ten-million-image catalog.

## Ten-million-vector target

Uncompressed float32 vectors alone require the following storage for one copy:

| Dimensions | 10,000,000 vectors |
|---:|---:|
| 384 | 15.36 GB |
| 768 | 30.72 GB |
| 1,536 | 61.44 GB |
| 3,584 | 143.36 GB |

For example, [Nomic Embed Code's published pooling configuration](https://huggingface.co/nomic-ai/nomic-embed-code/blob/main/1_Pooling/config.json)
uses 3,584 dimensions. The exact embedding model/output configuration determines
the collection dimension; the answer-generating LLM's parameter count does not.
The [live adapter smoke tests](../tests/backends/test_qdrant_index_recovery.py)
also cover 3,584-D vectors, scope/metadata filters and reopening a collection.
That verifies dimensional compatibility, not throughput at this dimension.

These arithmetic totals exclude HNSW, payloads, payload indexes, IDs, WAL,
replicas, snapshots, optimizer temporary space and the document/blob catalog.
They are not RAM recommendations. On-disk vectors, memory mapping and optional
quantization change RAM requirements and may change latency or recall.

The validation progression is 100k, 1M, then 10M vectors on provisioned test
hardware. At each stage record:

- Exact-ground-truth recall@k for unfiltered queries and several filter
  selectivities, including overlapping entity names and disjoint tenants.
- p50/p95/p99 latency and throughput under concurrent reads and ingestion.
- HNSW build progress/time, actual indexed counts, CPU, resident memory and disk
  high-water marks, including optimizer and snapshot overhead.
- Ingest batching/backpressure, delete/update visibility, retry behavior, restart
  recovery and replica/shard failures.
- The same metrics across HNSW search breadth, graph/build settings, storage
  placement and any quantization. Do not select configurations by latency alone.

Shards distribute storage and CPU; replicas provide availability and read
capacity while increasing storage and write cost. Their counts must follow these
measurements and the failure model, not a universal preset. Scone can connect to
operator-provisioned collections; it must not silently reconfigure existing
collections or trade recall for speed.

There is no ten-million-vector capacity result yet. OpenSearch and ElastiCache
command-contract tests also do not establish managed-service production capacity.

References: [Qdrant indexing](https://qdrant.tech/documentation/manage-data/indexing/),
[capacity planning](https://qdrant.tech/documentation/capacity-planning/), and
[S3 HEAD/checksum semantics](https://docs.aws.amazon.com/AmazonS3/latest/API/API_HeadObject.html).
