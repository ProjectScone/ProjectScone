# Storage adapters and validation scope

Scone separates document/evidence storage, vector retrieval and raw attachment
storage. Image context and entity associations use the same interfaces as other
sources. An S3 bucket stores originals; it is not itself a vector index. A service
sharing a wire protocol is not automatically validated for every engine/version.

## Implemented adapters

| Interface | Implementations |
|---|---|
| Document store | In-memory, SQLite, MongoDB, PostgreSQL, Elasticsearch |
| Vector index | In-memory, SQLite, Qdrant, Chroma, LanceDB, Milvus, PostgreSQL/pgvector, Redis Stack, Elasticsearch, OpenSearch, ElastiCache Valkey Search, LangChain bridge |
| Blob store | In-memory, filesystem, S3 with DynamoDB attachment metadata |
| Evidence/event log | In-memory, SQLite, MongoDB, PostgreSQL, Elasticsearch |

Each adapter must preserve the framework's cosine-score contract, tenant scope,
all required tags, exact scalar metadata equality, time bounds, dimensions and
write/delete behavior. Nested image attributes and entity evidence remain in
retained context manifests; vector stores receive small filterable fields.
The LangChain bridge depends on the connected store's filtering capabilities;
its bounded postfilter path can underfill results.

## AWS products: explicit support boundaries

| AWS service | Current integration | Validation boundary |
|---|---|---|
| S3 | Original bytes through `S3BlobStore` | Moto integrity/recovery tests; compatible-provider fallback; no live AWS run in this change |
| DynamoDB | S3 attachment ownership/catalog metadata | This is not a general `DynamoDocumentStore` or DynamoDB vector adapter |
| OpenSearch Service domains | Native `OpenSearchVectorIndex`; optional explicit SigV4 `es` authentication | Offline signing and command-contract tests; live domain/ANN behavior must be validated separately |
| OpenSearch Serverless | Not supported by this vector mapping | The adapter rejects `aoss` signing scope; Serverless has different vector-engine constraints |
| ElastiCache | Explicit Valkey Search vector adapter | Requires the documented search-capable deployment; plain Redis OSS/Memcached are not interchangeable vector stores |
| MemoryDB | No dedicated validated adapter yet | Similar search commands are not sufficient proof of compatibility |
| RDS/Aurora PostgreSQL | Existing PostgreSQL/pgvector protocol adapter | Provision pgvector and test the selected deployment/version; no separate AWS lifecycle or IAM adapter claim |
| DocumentDB | Not certified through the MongoDB adapter | MongoDB compatibility must be tested, not assumed |
| S3 Vectors, Neptune, other database products | No native adapter claim yet | Separate APIs/capabilities require implementations and contract suites |

SDK construction and adapter configuration do not provision AWS resources or
select/download models. Cloud endpoints and credentials require explicit caller
configuration. Defaults remain self-managed and inactive AWS placeholders stay
in `.env.example`. Secrets belong in ignored environment files or the caller's
secret-management system.

## OpenSearch

Install `scone-memory[opensearch]`, set `SCONE_VECTORS=opensearch`, and configure
`SCONE_OPENSEARCH_URL`. The adapter uses a pooled HTTP client, Lucene HNSW cosine
vectors, filters inside the k-NN query, and bounded bulk operations. It validates
existing mappings before reuse and rejects incompatible vector dimensions/metrics.
Optional basic authentication uses separate username/password settings; credentials
must not be embedded in endpoint URLs.

AWS domain authentication can use explicit SigV4 credentials. No default AWS
credential discovery, instance metadata lookup, role assumption, or resource
provisioning occurs. Refer to `.env.example` for the available settings. The
shared signer can sign allowed AWS service scopes; that does not make every
service compatible with the OpenSearch vector adapter.

Bulk writes are not atomic across documents or batches. Immediate visibility is
preserved by default; disabling refresh is an explicit ingestion tradeoff. Contract
tests inspect native requests and score/filter behavior; mocked responses cannot
validate real ANN accuracy or production throughput. A loopback-only opt-in live
OpenSearch test is available when `SCONE_TEST_OPENSEARCH_URL` is supplied.

## ElastiCache Valkey

ElastiCache Search has its own command details, including index metadata and
index deletion semantics. The adapter validates capabilities/schema, encodes
filter values without separator collisions, and avoids cross-slot multi-key
operations. Its engine settings and explicit endpoint are separate from the
Redis Stack adapter. ElastiCache vector search is documented for Valkey 8.2
node-based clusters; do not assume Redis OSS, Memcached or other deployment modes
support those commands.

## Fact placement index

Asserting a fact places it in its slot: the facts that share its space, subject
and predicate. Placing needs two things from the slot's history: the facts that
cover the new fact's start, and the first fact that starts after it. A document
store that only offers `facts_for` hands the engine the slot's whole history on
every assert, so filling a slot with N facts reads on the order of N² rows.

A document store can implement the optional `FactPlacementIndex` port
(`core/ports.py`) and answer that question from an index:

```python
async def facts_placing(space, subject, predicate, start, *, object=None, exclude_id=None) -> list[Fact]
```

The contract is a superset. The result must contain every ledger fact that
covers `start` and the earliest ledger fact that starts after it; it may contain
more, and the engine applies its exact rule to what comes back. The engine
passes `object` for a predicate that holds many objects at once, and the store
then returns that object's facts only. A store without the port is still
scanned, so a third-party store keeps working unchanged.

All five bundled document stores implement the port.

| Store | How the lookup is indexed |
|---|---|
| In-memory | Per slot, and per slot and object, two sorted lists (starts, ends) searched by bisection. |
| SQLite | Four expression indexes on the canonical start and end text; every statement names its index with `INDEXED BY`. The database maintains them, so rows written by another program on the same file are indexed too. |
| PostgreSQL | Four expression indexes keyed by a 64-bit hash of the slot (or slot and object). The statement contains only conditions an index answers; rows of another slot with the same hash and rows outside the ledger are dropped by the adapter. |
| MongoDB | Stored integer keys (`valid_from_us`, `valid_until_us`, microseconds since the epoch) with four compound indexes; every find names its index with `hint`. |
| Elasticsearch | The same two stored integer keys, queried with range filters. |

A timestamp is indexed as text only when it has the exact shape Scone writes
(`YYYY-MM-DDTHH:MM:SS.mmmZ`), because only that shape sorts as text the way it
sorts in time. A fact stored with another shape (a custom clock, an import) has
no key and is always returned as a candidate for the engine to compare as an
instant. MongoDB and Elasticsearch key existing facts when the store is opened;
until a fact is keyed it is returned as a candidate, never skipped.

### Measured

One slot, values asserted in time order through `MemoryEngine.assert_fact`, on a
laptop with the servers in local containers. "Before" is the scan.

| Case | Before | After |
|---|---:|---:|
| In-memory, 4,000 asserts | 6.93 s | 0.11 s |
| In-memory, 20,000 asserts | not run | 0.59 s (0.03 s per thousand throughout) |
| SQLite, 4,000 asserts | 37.79 s | 1.00 s |
| SQLite, 20,000 asserts | not run | 5.09 s (0.22–0.28 s per thousand throughout) |
| SQLite, one predicate holding 6,000 open objects | 69.65 s | 1.16 s |
| In-memory, one predicate holding 6,000 open objects | 2.92 s | 0.12 s |
| MongoDB, per 500 asserts | 2.5 s rising to 4.9 s by 3,000 | 3.2 s, flat to 5,000 |
| Elasticsearch, per 500 asserts | rising | 8.9–9.6 s to 1,500 |
| SQLite, one fact dated into the middle of 8,000 | 44.98 ms | 0.47 ms |
| In-memory, one fact dated into the middle of 8,000 | 7.58 ms | 0.22 ms |

The lookup alone, on a slot loaded in bulk:

| Facts in the slot | PostgreSQL, at the end | PostgreSQL, mid-history | MongoDB, at the end | MongoDB, mid-history |
|---:|---:|---:|---:|---:|
| 2,000 | 0.35 ms | 1.60 ms | 0.74 ms | 1.61 ms |
| 20,000 | 0.49 ms | 0.68 ms | 0.75 ms | 8.70 ms |
| 200,000 | 0.60 ms | 3.75 ms | 0.79 ms | 82.37 ms |

### Limits

- **A fact dated in the past is not logarithmic.** Appending reads the open
  fact and nothing else. A fact dated before existing facts also walks the
  index entries of the facts that end after it, without reading their rows:
  100,000 later facts cost 3.75 ms on PostgreSQL and 82 ms on MongoDB above. A
  reading that arrives a few facts late costs a few entries.
- **Writes maintain four more indexes.** On SQLite an append takes about a
  quarter longer with four placement indexes than with two.
- **PostgreSQL has no index hints.** Its plan is checked by a test on a slot of
  30,000 facts, with and without table statistics, that fails if the statement
  sorts or reads more than six rows. An earlier form of the statement returned
  the right facts while reading 9,999 rows to find one.
- **PostgreSQL asserts end to end were not flat.** Over 10,000 asserts in one
  slot the time per assert rose by about half (3.87 s for the first 2,500,
  6.01 s for the last). Timed per statement, every statement slowed by a
  similar factor, including single-row reads of tables that stayed empty, so
  the growth is not in the placement lookup (0.19 ms rising to 0.27 ms). The
  update that closes the previous fact slowed most, 0.22 ms to 0.49 ms; the
  run left 9,737 dead rows in `facts`. The cause has not been established.
- **Opening an existing store builds the indexes.** SQLite and PostgreSQL
  create them on open; MongoDB and Elasticsearch also write the keys into
  every existing fact. On a large store the first open after upgrading takes
  correspondingly long.
- **Elasticsearch is dominated by its refresh.** Each assert waits for the
  index to refresh; the placement lookup is a small part of the 18 ms.
- **The scan path on Elasticsearch reads at most 10,000 facts per slot.**
  `facts_for` there has a fixed size. The engine no longer uses it for
  placement, but other callers of `facts_for` on a longer slot get a truncated
  history without being told.

## Performance evidence and outstanding work

See [scaling validation](scaling-validation.md) for measured Qdrant payload-filter
latency, S3 request/byte reductions, remaining catalog limits, and the unverified
ten-million-vector capacity target. No adapter is described as universally optimal.
Supported filters can still interact with bounded candidate budgets and ANN recall;
measure them against exact references for the real workload.

The shared image/backend contract test runs on configured available backend pairs.
Services that are absent are skipped and remain unverified in that run. Optional
request-contract tests cover additional adapters without contacting hosted services.

References: [ElastiCache Search features and limits](https://docs.aws.amazon.com/AmazonElastiCache/latest/dg/search-features-limits.html),
[OpenSearch efficient k-NN filtering](https://docs.opensearch.org/latest/vector-search/filter-search-knn/efficient-knn-filtering/),
and [S3 HEAD semantics](https://docs.aws.amazon.com/AmazonS3/latest/API/API_HeadObject.html).
