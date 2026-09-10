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
