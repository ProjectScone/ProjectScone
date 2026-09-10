# Optional AWS backend map

The first end-to-end slice is `S3BlobStore` behind the existing attachment
port, with the current document, vector and event backends unchanged. Attachment
bytes belong in S3; DynamoDB stores digest identity, immutable generation keys,
per-space holds, episode links and recoverable upload/delete intents. The table
uses string `pk`/`sk` and no GSI. Knowing a shared digest must never grant a space
access: every operation checks that space's hold. A root-version compare-and-swap
and a metadata transaction fence concurrent acquisition and release. S3 and
DynamoDB cannot share a transaction, so an interrupted upload or cleanup must
remain recoverable through durable intent records.

The package ports are in `packages/memory/src/scone_memory/core/ports.py`; the
attachment port is in `backends/blobs.py`. `runtime/config.py` composes concrete
adapters into `memory/engine.py`. AWS types and SDK configuration should stay in
the adapter and composition layer; domain authorization, provenance and evidence
validation remain backend-independent.

| Port / future implementation | Access patterns and invariants |
| --- | --- |
| BlobStore (`backends/aws_blobs.py`) | Digest lookup plus space hold check; per-episode ordered references; first metadata per space; all holds/links released on space erasure; immutable generation upload and version-specific deletion after the final hold; durable retryable intent state. |
| DocumentStore (`backends/dynamodb.py`, future) | Episodes by ID and content hash; bounded recent/source inventory; chunks by ID and episode ordinal; globally unique chunk IDs for vector deletion; facts by subject/predicate and status/history; both-end link adjacency and idempotent link identity; deletion tombstones, revisions, jobs and inflight recovery. |
| EventLog (`backends/dynamodb_events.py`, future) | Stable committed cursor order, deduplication, pagination and retention. Couple counter CAS, event publication and dedup state transactionally: publishing a lower allocated ID after readers advance would skip that event permanently. |
| Lexical search / FactSearch (future) | Exact existing tokenization, filtering and deterministic ranking; bounded indexed postings and rebuild/recovery metadata, or a separate search adapter. A paginated scan may provide diagnostic parity but is not a scalable search design. |
| VectorIndex (existing separate port) | Continue using an explicit existing vector backend; S3 and ordinary DynamoDB key lookups do not replace embedding similarity search. Preserve IDs, namespaces, filters and deletion behavior. |

Scone accepts episodes up to 2,000,000 UTF-8 bytes. DynamoDB's 400 KB item limit
requires an explicit overflow design for episode bodies rather than silently
shrinking the public limit. DynamoDB transactions also have item-count and
aggregate-size bounds, so large inserts and space deletions need durable progress
and a deny-first tombstone/revision fence. [DynamoDB constraints](https://docs.aws.amazon.com/amazondynamodb/latest/developerguide/Constraints.html)
and [S3 overflow guidance](https://docs.aws.amazon.com/amazondynamodb/latest/developerguide/bp-use-s3-too.html).

Apply scope/session, tags, literal source prefixes, time bounds, history/status
and structured conditions with the same semantics as the current stores.
DynamoDB filters run after its query read limit; iterate pages until eligibility
and requested limits are satisfied. Authorization, tombstones and source revision
checks need strongly consistent base-key reads; an eventually consistent GSI
cannot establish permission. [Query API behavior](https://docs.aws.amazon.com/amazondynamodb/latest/APIReference/API_Query.html).

Deletion is a state transition before asynchronous physical cleanup. Expired TTL
items can still be returned, so TTL cannot authorize reads or establish completed
erasure. Versioned S3 deletion must remove the referenced version, not just add
a marker. Backup/PITR retention must be part of the operator's erasure contract.
[TTL behavior](https://docs.aws.amazon.com/amazondynamodb/latest/developerguide/ttl-expired-items.html)
and [S3 delete markers](https://docs.aws.amazon.com/AmazonS3/latest/userguide/ManagingDelMarkers.html).

Use a loopback-only AWS emulator with synthetic credentials and input. Run the
existing attachment and backend contracts against it, persist emulator storage
across restart, and test cross-space shared digests, unauthorized known-digest
reads, concurrent same-byte writes, last-holder deletion/reacquisition races,
ambiguous S3 success, failed metadata commits, idempotent cleanup and versioned
physical deletion. For later document/event adapters, reuse the contract,
inventory, source, retention, tombstone and recovery suites. Force pagination,
400 KB overflow and large operation boundaries. Inject throttling, transaction
conflicts and partial batch failures: local emulators do not faithfully reproduce
all cloud throughput/concurrency behavior. [DynamoDB Local differences](https://docs.aws.amazon.com/amazondynamodb/latest/developerguide/DynamoDBLocal.UsageNotes.html).

Cloud deployment still needs the planned document and journal adapters, TLS
ingress, secrets delivery, compute sizing and persistence, task execution role,
monitoring, backup/restore exercises and tests under actual AWS IAM/KMS behavior.
The Terraform foundation and synthetic container smoke prove none of
those production properties by themselves.

The deployment target after these ports are implemented is S3 raw/artifact data,
DynamoDB metadata/jobs/conversations, SQS ingestion, Fargate API/workers from ECR,
and static frontend assets through private S3/CloudFront with same-origin API
routing. The recommended split keeps API/workers on Fargate and puts only Qdrant
on ECS EC2 capacity, in separate services/tasks in the same cluster. Qdrant needs
a POSIX-compatible block filesystem, not S3 or NFS/EFS. Its encrypted EBS volume
must outlive tasks and instances, with delete-on-termination disabled,
same-AZ reattachment, single-writer fencing, mount/readiness checks and snapshots.
Replication and tested recovery objectives remain necessary for availability.
An all-Fargate alternative requires immutable S3 snapshots and a durable
DynamoDB/S3 indexing outbox, consistent snapshot watermarks, idempotent replay
including deletions, and restore-before-ready replacement tasks: ephemeral and
service-managed EBS storage cannot be its sole durable boundary. These compute
resources, snapshot/replay controllers and recovery tests are future work;
the current attachment module deploys neither topology.
[Qdrant storage requirements](https://qdrant.tech/documentation/installation/)
and [ECS EBS lifecycle](https://docs.aws.amazon.com/AmazonECS/latest/developerguide/configure-ebs-volume.html).
