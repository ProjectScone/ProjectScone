# Pending source cleanup records

`MemoryEngine.forget()` writes a durable source cleanup intent before deleting
rows, vectors or attachment references. Retrying the same episode resumes that
intent, including when the original row is already gone. This is separate from
public tombstones and export data; it is not a transaction across independent
stores. Claims and links that cite the source continue to stand.

The document backends expose `scone_memory.core.retirement.RetirementStore` for
these intents. Custom document stores must implement this optional seam to use
`forget()`; unsupported stores are refused before any destructive operation.

A `Retirement` retains the source's space, episode ID, content hash, request time,
original chunk IDs and impact preview. It retains no source text or attachment
bytes. The first record for `(space, episode_id)` wins until explicitly cleared;
a retry cannot replace its cleanup targets. Returned records are detached, and
copied models are revalidated before persistence. Invalid identities, duplicate
chunk IDs and mismatched receipt counts are refused. Stored payload identities
must agree with their catalog keys.

`page_retirements(after, limit)` orders by `(space, episode_id)`. The limit is
1–101 and the cursor is exclusive, including when its record has been removed.
`retirement(space, episode_id)` reads one intent. `clear_retirement` acknowledges
finished cleanup and is idempotent. Whole-space deletion also clears its intents.

Memory stores retain records for their object's lifetime. SQLite creates an
additive table after schema validation; PostgreSQL creates a table, MongoDB a
collection with a unique compound index, and Elasticsearch a dedicated index.
Elasticsearch forces refresh for intent writes and acknowledgements even when
bulk data writes use delayed refresh. Existing source rows and public tombstone
schemas are unchanged. Persistence inherits the configured catalog's durability
and encryption; the intent is not a separately encrypted backup.

Callers serialize source mutations and recovery for a store. After a failed
operation, recover before resuming source writes. A normal `forget()` refuses a
source with unfinished indexing. The intent is cleared only after row removal
has been checked, vectors and attachments are cleaned, the tombstone is written,
the revision is invalidated and the configured event sink acknowledges the event.
Failures propagate and retain the intent for retry. File attachment cleanup keeps
its original target list until every filesystem deletion has succeeded.

`await memory.recover(retirement_limit=100)` finishes pending source deletions
before ingestion repair. The limit accepts 1–1000 deletions per call. Its
`RecoveryReport.retired` counts completed deletions and `retirements_pending`
reports a remaining backlog. With a backlog, ingestion repair waits. A store
failure raises and leaves its intent. `open()` runs the default pass and refuses
to return an engine while cleanup remains; call `recover()` again and reopen.
Recovery must use the same document, vector, blob and event stores.

The final receipt retains the original impact preview, with `forgotten_at` set to
the recorded deletion-request time after cleanup succeeds. Its source citations
and planned attachment releases describe that preview. The forget event uses a
stable deduplication key and payload, so a lost acknowledgement can be retried. It
does not claim a latency across a process interruption. A repeated forget after
the intent has been acknowledged retains the existing `Gone` behavior.

Directory synchronization resumes catalog intents only after matching the exact
managed source identity. Missing rows without an intent or tombstone still fail
closed. Records lost before this feature cannot be reconstructed automatically.
This does not provide distributed ownership, a background retry service,
transactional whole-space deletion or a backup that spans all stores. Keep a
catalog's pending intents with its vector and blob data when recovering it.

SQLite chunk IDs now have a persistent monotonic counter, initialized under a
write transaction when a catalog opens and committed with each inserted batch.
It is seeded above existing chunks and locally stored vectors. This prevents
delayed cleanup from targeting a newer chunk after deletion or restart. Upgrade
cannot reconstruct IDs already erased by an older build, or orphan vector IDs
held only in a separate external vector service. Keep older writers stopped
while upgrading and recovering a catalog.
