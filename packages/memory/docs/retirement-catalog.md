# Pending source cleanup records

The document backends expose `scone_memory.core.retirement.RetirementStore` for
durable source cleanup intents. This is an internal recovery seam, separate from
public tombstones and export data. It does not by itself change `forget()` or
provide a transaction across document, vector, blob and event stores.

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

Callers serialize source mutations and recovery for a store. This seam provides
neither distributed ownership nor automatic retry scheduling. Clearing an intent
before every cleanup operation completes loses the retry record.

SQLite chunk IDs now have a persistent monotonic counter, initialized under a
write transaction when a catalog opens and committed with each inserted batch.
It is seeded above existing chunks and locally stored vectors. This prevents
delayed cleanup from targeting a newer chunk after deletion or restart. Upgrade
cannot reconstruct IDs already erased by an older build, or orphan vector IDs
held only in a separate external vector service. Keep older writers stopped
while upgrading and recovering a catalog.
