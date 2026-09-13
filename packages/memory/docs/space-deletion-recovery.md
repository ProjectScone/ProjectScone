# Recover interrupted space deletion

`await engine.delete_space("old")` durably accepts deletion before removing any
attachment holds or source rows. Once accepted, the space refuses further engine
writes and HTTP access. The cleanup record survives deletion of the space's own
rows and is removed only after attachment, document, vector and event cleanup
all acknowledge success.

If a storage operation fails, repair its availability and call
`await engine.delete_space("old")` again while its cleanup record remains, or
`await engine.recover()`. Opening the engine also processes pending space
cleanup, before vector identity settlement can rebuild embeddings from deleted
source content. Use the same storage configuration when recovering; an intent
cannot find bytes or vectors in a backend that has been disconnected or replaced.
HTTP credentials for a closed space remain refused; recovery runs through the
locally managed engine, not that space's data routes.

```python
report = await engine.recover(space_deletion_limit=100, retirement_limit=100)
while report.space_deletions_pending or report.retirements_pending:
    report = await engine.recover(space_deletion_limit=100, retirement_limit=100)
```

Each limit accepts integers from 1 through 1000. Whole-space cleanup runs before
individual source cleanup and ingestion repair. `spaces_deleted` counts space
intents completed by that recovery call; `space_deletions_pending` discloses a
remaining backlog. An unfinished space backlog defers source and ingestion
recovery. `open()` refuses to serve a remaining backlog after its bounded pass;
run additional recovery calls before opening again. Storage failures propagate
and retain the intent, so a caller controls retries and backoff.

The deletion receipt retains the accepted impact snapshot and original timestamp
across retries. Counts describe the data accepted for deletion, including data
already removed by an earlier attempt; they are not counts of writes performed
by the last retry. Attachment released/kept lists describe the original hold
snapshot. `deleted_at` is returned only after cleanup completes, using the
original acceptance time. If final acknowledgment succeeds but its response is
lost, a later deletion call sees an already closed space; there is no permanent
receipt archive.

## Storage contract

All built-in document stores implement the `SpaceDeletionStore` catalog. SQLite,
PostgreSQL, MongoDB and Elasticsearch retain it in a separate table, collection
or index; memory storage retains it for its object lifetime. Existing SQLite
catalogs get the table additively on open. Elasticsearch forces catalog write and
clear visibility even when bulk refresh is disabled. Catalog reads validate both
payloads and their stored identities; engine consumption additionally validates
custom-store responses against the requested space.

The intent contains identities, the original impact receipt and timestamp, never
source text or attachment bytes. It retains live chunk IDs and IDs from pending
individual source cleanups separately. Vector adapters without whole-space
removal can therefore finish cleanup after the source rows and retirement records
have disappeared. Their fallback removes these known IDs; it does not discover
pre-existing orphan vectors with no document or retirement identity.

File blob cleanup removes unshared bytes before discarding the space's attachment
metadata. A failed byte unlink therefore leaves durable targets for another
attempt. Shared bytes remain while another space holds them. Metadata removal
errors propagate; they are not treated as successful cleanup.

Custom document stores must implement callable catalog methods, `delete_space`
and a complete `chunk_index` before accepting new space deletion. Catalog writes
must preserve the first intent until acknowledgment, survive space deletion,
return detached validated values and support bounded keyset pages. Unsupported
stores refuse before destructive writes. This does not reconstruct deletions
that began on an older implementation without an intent.

## Consistency and space merge

Serialize deletion and recovery with writers to the affected spaces and attachment
holds. The catalog provides restart recovery across independent stores. It is
not a distributed transaction, concurrent writer lock, atomic authorization
fence or rollback mechanism. Reads already in flight are not canceled.

After a merge verifies copied evidence and starts source deletion, interrupted
cleanup now uses this journal. Recovery finishes source erasure; it does not
re-run destination copying or reconstruct a lost merge receipt. Copy/link failures
before source deletion retain the ordinary merge retry path. See
[space movement](space-merge.md) for those limits.
