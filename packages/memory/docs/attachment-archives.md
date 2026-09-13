# Export and import retained attachments

Use `engine.export(space, include_attachments=True)` or
`scone-memory --space alpha export --include-attachments` to select
`scone.archive/2`. The default export stays `scone.archive/1`, which omits
attachment bytes. `engine.import_records` and the CLI `import` command recognize
both profiles automatically.

```python
records = [row async for row in source.export("alpha", include_attachments=True)]
summary = await target.import_records("beta", records)
print(summary.attachments, summary.attachment_links)
```

Profile 2 carries episodes, facts, fact links and affirmations using the existing
ledger import rules, plus verified attachment bytes and ordered episode links.
An attachment shared by several episodes is encoded once. Episode and fact IDs
are remapped for the destination. Text is re-chunked and re-embedded through
normal ingestion using the destination's configured embedder. Textless retained
video sources use the existing manifest verification path; they require no
parser, decoder, OCR or language-model call during import.

Each attachment row contains its SHA-256 identity, media type, byte count,
filename and canonical base64 bytes. Episode rows contain `attachment_ids`.
Import checks the complete attachment graph, source spaces, duplicate ledger
identities and dangling ledger references before any write. Unknown fields and
profiles refuse. Caller-owned nested data is detached before asynchronous work.
A source whose ledger points to an episode or fact missing from the archive
cannot be exported using profile 2; resolve those references or use the legacy
profile with its existing lossy reference remapping.

Export verifies attachment hashes and metadata, then rechecks episode identities
and attachment links before yielding its header. This is a consistency check
across separate reads, **not an atomic database snapshot**. Quiesce source writes
when a consistent whole-space backup is required. Unlinked attachment holds are
not transferred; their count appears as `not_carried.unlinked_attachments`.
Chunks, vectors, jobs, tombstones, event logs and runtime configuration do not
travel in either profile. Keep document and blob storage backups for full-store
recovery.

The ledger import policies remain the same as profile 1: target exclusions are
preserved, [fact replacement edges are remapped](archive-replacement-history.md), and source references
to episodes skipped by the destination's tombstones are dropped. Facts and quotes
remain under those existing rules. Profile 2 is an attachment transfer format,
not a lossless ledger or policy snapshot. For whole-space movement, [space merge](space-merge.md) additionally carries
unlinked attachment holds, reports known forgotten-source reference omissions,
and verifies retained evidence before closing the source.

## Limits and retries

Both export and import enforce 100,000 records, 10,000 distinct attachments and
256 MiB of aggregate decoded attachment bytes. The engine's
`max_attachment_bytes` limit also applies to each attachment (25 MiB by default).
Filenames must be valid UTF-8 and at most 4,096 characters. Archives are buffered
and preflighted in memory; base64 increases their serialized size. This is not a
streaming format for arbitrarily large repositories.

Import first checks destination episode identities and attachment collisions,
then stages bytes, ingests episodes and repairs their links before importing the
ledger. Existing content-addressed bytes with different metadata or corrupted
content refuse; import does not silently rename or relabel evidence. A forgotten
episode does not regain its attachments unless `resurrect=True` was explicitly
requested. The tombstone itself remains.

A failure is reported, but the import is **not a transaction across stores**.
Bytes or episodes can remain after a later operation fails. Retry the same archive
to deduplicate completed writes and repair missing links. Do not delete the source
until the import has returned successfully and its evidence has been checked.
The summary counts verified attachments and episode links, including those already
present; these counts do not represent newly allocated storage.

Base64 is an encoding, not encryption. The export API yields records and the CLI
writes JSON lines to stdout. Store sensitive archives through your encrypted
storage or backup pipeline; the archive format does not manage encryption keys.
