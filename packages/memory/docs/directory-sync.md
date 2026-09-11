# Incremental local directory ingestion

`DirectorySync` manages a collection of local files through an encrypted journal.
It indexes new files, replaces changed revisions, and reuses unchanged sources
without parsing or embedding them again. `delete_missing=True` additionally
retires managed episodes whose files are absent from a complete, stable scan.
The original files are never modified or deleted.

Install `scone-memory[document-workflows]`. This uses the existing document
parsers and local POSIX filesystem facilities; it starts no server or model
service. The native API accepts a configured parser, including an explicitly
configured OCR PDF parser. See [format coverage](file-ingestion.md) for extraction
limitations and optional converter dependencies. Unsupported suffixes are counted
as skipped; a supported file that cannot be parsed produces a failed receipt.

## CLI

Keep the key, journal and memory stores outside the source directory. Create a
private state directory and a random key once; preserve both key and journal for
subsequent runs:

```sh
mkdir -m 700 ./source-state
python3 - <<'PY'
import os
fd = os.open('./source-state/journal.key', os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
with os.fdopen(fd, 'wb') as key:
    key.write(os.urandom(32))
PY

SCONE_SQLITE_PATH=./source-state/memory.db scone-memory sync-directory ./documents \
  --journal ./source-state/directory.journal \
  --key-file ./source-state/journal.key \
  --store-id project-documents-local \
  --parser-revision builtin-v1 --json
```

Use the same command with `--delete-missing` to retire files removed from the
directory. Without it, missing sources remain in memory. Exit status is `0` for
a completed run, `1` for a partial run with structured receipts, and `2` for an
invalid invocation or an unreadable journal. Text output is available by omitting
`--json`.

An interrupted deletion remains partial with a `deletion_pending` receipt when
a retry omits `--delete-missing`. Repeat with that flag and a complete scan to
finish it. If the file reappears first, the runner cancels the pending deletion
and reconciles its current bytes.

The caller supplies `--store-id`: a stable name for the exact memory catalog and
attachment store. Change it when switching catalogs. The journal also binds the
space, canonical root path, root inode and selected suffixes. Reusing it with a
different binding or encryption key is refused. This binding is not a backend
authentication mechanism; callers must associate the name with the right stores.

`--max-files` defaults to 1,000 and `--max-total-bytes` to 256,000,000. Repeat
`--extension .txt --extension .md` to restrict the supported suffixes. Changing
the suffix selection requires a new collection, so narrowing it cannot silently
delete previously managed documents. Never discard a journal to bypass a refusal:
a new journal creates a different collection, with different source identities.

## Native Python

```python
from pathlib import Path
from scone_memory.ingestion import DirectorySync, ScanLimits

# memory is an opened MemoryEngine configured with local persistent stores.
sync = DirectorySync(
    memory, Path('./documents'),
    space='research',
    journal=Path('./source-state/directory.journal'),
    key=Path('./source-state/journal.key').read_bytes(),
    store_id='project-documents-local',
    parser_revision='builtin-v1',
    scan_limits=ScanLimits(max_files=500, max_total_bytes=100_000_000),
)
result = await sync.synchronize(delete_missing=True)
for receipt in result.receipts:
    print(receipt.path, receipt.status, receipt.episode_id, receipt.code)
```

`parser_revision` identifies the full ingestion configuration. Change it when
changing extraction, OCR, chunking or embedding behavior, even if the source
bytes are unchanged. Existing revisions keep their original parser identity.
Retries of an unfinished transition use its retained extraction before applying
the newly selected revision. The journal is bounded to 10,000 tracked paths and
8 MB, including missing and suppressed paths.

## Source ownership and replacement

Each journal owns a random collection ID. Within it, the exact relative path
identifies a source; case and Unicode are preserved. Different paths with equal
bytes have separate episodes while sharing content-addressed attachments.
A revision key includes the source, parser revision, original and manifest
digests, and a journal-controlled generation. A later A → B → A edit therefore
creates a new episode without resurrecting the tombstone for the first A.

The runner retains both inputs, records a pending transition, indexes the new
episode, and verifies ownership, linked evidence, extracted text and indexed
chunks before recording retirement intent. Only then does it forget the previous
owned episode. Pending transitions replay using the exact retained extraction;
they do not re-embed an already indexed revision. Restarting with the same journal
also finishes pending updates whose source file has since disappeared.

An externally forgotten managed source becomes suppressed, including when its
file later changes. If either side of an interrupted replacement was externally
forgotten, a durable suppression transition cleans up the other known owned
revision before reporting success. Failed cleanup keeps both revision identities
for retry. Missing-file deletion performed by this runner instead records an
absent source, which can be ingested at a new generation if its file returns.

## Scan and recovery limits

Scans do not follow symlinks. Special files, unreadable entries, invalid paths,
changed files and exhausted budgets make the inventory incomplete. Defaults
also limit each file to 25 MB, traversal to 20,000 entries and 32 nested levels,
and each scan to a cooperative 30-second deadline. This does not interrupt a
blocked operating-system read. A selected file is reopened beneath the pinned root
and checked against its observed identity and bytes before ingestion.

Missing-file deletion requires two matching complete inventories. Absence is
checked again immediately before forgetting each source. These checks are not an
atomic filesystem snapshot: serialize source-tree changes and other writes to
managed episodes with synchronization. The journal lock excludes other runners
using that same journal; it does not lock the memory API or the source tree.

Across memory, vector and attachment stores, replacement is a sequence of writes,
not a global transaction. Both revisions can be visible before retirement. The
runner recovers interruptions between recorded stages and repairs replayable
attachment links. It refuses retirement when indexing remains marked in flight
or retained evidence is missing. Reopen/recover the engine to finish its own
in-flight indexing before retrying. A storage failure inside forgetting that
removes a record without its tombstone is reported as unresolved, never guessed
to be a completed deletion. Bytes retained before a failed intent save can remain
unlinked; the runner does not garbage-collect them.

Forgetting a source removes its episode, chunks and releasable attachment links.
Extracted claims and their relationships **continue to stand**, following the
existing retention policy. This workflow does not retract or recompute claims,
retain a browsable archive of retired revisions, synchronize remote connectors,
or expose arbitrary local filesystem paths through HTTP.
