# Durable directory synchronization history

`DirectoryRunStore` retains encrypted sync requests, control revisions and
historical per-source results. It is a registry for a host that executes
`DirectorySync`; opening or reading it does not scan files or invoke models.
Background execution, configured HTTP collections and browser controls are not
provided by this registry alone.

A request binds the memory space, run identifier, collection identifier, exact
configuration fingerprint, deletion policy, deadline and attempt limit. Registering
that same request again returns its existing state. Reusing its identifier with
changed settings is a conflict. The host must fingerprint the root identity,
store, parser, scan/document limits and suffix configuration consistently.

Call `start_attempt` with the observed `expected_revision` before executing the
sync. It records intent; it is not an execution lock. The host must separately own
both the run and collection while executing, including across processes. An
interrupted attempt requires `resume=True`, and a cancelled registration also
requires explicit resume. Cancellation is durable intent until the owner stops
and records the attempt's outcome. Revisions prevent a late success from replacing
a newer cancellation.

After `DirectorySync.synchronize` returns, `finish` publishes its source outcomes,
scan issues and terminal summary in one SQLite transaction. A failed write leaves
no partially published result. It does not undo source changes already performed
by the sync. `fail` records an unsuccessful attempt; it does not claim that no
source changes happened.

`outcomes(space, run_id, limit=20, after=None)` returns a bounded page and its
`next_after` cursor. Source paths are canonical relative paths, successful source
outcomes retain their episode identities, and duplicate conflicting paths are
refused. An invalid-Unicode filesystem name in a scan issue is retained as a JSON
string literal with `path_escaped=True`; it is display-only diagnostic text,
not a source locator. Ordinary Unicode issue paths retain their original text.

Completed and partial results are immutable. Use a new run identifier for another
finished scan. A resumed unfinished scan reconciles the source journal and reads
the current directory; it does not reconstruct the initial filesystem snapshot.
History reports what that run observed. An old `added` receipt does not prove its
episode is still readable, and following an episode link requires current memory
access. No extracted document text or absolute source root is stored in outcomes.

The containing directory must be locally owned and not writable by other users.
Rows use authenticated encryption and space/run-bound opaque keys. The registry
bounds run capacity, individual records and result pages; it never evicts history
implicitly. Wrong keys, substituted rows and missing outcome sequences fail
explicitly. Close the registry when the owning host has stopped its work; the
memory engine has its own lifetime.

`DirectorySyncService` adds opt-in local execution around configured
`DirectoryCollection` instances. The host supplies an already-open engine,
private state directory, 32-byte key and a tuple of collections. Each collection
supplies its existing `DirectorySync`, a public identifier and label, and an
explicit `allow_delete_missing` policy. The service owns its workers and run
registry; the host retains ownership of the engine.

```python
from scone_memory.ingestion.directory_service import (
    DirectoryCollection, DirectorySyncService,
)

service = DirectorySyncService(
    private_state_directory, key=local_key, memory=memory,
    collections=(DirectoryCollection("notes", "Team notes", sync),),
)
try:
    admitted = await service.start("team", "scan-2026-09-12", collection_id="notes")
    # Admission returns before the scan finishes. Poll status; reads never replay.
    status = await service.status("team", admitted.record.run_id)
finally:
    await service.aclose()
```

The configured `sync.space` must match the requested space (`team` above).
`catalog(space)` exposes collection labels, policy and configuration digests;
absolute source and state paths are not included. `request`, `status` and `list`
read durable run state. `result` pages historical outcomes after completed or
partial scans. These methods refuse access to deleted spaces.

`start` is idempotent for the same run identifier and immutable request. It never
restarts an interrupted attempt. `resume` requires the current record revision,
original configuration and remaining attempt budget. It reconciles unfinished
source transitions and scans the **current** directory. Completed and partial
results require a new run identifier for another scan. `cancel` also requires the
current revision. Cancellation does not undo previously accepted source changes;
`outcome_unknown` identifies inactive unfinished attempts whose source effects
need reconciliation. A deadline is persisted as a failure without automatic retry.

All cooperating service instances must use the same private service directory
and key. Per-run locks prevent overlapping attempts; per-collection locks prevent
simultaneous runs against a source root. `max_active` limits workers per service
instance, not across a distributed fleet. An active owner in another process is
reported with `active_elsewhere`; cancellation from this process is refused.
SourceJournal separately guards direct CLI access. Host configuration must remain
stable while the service is open; changing engine, space, parser, scanner, journal
or bound settings requires a new service. Each admitted execution pins its
coordinator, scanner and journal fields; a later host mutation cannot redirect
that execution to a different space. A final configuration check also refuses
result publication after a detected change. Parser revisions remain the host's
contract for changes inside provider or parser implementations.

Shutdown cancels and joins owned workers before closing the registry. If its
caller is cancelled, shutdown finishes draining and then propagates cancellation.
This includes provider cancellation cleanup; providers must cooperate with
cancellation. Blocking filesystem reads already dispatched to threads can finish
independently, but cannot publish a successful run result. If recording a cancel
intent fails, the owned worker is still signalled and the storage error is raised.
A stale control revision never cancels a newer attempt. Storage failure cannot
manufacture a successful result: an unpublished attempt remains recoverable and
is reported as interrupted once ownership ends.

Hosts can inject the service through `create_app(directory_sync_service=service)`
or `create_conversation_app(..., directory_sync_service=service)`. Both advertise
`documents.sync` and own service shutdown. Without the service, the capability is
false and the routes are absent. Cleanup drains before propagating cancellation;
a worker startup failure also closes the registry.

The authenticated HTTP interface exposes:

| Method | Route | Behavior |
| --- | --- | --- |
| GET | `/v1/sync-collections` | Configured collection catalog in `items` |
| POST | `/v1/sync-runs` | Admit `{run_id, collection_id, delete_missing?}` |
| GET | `/v1/sync-runs` | Status history with `limit` and opaque `after` cursor |
| GET | `/v1/sync-runs/{run_id}` | Status and immutable request in `record` |
| GET | `/v1/sync-runs/{run_id}/request` | Durable request and control record |
| GET | `/v1/sync-runs/{run_id}/result` | Historical outcomes with `limit` and integer `after` |
| POST | `/v1/sync-runs/{run_id}/resume` | Explicit resume with `{expected_revision}` |
| POST | `/v1/sync-runs/{run_id}/cancel` | Cancellation intent with `{expected_revision}` |

Responses carry `Cache-Control: no-store`. Reads require read permission and
controls require write permission. The host rechecks credential scope after
asynchronous reads and before durable admission. Requests are strict JSON objects
of at most 8 KiB; duplicate keys, unknown fields, arbitrary paths and supplied
spaces are refused. Busy admission returns 429 with `Retry-After: 1`; stale
revisions, foreign ownership and unavailable results return 409. Missing runs or
collections return 404. POST success returns 202: cancellation acknowledges the
intent, so poll status for worker completion and its updated revision before
resuming. A historical episode identifier still needs current source authorization
and may now return Gone.

The standard launcher enables this service only when
`SCONE_DIRECTORY_SYNC_CONFIG` names a private configuration file. It applies to
both the memory-only host and the composed conversation host. For example:

```json
{
  "schema_version": 1,
  "state_dir": "directory-state",
  "key_env": "SCONE_DIRECTORY_KEY",
  "store_id": "my-local-memory-store-v1",
  "collections": [
    {
      "collection_id": "notes",
      "label": "Local notes",
      "space": "default",
      "root": "notes",
      "parser_revision": "installed-parser-v1",
      "allow_delete_missing": false
    }
  ]
}
```

The file must be owned by the server user, mode 0600, a regular file with one
hard link, and at most 128 KiB. Duplicate or unsupported keys are refused. Relative
`root` and `state_dir` paths resolve beside the configuration file. Source roots
must already exist; the loader creates only the private state directory, which
must be outside every source root. Duplicate roots in the same space are refused.
Keep `store_id` stable for the same memory catalog. A replacement catalog needs
a new identity and a fresh state directory: registry encryption is derived from
both the master key and the declared catalog identity, so existing history cannot
be reopened as belonging to a different catalog. The referenced environment variable contains a persistent 32-byte
key encoded as 64 hexadecimal characters. Changing this key without migrating
state is refused as an integrity error; it does not create fresh history silently.

Set `SCONE_DIRECTORY_SYNC_CONFIG` before running `scone-memory serve`. Loading
opens private state without scanning or invoking parsers/models. Existing API
keys and roles determine which configured spaces and controls a caller can use.
Startup failure and shutdown close the service before the engine, including when
server construction fails before the app lifespan begins.

Optional global controls are `max_active` (1–16, default 2), `max_runs`
(1–100000, default 4096), `max_attempts` (1–4, default 3), and `deadline_s`
(0–3600 exclusive of zero, default 300). Each collection can configure existing
`DocumentLimits` through `limits`, `ScanLimits` through `scan_limits`, and an
explicit `extensions` array of lowercase dotted suffixes. Without `extensions`,
the collection uses built-in formats plus audio/video formats when the host has
explicitly configured document media.

A collection can opt into PDF OCR using
`"pdf_ocr": {"mode": "all_pages", "reading_order": "provider"}` (the same choices
as document ingestion). This requires configured host OCR; it does not select a
recognizer or download a model. OCR applies only to PDFs, while native text and
configured media retain their own parsers. `SCONE_DOCUMENT_MEDIA_CONFIG` supplies
the host's explicit local transcriber/model selection. Parser identity includes
the operator revision, installed parser package versions, OCR choices/settings
and media revision. Change `parser_revision` when OCR binaries, trained data or
custom parser behavior change. Resuming old work under changed configuration is
refused; creating a new run reconciles source revisions through the existing
journal.

Documents controls, standalone client methods, scheduling, remote connectors and
distributed workers remain separate work.

Clients can include `expected_configuration` from collection discovery in a start request. The host refuses a changed collection with `409 sync_configuration_changed` before registering or starting the run. Clients should retain the original run ID and this configuration across uncertain responses. Omitting the field preserves programmatic callers that intentionally select the current host configuration.
