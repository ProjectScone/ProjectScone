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
