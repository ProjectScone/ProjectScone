# Encrypted agent event history

`AgentEventHistoryStore` stores native agent metadata observations in a private,
authenticated encrypted SQLite database. Each history belongs to an immutable
`AgentRunRequest`, including its saved plan/model bindings, question and scope.
The database contains no plaintext prompts, arguments or model output. It stores
only validated [native progress events](agent-execution-events.md), their step
and selection identity, an encrypted run digest and retention metadata. Space and
run identifiers in row keys are HMAC-derived.

This is a native storage API. Automatic collection, scheduler/run-service wiring,
collection lifecycle markers, authenticated HTTP/SDK replay and a Console timeline
remain required follow-up work. The store does not execute a saved run, enforce
recipient authorization, inspect current source retention or prove that collection
was complete. The host must perform those checks before using or exposing history.

## Store and replay observations

The caller supplies a validated saved request and events from an invocation using
that request's selected agent/model binding. For a task plan, the step and selection
IDs both name the task; for a handoff plan, the step names the hop (`hop-01`, etc.)
and the selection names the agent. Handoff selections must be reachable at that
exact depth in the saved plan. This structural check cannot establish which route
was actually taken; that remains the collector and execution journal's concern.

```python
from scone_memory.agents.event_history import AgentEventHistoryStore

# history_path is in a private host-managed directory; key is 32 secret bytes.
# request comes from AgentRunStore, never an untrusted HTTP body.
history = AgentEventHistoryStore(history_path, key=key, max_events=512)
try:
    # observed_events comes from AgentEventStream. Reading this sequence does
    # not re-execute the agent; stream gaps are stored as explicit gap records.
    for event in observed_events:
        history.append(request, step_id="find", selection_id="find", event=event)

    page = history.read(request, limit=50)
    if not page.available:
        print("No retained observations are available")
    elif page.omitted is not None:
        print("History positions removed by retention:", page.omitted)
    for entry in page.items:
        print(entry.position, entry.step_id, entry.event)
    cursor = page.next_after
finally:
    history.close()
```

Reopen with the same key and call `read(request, after=cursor)`. A cursor is bound
to the space/run, immutable request digest, history generation and last returned
position. It can read future appends without causing model requests or tool
calls. The final page still returns a cursor; repeated reads after it return an
empty page until more observations arrive. Page limits are integers in 1..100.
No-history pages have `available=False`, no items and no cursor. This applies to
old runs that were never observed and histories explicitly purged.

Each `AgentHistoryEntry` has a global append `position`, `step_id`, `selection_id`
and its original immutable progress event or gap. Append positions describe
transaction commit order, not a reconstructed global clock across concurrent
invocations. Original event sequences and elapsed times stay invocation-local.
A retained start/completion event describes the native invocation's own checks;
it is not a persisted assertion that the entire workflow or collection completed.

## Retention, integrity and failure

`max_events` is 1..4096, default 512, per history. `max_histories` is 1..100000,
default 4096. A new history refuses when the history count is full; existing
histories can continue appending. Reopening with a smaller event capacity applies
the new bound at the next append. The store does not silently expand an older
retention boundary when reopening with a larger capacity.

Appending one entry, advancing the encrypted manifest and removing expired
entries are one transaction. Concurrent connections allocate distinct contiguous
positions. A failed transaction preserves the prior manifest, rows and cursor.
An I/O failure with an uncertain commit outcome is not a reason to retry an
external effect; append itself does not provide an idempotent retry key.

`retained_from` is the first retained append position. `omitted` gives an inclusive
range removed before the requested page. This is separate from a native
`AgentProgressGap`, which describes observation loss within one invocation before
persistence. Missing or altered rows *inside the requested retained page* raise
`history_key_or_integrity`; they are not reported as normal retention loss.

`purge(request)` deletes this history and invalidates its cursors. A later append
creates a fresh generation; an old cursor or old-generation event ciphertext
cannot be transplanted into it. Purge is logical deletion, not secure erasure of
SQLite free pages, filesystem snapshots or backups. The host owns key management,
backup retention and source/space-deletion coordination. Authenticated encryption
does not detect replacement of the entire database with a previously valid
snapshot; no external rollback anchor is claimed.

Requests copied without validation are revalidated without logging their private
values. Event fields must be bounded primitives with valid native shape, coherent
reuse flags and allowed diagnostic codes. Invalid metadata, selection mismatches,
request conflicts, cursors and storage failures use fixed `WorkflowError` codes.
Model factories and tool callbacks are never consulted by this API.

## Remaining integration

A durable collector must record collection starts, completion/failure and process
interruption coverage, attach stable activation/task/hop identities, and join its
owned work during cancellation. Reading an incomplete history must not invent
missing timings or infer whether an external action happened. Saved-run service
reads still need current space/scope/source checks, followed by authenticated live
HTTP delivery, SDK reconnect and Console rendering. This storage contract supplies
the encrypted persistence and cursor layer for that work; it does not replace it.
