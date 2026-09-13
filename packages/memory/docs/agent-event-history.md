# Encrypted agent event history

`AgentEventHistoryStore` stores native agent metadata observations in a private,
authenticated encrypted SQLite database. Each history belongs to an immutable
`AgentRunRequest`, including its saved plan/model bindings, question and scope.
The database contains no plaintext prompts, arguments or model output. It stores
only validated [native progress events](agent-execution-events.md), collection markers, their step
and selection identity, an encrypted run digest and retention metadata. Space and
run identifiers in row keys are HMAC-derived.

`AgentRunService` automatically collects metadata for saved task, handoff and
interactive workflows, including parallel tasks and approval resumptions. Native
replay checks current space, scope, catalog bindings and an optional host admission
guard. Authenticated HTTP replay is available; live delivery, strict SDK replay
and a Console timeline remain follow-up work.
The store itself trusts the host; neither storage nor native metadata replay
revalidates answers or grants recipient authorization. The HTTP delivery path
adds current recipient and committed/paused-source verification, described below.

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
and its original immutable progress event, gap or collection marker. Collected
entries also carry `collection_id` and an optional saved `activation_id`; old rows
without these fields remain readable. Append positions describe
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

## Automatic collection and lifecycle

Each actual invocation receives a distinct `collection_id`. The collector checks
the saved workflow signature, question, scope, selected model fingerprint and
scoped-tool binding before model dispatch. Approval or input continuations attach
their saved activation ID. Replaying a completed workflow does not invent another
model invocation or collection.

`collection_started` records admission to observation. `collection_finished`
records reader completion, with the native invocation ID, terminal kind, last
sequence, observed event count and lost event count. A finished collection can
contain a paused, failed or cancelled turn; it does not mean the workflow succeeded.
`observed_events + lost_events == last_sequence` describes consumed stream
coverage, including explicit buffer gaps. It does not claim every event was saved.

`collection_failed` carries only `history_unavailable` or `collection_interrupted`.
After a storage failure, the collector drains without retrying model or tool work
and attempts a final failure marker. If even that write fails, the retained history
remains incomplete. An uncertain first append is not retried: without an
acknowledged generation, it cannot safely distinguish a purge from lost commit
acknowledgement. Later writes pin the acknowledged generation, so purging or
replacing history while a reader runs cannot make it repopulate the old history.

Cancellation joins the owned reader before resources close, including repeated
cancellation during service shutdown. Collection shares the workflow's original
deadline. A slow initial write cannot dispatch the model after that deadline; a
slow final write cannot authorize a late completed workflow receipt. An attempt
that ran but did not commit a receipt remains uncertain. A killed process leaves
an unmatched start or partial stream; readers must not invent a completion time.

```python
# Native host API: reads existing observations and does not execute the run.
page = await service.history(
    "research", "run-1", after=cursor, limit=50,
    admission_guard=check_current_host_authorization,
)
```

For hosts constructing workflows directly, `AgentWorkflow`,
`AgentHandoffWorkflow` and `InteractiveAgentWorkflow` accept an optional
`AgentRunHistory` bound to the exact saved `AgentRunRequest`. The host owns its
`AgentEventHistoryStore` lifetime. Omitting history preserves unobserved execution.

## Authenticated HTTP replay

Hosts mounting `AgentRunService` advertise `agents.history` and expose
`GET /v1/agent-runs/{run_id}/history?limit=50&after=<cursor>`. The space comes from
the current bearer key. Read roles can replay history; cross-space runs remain
unavailable. Limits are canonical decimal integers in 1..100. Unknown or duplicate
query fields, empty cursors and oversized cursors are rejected without echoing
private values. Responses use `Cache-Control: no-store`.

Pages contain `space`, `run_id`, `available`, `items`, `next_after`, `retained_from`
and `omitted`. Entry/event shapes and cursor behavior match native storage.
Old or purged runs return unavailable observations rather than invented events.
Reconnecting with a cursor reads observations only; it never continues a workflow.

The delivery service reads a bounded page, then verifies committed evidence and
exact paused-turn sources through the existing read-only workflow inspection
path. Inspection is the final awaited operation. Before returning it synchronously
checks the current request/key binding, execution-root/file identity and history
generation/retention. Forgotten evidence refuses delivery without changing the
executing owner's journal. A page purged or evicted during verification returns
`history_changed` (409); restart the observation read with current state.

Available history requires its authenticated execution proof. Missing journals,
missing workflow roots, foreign schemas and wrong keys refuse delivery. Inspection
opens SQLite in read-only mode, creates neither journals nor lock files, and
cannot execute work or perform the mutating final-result read. Hosts can explicitly
select `read_only=True` when constructing native workflows for inspection.

This is past execution metadata, not an answer or source content. A successful
read verifies a snapshot; it does not promise sources remain unchanged after the
response reaches the client or provide an atomic transaction across independent
storage backends. Final answers must still use the verified result endpoint.

## Remaining delivery work

Authenticated live delivery, strict standalone SDK history/reconnect models,
Console rendering and genuine provider public-text streaming remain required.
Cursor replay does not substitute for these features or simulate token streaming.
