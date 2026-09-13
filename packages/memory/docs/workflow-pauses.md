# Checkpoint-backed workflow pauses

[Package overview](../README.md) · [Agent tools](agent-tools.md) · [Durable agent turns](agent-turn-journal.md)

`WorkflowPausableStep` lets a trusted native workflow callback return
`WorkflowPaused(checkpoint)` instead of a completed JSON result. The named
checkpoint must already contain nonempty bytes in that callback's
`StepCheckpoints`. The runner seals the checkpoint key and SHA-256 digest into
its existing encrypted, invocation-bound SQLite journal.

A later explicit `run()` with the same run ID, space, scope, inputs and workflow
revision verifies sources and the exact checkpoint bytes. It consumes the pause
record and records the next attempt in one durable write **before** entering the
callback. A started attempt without a completed result or acknowledged pause
remains `outcome_unknown`; a checkpoint by itself never permits replay.

## Native example

This example snapshots a manifest and waits for the host to make a local
prerequisite available. It performs no external writes or model calls.

```python
from scone_memory.agents import (
    StepContext, WorkflowPaused, WorkflowPausableStep, WorkflowRunner,
)

prerequisite_ready = False
manifest = b'{"version":1,"document_ids":["doc-1"]}'

async def verify_sources(context: StepContext) -> bool:
    # A real host checks current authorization and retained sources here.
    return True

async def prepare(context: StepContext) -> str | WorkflowPaused:
    assert context.checkpoints is not None
    saved = context.checkpoints.get("manifest")
    if saved is None:
        context.checkpoints.put("manifest", manifest)
    elif saved != manifest:
        raise ValueError("Manifest no longer matches this implementation")
    if not prerequisite_ready:
        return WorkflowPaused("manifest")
    return "manifest ready"

# Supply a stable, privately stored 32-byte key; never put it in source control.
runner = WorkflowRunner(
    "workflow.sqlite3", key=host_journal_key,
    steps=[WorkflowPausableStep("prepare", "manifest-v1", prepare)],
    source_verifier=verify_sources,
)
try:
    first = await runner.run("run-1", space="default", scope={}, inputs=None)
    assert first.status == "paused"
    assert first.results == {}
    prerequisite_ready = True
    completed = await runner.run("run-1", space="default", scope={}, inputs=None)
    assert completed.results == {"prepare": "manifest ready"}
finally:
    runner.close()
```

Closing and reopening the runner with the same key and definition preserves
pending pauses. Host policy and prerequisite state remain the host's
responsibility; the callback must validate its own checkpoint format and phase.
Changing the step version, kind or resume budget changes the workflow revision.

## Scheduling and reads

- Sequential execution returns when a step pauses. Dependency workflows let
  independent siblings finish while paused nodes block their dependents. A
  callback runs at most once per explicit invocation of `run()`.
- `max_resumes` defaults to 32 and accepts strict integers from 1 through 128.
  The initial attempt does not consume a resumption. Exhaustion raises
  `resumes_exhausted` and leaves the pending pause inspectable.
- `status().paused_steps` lists pending pauses. `waiting_steps` still refers
  exclusively to `WorkflowInputStep` human-input nodes; both can coexist.
- `inspect_completed()` verifies and returns committed sibling results without
  executing callbacks. `read_result()` refuses incomplete or paused workflows.
  A completion policy cannot discard an already attempted, paused step.
- Source verification outages preserve pause records. Confirmed source
  invalidation clears their checkpoints and permanently invalidates the run.
- Old checkpoint handles expire when their callback returns, fails or is
  cancelled. Only the current attempt can read or change its checkpoints.

## Failure boundary

The callback must own its phase journal and external-operation reconciliation.
The runner proves checkpoint identity; it does not prove that an external action
succeeded, that a user approved it, or that arbitrary callback code is replayable.
Ordinary `WorkflowStep` callbacks cannot return a pause to acquire resumption
permission. Existing ordinary-step and input-step revision identities are
unchanged.

Cancellation cleanup reloads committed state before preserving finished siblings.
It cannot restore a consumed pause record from an older in-memory snapshot, and
it never creates a pause from a late callback result during cleanup. Failed
admission writes do not dispatch the callback. A write that committed before an
interruption remains authoritative.

Pausable workflows check the execution deadline before admission, after source
verification and callback return, and around pause acknowledgement. A synchronous
callback can still block the event loop; it cannot be forcibly preempted by this
runner. Once control returns, an expired invocation refuses successful pause
acknowledgement. If the pause record committed while a journal write crossed the
deadline, it remains available for inspection and later explicit continuation.

This API is a native workflow primitive. [Durable agent turns](agent-turn-journal.md)
add ordered model/tool receipts and native scheduling yields on top of it.
Tool-specific approval requests, decisions, HTTP/SDK continuation and console
approval controls still require further integration.
