# Durable native agent turns

[Agent models](agent-models.md) · [Application tools](agent-application-tools.md) · [Workflow pauses](workflow-pauses.md)

A native `BoundAgent` can journal a turn through its current workflow step's
`StepCheckpoints`. Accepted model proposals and completed tool results then
survive a deliberate workflow pause. Resumption reconstructs the original
transcript without repeating those model requests or application calls.

`max_new_operations` optionally bounds work in one activation. A model request,
fresh memory preparation or application-tool invocation counts as an operation.
Previously completed operations replay without consuming this scheduling
allowance. Reaching the allowance raises `TurnJournalPaused` before starting
another operation. A pausable workflow callback returns that exception's
`pause` value to let the workflow runner acknowledge the checkpoint.

```python
from scone_memory.agents import StepContext, WorkflowPausableStep, WorkflowRunner
from scone_memory.agents.turn_journal import TurnJournalPaused

# The host owns catalog, scoped_memory_tools, source_verifier and journal_key.
# Both the agent and this explicit model choice must exist in that catalog.
agent = catalog.bind("research", model_id="selected-local-model")

async def run_agent(context: StepContext):
    assert context.checkpoints is not None
    assert isinstance(context.inputs, str)
    try:
        result = await agent.run(
            context.inputs,
            tools=scoped_memory_tools,
            checkpoints=context.checkpoints,
            max_new_operations=1,
        )
    except TurnJournalPaused as suspended:
        return suspended.pause
    return result.output.text

workflow = WorkflowRunner(
    "agent.sqlite3", key=journal_key,
    steps=[WorkflowPausableStep("research", "agent-turn-v1", run_agent)],
    source_verifier=source_verifier,
)
try:
    progress = await workflow.run(
        "run-1", space="default", scope={}, inputs="Summarize the retained sources."
    )
    # A later explicit run with this same invocation resumes a paused turn.
    # The host decides when to activate it; reading status never starts work.
finally:
    workflow.close()
```

The host must align `source_verifier`, the workflow's scope and
`scoped_memory_tools` with the caller's authorization. Closing and reopening
`WorkflowRunner` with its original key and definition preserves pending work.
`max_new_operations` accepts strict integers from 1 through 64; omitting it runs
through under the existing agent limits. It requires checkpoints and can vary
between activations because it changes scheduling, not the tool proposal.

## What is retained

The journal uses one encrypted, run/step-bound checkpoint. Each ordered entry
contains an operation kind, a digest of its exact request, and either a started
marker or a completed, detached JSON result. It saves started **before** dispatch
and completion only after the result passes its boundary validation.

Model entries retain the original accepted content, tool IDs, arguments, order
and provider-reported usage. Memory entries retain the original packet and a
fingerprint of its scoped source snapshots and document revision. Application
entries retain the validated tool result. Transcript reconstruction reapplies
call, round, output, schema, initial-search, repeated-read and search-compaction
rules. Direct results retain their original publication checks and skipped calls.

The selected agent/model revision, custom registrations, original messages,
answer requirements, recall scope, session exclusion, memory-tool settings and
execution limits are bound before replay. Scope drift during a model call or
factory initialization is rejected. Model factories remain trusted host code;
change their registered revision when changing model endpoints or configuration.

## Recovery limits

An operation recorded as started without a completed result has an unknown
outcome. Reopening refuses it before dispatch, including when a callback failed,
was cancelled, returned invalid data, or its completion could not be persisted.
A completed record that committed despite a lost acknowledgement can be reused;
it does not require repeating the operation. A workflow still needs its separately
acknowledged pause before admitting another callback attempt.

Restoring memory evidence reads the original episodes, chunks, facts and links;
it does not rerun search. Changed or removed evidence, changed document revision,
or a storage outage stops continuation. Restored evidence is checked again before
new application effects and final publication. Storage failures carry fixed
messages, not private filesystem or database diagnostics. This checks retention
and scope, not factual accuracy or the correctness of the model's decision.

Counts and token usage describe the whole accepted turn, including operations
completed before the pause. Reusing a response does not add another provider
usage report. Execution time is accumulated at journal boundaries; time between
activations is excluded. A larger low-level journal timeout cannot reset the
agent loop's smaller cumulative budget. Checkpoint persistence and replay decoding
also check cancellation and the remaining deadline before acknowledging success.

The low-level `ToolTurnJournal` is available for trusted native hosts. Its
`binding` must identify their chosen model and revision; `EvidenceToolLoop` binds
its own invocation configuration additionally. The default journal cap is 64
operations and 4 MiB, with strict limits up to 64 operations and 8 MiB. Stale
journal instances cannot overwrite a newer checkpoint. `pause()` and `finish()`
require all recorded operations to have been consumed in order; finished turns
cannot acquire new operations.

This implements native resumable turns and scheduling yields. It does not yet
implement tool-specific approval requests, user decisions, activation records,
HTTP/SDK continuation or console approval controls. Existing task/handoff services
keep their current behavior until that integration is implemented. A scheduling
yield is never evidence of user approval.
