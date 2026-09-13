# Observe a native agent invocation

`BoundAgent.run(..., events=stream)` exposes live metadata for one invocation.
The caller still selects the model through `AgentCatalog.bind`; observation
cannot change that selection, memory scope, approvals, budgets or checkpoint
replay. This implementation uses the existing native execution boundaries and
contains no reference framework code.

Using `selected` and `tools` from [model selection](agent-models.md):

```python
import asyncio
from scone_memory.agents import AgentEventStream, AgentProgressGap

async def answer_with_progress(selected, tools, question):
    stream = AgentEventStream(max_events=128)
    running = asyncio.create_task(selected.run(question, tools=tools, events=stream))
    try:
        async for event in stream:
            if isinstance(event, AgentProgressGap):
                print("Events omitted:", event.first_sequence, event.last_sequence)
            else:
                print(event.sequence, event.model_id, event.kind, event.tool_name)
        result = await running
        return result
    finally:
        # This host explicitly owns execution; leaving the reader alone never
        # cancels the agent. Always collect the owned task's result or exception.
        if not running.done():
            running.cancel()
        await asyncio.gather(running, return_exceptions=True)
```

Create the task and consume concurrently to see progress while the model waits.
Awaiting the run before reading is also supported, subject to buffer loss. Each
stream accepts one invocation and one active iterator, all on the same event
loop. A second invocation is rejected before model creation. Closing or
cancelling the iterator detaches observation; another iterator can consume what
remains on the original loop. Explicitly cancel the execution task to stop it.

## Event contract

`AgentProgressEvent` and `AgentProgressGap` are immutable dataclasses. Events
carry a generated `invocation_id`, a sequence beginning at 1, the selected
`agent_id`, `model_id` and configuration `binding`, a UTC `occurred_at` timestamp,
and monotonic `elapsed_s` since this invocation began. These identifiers are
metadata; hosts must still authorize recipients before exposing them.

| Kind | Meaning |
| --- | --- |
| `turn_started` | The stream is reserved, before question/context validation or factory creation. |
| `operation_started` | A model, memory or custom operation callback began. |
| `operation_completed` | That operation completed its local acceptance checks. |
| `operation_failed` | A started operation raised or was cancelled. |
| `operation_reused` | An accepted journal receipt replaced a new operation callback. |
| `tool_proposed` | The loop encountered a host/model tool call, before dispatch or refusal. |
| `tool_result` | The loop accepted a sanitized outcome, including denials and skips. |
| `turn_completed` | The native invocation returned successfully after owned model cleanup and required source/deadline checks. |
| `turn_paused` | Scheduling or approval paused this invocation. |
| `turn_failed` | Validation, execution, source checks or cleanup failed. |
| `turn_cancelled` | Caller cancellation propagated after owned cleanup. |

Operation events share an invocation-local `operation_id` and `operation_kind`
(`model`, `memory`, `custom`). Started operations report `duration_s` at completion
or failure. This measures callback execution and local receipt acceptance; it is
not provider-only latency. Reused operations have no invented duration. Sequence
and operation IDs restart for a new invocation, including a resumed journal.

Tool events use a host-assigned `tool_index` to connect proposal, operation and
result. A registered tool name is preserved; unknown model-authored names become
`unknown_tool`. Events exclude call IDs, arguments, prompts, context, evidence
text, model output and provider exception messages. Results contain only the
existing allowlisted status/error code, byte count and host/model origin.

A proposal is not proof of execution. A custom operation can resolve an approval
denial or reject arguments without entering the application handler. A completed
operation is not proof of an external side effect or a correct answer. Inspect
the result status/error and the final validated result. A journal replay may
perform source-retention checks while avoiding the original retrieval/provider
request or custom effect.

Tool-result `journal_reused` identifies saved operation receipts;
`presentation_reused` identifies an existing evidence presentation reference.
`reused` is true if either is true. A repeated search can execute again and still
reuse a presentation. These event fields do not change the older
`ToolOutcome.reused` presentation semantics or serialized execution results.

## Buffering and terminal states

Capacity is an integer in 1..1024, default 128. Producers never wait for consumers
or execute consumer callbacks. A slow consumer loses the oldest buffered events.
It receives an explicit inclusive `AgentProgressGap` for every omitted sequence
range; `stream.dropped_events` counts dropped events. A gap can include a start
or result event, so missing observations must not be interpreted as missing
execution. The final terminal event remains buffered after production ends.

A terminal event describes this invocation, not the whole durable workflow.
Approval/scheduling pauses still raise their normal exception; a later resume
uses a new stream and can emit reused operations. Cancellation and failure still
propagate from the task. The metadata stream does not carry exception details or
an answer. Consume the task result separately and revalidate evidence before
later publication under the original deadline.

## Durable history and remaining delivery work

Saved task, handoff and interactive workflows now collect these observations into
[encrypted event history](agent-event-history.md), including explicit collection
coverage, stable task/hop/activation identity and native cursor replay. Direct
`BoundAgent` callers can still opt into the bounded stream independently.

Authenticated HTTP/live delivery, standalone SDK reconnect and Console rendering
remain required. Real provider public-text streaming also remains required; these
events neither simulate token streaming nor expose hidden reasoning. Old journals
do not acquire historical timing data retrospectively. History is metadata, not
an answer or proof of current source/recipient authorization.
