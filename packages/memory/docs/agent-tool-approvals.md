# Approval of exact tool calls

An application tool can declare `requires_approval=True` on `AgentTool` or
`function_tool`. The default remains false and is omitted from registration
metadata, preserving existing agent fingerprints. Changing the policy changes
the bound agent fingerprint. Register tools and select models explicitly through
`AgentCatalog`; the model cannot enable a tool or choose its approval policy.

Saved task, interactive, and handoff runs support encrypted requests, immutable
decisions, explicit activation, and single execution claims through
`AgentRunService`, authenticated HTTP, and the standalone Python SDK. The console
approval interface is still pending. A guarded agent invoked directly without the
required native context refuses before its model factory runs. Existing unguarded
workflow signatures and fingerprints retain their behavior.

## State and identity

`AgentApprovalStore(runs)` shares an `AgentRunStore`'s encrypted database,
transactions, encryption key, and lifetime. It has no separate close operation.
Each request binds the immutable registered invocation, selected task or handoff
hop, agent/model fingerprint, tool revision and full registration digest, canonical
original argument JSON, and ordered operation identity from the encrypted turn
journal. Two proposals with identical arguments still require separate requests
when their ordered call identities differ. Invalid arguments produce the normal
`invalid_arguments` tool result without creating an approval request.

| Revision | Meaning |
| --- | --- |
| 1 | Exact call requested; no decision |
| 2 | Immutable approve/deny decision and host-supplied actor recorded |
| 3 | Decision included in an explicit named continuation activation |
| 4 | Activation claimed once; execution admitted, not necessarily completed |

Decisions and matching activation retries are idempotent. Reusing either with
changed content refuses. A second claim always refuses. Activation can include
1–32 decided requests atomically; failed writes roll back the whole batch.
Requests and activation receipts are bounded to 512 each per run and do not count
against run capacity or appear in run pagination. Arguments remain encrypted in
storage and returned only through the host's scoped inspection boundary.

## Saved runs and HTTP

`AgentRunService` supplies the approval context for registered guarded tools.
The saved plan retains the selected LLM for every task or handoff agent. A run
waiting for a decision exposes `status="paused"` and `paused_steps`; it is not an
unknown attempt. Reading status or approvals, recording a decision, restarting the
service, or repeating `start` does not resume it.

| HTTP operation | Required role | Effect |
| --- | --- | --- |
| `GET /v1/agent-runs/{run_id}/approvals` | Any authenticated role in the space | Inspect current requests and visible consumed history |
| `POST /v1/agent-runs/{run_id}/approvals/{request_id}/decision` | `review` or `full` | Persist `decision` and `expected_revision: 1` |
| `POST /v1/agent-runs/{run_id}/approval-continuations` | `write` or `full` | Activate the exact `continuation_id` and `decisions: {request_id: 2}` batch |

Hosts advertise `agents.approvals` when these operations are mounted. Bodies are
bounded to 8 KiB, reject duplicate JSON keys at every depth, and cannot supply the
decision actor. The host derives a stable `key:<digest>` actor using a keyed hash
of the currently authenticated bearer key; the bearer secret is never stored in
the approval record. Authorization is checked again after awaited verification.
Responses carry `Cache-Control: no-store`.

Inspection authenticates the paused workflow ticket and turn checkpoint, checks
the first unfinished application call, and restores retained evidence with scoped
point reads. It does not search again, invoke a model, acquire execution authority,
or create an execution journal for an unstarted run. Changed source or checkpoint
state refuses inspection. Unknown operations remain non-replayable.

Only selected paused steps resume. An unrelated tool pause retains its ticket and
attempt count. Human-input continuation is separate: a tool continuation cannot
consume a human response whose earlier admission failed. Admission capacity is
reserved before awaited source checks, and process ownership is released even if
workflow cleanup raises. A committed activation followed by failed admission can
be reconciled only by the same explicit activation ID and batch.

The continuation response contains `status` and an immutable `activation` receipt.
Each decided request exposes `decision_digest`; the activation's
`decision_digests` must match the selected records. These opaque hashes allow the
client to validate the exact acknowledgment without exposing the private
invocation digest. A consumed approval still proves admission only; use the
workflow result to establish completion.

## Standalone Python client

The SDK supports Python 3.9 and later and needs no native framework import.
After inspecting the literal call and obtaining the operator's decision:

```python
agents = client.agents(expected_space="alpha")
pending, = agents.approvals("run-1")
print(pending.call.model_id, pending.call.tool_name, pending.call.arguments())

# Explicit review action; this does not execute the tool.
decided = agents.decide_tool(pending, decision="approve")  # or "deny"

# Separate explicit execution action, using a write/full credential.
continuation = agents.continue_tools(
    "run-1", continuation_id="review-1", decisions=(decided,),
)
print(continuation.activation.activation_id, continuation.status.status)
```

The client validates selected models, saved bindings, canonical argument JSON,
revision state, exact decision hashes and the activation batch. It rejects agents
unreachable at the claimed handoff depth; the server verifies the actual receipt
history. It issues one POST per explicit operation and does not retry an ambiguous
failure automatically. Retrying the same completed activation returns its original
receipt without rerunning the call. `agents.continue_run(..., responses=...)`
remains the separate human-input protocol.

## Low-level host integration

The host first saves a plan with `AgentPlanStore`, registers its immutable question
and recall scope with `AgentRunStore`, and obtains a `BoundAgent` matching that saved
selection. The host owns authorization, workflow execution ownership, and source
verification. Its `WorkflowPausableStep` callback receives the actual `StepContext`:

```python
from scone_memory.agents.approval_context import ApprovalContext
from scone_memory.agents.turn_journal import TurnJournalPaused

async def execute(context):
    approval = ApprovalContext(
        approval_store,
        context,
        step_id="calculate",
        selection_id="calculate",
        activation_id=current_activation_id,
    )
    try:
        result = await selected_agent.run(
            model_question,
            tools=scoped_memory_tools,
            checkpoints=context.checkpoints,
            approval=approval,
        )
    except TurnJournalPaused as pause:
        return pause.pause
    return result.output.text
```

`context.inputs` must contain the registered original question. The supplied
checkpoints must be the same object as `context.checkpoints`; scope, exclusion,
agent, and selected model must match the registered invocation. The turn journal
also binds the actual model question and workflow context. For task plans,
`step_id` and `selection_id` identify the model task. For handoffs, `selection_id`
is the selected agent and `step_id` is the next `hop-NN`; the complete preceding
receipt prefix, selected models, fingerprints, and allowed transitions are checked.
Human-input nodes cannot be selected as model tasks.

Only the selected task's dependency receipts participate in its durable context
identity. Independent DAG siblings may finish while it waits. Within an active
invocation, a separate complete context snapshot detects host mutation.

A missing decision or mismatched activation raises `ToolApprovalPaused`, a subclass
of `TurnJournalPaused` with an opaque `request_id`. It contains a sealed workflow
pause and occurs **before** the custom operation is marked started. A scheduling
pause from `max_new_operations` remains separate and grants no approval.

Raw `approval_store.list(space, run_id)` and `get` authenticate stored records;
they do not establish that a call is currently pending or its sources retained.
After verifying the current workflow checkpoint and sources (or using the saved-run
service inspection above), the authorized host can record a decision without execution:

```python
approval_store.decide(
    space, run_id, request_id,
    decision="approve",  # or "deny"
    actor=authorized_actor,
    expected_revision=1,
)
```

An explicit continuation then activates the decision:

```python
approval_store.activate(
    space, run_id, activation_id,
    decisions={request_id: 2},
)
```

The host passes that exact activation ID to the callback during an explicitly
resumed workflow run. Merely saving the decision, saving an activation, or polling
status does not execute the call. The runtime claims the activation; application
code should not pre-claim it. Approved calls invoke the exact canonical approved
arguments. Denied calls record `approval_denied` without invoking the handler and
allow the model to continue. Function adapter defaults are supplied by the same
revisioned registration. Direct-return tools still use the normal validated
output contract.

## Recovery and limits

Before publishing a new request and before admitting its execution, the runtime
revalidates retained sources, scope, cancellation, deadline, and checkpoint lease.
The execution path persists an operation-start receipt, revalidates sources and
lease, then claims the activation before dispatch. Completed operation receipts
replay without another model request, effect, or claim; historical usage is retained.
Unknown started operations never replay automatically.

Claim and run cancellation serialize in the shared database. They cannot be atomic
with a separate workflow journal or an external side effect. Cancellation after a
successful claim cannot undo an in-flight handler. A lost claim acknowledgement
leaves revision four and prevents automatic dispatch or reclaim. A handler failure
or lost completion acknowledgement is treated according to the turn journal's
unknown/completed receipt, never inferred from the approval record. The host must
not use an approval receipt to override an uncertain workflow attempt.

Handlers and model factories remain trusted host code. This is policy enforcement
in the bound agent execution path, not a sandbox around arbitrary direct Python
calls to a handler. No cloud integration, service discovery, or model fallback is
introduced.
