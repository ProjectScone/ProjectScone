# Named agents and model selection

`AgentCatalog` connects a named agent's instructions and execution limits to an
explicit set of host-registered models. A caller selects a model by its catalog
ID, or uses that agent's declared default. Unknown or disallowed choices fail
before any model is created. A failed selected model never triggers a fallback.

This is a native Python interface. It builds on `EvidenceToolLoop` and
`ScopedMemoryTools`; it does not provision services, discover models, install
providers, or grant access to a memory space.

## Register and select models

Factories implement `ToolModel.complete`. Existing self-hosted native tool-call
and structured-action adapters can be registered independently. Select the
protocol supported by your configured model explicitly. Each factory must return
a fresh model instance. The model identifiers and local endpoint below are
examples; replace them with services and models you operate.

```python
from scone_memory.agents.catalog import AgentCatalog, AgentDefinition, AgentModel
from scone_memory.agents.evidence_loop import ToolLoopLimits
from scone_memory.integrations.scoped_tools import ScopedMemoryTools
from scone_memory.providers.tool_chat import SelfHostedToolChat
from scone_memory.retrieval.recall_scope import RecallScope

agents = AgentCatalog(
    models=[
        AgentModel("fast", "Fast local model", "config-1", lambda: SelfHostedToolChat(
            "http://127.0.0.1:8000/v1", "model-a")),
        AgentModel("careful", "Careful local model", "config-2", lambda: SelfHostedToolChat(
            "http://127.0.0.1:8000/v1", "model-b")),
    ],
    agents=[AgentDefinition(
        agent_id="research", instructions="Answer using authorized retained evidence.",
        models=("fast", "careful"), default_model="fast",
        limits=ToolLoopLimits(max_tool_calls=4, max_tool_rounds=4, timeout_s=90.0),
    )],
)

# A UI or native caller chooses among these public entries.
choices = agents.choices("research")
selected = agents.bind("research", model_id="careful")
tools = ScopedMemoryTools(memory, "team-space", scope=RecallScope.validated(
    where={"project": "approved-project"}))
result = await selected.run("What did we decide?", tools=tools)
assert result.model_id == "careful"
print(result.output.text)
```

`describe()` returns agent IDs, defaults and allowed model metadata. It excludes
instructions, endpoints, credentials and factories. The host must filter the
catalog for the caller's permissions before exposing it through an application.
The agent cannot change its own model selection or the supplied memory scope.
Tools remain read-only; custom write tools and approval handling are not enabled
by this catalog.

The result records `agent_id`, `model_id`, and a configuration `binding` alongside
the existing tool-loop evidence, budgets and validation result. Grounded evidence
identifies retained source material; it does not establish answer correctness.
The loop's deadline and cancellation contract remain in effect. Validate again
before delayed publication using `await result.output.validate()`; this validator
uses the original run deadline and is not a durable checkpoint validator.

## Bind workflow checkpoints to the selected model

`selected.fingerprint` hashes the agent instructions, allowed/default models,
limits, initial-search policy and selected model metadata. Use it as the version
of a `WorkflowStep` that executes this bound agent. The existing `WorkflowRunner`
then rejects reopening the same run with a different model or agent policy:

```python
from scone_memory.agents.workflow import WorkflowStep

async def answer(context):
    if not isinstance(context.inputs, str):
        raise ValueError("a question is required")
    outcome = await selected.run(context.inputs, tools=tools)
    return {"agent_id": outcome.agent_id, "model_id": outcome.model_id,
            "binding": outcome.binding, "text": outcome.output.text}

step = WorkflowStep("answer", selected.fingerprint, answer)
```

The host owns the workflow's authorization and source verifier. It must verify
all retained evidence used by completed outputs before reuse, and bind the
correct space, scope and input to the journal. The small step above demonstrates
model identity only: retain the required evidence receipts and supply a matching
verifier when checkpointing source-grounded answers. A `ToolLoopResult` closure
must not be serialized or treated as a verifier after process restart.

Model factories are trusted application code. Their declared revision must change
when their endpoint, model, protocol or other behavior changes. A fingerprint
cannot inspect closure state, provider weights or an endpoint changed in place.
Workflow callbacks, tool binding and evidence policy still need their own version
when changed. Model calls are not assumed idempotent: an interrupted step with an
unknown outcome is not automatically replayed.

## Execute a declared task graph

`AgentWorkflow` composes named agents into a sequential, checkpointed task graph.
Each task selects an allowed model and declares which predecessor outputs it
receives. Unrelated outputs never enter its model context. For example, using
`agents`, `memory`, and a host-managed encryption key:

```python
from scone_memory.agents.task_workflow import AgentTask, AgentTaskPlan, AgentWorkflow

plan = AgentTaskPlan(workflow_id="research-report", tasks=(
    AgentTask(task_id="find", agent_id="research", model_id="fast",
              prompt="Find the relevant decisions."),
    AgentTask(task_id="summarize", agent_id="research", model_id="careful",
              prompt="Summarize the decisions and identify uncertainty.",
              depends_on=("find",)),
))
workflow = AgentWorkflow("report.sqlite", key=key, catalog=agents, plan=plan,
    memory=memory, space="team-space", scope=RecallScope.validated(
        where={"project": "approved-project"}))
try:
    result = await workflow.run("request-1", "What should our team do next?")
    status = workflow.status("request-1", "What should our team do next?")
finally:
    workflow.close()
```

A repeated run with identical input, plan, model bindings and scope reuses
completed task receipts only after rechecking their retained evidence. Evidence
is checked again before each task starts and before results are published.
Confirmed missing or changed evidence invalidates saved outputs. A temporary
storage outage pauses verification and preserves completed work. Cancellation
or interruption during a model call leaves an uncertain outcome that cannot be
automatically replayed; use a new run ID for an intentional new execution.
`status` returns progress metadata, not revalidated answer content.

Handoffs are bounded JSON in a separate user message marked as untrusted data.
They cannot change system instructions, model selection or the fixed tool scope.
A receipt's `source_status` describes its own tool evidence, not independent
verification of its prose or inherited conclusions. Follow `depends_on` to trace
which prior agent outputs a task consumed. Prompt injection remains possible in
model-generated text; read-only scoped tools bound its available actions.

Plans allow 1–32 tasks, reject unknown dependencies and cycles, and run in stable
topological order. Task prompts are limited to 2,000 UTF-8 bytes, the run question
to 4,000 bytes, and each handoff to 32,000 bytes. Oversized handoffs fail before the
receiving model is constructed. Journals retain at most 128 evidence packets
within the configured encrypted payload budget.

## Save model choices and task plans

`AgentPlanStore` stores encrypted plans separately from execution journals. It
resolves defaults into explicit model IDs and records the bound configuration for
every task. A host configuration change makes `checked_plan` refuse execution
until the caller reviews and saves a new revision. Editing invokes no model.

```python
from scone_memory.agents.plan_store import AgentPlanStore

store = AgentPlanStore("agent-plans.sqlite", key=key)
try:
    saved = store.save("team-space", plan, catalog=agents, expected_revision=0)
    # A later edit must supply the last revision the caller actually read.
    stored = store.get("team-space", "research-report")
    assert stored is not None
    checked_plan = stored.checked_plan(agents)
    # Pass checked_plan to AgentWorkflow with the authorized space and scope.
finally:
    store.close()
```

Names are HMAC-indexed; the full record uses authenticated encryption. The host
owns the 32-byte key, access decisions, backup retention and store lifecycle.
Use a directory owned by the current OS user that others cannot write to. Files
must be private, regular and unlinked elsewhere; symlinks and unrelated databases
are refused. Concurrent path replacement by the same OS user is unsupported.
The store bounds payloads to 128,000 bytes, contains at most 4,096 plans by default,
and pages up to 100 at a time with space-bound cursors. Two editors cannot silently
overwrite each other: stale revisions raise `PlanConflict`.

## Expose authenticated plan configuration

An application can opt into the configuration routes using its host-registered
catalog and caller-owned plan store:

```python
from scone_memory.api.app import create_app

app = create_app(memory, keys, roles=roles,
                 agent_catalog=agents, agent_plan_store=store)
```

Both arguments are required together. The routes and capability flags are absent
when they are not configured. Bearer keys determine the space. Read and review
roles may inspect plans; write and full roles may save. Authorization and space
identity are checked again after an uploaded body is consumed, before saving.
The supplied catalog is shared across this application's authorized spaces; the
host must expose only the models and agent definitions intended for those users.

| Route | Behavior |
| --- | --- |
| `GET /v1/agents/catalog` | Public model-choice metadata; no system instructions or provider credentials |
| `GET /v1/agent-plans?limit=50&after=...` | Page this key's saved plans |
| `GET /v1/agent-plans/{workflow_id}` | Read a saved plan and whether its configuration is current |
| `PUT /v1/agent-plans/{workflow_id}` | Save `{expected_revision, plan}`; return 409 for a stale revision |

Responses carry `Cache-Control: no-store`. PUT bodies are bounded before parsing,
unknown fields and invalid task graphs are refused, and the path must match the
plan identity. These routes edit plans only; they do not start model execution.
The application closes its plan store when it shuts down. Deleting a memory space
blocks HTTP access but does not physically erase its separate plan store or
backups; the host must include those in its retention policy.

## Current boundary

The catalog supports up to 32 agents, 64 models and 64 allowed models per agent.
Agent instructions are limited to 32,000 UTF-8 bytes and questions to 8,000 bytes;
the tool-loop budgets bound model rounds, tool calls and retained transcript data.
Factories should be quick synchronous constructors; asynchronous model work
belongs in `complete`, where cancellation is enforced cooperatively.

Declared dependency handoffs and sequential recovery are implemented natively.
Parallel workflow scheduling, dynamic handoffs and HTTP run management remain
separate work. Saved plan editing is available through the native store and
authenticated HTTP configuration routes. Catalog factories
remain host-managed application code.
