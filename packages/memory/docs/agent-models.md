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
`await workflow.read_result(run_id, question)` verifies completed results without
executing any task. Missing runs return `None`; incomplete or uncertain runs are
refused. Verification outages preserve receipts, and successful completion also
cleans intermediate checkpoints. Cancelling a read does not cancel the saved run.

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

## Retain a run request independently of later edits

`AgentRunStore` records an immutable invocation snapshot: run ID, space, selected
plan revision and model bindings, question, fixed recall scope, optional excluded
session and creation time. The original plan remains attached to the run even
when the editable plan changes. Registering the same ID and identical invocation
returns the original record; changing any bound input raises `RunConflict`.

```python
from scone_memory.agents.run_store import AgentRunStore

runs = AgentRunStore("agent-runs.sqlite", key=key)
try:
    request = runs.register("team-space", "request-1", plan=stored,
        question="What should our team do next?",
        scope=RecallScope.validated(where={"project": "approved-project"}))
finally:
    runs.close()
```

The host must authorize the space and call `request.plan.checked_plan(agents)`
before execution. Registration itself calls no model and does not start or resume
work. Run requests and execution receipts use separate stores. The registry shares
the plan store's private-file, encryption, payload and pagination protections,
with a distinct schema and encryption domain. It defaults to 4,096 run records.
The host owns retention and cleanup; records are not automatically expired. A
separate `cancel_requested_at` timestamp records cancellation intent without
changing the bound invocation. Repeating registration preserves that marker.

## Own bounded background runs

`AgentRunService` owns local background tasks, durable requests and execution
journals. The host supplies an allowed catalog, plan store, memory engine and
scope resolver; it remains responsible for authentication and lifecycle.

```python
from scone_memory.agents.run_service import AgentRunService

service = AgentRunService("agent-run-data", key=key, catalog=agents,
    plans=store, memory=memory, scope_for=lambda space: RecallScope.validated(
        where={"project": "approved-project"}), max_active=4, max_parallel_tasks=2)
try:
    admitted = await service.start("team-space", "request-1",
        workflow_id="research-report", plan_revision=stored.revision,
        question="What should our team do next?", max_parallel=2)
    progress = await service.wait("team-space", "request-1")
    answer = await service.result("team-space", "request-1")
finally:
    await service.aclose()
```

Starting a run does not await its model call. The owned task outlives its caller;
cancelling a `wait` caller does not cancel that task. `status` reports committed
progress and whether this process owns an active task. It does not verify answer
content. `result` uses the non-executing evidence verifier. A saved plan edit does
not change an admitted run; changed host model configuration or recall scope
refuses execution/result reuse under the old binding.

`cancel(space, run_id)` persists intent and stops the locally owned task. An
interrupted model call remains an unknown outcome and cannot be automatically
replayed. Cancellation before execution remains recorded after restart. Even if
intent persistence fails, cancellation still stops the local task and reports the
storage failure. Starting a cancelled request requires a new run ID.

Admission is bounded per service instance and excess new work raises `run_busy`;
there is no unbounded queue. A private process-held lock protects each run from
admission through completion, including the time before its coroutine starts.
Another instance cannot acknowledge cancellation of that owned work. A dead
process releases its lock; uncertain journal attempts still prevent replay.
`aclose` rejects new admissions and waits for cooperative task cancellation.
Use one owning process for served cancellation; this is not distributed execution.

## Serve run controls

Pass `agent_run_service=service` alongside the same `agent_catalog` and
`agent_plan_store` objects to `create_app`. The service must use that app's memory
engine. The app advertises `agents.runs` only when this service is mounted and
closes its owned tasks during lifespan shutdown; the caller still owns the engine
and plan store.

- `POST /v1/agent-runs` accepts `run_id`, `workflow_id`, `plan_revision` and
  `question` and optional `max_parallel` (default 1), returning 202 after bounded admission. It requires a write key.
- `GET /v1/agent-runs?limit=20&after=...` lists space-scoped progress; individual
  status is at `/v1/agent-runs/{run_id}`.
- `GET /v1/agent-runs/{run_id}/request` returns the original invocation snapshot.
- `GET /v1/agent-runs/{run_id}/result` verifies retained sources and current host
  scope/model bindings without calling a model or resuming tasks.
- `POST /v1/agent-runs/{run_id}/cancel` accepts an empty body or `{}` and requires
  a write key. Unknown cancellation options are rejected.

Reads accept read keys. All routes enforce the current key's space, reject
deleted spaces and return non-cacheable responses. Full admission returns 429;
stale plans, changed scope/configuration and uncertain outcomes return 409.
After a lost start response, inspect the original run ID before deciding to
submit again. Never automatically replay an interrupted model call.

## Run independent branches concurrently

Pass `max_parallel=2` (up to 8) to `AgentWorkflow` to execute ready independent
branches together. The default remains 1 and retains existing sequential journal
bindings. A task starts only after every declared dependency has a saved result;
its model receives only those declared outputs. The concurrency/dependency
schedule is part of the journal binding, so changing it refuses old run reuse.

```python
workflow = AgentWorkflow("parallel-report.sqlite", key=key, catalog=agents,
    plan=plan, memory=memory, space="team-space", scope=RecallScope.validated(),
    max_parallel=2)
try:
    result = await workflow.run("report-1", "Compare the evidence.")
finally:
    workflow.close()
```

One scheduler writes run state while worker callbacks execute concurrently.
Successful branches are committed independently, including results already
finished when a sibling fails or verification becomes temporarily unavailable.
Before any recovery admission, unresolved prior attempts refuse replay. An
interrupted model call is not evidence that nothing happened externally.

Cancellation, deadline, storage failure and invalid outputs stop admissions and
wait for every owned worker before releasing journal ownership. Cancellation is
cooperative. Confirmed source invalidation discards receipts and revokes active
checkpoint leases; worker cleanup cannot recreate invalid-source checkpoints.
Each step has its own lease and shared aggregate checkpoint budget.

Native `WorkflowStatus.inflight_steps` reports every recorded in-flight step;
`inflight` remains the first for existing consumers. Progress lists retain
plan order regardless of completion order. Generic `WorkflowRunner` also accepts
an explicit complete `dependencies` map and `max_parallel`; retryable steps are
rejected in this scheduling mode. For served runs, configure `AgentRunService(max_parallel_tasks=2)` as the host
ceiling and select `max_parallel` when starting a run. The default ceiling and
request width are 1. Values above the ceiling are rejected before registration;
excess whole-run admission is still bounded by `max_active`. Thus the configured
upper bound on simultaneous task callbacks is their product, at most 32 × 8.
`GET /v1/agents/run-policy` reports these limits within the authenticated space.
`agents.parallel` is advertised only when the ceiling exceeds 1. The original
run request retains its chosen width; changing it under an existing run ID
conflicts. Sequential records retain their previous encrypted payload shape so
older local readers can still inspect them. New parallel records require a reader
that supports scheduling.

## Finish a sequential workflow early

Native `WorkflowRunner` accepts `completion=WorkflowCompletion(version, when)`.
The synchronous predicate receives detached invocation data and completed results;
it must return a boolean and depend only on that data. Change its version when
the policy changes. It must not call a model or perform external work.

A true condition stops remaining steps. The actual completed prefix is retained,
and `read_result` verifies that prefix and its completion condition without
executing omitted steps. Unknown prior attempts still prevent completion/replay.
Default workflows retain their existing journal signatures. Completion conditions
cannot currently be combined with parallel dependency scheduling.

## Let an agent hand off within a fixed policy

Native `AgentHandoffWorkflow` lets a model choose the next agent from a
host-reviewed set of edges. Each agent still uses its configured model and the
same fixed memory space, recall scope and tool set. A model cannot select a
different provider or grant permissions through its answer.

```python
from scone_memory.agents.handoff_workflow import (
    AgentHandoffPlan, AgentHandoffWorkflow, HandoffAgent,
)

handoffs = AgentHandoffPlan(
    workflow_id="research-report", root_agent="research", max_handoffs=3,
    agents=(
        HandoffAgent(agent_id="research", model_id="careful",
                     can_handoff_to=("writer",)),
        HandoffAgent(agent_id="writer", model_id="fast"),
    ),
)
workflow = AgentHandoffWorkflow("handoffs.sqlite", key=key, catalog=agents,
    plan=handoffs, memory=memory, space="team-space", scope=RecallScope.validated())
try:
    result = await workflow.run("report-1", "Explain the decision.")
    if result.status == "completed":
        assert result.final is not None
        print(result.final.text)
    else:
        # Partial hops remain inspectable, but there is no final answer.
        assert result.status == "handoff_limit" and result.final is None
finally:
    workflow.close()
```

The host catalog must contain these agents and their allowed model IDs. Omitting
`model_id` binds the catalog's current explicit default. The selected bindings,
root, edges and budget are recorded in the encrypted journal's signature;
changing them prevents reuse of a prior run.

Each model receives an output contract requiring exactly `answer` and
`handoff_to`. A null target finishes the chain; otherwise the target must be on
that agent's allowed list. Invalid JSON, duplicate keys, extra fields and
disallowed targets fail before another agent is called. They are not silently
repaired or retried. Explicit cycles, including self-edges, are allowed, but
`max_handoffs` (0–31) permits only 1–32 total hops. Exhausting the budget while
requesting another handoff returns `handoff_limit`, with `final=None`.

Only the immediately preceding answer reaches the next model, as untrusted
context bounded to 32,000 UTF-8 bytes. Each hop retains its own evidence; earlier
evidence does not automatically become evidence for a later answer. Actual
retained sources are revalidated before downstream work and result publication.
A final result can have `source_status="none"`; workflow completion is not a
claim of factual correctness.

`read_result` freshly validates saved hops without invoking models. `progress`
returns journal metadata only: its `completed` status means execution stopped,
and callers must inspect the verified result for `completed` versus
`handoff_limit`. Completed hops are reusable after reopening; interrupted model
attempts refuse automatic replay. Temporary verification outages preserve
receipts, while confirmed invalid evidence prevents their reuse.

Handoff plans can also be saved through `AgentPlanStore` and the same authenticated
`/v1/agent-plans` routes. The plan body uses `agents`, `root_agent` and
`max_handoffs` instead of `tasks`; mixed shapes and unknown fields are rejected.
Saved models are explicit, with bindings keyed by agent ID. Revisions and immutable
run snapshots behave the same as task plans. Existing task records retain their
format; new handoff records require an updated reader.

The host advertises `agents.handoffs` when agent configuration is mounted.
When `agents.runs` is also enabled, start, status, original request, result and
cancellation use the existing run routes and share the same bounded admission
pool. Handoffs require `max_parallel=1`; other widths are rejected before request
registration. A handoff result contains `status`, `final`, `hops` and
`reused_hops`, rather than the task workflow's `results` and `reused_steps`.
As with native progress, run status describes execution; inspect the verified
result to distinguish a final answer from exhausted partial work. Result reads
recheck the current host scope and model bindings before publication. The browser
handoff editor is being implemented separately.

## Current boundary

The catalog supports up to 32 agents, 64 models and 64 allowed models per agent.
Agent instructions are limited to 32,000 UTF-8 bytes and questions to 8,000 bytes;
the tool-loop budgets bound model rounds, tool calls and retained transcript data.
Factories should be quick synchronous constructors; asynchronous model work
belongs in `complete`, where cancellation is enforced cooperatively.

Declared dependencies, bounded parallel execution, controlled dynamic handoffs
and sequential recovery are implemented natively.
Saved plan editing is available through the native store and
authenticated HTTP configuration routes. Catalog factories
remain host-managed application code.
