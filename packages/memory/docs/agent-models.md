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

## Current boundary

The catalog supports up to 32 agents, 64 models and 64 allowed models per agent.
Agent instructions are limited to 32,000 UTF-8 bytes and questions to 8,000 bytes;
the tool-loop budgets bound model rounds, tool calls and retained transcript data.
Factories should be quick synchronous constructors; asynchronous model work
belongs in `complete`, where cancellation is enforced cooperatively.

Multi-agent handoffs, parallel workflow scheduling, persisted catalog editing,
HTTP run management and a browser model selector for these named agents remain
separate work. Existing conversational model connections and workflow checkpoints
are foundations, not a claim that those agent orchestration features are complete.
