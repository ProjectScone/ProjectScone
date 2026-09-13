# Registered application tools

[Agent models](agent-models.md) · [Scoped memory tools](agent-tools.md)

Native agents can call host-registered synchronous or asynchronous functions
alongside scoped memory reads. Install `scone-memory[structured-output]` for
argument schema validation. The host supplies Python handlers; model responses
and workflow JSON cannot install code or choose an unregistered function.

## Register a function and choose its model

```python
from scone_memory.agents.catalog import AgentCatalog, AgentDefinition, AgentModel
from scone_memory.agents.custom_tools import AgentTool, ToolContext
from scone_memory.providers.tool_chat import SelfHostedToolChat
from scone_memory.integrations.scoped_tools import ScopedMemoryTools
from scone_memory.retrieval.recall_scope import RecallScope


def double_count(arguments: dict[str, object], context: ToolContext) -> object:
    count = arguments["count"]
    if not isinstance(count, int) or isinstance(count, bool):
        raise ValueError("count must be an integer")
    return {"doubled": 2 * count, "space": context.space}


count_tool = AgentTool(
    name="double_count",
    description="Double a supplied count between one and ten.",
    revision="1",
    parameters={
        "type": "object",
        "properties": {"count": {"type": "integer", "minimum": 1, "maximum": 10}},
        "required": ["count"],
        "additionalProperties": False,
    },
    handler=double_count,
)

catalog = AgentCatalog(
    models=[AgentModel(
        "local", "Installed local model", "configuration-1",
        lambda: SelfHostedToolChat("http://127.0.0.1:11434/v1", "your-installed-model"),
    )],
    agents=[AgentDefinition(
        agent_id="counter", instructions="Use the count tool when appropriate.",
        models=("local",), default_model="local", initial_search=False,
        tools=("double_count",),
    )],
    tools=[count_tool],
)

# The host supplies an open engine and the caller's authorized space and scope.
memory_tools = ScopedMemoryTools(
    engine, "default", scope=RecallScope.validated(where={"team": "blue"}),
)
result = await catalog.bind("counter", model_id="local").run(
    "Double three.", tools=memory_tools,
)
print(result.output.text)
```

Register additional `AgentModel` factories and include their IDs in the agent's
`models` tuple to let users choose among those models. Each factory must return
a fresh configured client. `AgentTask.model_id` selects the model for a saved
task workflow; handoff and interactive workflows use the same catalog binding.
No model is discovered, downloaded, or substituted automatically.

`SelfHostedStructuredToolChat` supports the same registrations for local models
using schema-constrained actions. Its custom action is
`{"action":"double_count","arguments":{"count":3}}`. Native tool providers
receive ordinary function schemas. Both paths validate arguments on the host
before calling the handler, even if the provider ignores the schema.

## Contracts and scope

An agent sees only the tools named in its definition. A catalog accepts at most
32 registrations and rejects duplicate, reserved, or unknown selected names.
Names contain 1–64 ASCII letters, digits, underscores or hyphens. Descriptions
are limited to 4,000 UTF-8 bytes and offered custom metadata to 128,000 bytes.
Parameters use the existing bounded object-root JSON Schema compiler: local
acyclic references may be expanded; external references are refused without
network access. Public schema metadata is immutable, and provider schemas and
handler argument dictionaries are detached copies.

Handlers receive `ToolContext(space, scope, exclude_session_id, deadline)` from
the host's scoped memory binding. The context contains no engine or credentials.
It does not grant access to arbitrary resources. A handler that accesses storage
must enforce the context's scope and its own authorization policy. Handlers
are trusted application code, not sandboxed code.

Successful results contain strict JSON values and are wrapped as unverified
application data. The result can inform an answer, but cannot create retained
memory evidence, authorize a trace/read seed, or certify factual accuracy.
Fields such as `facts` inside an application result remain application data.
Real memory evidence is still revalidated before the answer is returned.

## Budgets, failures and saved runs

Custom calls share the existing call, round, transcript, output-byte and turn
deadline budgets. Arguments have a 16,000-byte limit. Each result wrapper has a
16,000-byte default cap, configurable up to 64,000 bytes, and must also fit the
remaining aggregate tool budget. An expired deadline, an already oversized
transcript, or insufficient space for even the smallest successful result stops
execution before the handler starts. Actual result size is checked afterward;
unknown result sizes cannot be guaranteed to fit in advance.

Invalid arguments return an `invalid_arguments` denial without executing the
handler. Handler exceptions, invalid or oversized output, timeout, or cancellation
stop the turn. The framework does not automatically retry a failed handler.
Synchronous handlers run in a worker thread; cancellation cannot force-stop
an executing handler or undo its effects. A queued worker checks cancellation
and the deadline again immediately before entering the handler. A coroutine returned after cancellation is
closed instead of being executed. Async handlers must cooperate with cancellation.

Durable workflows record an interrupted attempt as an uncertain outcome and
refuse to replay it automatically after restart. A completed saved step is reused
without invoking the function again. This is not an exactly-once guarantee for
external effects; applications must reconcile an uncertain operation using their
own storage and idempotency policy.

Tool names, descriptions, argument schemas, output caps and declared revisions
participate in the saved agent binding. Change `revision` whenever handler code
or its external configuration changes. Python function bodies and mutable
closures are not hashed. A changed registration refuses reuse of an old run.
Agents with no selected application tools retain their historical serialization
and binding identity.

This API registers functions explicitly. Automatic signature inference,
runtime plugin loading, custom tool management in the console, and public SDK
registration are separate capabilities and are not provided by this change.
