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

### Infer a schema from a typed function

Use `function_tool` when the Python signature describes the arguments:

```python
from typing import Annotated
from scone_memory.agents.function_tools import function_tool


def multiply(
    count: Annotated[int, "Number of items."],
    context: ToolContext,
    factor: int = 2,
) -> object:
    """Multiply the supplied count by a factor."""
    return {"total": count * factor, "space": context.space}


count_tool = function_tool(multiply, revision="1", context_parameter="context")
# Register count_tool in AgentCatalog.tools, selecting tools=("multiply",)
# in the agent definition. Model selection works exactly as above.
```

The name and description default to the function name and docstring; explicit
`name` and `description` overrides are available. Both synchronous and async
functions work. Bound methods and `functools.partial` work with inspectable,
annotated signatures; supply explicit metadata when the callable has no name or
docstring. The return value must satisfy the existing strict JSON result contract;
return annotations do not add output validation or convert Python objects.

Pass `describe_from_docstring=True` to give each parameter the description its
docstring writes, in Google (`Args:` then `name (type): text`), NumPy (a
`Parameters` heading underlined with dashes) or Sphinx (`:param name: text`)
style. A description continued on further lines becomes one line. An
`Annotated[T, "description"]` still wins. When the tool's description comes from
the docstring, the parameter section is left out of it, since the same text is now
in the schema; an explicit `description` is kept whole. It is off by default,
because a tool's schema is part of the digest a pending approval is bound to, so
turning it on for an existing tool invalidates approvals waiting on that tool.

Supported parameter annotations are `str`, `int`, `float`, `bool`, `None`,
`list[T]`, `dict[str, T]`, fixed tuples, `tuple[T, ...]`, unions/`Optional`,
`Literal`, scalar-valued enums, and `Annotated[T, "description"]`. JSON arrays
become tuples when the declared type requires one; enum values become members.
Booleans do not satisfy integer parameters. Union conversion first prefers a
branch preserving the input's exact Python types, then a unique conversion.
For example, `int | float` preserves an integer, while `Color | str` receives a
plain string. Ambiguous conversions are refused rather than choosing by order.

Parameters with defaults are optional in the schema. Registration snapshots each
default, includes it in the saved binding, and creates fresh values for every
invocation. Mutating the function's defaults later cannot change the registration.
A default must round-trip without changing its Python type or value: use `1.0`
for a float default, and avoid an enum default in `Color | str` or a tuple default
in `tuple[int, ...] | list[int]`. These ambiguous defaults fail registration.

Every model-supplied parameter requires an annotation. Variadic arguments,
`Any`, arbitrary classes, dataclasses, and Pydantic parameters are not inferred.
Use explicit `AgentTool` schemas and a host-owned conversion function for those
contracts. Constraints such as numeric bounds also require an explicit schema.
At most 32 parameters are accepted, including an injected context.

Only the explicitly named `context_parameter` receives the host context; it is
excluded from the model schema. If annotated, it must resolve to `ToolContext`.
Positional-only and keyword-only parameters retain their Python calling semantics.
String annotations use a bounded syntax parser, without `eval`, imports, calls,
or arbitrary attribute access. `annotation_namespace={"Color": Color}` supplies
local type aliases that are absent from the function's module namespace.
Known `typing` forms are supported; module-qualified application types should
be supplied under a direct alias. Schemas remain bounded to 32,768 UTF-8 bytes.
Python 3.14 deferred annotations require readable Python function source;
registration refuses dynamically created deferred functions without source.
The bounded source module is parsed and compiled without execution, and its
annotation code must match the loaded function. Editing an annotation on disk
without reloading the function therefore refuses registration. This avoids
executing Python's deferred annotation machinery during inspection.

Conversion failures stop execution through the existing sanitized tool-error
path; they never invoke the application function. Schema validation still rejects
invalid arguments before dispatch. The adapter shares the existing deadlines,
cancellation behavior, model selection, and durable-run rules below.

An embedded host can combine its existing private runtime configuration with
trusted Python registrations:

```python
from scone_memory.runtime.agent_runtime import load_agent_runtime

runtime = load_agent_runtime("/path/to/private/agents.json", engine, tools=[count_tool])
```

The JSON agent entry selects `"tools": ["double_count"]`. Unknown names fail
before opening runtime state. The ordinary CLI has no registrations by default;
JSON cannot import Python modules or install handlers. The host owns runtime
shutdown through `await runtime.aclose()` or its application lifespan.

Authenticated `GET /v1/agents/catalog` responses contain a shared `tools` table
with each selected tool's `name`, `description`, and `revision`, and a `tools`
name list for each agent. `agents.tools` advertises this catalog support. Only
selected registrations are published; schemas and executable state stay private.
Descriptions are public to callers authorized to read that host's agent catalog,
so keep credentials and private configuration out of them.

The Python SDK resolves these summaries to immutable `AgentChoice.tools` values.
The console shows them in task and handoff editors alongside model selection.
An explicit empty list means no application tools are configured; absent metadata
on older hosts means availability is unknown. Reading summaries never executes
a model or function and does not let clients change the host's tool policy.

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

### Return a tool result directly

Set `return_direct=True` on either `AgentTool(...)` or `function_tool(...)` to
finish a turn with that tool's successful result. The model selects the tool;
the framework then returns the result without another model request or rewrite.
For example, `function_tool(multiply, revision="1", context_parameter="context",
return_direct=True)` returns the function's JSON object as the final answer.
String results retain their exact text; other JSON values are compact JSON.
Empty strings still fail the final nonempty-answer check.

The default is `False`. Successful direct return removes the final model call;
it does not estimate token savings or synthesize a usage report for that call.
Changing this flag changes the saved agent binding. Explicit `False` preserves
the historical registration format and identity.

Calls in one model response execute in order. After the first successful direct
result, later calls receive `direct_return` denials and do not execute. Earlier
calls retain their outcomes and evidence. Denied invalid arguments do not finish
the turn; exceptions and uncertain outcomes keep the existing stop/no-retry
behavior. If a successful direct result fails its final format or size check,
it is withheld and later calls remain skipped; the framework does not ask a
model to repair or rewrite it.

Direct results still obey reply/transcript/tool-byte limits, deadlines, and
the caller's `AnswerRequirements`. Retained memory evidence is revalidated
before publication and can be checked again later. Application results remain
unverified data, even if their fields claim otherwise. A handoff workflow still
requires its existing `{ "answer": ..., "handoff_to": ... }` decision envelope;
direct tools used there must supply that contract, including any final answer
schema. Direct return finishes the current agent turn, not the entire task DAG
or every subsequent handoff selected by a valid decision.

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

Runtime plugin loading, custom tool management in the console, streaming custom
results, and public SDK registration remain separate capabilities. Inferred
function tools still require explicit host registration and agent selection.


Tools may opt into [native exact-call approval](agent-tool-approvals.md) with
`requires_approval=True`. Guarded agents require an active workflow checkpoint and
an approval context matching the saved plan and selected model. A recorded decision
alone never invokes the handler; an explicit continuation activates and claims the
exact proposed call. The default remains false and preserves prior bindings.
