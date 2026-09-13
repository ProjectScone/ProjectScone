# Token accounting for native agent turns

`EvidenceToolLoop.run()` returns ordered usage reports for its accepted model
responses in `result.usage.calls`. A named agent exposes the same reports through
`AgentResult.output.usage`, alongside its selected `model_id` and binding.

```python
selected = catalog.bind("research", model_id="careful")
result = await selected.run("What changed?", tools=scoped_tools)
usage = result.output.usage
assert len(usage.calls) == result.output.model_calls

# These are optional provider reports, not estimates from answer length.
print(result.model_id, usage.prompt_tokens, usage.completion_tokens, usage.total_tokens)
for index, report in enumerate(usage.calls, start=1):
    print(index, report.model_dump())
```

Each `ModelTokenUsage` contains optional `prompt_tokens`, `completion_tokens`,
and `total_tokens`. `None` means unknown; a reported integer zero remains zero.
For each category, the turn total is available only when every model call
reports that category. Partial observations remain accessible in `calls`, in
request order. A missing total is never inferred by adding prompt and completion
counts. An empty usage object on a manually constructed legacy result also has
unknown totals.

`SelfHostedToolChat` and `SelfHostedStructuredToolChat` read these fields from the
response's top-level `usage` metadata. This covers native tools, structured
actions, prose answers, and constrained JSON answers. Answer text cannot supply
usage counts. Extra provider metadata is discarded. Counts must be strict
integers in `0..1_000_000_000`; booleans, strings, fractions, negatives, and larger
values become unknown. A reported total smaller than known components, or
unequal to their sum when both components are known, becomes unknown while
valid individual components remain available. Invalid optional telemetry does
not discard an otherwise accepted answer. Private request diagnostics use the
same validation.

Custom adapters can supply the same strict contract:

```python
from scone_memory.agents.evidence_loop import ToolStep
from scone_memory.agents.usage import ModelTokenUsage

step = ToolStep(content="The source says...", usage=ModelTokenUsage(
    prompt_tokens=120, completion_tokens=18, total_tokens=138,
))
```

Existing adapters that return `ToolStep(content=...)` continue to work and report
unknown counts. Invalid custom usage is rejected by the loop as an invalid tool
protocol, including objects constructed by bypassing normal model validation.
Reports are detached after each response so a later adapter call cannot rewrite
an earlier observation. The loop's existing limit permits at most 17 model
calls, and their aggregate can exceed the per-response count limit.

Host-initiated searches and tool executions do not add model usage entries.
Source validation, output-format checks, deadlines, and cancellation still gate
publication of the completed result.

## Scope

This is native completed-turn telemetry. Counts are provider assertions, not
independent measurements, billing receipts, prices, or answer-quality scores.
Requests that fail, time out, or lose their response can consume tokens without
producing a completed turn result; these totals cannot represent all attempted
work. Existing private provider diagnostics remain available for failures when
metadata was received.

This change does not add usage to persisted `AgentTaskReceipt` records, workflow
HTTP responses, standalone client result types, or console views. Those surfaces
need a coordinated contract extension before they can display durable per-model
usage. Native users can serialize `usage.model_dump_json()` themselves; that
representation contains the ordered reports, and the aggregate properties are
recomputed when read with `ToolTokenUsage.model_validate_json()`.
