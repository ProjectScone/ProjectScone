# Per-task output requirements

Task and interactive workflows accept optional `AgentTask.answer_requirements`.
Each model task still selects a model from the host's registered catalog. Its
requirements guide every model round and gate the final answer before it can be
saved or passed to dependent tasks. Human-input tasks retain their plain-text
response contract.

```python
from scone_memory.agents.task_requirements import TaskAnswerRequirements
from scone_memory.agents.task_workflow import AgentTask, AgentTaskPlan

plan = AgentTaskPlan(
    workflow_id='extract-name',
    tasks=(AgentTask(
        task_id='extract', agent_id='research', model_id='local',
        prompt='Extract the supported name from the supplied evidence.',
        answer_requirements=TaskAnswerRequirements(
            format='json_object', max_bytes=4000, max_lines=1,
            output_schema={
                'type': 'object',
                'properties': {'name': {'type': 'string', 'minLength': 1}},
                'required': ['name'],
                'additionalProperties': False,
            },
        ),
    ),),
)
```

`instructions` defaults to an empty string and accepts at most 8000 UTF-8 bytes.
`max_bytes` defaults to 64000 and accepts 1–128000. `max_lines` defaults to no
additional line limit and accepts 1–1000. The format is `text` (default) or
`json_object`. Existing agent transcript, answer and workflow payload budgets
still apply; these settings cannot increase a host's execution limits.

JSON-object output must be one object, without fences, duplicate keys or
nonstandard numeric constants. Optional `output_schema` additionally checks its
fields and values. Schemas require the `structured-output` extra, use the
existing bounded Draft 2020-12 compiler, and may reference local definitions.
External references and recursive schemas are refused. Authored schemas are
limited to 32768 UTF-8 bytes, depth 32 and 4096 JSON values; expanded execution
schemas are additionally bounded by the compiler. Validation is synchronous:
these limits do not impose a CPU deadline on complex patterns or combinations.
Only trusted plan authors should be granted plan-writing credentials. Schema
validation establishes output shape, not factual accuracy; source retention and
scope checks still apply.

Plan save/get responses preserve the authored schema, including `$defs` and
`$ref`. Execution compiles a detached copy. This keeps saved acknowledgments
faithful to the submitted contract and prevents a provider from weakening the
stored requirements. The complete contract participates in run identity:
changing it requires a new run. Invalid output is withheld without automatic
retry, truncation or repair. Result text remains the original validated string.
Completed task receipts are rechecked against the contract on reuse and read.

Omitted requirements preserve historical task serialization and run identity.
Old journals can be reopened without model calls. Older servers and clients do
not understand this field: check `agents.output_requirements` before authoring
any contract, and `agents.output_schema` before authoring a schema. These HTTP
capabilities describe mounted plan support and optional validator availability;
simple text/JSON-object requirements work without the schema dependency.

Task and interactive DAGs use per-task contracts. Handoff workflows instead use
a [workflow-wide final contract](handoff-output-contracts.md). Custom output
transformation code and typed partial streaming remain separate capabilities.
