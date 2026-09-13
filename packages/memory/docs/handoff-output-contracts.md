# Final output requirements for handoff workflows

`AgentHandoffPlan.answer_requirements` applies a workflow-wide contract to the
terminal answer. Each agent still uses its host-approved model and may delegate
only along the plan's declared edges. Intermediate notes remain ordinary text;
a chain that exhausts its handoff budget has no final answer.

```python
from scone_memory.agents.handoff_workflow import AgentHandoffPlan, HandoffAgent
from scone_memory.agents.task_requirements import TaskAnswerRequirements

plan = AgentHandoffPlan(
    workflow_id='research-report', root_agent='research', max_handoffs=3,
    agents=(
        HandoffAgent(agent_id='research', model_id='careful', can_handoff_to=('write',)),
        HandoffAgent(agent_id='write', model_id='fast'),
    ),
    answer_requirements=TaskAnswerRequirements(
        format='json_object', max_bytes=4000, max_lines=1,
        output_schema={
            'properties': {'summary': {'type': 'string'}},
            'required': ['summary'], 'additionalProperties': False,
        },
    ),
)
```

The requirements use the same fields and bounds as
[task output contracts](task-output-contracts.md). With a JSON contract, a model
finishes with `{"answer":{"summary":"..."},"handoff_to":null}` and delegates
with `{"answer":"Research notes","handoff_to":"write"}`. Text contracts retain
a string-valued terminal `answer`. The provider receives both the routing schema
and final contract; the host checks the terminal result before persistence.

The result API continues to return `final.text` as a string. JSON object output
is compacted by removing only structural whitespace: decimal precision,
exponent spelling, escapes, key order and string whitespace are preserved.
Byte and line limits apply to this compact final string. Limits on the outer
model response, transcript, handoff context and journal still apply. A final
contract cannot increase those host budgets.

Saved plans preserve the authored schema and local references. Execution
compiles a detached copy. Admission also compiles the routing envelope, so a
schema that fits by itself may be refused if wrapping it exceeds the schema
byte, depth or node budget. Handoff contracts require the `structured-output`
extra, including text-only contracts, because routing uses an outer schema.
Check `agents.handoffs.output_requirements` before authoring a handoff contract
and `agents.output_schema` when supplying a schema. Older task-contract servers
may advertise task support without handoff support.

Changing any contract field changes run identity. Saved terminal receipts are
checked against the contract again on read and reuse, together with their
original model bindings, memory scope and source retention. Rejected output is
withheld without automatic retries or repair. An omitted contract keeps the
historical plan shape, routing protocol and run identity.

Schema validation is synchronous and uses bounded, local Draft 2020-12
references; its structural limits do not impose a CPU deadline on complex
patterns. Plan-writing credentials remain for trusted authors. A conforming
schema proves output shape, not factual accuracy. Custom output transformations
and typed partial streaming are separate capabilities.
