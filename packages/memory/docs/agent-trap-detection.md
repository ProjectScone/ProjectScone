# Detect repeated agent observations

An agent can spend its budget issuing the same search under new call IDs, or
cycling between searches whose observations do not change. Configure
`ToolLoopLimits(max_repeated_rounds=3)` to stop after three consecutive copies of
such a pattern. This is an opt-in intervention signal, not a hallucination or
factual-accuracy detector.

```python
from scone_memory.agents.catalog import AgentDefinition
from scone_memory.agents.evidence_loop import ToolLoopLimits
from scone_memory.agents.traps import AgentTrapDetected

worker = AgentDefinition(
    agent_id='researcher',
    instructions='Find evidence for the requested answer.',
    models=('local-careful',),
    default_model='local-careful',
    limits=ToolLoopLimits(
        max_tool_calls=12,
        max_tool_rounds=12,
        max_repeated_rounds=3,
    ),
)

# Register worker and its host-owned model factory in the normal AgentCatalog.
# selected = catalog.bind('researcher', model_id='local-careful')
try:
    result = await selected.run(question, tools=scoped_memory_tools)
except AgentTrapDetected as intervention:
    graph = intervention.graph
    # graph.path: observed node IDs in order
    # graph.pattern: the repeating suffix, e.g. (1,) or (1, 2)
    # graph.nodes: first observed round, visits and comparability
    # graph.edges: directed transitions and their occurrence counts
    # The host decides whether to ask for guidance or launch different work.
```

Each completed model tool round is one graph observation. Comparison includes the
read-only memory tools or rejected unknown tools, their JSON arguments and returned observations. Object key
order and provider call IDs do not create new observations. Changed arguments or
results do. Search compaction and cached read presentation compare against their
underlying evidence, so formatting a previous result as a reference cannot hide
repetition. A round containing an application tool breaks the comparable sequence:
identical application output does not prove that the application made no progress.
Repeated unknown-tool rejections can form a trap because no application handler
was invoked; inventing a new call ID does not make the rejected operation new.

The detector checks exact repeated suffixes, including multi-node cycles. Three
visits to the same node at unrelated points are not automatically a trap. It does
not detect semantic paraphrases, unobserved external progress, repeated reasoning
without tools, or cycles spanning separate agent invocations. Legitimate polling
can intentionally return the same observation, so enable the policy only where
this intervention is appropriate. The threshold accepts integers from 2 to 16;
tool and time budgets still apply and may stop execution first.

`AgentTrapDetected` stops before another model call and carries only numeric graph
identities, visit counts and transitions. Raw queries, arguments, result text and
internal signatures are absent from that report. The exception message is the
fixed string `agent_no_progress`. This diagnostic graph is not written into the
factual entity ledger. Source validation, cleanup and cancellation rules remain in
force. No answer is returned by the intervention, and no tool effect is retried.

The default `None` disables the detector and is omitted from serialized limits,
preserving existing configuration identities. An enabled threshold participates
in agent and turn-journal bindings. Journal replay reconstructs observations from
restored receipts instead of issuing already-completed searches again.

The native exception and `trap_detected` progress event carry the graph. When run
history collection is enabled, this event is persisted in encrypted history before
`turn_failed`. The existing authorized history HTTP and event-stream routes expose
its `trap_graph` field after restart, without rerunning the model or tools. Ordinary
events omit that field. History retention and explicit event-loss gaps still apply;
a slow observer is not guaranteed to retain every diagnostic. Validation checks
node visits, directed transition counts and the repeating suffix against the
bounded path. Raw tool observations are never part of this report.

Configurable recovery, cross-run matching and a Console graph view remain separate
required work. Defensive attack/decoy graphs are a different
application and are not implemented by this detector.
