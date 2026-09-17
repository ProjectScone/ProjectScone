"""A repeated observation is grounds for intervention, not proof of hallucination."""

import pytest

from scone_memory import HashEmbedder, InMemoryDocumentStore, InMemoryVectorIndex, MemoryEngine
from scone_memory.agents.evidence_loop import EvidenceToolLoop, ToolLoopLimits, ToolStep
from tests.agents.test_evidence_tool_loop import Script, binding, search


@pytest.fixture
async def memory():
    engine = await MemoryEngine(InMemoryDocumentStore(), InMemoryVectorIndex(), HashEmbedder()).open()
    await engine.remember('alpha', 'Juniper uses Polaris.', metadata={'team': 'blue'})
    try:
        yield engine
    finally:
        await engine.close()


async def test_repeated_search_observations_stop_before_another_model_call(memory):
    model = Script(*(search('call-' + str(i)) for i in range(4)), ToolStep(content='Invented progress.'))
    loop = EvidenceToolLoop(model, binding(memory), limits=ToolLoopLimits(
        max_tool_calls=8, max_tool_rounds=8, max_repeated_rounds=3))
    with pytest.raises(RuntimeError, match='agent_no_progress') as caught:
        await loop.run([{'role': 'user', 'content': 'Find the answer.'}])
    assert len(model.requests) == 3
    report = caught.value.graph
    assert report.path == (1, 1, 1)
    assert report.pattern == (1,) and report.repetitions == 3
    assert report.nodes[0].visits == 3
    assert [(edge.source, edge.target, edge.count) for edge in report.edges] == [(1, 1, 2)]
    assert 'Juniper' not in repr(report) and 'Polaris' not in repr(report)


async def test_disabled_policy_preserves_limited_retries(memory):
    model = Script(*(search('call-' + str(i)) for i in range(4)), ToolStep(content='Juniper uses Polaris.'))
    result = await EvidenceToolLoop(model, binding(memory)).run([{'role': 'user', 'content': 'Find the answer.'}])
    assert result.model_calls == 5 and await result.validate()
    assert 'max_repeated_rounds' not in ToolLoopLimits().model_dump()


async def test_changed_search_observations_are_not_a_trap(memory):
    model = Script()

    async def changed_search():
        await memory.remember('alpha', 'Juniper operates in Oregon.', metadata={'team': 'blue'})
        return search('second')

    model.steps = [search('first'), changed_search, ToolStep(content='Juniper uses Polaris.')]
    # New memory changes the source snapshot. It is not a repetition trap,
    # but the existing final source-validation rule must still refuse it.
    with pytest.raises(RuntimeError, match='evidence changed'):
        await EvidenceToolLoop(model, binding(memory), limits=ToolLoopLimits(
            max_repeated_rounds=2)).run([{'role': 'user', 'content': 'Find the answer.'}])
    assert len(model.requests) == 3


@pytest.mark.parametrize('limit', [True, 1, 17, 2.5, '3'])
def test_policy_refuses_ambiguous_or_unbounded_thresholds(limit):
    with pytest.raises(ValueError):
        ToolLoopLimits(max_repeated_rounds=limit)


def test_alternating_cycle_has_directed_edges_and_repetition_evidence():
    from scone_memory.agents.traps import ObservationGraph

    graph = ObservationGraph(3)
    for observation in [b'a', b'b', b'a', b'b', b'a']:
        assert graph.observe(observation) is None
    trap = graph.observe(b'b')
    assert trap is not None and trap.pattern == (1, 2)
    assert trap.path == (1, 2, 1, 2, 1, 2)
    assert [(edge.source, edge.target, edge.count) for edge in trap.edges] == [(1, 2, 3), (2, 1, 2)]


def test_unobserved_application_progress_breaks_repetition():
    from scone_memory.agents.traps import ObservationGraph

    graph = ObservationGraph(2)
    assert graph.observe(b'same retrieval') is None
    assert graph.observe(None) is None
    assert graph.observe(b'same retrieval') is None
    assert graph.snapshot().nodes[1].comparable is False


def test_policy_changes_agent_binding_but_disabled_preserves_old_limits():
    from scone_memory.agents.catalog import AgentDefinition, AgentModel, BoundAgent

    definition = AgentDefinition(agent_id='worker', instructions='Find evidence.',
                                 models=('chosen',), default_model='chosen')
    model = AgentModel('chosen', 'Chosen', '1', lambda: Script())
    before = BoundAgent(definition, model)
    enabled = BoundAgent(definition.model_copy(update={'limits': ToolLoopLimits(max_repeated_rounds=3)}), model)
    assert before.fingerprint != enabled.fingerprint
    assert before.definition.limits.model_dump() == {
        'max_tool_calls': 4, 'max_tool_rounds': 4, 'timeout_s': 120.0,
        'max_transcript_bytes': 256000, 'max_tool_bytes': 128000, 'max_reply_bytes': 16000,
    }


async def test_application_tool_activity_breaks_the_memory_repetition_sequence(memory):
    from tests.agents.test_custom_tools import call, tool

    effects = []
    model = Script(search('first'), call(), search('second'), ToolStep(content='Completed.'))
    registered = tool(lambda arguments, context: effects.append(arguments['count']))
    result = await EvidenceToolLoop(model, binding(memory), custom_tools=[registered],
                                   limits=ToolLoopLimits(max_repeated_rounds=2)).run(
                                       [{'role': 'user', 'content': 'Find the answer.'}])
    assert effects == [3] and result.model_calls == 4 and await result.validate()


async def test_compacted_search_presentation_does_not_hide_repetition(memory):
    await memory.remember('alpha', 'Juniper manual ' + 'documentation ' * 200, metadata={'team': 'blue'})
    model = Script(search('first'), search('second'), search('third'))
    with pytest.raises(RuntimeError, match='agent_no_progress') as caught:
        await EvidenceToolLoop(model, binding(memory), compact_search_results=True,
                               limits=ToolLoopLimits(max_repeated_rounds=3)).run(
                                   [{'role': 'user', 'content': 'Find the answer.'}])
    assert caught.value.graph.path == (1, 1, 1)
    assert '"reuse"' in model.requests[-1][0][-1]['content']


async def test_chosen_agent_closes_model_and_emits_failure_on_intervention(memory):
    from scone_memory.agents.catalog import AgentDefinition, AgentModel, BoundAgent
    from scone_memory.agents.progress import AgentEventStream

    closed = []
    model = Script(search('first'), search('second'))

    async def close():
        closed.append(True)

    model.aclose = close
    agent = BoundAgent(AgentDefinition(
        agent_id='worker', instructions='Find evidence.', models=('chosen',), default_model='chosen',
        initial_search=False, limits=ToolLoopLimits(max_repeated_rounds=2)),
        AgentModel('chosen', 'Chosen', '1', lambda: model))
    events = AgentEventStream()
    with pytest.raises(RuntimeError, match='agent_no_progress'):
        await agent.run('Find the answer.', tools=binding(memory), events=events)
    recorded = [event async for event in events]
    assert closed == [True] and len(model.requests) == 2
    assert recorded[-1].kind == 'turn_failed'
    assert all(event.model_id == 'chosen' for event in recorded)


async def test_checkpoint_replay_reconstructs_repetition_without_repeating_searches(memory, tmp_path):
    from scone_memory.agents.catalog import AgentDefinition, AgentModel, BoundAgent
    from scone_memory.agents.traps import AgentTrapDetected
    from scone_memory.agents.turn_journal import TurnJournalPaused
    from scone_memory.agents.workflow import WorkflowError, WorkflowPausableStep, WorkflowRunner

    model = Script(search('first'), search('second'), search('third'))
    agent = BoundAgent(AgentDefinition(
        agent_id='worker', instructions='Find evidence.', models=('chosen',), default_model='chosen',
        initial_search=False, limits=ToolLoopLimits(max_repeated_rounds=3)),
        AgentModel('chosen', 'Chosen', '1', lambda: model))
    scoped = binding(memory)
    prepare = scoped.prepare
    searches = []
    reports = []

    async def count_search(*args):
        searches.append(True)
        return await prepare(*args)

    scoped.prepare = count_search

    async def verify(context):
        return True

    async def execute(context):
        try:
            return (await agent.run('Find the answer.', tools=scoped, checkpoints=context.checkpoints,
                                    max_new_operations=2)).output.text
        except TurnJournalPaused as pause:
            return pause.pause
        except AgentTrapDetected as trap:
            reports.append(trap.graph)
            raise

    config = {'path': tmp_path / 'journal', 'key': b'k' * 32,
              'steps': [WorkflowPausableStep('agent', '1', execute)], 'source_verifier': verify}
    for attempt in range(3):
        job = WorkflowRunner(**config)
        try:
            if attempt < 2:
                assert (await job.run('one', space='alpha', scope={}, inputs=None)).status == 'paused'
            else:
                with pytest.raises(WorkflowError):
                    await job.run('one', space='alpha', scope={}, inputs=None)
        finally:
            job.close()
    assert len(searches) == 3 and len(model.requests) == 3
    assert reports[0].path == (1, 1, 1)


async def test_alternating_search_cycle_stops_with_a_two_node_pattern(memory):
    model = Script(*(search('call-' + str(i), query=('Juniper' if i % 2 == 0 else 'Polaris'))
                     for i in range(6)))
    with pytest.raises(RuntimeError, match='agent_no_progress') as caught:
        await EvidenceToolLoop(model, binding(memory), limits=ToolLoopLimits(
            max_tool_calls=8, max_tool_rounds=8, max_repeated_rounds=3)).run(
                [{'role': 'user', 'content': 'Find the answer.'}])
    assert len(model.requests) == 6 and caught.value.graph.pattern == (1, 2)


def test_graph_memory_is_bounded_to_the_native_round_limit():
    from scone_memory.agents.traps import ObservationGraph

    graph = ObservationGraph(3)
    for index in range(16):
        assert graph.observe(str(index).encode()) is None
    with pytest.raises(ValueError, match='round limit'):
        graph.observe(b'seventeenth')
    assert len(graph.snapshot().nodes) == 16


async def test_repeated_unknown_tool_failures_stop_without_executing_a_tool(memory):
    from scone_memory.agents.evidence_loop import ToolCall

    model = Script(*(ToolStep(calls=(ToolCall(id='missing-' + str(i), name='imaginary_tool',
                                              arguments={'query': 'private request'}),))
                     for i in range(2)), ToolStep(content='Invented success.'))
    with pytest.raises(RuntimeError, match='agent_no_progress') as caught:
        await EvidenceToolLoop(model, binding(memory), limits=ToolLoopLimits(max_repeated_rounds=2)).run(
            [{'role': 'user', 'content': 'Find the answer.'}])
    assert len(model.requests) == 2 and caught.value.graph.path == (1, 1)
    assert 'imaginary_tool' not in repr(caught.value.graph)
