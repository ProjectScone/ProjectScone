"""Live progress reflects actual operations without exposing private content."""

import asyncio
from dataclasses import asdict
import json

import pytest

from scone_memory.agents.catalog import AgentCatalog, AgentDefinition, AgentModel
from scone_memory.agents.evidence_loop import ToolStep
from scone_memory.agents.progress import AgentEventStream, AgentProgressGap
from scone_memory.agents.turn_journal import TurnJournalPaused
from .test_agent_turn_recovery import storage
from .test_custom_tools import call, tool
from .test_evidence_tool_loop import binding
from .test_task_workflow import memory


def agent(factory, registered=(), initial_search=False):
    return AgentCatalog(
        models=[AgentModel('chosen', 'Chosen', 'v1', factory)],
        agents=[
            AgentDefinition(
                agent_id='worker',
                instructions='PRIVATE instructions',
                models=('chosen',),
                default_model='chosen',
                initial_search=initial_search,
                tools=tuple(item.name for item in registered),
            )
        ],
        tools=registered,
    ).bind('worker')


async def test_live_model_span_and_terminal_follow_cleanup(memory):
    entered = asyncio.Event()
    release = asyncio.Event()
    closed = []

    class Model:
        async def complete(self, messages, tools):
            entered.set()
            await release.wait()
            return ToolStep(content='PRIVATE answer')

        async def aclose(self):
            closed.append(True)

    selected = agent(Model)
    events = AgentEventStream()
    running = asyncio.create_task(selected.run('PRIVATE question', tools=binding(memory), events=events))
    reader = events.__aiter__()
    try:
        first = await asyncio.wait_for(anext(reader), 1)
        started = await asyncio.wait_for(anext(reader), 1)
        assert first.kind == 'turn_started' and started.kind == 'operation_started'
        assert started.operation_kind == 'model' and not running.done()
        await entered.wait()
        release.set()
        result = await running
        remaining = [event async for event in reader]
        assert [event.kind for event in remaining] == ['operation_completed', 'turn_completed']
        assert closed == [True] and result.output.text == 'PRIVATE answer'
        records = [first, started, *remaining]
        assert all(
            event.agent_id == 'worker'
            and event.model_id == 'chosen'
            and event.binding == selected.fingerprint
            for event in records
        )
        assert [event.sequence for event in records] == list(range(1, 5))
        assert remaining[0].operation_id == started.operation_id and remaining[0].duration_s >= 0
        assert 'PRIVATE' not in json.dumps([asdict(event) for event in records])
    finally:
        release.set()
        await asyncio.gather(running, return_exceptions=True)
        await reader.aclose()


async def test_slow_consumer_gets_exact_gap_and_terminal(memory):
    class Model:
        async def complete(self, messages, tools):
            return ToolStep(content='Done')

    events = AgentEventStream(max_events=1)
    await agent(Model).run('Question', tools=binding(memory), events=events)
    records = [event async for event in events]
    assert len(records) == 2 and isinstance(records[0], AgentProgressGap)
    assert (records[0].first_sequence, records[0].last_sequence) == (1, 3)
    assert records[1].kind == 'turn_completed' and records[1].sequence == 4
    assert events.dropped_events == 3


async def test_cancelled_consumer_does_not_cancel_model(memory):
    entered = asyncio.Event()
    release = asyncio.Event()

    class Model:
        async def complete(self, messages, tools):
            entered.set()
            await release.wait()
            return ToolStep(content='Done')

    events = AgentEventStream()
    running = asyncio.create_task(agent(Model).run('Question', tools=binding(memory), events=events))

    async def consume():
        return [event async for event in events]

    reader = asyncio.create_task(consume())
    try:
        await entered.wait()
        reader.cancel()
        with pytest.raises(asyncio.CancelledError):
            await reader
        assert not running.done()
        release.set()
        assert (await running).output.text == 'Done'
        assert [event.kind async for event in events][-1] == 'turn_completed'
    finally:
        release.set()
        await asyncio.gather(running, reader, return_exceptions=True)


async def test_only_one_reader_and_one_invocation_per_stream(memory):
    events = AgentEventStream()
    first = events.__aiter__()
    waiting = asyncio.create_task(anext(first))
    await asyncio.sleep(0)
    second = events.__aiter__()
    with pytest.raises(RuntimeError, match='reader'):
        await anext(second)

    class Model:
        async def complete(self, messages, tools):
            return ToolStep(content='Done')

    selected = agent(Model)
    try:
        await selected.run('Question', tools=binding(memory), events=events)
        await waiting
        with pytest.raises(ValueError, match='already'):
            await selected.run('Again', tools=binding(memory), events=events)
    finally:
        await first.aclose()
        await second.aclose()


@pytest.mark.parametrize('failure', ['question', 'provider', 'cleanup'])
async def test_errors_close_stream_without_private_exception_text(memory, failure):
    class Model:
        async def complete(self, messages, tools):
            if failure == 'provider':
                raise RuntimeError('PRIVATE provider credential')
            return ToolStep(content='PRIVATE answer')

        async def aclose(self):
            if failure == 'cleanup':
                raise RuntimeError('PRIVATE cleanup credential')

    events = AgentEventStream()
    with pytest.raises((RuntimeError, ValueError)):
        await agent(Model).run(
            '' if failure == 'question' else 'PRIVATE question', tools=binding(memory), events=events
        )
    records = [event async for event in events]
    assert records[-1].kind == 'turn_failed'
    assert 'PRIVATE' not in json.dumps([asdict(event) for event in records])


async def test_journal_reuse_is_not_reported_as_another_model_execution(memory):
    requests = []
    effects = []

    class Model:
        async def complete(self, messages, tools):
            requests.append(1)
            return (
                ToolStep(content='Done')
                if any(row['role'] == 'tool' for row in messages)
                else call(call_id='PRIVATE-call-ID')
            )

    selected = agent(Model, [tool(lambda args, ctx: effects.append(args['count']))])
    points, _ = storage()
    first = AgentEventStream()
    with pytest.raises(TurnJournalPaused):
        await selected.run(
            'PRIVATE question', tools=binding(memory), checkpoints=points, max_new_operations=1, events=first
        )
    before = [event async for event in first]
    assert before[-1].kind == 'turn_paused' and effects == [] and len(requests) == 1
    second = AgentEventStream()
    await selected.run('PRIVATE question', tools=binding(memory), checkpoints=points, events=second)
    after = [event async for event in second]
    assert len(requests) == 2 and effects == [3]
    assert [(event.operation_kind, event.kind) for event in after if event.kind == 'operation_reused'] == [
        ('model', 'operation_reused')
    ]
    assert (
        len(
            [
                event
                for event in after
                if event.kind == 'operation_started' and event.operation_kind == 'model'
            ]
        )
        == 1
    )
    assert [event.kind for event in after].count('tool_proposed') == 1
    assert [event.kind for event in after].count('tool_result') == 1
    assert 'PRIVATE' not in json.dumps([asdict(event) for event in before + after])


@pytest.mark.parametrize('maximum', [0, -1, True, 1025, 1.5])
def test_stream_capacity_is_bounded(maximum):
    with pytest.raises(ValueError):
        AgentEventStream(max_events=maximum)


async def test_host_search_and_direct_return_skips_have_truthful_metadata(memory):
    from scone_memory.agents.evidence_loop import ToolCall

    await memory.remember('alpha', 'PRIVATE Juniper source.', metadata={'team': 'blue'})
    effects = []

    class Model:
        async def complete(self, messages, tools):
            return ToolStep(
                calls=(
                    ToolCall(id='PRIVATE-first', name='double_count', arguments={'count': 3}),
                    ToolCall(id='PRIVATE-second', name='double_count', arguments={'count': 7}),
                )
            )

    def invoke(arguments, context):
        effects.append(arguments['count'])
        return 'PRIVATE direct answer'

    selected = agent(Model, [tool(invoke, return_direct=True)], initial_search=True)
    events = AgentEventStream()
    result = await selected.run('PRIVATE Juniper query', tools=binding(memory), events=events)
    records = [event async for event in events]
    assert result.output.text == 'PRIVATE direct answer' and effects == [3]
    proposed = [event for event in records if event.kind == 'tool_proposed']
    assert [(event.tool_index, event.tool_name, event.origin) for event in proposed] == [
        (1, 'search_memory', 'host'),
        (2, 'double_count', 'model'),
        (3, 'double_count', 'model'),
    ]
    results = [event for event in records if event.kind == 'tool_result']
    assert results[-1].error == 'direct_return' and results[-1].status == 'unavailable'
    assert (
        len(
            [
                event
                for event in records
                if event.kind == 'operation_started' and event.operation_kind == 'custom'
            ]
        )
        == 1
    )
    assert (
        len(
            [
                event
                for event in records
                if event.kind == 'operation_started' and event.operation_kind == 'model'
            ]
        )
        == 1
    )
    assert 'PRIVATE' not in json.dumps([asdict(event) for event in records])


async def test_cancellation_terminal_waits_for_model_disposal(memory):
    entered = asyncio.Event()
    closing = asyncio.Event()
    release = asyncio.Event()

    class Model:
        async def complete(self, messages, tools):
            entered.set()
            await asyncio.Event().wait()

        async def aclose(self):
            closing.set()
            await release.wait()

    events = AgentEventStream()
    running = asyncio.create_task(agent(Model).run('Question', tools=binding(memory), events=events))
    observed = []

    async def consume():
        async for event in events:
            observed.append(event)

    reader = asyncio.create_task(consume())
    try:
        await entered.wait()
        running.cancel()
        await closing.wait()
        await asyncio.sleep(0)
        assert not running.done() and not reader.done()
        assert all(event.kind != 'turn_cancelled' for event in observed)
        release.set()
        with pytest.raises(asyncio.CancelledError):
            await running
        await reader
        assert observed[-1].kind == 'turn_cancelled'
        assert [event.kind for event in observed].count('operation_failed') == 1
    finally:
        release.set()
        await asyncio.gather(running, reader, return_exceptions=True)


@pytest.mark.parametrize('decision', ['approve', 'deny'])
async def test_approval_events_do_not_advance_actions_and_reuse_original_model(memory, tmp_path, decision):
    from scone_memory.agents.approval_context import ApprovalContext
    from scone_memory.agents.approval_store import AgentApprovalStore
    from scone_memory.agents.plan_store import AgentPlanStore
    from scone_memory.agents.run_store import AgentRunStore
    from scone_memory.agents.task_workflow import AgentTask, AgentTaskPlan
    from scone_memory.agents.workflow import WorkflowPausableStep, WorkflowRunner
    from scone_memory.retrieval.recall_scope import RecallScope

    effects = []
    requests = []
    streams = []

    class Model:
        async def complete(self, messages, tools):
            requests.append(1)
            return ToolStep(content='Done') if any(row['role'] == 'tool' for row in messages) else call()

    selected = agent(Model, [tool(lambda args, ctx: effects.append(args['count']), requires_approval=True)])
    catalog = AgentCatalog(models=[selected.model], agents=[selected.definition], tools=selected.tools)
    plans = AgentPlanStore(tmp_path / 'plans', key=b'k' * 32)
    runs = AgentRunStore(tmp_path / 'runs', key=b'k' * 32)
    store = AgentApprovalStore(runs)
    plan = AgentTaskPlan(
        workflow_id='job', tasks=(AgentTask(task_id='work', agent_id='worker', prompt='Count'),)
    )
    saved = plans.save('alpha', plan, catalog=catalog, expected_revision=0)
    runs.register(
        'alpha',
        'one',
        plan=saved,
        question='Count',
        scope=RecallScope.validated(where={'team': 'blue'}),
        exclude_session_id='current',
    )
    activation = None

    async def valid(context):
        return True

    async def execute(context):
        stream = AgentEventStream()
        streams.append(stream)
        approval = ApprovalContext(
            store, context, step_id='work', selection_id='work', activation_id=activation
        )
        try:
            result = await selected.run(
                'Count',
                tools=binding(memory),
                checkpoints=context.checkpoints,
                approval=approval,
                events=stream,
            )
            return result.output.text
        except TurnJournalPaused as pause:
            return pause.pause

    workflow = WorkflowRunner(
        tmp_path / 'workflow',
        key=b'k' * 32,
        steps=[WorkflowPausableStep('work', '1', execute)],
        source_verifier=valid,
    )

    async def run_once():
        return await workflow.run('one', space='alpha', scope={}, inputs='Count')

    try:
        assert (await run_once()).status == 'paused'
        first = [event async for event in streams[-1]]
        assert first[-1].kind == 'turn_paused' and effects == [] and len(requests) == 1
        assert not any(event.operation_kind == 'custom' for event in first)
        (pending,) = store.list('alpha', 'one')
        store.decide(
            'alpha', 'one', pending.request_id, decision=decision, actor='owner', expected_revision=1
        )
        assert effects == [] and len(requests) == 1
        store.activate('alpha', 'one', 'go', decisions={pending.request_id: 2})
        activation = 'go'
        assert (await run_once()).status == 'completed'
        second = [event async for event in streams[-1]]
        assert second[-1].kind == 'turn_completed' and len(requests) == 2
        assert effects == ([3] if decision == 'approve' else [])
        assert any(event.kind == 'operation_reused' and event.operation_kind == 'model' for event in second)
        outcome = next(event for event in second if event.kind == 'tool_result')
        assert outcome.error == (None if decision == 'approve' else 'approval_denied')
    finally:
        workflow.close()
        plans.close()
        runs.close()


async def test_cross_loop_reader_refuses_without_consuming_events(memory):
    class Model:
        async def complete(self, messages, tools):
            return ToolStep(content='Done')

    events = AgentEventStream()
    await agent(Model).run('Question', tools=binding(memory), events=events)

    async def consume():
        return [event async for event in events]

    with pytest.raises(RuntimeError, match='another event loop'):
        await asyncio.to_thread(lambda: asyncio.run(consume()))
    assert [event.kind async for event in events][-1] == 'turn_completed'
