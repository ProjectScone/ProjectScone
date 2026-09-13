"""Independent recovery regressions promoted from boundary review."""

import asyncio
import json
import pytest
from .test_task_workflow import memory
from .test_evidence_tool_loop import Script, binding, search
from .test_custom_tools import agents, call, tool
from scone_memory.agents.evidence_loop import ToolStep
from scone_memory.agents.usage import ModelTokenUsage
from scone_memory.agents.turn_journal import TurnJournalPaused, TurnJournalError
from scone_memory.agents.workflow import StepCheckpoints


def storage():
    records = {}

    def put(key, value):
        records[key] = value

    return StepCheckpoints(records.get, put), records


async def test_scope_drift_during_model_await_refuses_effect(memory):
    scoped = binding(memory)
    seen = []

    class Mutating(Script):
        async def complete(self, messages, tools):
            await asyncio.sleep(0)
            scoped._space = 'bravo'
            return await super().complete(messages, tools)

    model = Mutating(call(), ToolStep(content='done'))
    agent = agents(model, [tool(lambda args, ctx: seen.append(ctx.space))]).bind('worker')
    points, _ = storage()
    with pytest.raises((TurnJournalError, RuntimeError, ValueError)):
        await agent.run('Count', tools=scoped, checkpoints=points)
    assert seen == []


@pytest.mark.parametrize('change', ['forget', 'outage'])
async def test_restored_evidence_failure_never_researches_or_dispatches(memory, change):
    source = await memory.remember('alpha', 'Juniper uses Polaris.', metadata={'team': 'blue'})
    effects = []
    model = Script(search(), call(), ToolStep(content='done'))
    agent = agents(model, [tool(lambda args, ctx: effects.append(1))]).bind('worker')
    scoped = binding(memory)
    points, _ = storage()
    with pytest.raises(TurnJournalPaused):
        await agent.run('Juniper', tools=scoped, checkpoints=points, max_new_operations=3)
    assert len(model.requests) == 2 and not effects

    async def forbidden(*args, **kwargs):
        pytest.fail('repeated search')

    scoped.run = forbidden
    original = memory.documents.get_episode

    async def outage(*args, **kwargs):
        raise OSError('private lookup detail')

    if change == 'forget':
        await memory.forget('alpha', source.episode_id)
    else:
        memory.documents.get_episode = outage
    try:
        with pytest.raises((RuntimeError, ValueError, OSError)):
            await agent.run('Juniper', tools=scoped, checkpoints=points, max_new_operations=3)
        assert len(model.requests) == 2 and not effects
    finally:
        memory.documents.get_episode = original


async def test_source_revoked_after_model_proposal_before_effect(memory):
    source = await memory.remember('alpha', 'Juniper uses Polaris.', metadata={'team': 'blue'})
    effects = []

    async def revoke():
        await memory.forget('alpha', source.episode_id)
        return call()

    model = Script(search(), revoke, ToolStep(content='done'))
    agent = agents(model, [tool(lambda args, ctx: effects.append(1))]).bind('worker')
    points, _ = storage()
    with pytest.raises((RuntimeError, ValueError)):
        await agent.run('Juniper', tools=binding(memory), checkpoints=points)
    assert len(model.requests) == 2 and not effects


async def test_direct_result_replay_keeps_usage_and_has_no_new_dispatch(memory):
    effects = []
    proposal = call().model_copy(
        update={'usage': ModelTokenUsage(prompt_tokens=7, completion_tokens=2, total_tokens=9)}
    )
    model = Script(proposal)
    registration = tool(lambda args, ctx: (effects.append(1) or 'exact direct output'), return_direct=True)
    agent = agents(model, [registration]).bind('worker')
    points, _ = storage()
    with pytest.raises(TurnJournalPaused):
        await agent.run('Count', tools=binding(memory), checkpoints=points, max_new_operations=1)
    result = await agent.run('Count', tools=binding(memory), checkpoints=points, max_new_operations=1)
    assert result.output.text == 'exact direct output'
    assert result.output.model_calls == 1 and result.output.tool_calls == 1
    assert result.output.usage.total_tokens == 9
    assert len(model.requests) == 1 and effects == [1]
    replay = await agent.run('Count', tools=binding(memory), checkpoints=points, max_new_operations=1)
    assert replay.output.text == result.output.text
    assert replay.output.usage == result.output.usage
    assert len(model.requests) == 1 and effects == [1]


async def test_restore_storage_outage_does_not_disclose_private_details(memory):
    await memory.remember('alpha', 'Juniper uses Polaris.', metadata={'team': 'blue'})
    model = Script(search(), call(), ToolStep(content='done'))
    agent = agents(model, [tool(lambda args, ctx: None)]).bind('worker')
    scoped = binding(memory)
    points, _ = storage()
    with pytest.raises(TurnJournalPaused):
        await agent.run('Juniper', tools=scoped, checkpoints=points, max_new_operations=3)
    original = memory.documents.get_episode
    secret = 'private storage path / credentials / query detail'

    async def outage(*args, **kwargs):
        raise OSError(secret)

    memory.documents.get_episode = outage
    try:
        with pytest.raises(Exception) as caught:
            await agent.run('Juniper', tools=scoped, checkpoints=points, max_new_operations=3)
        assert secret not in str(caught.value)
        assert len(model.requests) == 2
    finally:
        memory.documents.get_episode = original


async def test_replayed_read_reuse_and_search_compaction_keep_accounting(memory):
    from scone_memory.agents.evidence_loop import EvidenceToolLoop, ToolCall
    from scone_memory.agents.turn_journal import ToolTurnJournal

    await memory.remember(
        'alpha', 'Juniper uses Polaris. ' + ('Relevant Juniper evidence. ' * 40), metadata={'team': 'blue'}
    )
    scoped = binding(memory)
    actual = []
    original = scoped.run

    async def tracked(name, args):
        actual.append(name)
        return await original(name, args)

    scoped.run = tracked
    chunk = []
    model = Script()

    async def first_read():
        packet = json.loads(model.requests[-1][0][-1]['content'])
        chunk.append(packet['items'][0]['chunk_id'])
        return ToolStep(
            calls=(
                ToolCall(
                    id='context-1',
                    name='read_memory',
                    arguments={'chunk_id': chunk[0], 'before': 0, 'after': 0},
                ),
            )
        )

    async def same_read():
        return ToolStep(
            calls=(
                ToolCall(
                    id='context-2',
                    name='read_memory',
                    arguments={'chunk_id': chunk[0], 'before': 0, 'after': 0},
                ),
            )
        )

    model.steps = [
        search(),
        first_read,
        same_read,
        search(call_id='search-again'),
        ToolStep(content='Verified original evidence.'),
    ]
    points, _ = storage()
    result = None
    for _ in range(12):
        journal = ToolTurnJournal(points, binding={'model': 'local-v1'}, max_new_operations=1)
        try:
            result = await EvidenceToolLoop(model, scoped, journal=journal, compact_search_results=True).run(
                [{'role': 'user', 'content': 'Juniper?'}]
            )
            break
        except TurnJournalPaused:
            pass
    assert result is not None
    assert result.model_calls == 5 and result.tool_calls == 4
    assert len(model.requests) == 5 and actual == ['search_memory', 'read_memory', 'search_memory']
    assert result.tool_outcomes[2].reused and result.tool_outcomes[3].reused
    assert len(result.evidence_packets) == 3


async def test_cancelled_actual_custom_dispatch_never_replays(memory):
    entered = asyncio.Event()
    effects = []

    async def handler(args, ctx):
        effects.append(1)
        entered.set()
        await asyncio.Event().wait()

    model = Script(call(), ToolStep(content='must not execute'))
    agent = agents(model, [tool(handler)]).bind('worker')
    points, _ = storage()
    task = asyncio.create_task(agent.run('Count', tools=binding(memory), checkpoints=points))
    await entered.wait()
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    with pytest.raises(TurnJournalError, match='outcome_unknown'):
        await agent.run('Count', tools=binding(memory), checkpoints=points)
    assert effects == [1] and len(model.requests) == 1


async def test_factory_scope_drift_cannot_rebind_existing_journal_identity(memory):
    from scone_memory.agents.catalog import AgentCatalog, AgentDefinition, AgentModel

    scoped = binding(memory)
    effects = []
    model = Script(call(), ToolStep(content='done'))

    def factory():
        scoped._space = 'bravo'
        return model

    registration = tool(lambda args, ctx: effects.append(ctx.space))
    catalog = AgentCatalog(
        models=[AgentModel('local', 'Local', '1', factory)],
        tools=[registration],
        agents=[
            AgentDefinition(
                agent_id='worker',
                instructions='Use tool',
                models=('local',),
                default_model='local',
                initial_search=False,
                tools=('double_count',),
            )
        ],
    )
    points, _ = storage()
    with pytest.raises((TurnJournalError, RuntimeError, ValueError)):
        await catalog.bind('worker').run('Count', tools=scoped, checkpoints=points)
    assert effects == []
