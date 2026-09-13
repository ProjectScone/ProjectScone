"""Actual model/tool turns resume from exact encrypted operation receipts."""
import pytest

from scone_memory.agents.turn_journal import TurnJournalPaused
from scone_memory.agents.workflow import WorkflowPausableStep, WorkflowRunner
from scone_memory.agents.evidence_loop import ToolStep
from .test_custom_tools import agents, call, tool
from .test_evidence_tool_loop import Script, binding, search
from .test_task_workflow import memory


@pytest.mark.parametrize('parallel', [False, True])
async def test_pause_after_proposal_resumes_without_another_model_call(memory, tmp_path, parallel):
    effects = []
    model = Script(call(), ToolStep(content='Completed.'))
    agent = agents(model, [tool(lambda args, ctx: effects.append(args['count']))]).bind('worker')
    async def valid(context):
        return True
    async def execute(context):
        try:
            result = await agent.run('Count', tools=binding(memory), checkpoints=context.checkpoints,
                                     max_new_operations=1)
        except TurnJournalPaused as pause:
            return pause.pause
        return result.output.text
    step = WorkflowPausableStep('agent', '1', execute)
    args = {'path': tmp_path/'journal', 'key': b'k'*32, 'steps': [step], 'source_verifier': valid,
            'dependencies': {'agent': []} if parallel else None}
    first = WorkflowRunner(**args)
    assert (await first.run('one', space='alpha', scope={}, inputs=None)).status == 'paused'
    assert len(model.requests) == 1 and effects == []
    first.close()
    reopened = WorkflowRunner(**args)
    try:
        assert (await reopened.run('one', space='alpha', scope={}, inputs=None)).status == 'paused'
        assert len(model.requests) == 1 and effects == [3]
        final = await reopened.run('one', space='alpha', scope={}, inputs=None)
        assert final.results == {'agent': 'Completed.'}
        assert len(model.requests) == 2 and effects == [3]
    finally:
        reopened.close()


async def test_restored_search_does_not_repeat_retrieval_before_new_effect(memory, tmp_path):
    await memory.remember('alpha', 'Juniper uses Polaris.', metadata={'team': 'blue'})
    effects = []
    model = Script(search(), call(), ToolStep(content='Completed.'))
    agent = agents(model, [tool(lambda args, ctx: effects.append(1))]).bind('worker')
    scoped = binding(memory)
    async def valid(context):
        return True
    async def execute(context):
        try:
            result = await agent.run('Juniper', tools=scoped, checkpoints=context.checkpoints, max_new_operations=3)
        except TurnJournalPaused as pause:
            return pause.pause
        return result.output.text
    job = WorkflowRunner(tmp_path/'journal', key=b'k'*32,
        steps=[WorkflowPausableStep('agent', '1', execute)], source_verifier=valid)
    try:
        assert (await job.run('one', space='alpha', scope={}, inputs=None)).status == 'paused'
        assert len(model.requests) == 2 and effects == []
        async def forbidden(*args, **kwargs):
            pytest.fail('replayed search')
        scoped.run = forbidden
        assert (await job.run('one', space='alpha', scope={}, inputs=None)).status == 'completed'
        assert len(model.requests) == 3 and effects == [1]
    finally:
        job.close()


@pytest.mark.parametrize('change', ['delete', 'scope', 'model', 'tool_revision', 'question'])
async def test_changed_binding_or_evidence_refuses_before_new_effect(memory, tmp_path, change):
    from scone_memory.agents.catalog import AgentCatalog, AgentDefinition, AgentModel
    from scone_memory.agents.workflow import WorkflowError
    from scone_memory.integrations.scoped_tools import ScopedMemoryTools
    from scone_memory.retrieval.recall_scope import RecallScope
    episode = await memory.remember('alpha', 'Juniper uses Polaris.', metadata={'team': 'blue'})
    effects, factories = [], []
    model = Script(search(), call(), ToolStep(content='Completed.'))
    def create():
        factories.append(1)
        return model
    registration = tool(lambda args, ctx: effects.append(1))
    def select(revision='1', registered=registration):
        return AgentCatalog(models=[AgentModel('chosen', 'Chosen model', revision, create)],
            agents=[AgentDefinition(agent_id='worker', instructions='Use retained evidence.', models=('chosen',),
                default_model='chosen', initial_search=False, tools=('double_count',))], tools=[registered]).bind('worker')
    agent = select()
    scoped = binding(memory)
    question = 'Juniper'
    async def valid(context):
        return True
    async def execute(context):
        try:
            result = await agent.run(question, tools=scoped, checkpoints=context.checkpoints, max_new_operations=3)
        except TurnJournalPaused as pause:
            return pause.pause
        return result.output.text
    job = WorkflowRunner(tmp_path/'journal', key=b'k'*32,
        steps=[WorkflowPausableStep('agent', '1', execute)], source_verifier=valid)
    try:
        assert (await job.run('one', space='alpha', scope={}, inputs=None)).status == 'paused'
        if change == 'delete':
            await memory.documents.delete_episode('alpha', episode.episode_id)
        elif change == 'scope':
            scoped = ScopedMemoryTools(memory, 'alpha', scope=RecallScope.validated(where={'team': 'other'}))
        elif change == 'model':
            agent = select('2')
        elif change == 'tool_revision':
            from dataclasses import replace
            agent = select(registered=replace(registration, revision='2'))
        else:
            question = 'A different question'
        with pytest.raises(WorkflowError):
            await job.run('one', space='alpha', scope={}, inputs=None)
        assert not effects and len(model.requests) == 2
        if change != 'delete':
            assert factories == [1]
    finally:
        job.close()


async def test_ambiguous_application_effect_cannot_run_again(memory, tmp_path):
    from scone_memory.agents.workflow import WorkflowError
    effects = []
    async def effect(arguments, context):
        effects.append(1)
        raise RuntimeError('private handler error')
    model = Script(call())
    agent = agents(model, [tool(effect)]).bind('worker')
    async def valid(context):
        return True
    async def execute(context):
        try:
            result = await agent.run('Count', tools=binding(memory), checkpoints=context.checkpoints, max_new_operations=1)
        except TurnJournalPaused as pause:
            return pause.pause
        return result.output.text
    job = WorkflowRunner(tmp_path/'journal', key=b'k'*32,
        steps=[WorkflowPausableStep('agent', '1', execute)], source_verifier=valid)
    try:
        assert (await job.run('one', space='alpha', scope={}, inputs=None)).status == 'paused'
        with pytest.raises(WorkflowError, match='step_failed'):
            await job.run('one', space='alpha', scope={}, inputs=None)
        with pytest.raises(WorkflowError, match='outcome_unknown'):
            await job.run('one', space='alpha', scope={}, inputs=None)
        assert effects == [1] and len(model.requests) == 1
    finally:
        job.close()


@pytest.mark.parametrize('direct', [False, True])
async def test_selected_model_usage_and_direct_output_survive_scheduling_yields(memory, tmp_path, direct):
    from scone_memory.agents.catalog import AgentCatalog, AgentDefinition, AgentModel
    from scone_memory.agents.usage import ModelTokenUsage
    from scone_memory.realtime.answer_requirements import AnswerRequirements
    usage = ModelTokenUsage(prompt_tokens=9, completion_tokens=3, total_tokens=12)
    model = Script(ToolStep(calls=call().calls, usage=usage), ToolStep(content='{"count":6}', usage=usage))
    effects, results = [], []
    def effect(arguments, context):
        effects.append(1)
        return {'count': 6}
    registration = tool(effect, return_direct=direct)
    catalog = AgentCatalog(models=[AgentModel('default', 'Default', '1', lambda: pytest.fail('wrong model')),
        AgentModel('chosen', 'Chosen', '2', lambda: model)], tools=[registration],
        agents=[AgentDefinition(agent_id='worker', instructions='Count.', models=('default', 'chosen'),
            default_model='default', initial_search=False, tools=('double_count',))])
    agent = catalog.bind('worker', model_id='chosen')
    async def valid(context):
        return True
    async def execute(context):
        try:
            result = await agent.run('Count', tools=binding(memory), checkpoints=context.checkpoints,
                max_new_operations=1, answer_requirements=AnswerRequirements(format='json_object'))
        except TurnJournalPaused as pause:
            return pause.pause
        results.append(result)
        return result.output.text
    job = WorkflowRunner(tmp_path/'journal', key=b'k'*32,
        steps=[WorkflowPausableStep('agent', '1', execute)], source_verifier=valid)
    try:
        for _ in range(3):
            result = await job.run('one', space='alpha', scope={}, inputs=None)
            if result.status == 'completed':
                break
        assert result.status == 'completed' and result.results == {'agent': '{"count":6}'}
        count = 1 if direct else 2
        assert results[0].model_id == 'chosen'
        assert results[0].output.model_calls == len(model.requests) == count
        assert results[0].output.usage.total_tokens == count * 12 and effects == [1]
    finally:
        job.close()


async def test_cumulative_loop_budget_cannot_reset_with_a_larger_journal_budget(memory):
    from scone_memory.agents.evidence_loop import EvidenceToolLoop, ToolLoopLimits
    from scone_memory.agents.turn_journal import ToolTurnJournal, TurnJournalError
    from .test_turn_journal import checkpoints
    effects = []
    model = Script(call(), ToolStep(content='must not run'))
    registered = tool(lambda args, ctx: effects.append(1))
    storage, _ = checkpoints()
    first = ToolTurnJournal(storage, binding={'model': 'chosen'}, timeout_s=600, max_new_operations=1)
    first._base_elapsed = 70.0
    with pytest.raises(TurnJournalPaused):
        await EvidenceToolLoop(model, binding(memory), journal=first, custom_tools=[registered],
            limits=ToolLoopLimits(timeout_s=120)).run([{'role': 'user', 'content': 'Count'}])
    resumed = ToolTurnJournal(storage, binding={'model': 'chosen'}, timeout_s=600, max_new_operations=1)
    assert 70 <= resumed.elapsed_s < 80
    resumed._base_elapsed += 51.0
    assert resumed.remaining_s > 400
    with pytest.raises(TurnJournalError, match='deadline'):
        await EvidenceToolLoop(model, binding(memory), journal=resumed, custom_tools=[registered],
            limits=ToolLoopLimits(timeout_s=120)).run([{'role': 'user', 'content': 'Count'}])
    assert effects == [] and len(model.requests) == 1
