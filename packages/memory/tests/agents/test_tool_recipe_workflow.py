"""Selected models author and use tools; humans gate both version and call."""
import json

from scone_memory.agents.approval_context import ApprovalContext
from scone_memory.agents.approval_store import AgentApprovalStore
from scone_memory.agents.catalog import AgentCatalog, AgentDefinition, AgentModel
from scone_memory.agents.evidence_loop import ToolCall, ToolStep
from scone_memory.agents.plan_store import AgentPlanStore
from scone_memory.agents.run_store import AgentRunStore
from scone_memory.agents.task_workflow import AgentTask, AgentTaskPlan
from scone_memory.agents.tool_recipe_store import ToolRecipeStore
from scone_memory.agents.turn_journal import TurnJournalPaused
from scone_memory.agents.workflow import WorkflowPausableStep, WorkflowRunner
from scone_memory.retrieval.recall_scope import RecallScope
from tests.agents.test_evidence_tool_loop import Script, binding
from tests.agents.test_task_workflow import memory
from tests.agents.test_tool_recipes import capability, recipe


async def test_selected_agent_authors_then_humans_review_version_and_exact_call(memory, tmp_path):
    effects = []
    primitive = capability(effects)
    recipes = ToolRecipeStore(tmp_path / 'recipes', key=b'k' * 32)
    author = Script(
        ToolStep(calls=(ToolCall(id='discover', name='recipe_capabilities', arguments={}),)),
        ToolStep(calls=(ToolCall(id='propose', name='propose_tool_recipe', arguments={
            'proposal_id': 'quad-v1', 'recipe_json': recipe(primitive).model_dump_json()}),)),
        ToolStep(content='The proposed tool is awaiting review.'),
    )
    offered = [recipes.capabilities_tool(space='alpha', tools=[primitive]),
               recipes.proposal_tool(space='alpha', proposed_by='author', tools=[primitive])]
    catalog = AgentCatalog(models=[AgentModel('chosen', 'Chosen local model', '1', lambda: author)], tools=offered,
        agents=[AgentDefinition(agent_id='author', instructions='Propose a tool for the user requirements.',
            models=('chosen',), default_model='chosen', initial_search=False, tools=tuple(t.name for t in offered))])
    plans = AgentPlanStore(tmp_path / 'plans', key=b'k' * 32)
    runs = AgentRunStore(tmp_path / 'runs', key=b'k' * 32)
    approvals = AgentApprovalStore(runs)
    try:
        authored = await catalog.bind('author', model_id='chosen').run('Create a quadrupling tool.', tools=binding(memory))
        assert authored.model_id == 'chosen' and not effects
        discovery = json.loads(author.requests[1][0][-1]['content'])['result']
        assert discovery['tools'][0]['tool']['name'] == 'double'
        assert recipes.get('alpha', 'quad-v1').status == 'pending'
        recipes.decide('alpha', 'quad-v1', decision='approve', actor='reviewer',
                       reason='Verified arithmetic composition against the requirement.', expected_revision=1)
        approved = recipes.bind('alpha', 'quad-v1', tools=[primitive])
        executor = Script(ToolStep(calls=(ToolCall(id='execute', name='quadruple', arguments={'count': 3}),)),
                          ToolStep(content='The computed result is 12.'))
        executor_catalog = AgentCatalog(models=[AgentModel('chosen', 'Chosen local model', '1', lambda: executor)],
            tools=[approved], agents=[AgentDefinition(agent_id='worker', instructions='Use the reviewed tool.',
                models=('chosen',), default_model='chosen', initial_search=False, tools=('quadruple',))])
        agent = executor_catalog.bind('worker', model_id='chosen')
        saved = plans.save('alpha', AgentTaskPlan(workflow_id='job', tasks=(AgentTask(
            task_id='calculate', agent_id='worker', model_id='chosen', prompt='Calculate'),)),
            catalog=executor_catalog, expected_revision=0)
        runs.register('alpha', 'one', plan=saved, question='Calculate',
            scope=RecallScope.validated(where={'team': 'blue'}), exclude_session_id='current')
        activation = None
        outputs = []
        async def verify(ctx):
            return True
        async def execute(ctx):
            try:
                result = await agent.run('Calculate', tools=binding(memory), checkpoints=ctx.checkpoints,
                    approval=ApprovalContext(approvals, ctx, step_id='calculate', selection_id='calculate', activation_id=activation))
            except TurnJournalPaused as pause:
                return pause.pause
            outputs.append(result.output)
            return result.output.text
        async def run_once():
            runner = WorkflowRunner(tmp_path / 'workflow', key=b'k' * 32,
                steps=[WorkflowPausableStep('calculate', '1', execute)], source_verifier=verify)
            try:
                return await runner.run('one', space='alpha', scope={}, inputs='Calculate')
            finally:
                runner.close()
        assert (await run_once()).status == 'paused'
        assert effects == []
        (ticket,) = approvals.list('alpha', 'one')
        assert ticket.call.arguments_json == '{"count":3}'
        assert ticket.call.tool_revision == approved.revision
        approvals.decide('alpha', 'one', ticket.request_id, decision='approve', actor='reviewer', expected_revision=1)
        assert (await run_once()).status == 'paused' and not effects
        approvals.activate('alpha', 'one', 'continue', decisions={ticket.request_id: 2})
        activation = 'continue'
        assert (await run_once()).status == 'completed'
        assert effects == [(3, 'alpha'), (6, 'alpha')]
        assert outputs[0].evidence_ids == () and outputs[0].source_status == 'none'
        assert (await run_once()).status == 'completed'
        assert len(executor.requests) == 2 and len(effects) == 2
    finally:
        runs.close()
        plans.close()
        recipes.close()
