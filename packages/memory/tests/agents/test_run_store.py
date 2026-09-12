"""A durable run keeps its original invocation despite later plan edits."""
import pytest
from scone_memory.agents.catalog import AgentCatalog,AgentDefinition,AgentModel
from scone_memory.agents.plan_store import AgentPlanStore
from scone_memory.agents.run_store import AgentRunStore,RunConflict
from scone_memory.agents.task_workflow import AgentTask,AgentTaskPlan
from scone_memory.retrieval.recall_scope import RecallScope


def catalog():
    def forbidden():raise AssertionError('registration cannot invoke a model')
    return AgentCatalog(models=[AgentModel('local','Local','1',forbidden)],agents=[AgentDefinition(
        agent_id='research',instructions='Use authorized evidence.',models=('local',),default_model='local')])


def plan():return AgentTaskPlan(workflow_id='report',tasks=(AgentTask(task_id='find',agent_id='research',prompt='Find evidence.'),))


def test_run_snapshot_survives_plan_edit_and_process_restart(tmp_path):
    plans=AgentPlanStore(tmp_path/'plans.db',key=b'k'*32)
    initial=plans.save('alpha',plan(),catalog=catalog(),expected_revision=0)
    runs=AgentRunStore(tmp_path/'runs.db',key=b'k'*32)
    saved=runs.register('alpha','request-1',plan=initial,question='Secret question',scope=RecallScope.validated(where={'team':'approved'}))
    assert runs.register('alpha','request-1',plan=initial,question='Secret question',scope=RecallScope.validated(where={'team':'approved'}))==saved
    plans.save('alpha',plan(),catalog=catalog(),expected_revision=1)
    runs.close();reopened=AgentRunStore(tmp_path/'runs.db',key=b'k'*32)
    assert reopened.get('alpha','request-1')==saved
    assert saved.plan.revision==1
    assert reopened.get('bravo','request-1') is None
    assert b'Secret question' not in (tmp_path/'runs.db').read_bytes()
    reopened.close();plans.close()


def test_reused_request_id_cannot_change_input_or_scope(tmp_path):
    plans=AgentPlanStore(tmp_path/'plans.db',key=b'k'*32)
    saved=plans.save('alpha',plan(),catalog=catalog(),expected_revision=0)
    runs=AgentRunStore(tmp_path/'runs.db',key=b'k'*32)
    runs.register('alpha','one',plan=saved,question='Question',scope=RecallScope.validated())
    with pytest.raises(RunConflict):runs.register('alpha','one',plan=saved,question='Changed',scope=RecallScope.validated())
    with pytest.raises(RunConflict):runs.register('alpha','one',plan=saved,question='Question',scope=RecallScope.validated(where={'team':'other'}))
    with pytest.raises(ValueError):runs.register('bravo','one',plan=saved,question='Question',scope=RecallScope.validated())
    assert len(runs.list('alpha',limit=1).items)==1
    assert runs.list('bravo',limit=1).items==()
    runs.close();plans.close()


def test_registration_is_atomic_between_competing_callers(tmp_path):
    from concurrent.futures import ThreadPoolExecutor
    from threading import Barrier
    plans=AgentPlanStore(tmp_path/'plans.db',key=b'k'*32)
    saved=plans.save('alpha',plan(),catalog=catalog(),expected_revision=0);plans.close()
    path=tmp_path/'runs.db';runs=AgentRunStore(path,key=b'k'*32);runs.close()
    ready=Barrier(2)
    def register(question):
        store=AgentRunStore(path,key=b'k'*32)
        try:
            ready.wait(timeout=5)
            try:return store.register('alpha','one',plan=saved,question=question,scope=RecallScope.validated())
            except RunConflict:return None
        finally:store.close()
    with ThreadPoolExecutor(max_workers=2) as pool:results=list(pool.map(register,('First question','Second question')))
    winners=[item for item in results if item is not None];assert len(winners)==1
    reopened=AgentRunStore(path,key=b'k'*32);assert reopened.get('alpha','one')==winners[0];reopened.close()


def test_run_store_limit_paging_and_cipher_domains(tmp_path):
    from scone_memory.agents.workflow import WorkflowError
    plans=AgentPlanStore(tmp_path/'plans.db',key=b'k'*32)
    saved=plans.save('alpha',plan(),catalog=catalog(),expected_revision=0)
    runs=AgentRunStore(tmp_path/'runs.db',key=b'k'*32,max_runs=2)
    first=runs.register('alpha','one',plan=saved,question='One',scope=RecallScope.validated())
    runs.register('alpha','two',plan=saved,question='Two',scope=RecallScope.validated())
    assert runs.register('alpha','one',plan=saved,question='One',scope=RecallScope.validated())==first
    with pytest.raises(WorkflowError,match='run_store_limit'):
        runs.register('alpha','three',plan=saved,question='Three',scope=RecallScope.validated())
    page=runs.list('alpha',limit=1);assert len(page.items)==1 and page.next_after is not None
    assert len(runs.list('alpha',limit=1,after=page.next_after).items)==1
    with pytest.raises(WorkflowError,match='invalid_run_cursor'):runs.list('bravo',after=page.next_after)
    runs.close();plans.close()
    with pytest.raises(WorkflowError):AgentRunStore(tmp_path/'plans.db',key=b'k'*32)
    with pytest.raises(WorkflowError):AgentPlanStore(tmp_path/'runs.db',key=b'k'*32)
    with pytest.raises(WorkflowError):AgentRunStore(tmp_path/'runs.db',key=b'x'*32)
