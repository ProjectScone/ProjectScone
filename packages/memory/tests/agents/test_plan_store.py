"""Saved model selections are encrypted, isolated and revision checked."""
import pytest

from scone_memory.agents.catalog import AgentCatalog,AgentDefinition,AgentModel
from scone_memory.agents.evidence_loop import ToolStep
from scone_memory.agents.task_workflow import AgentTask,AgentTaskPlan
from scone_memory.agents.plan_store import AgentPlanStore,PlanConflict,PlanConfigurationChanged
from scone_memory.agents.workflow import WorkflowError

class Model:
    async def complete(self,messages,tools):return ToolStep(content='unused')

def catalog(revision='1'):
    return AgentCatalog(models=[AgentModel('local','Local model',revision,Model)],agents=[AgentDefinition(agent_id='a',instructions='Use evidence.',models=('local',),default_model='local')])

def plan(prompt='Secret task instructions'):
    return AgentTaskPlan(workflow_id='research',tasks=(AgentTask(task_id='find',agent_id='a',prompt=prompt),))


def test_choices_persist_as_explicit_models_and_files_hide_plan_content(tmp_path):
    path=tmp_path/'plans.db';store=AgentPlanStore(path,key=b'k'*32)
    saved=store.save('alpha',plan(),catalog=catalog(),expected_revision=0)
    assert saved.revision==1 and saved.plan.tasks[0].model_id=='local'
    store.close();reopened=AgentPlanStore(path,key=b'k'*32)
    assert reopened.get('alpha','research')==saved
    assert reopened.get('bravo','research') is None
    assert saved.checked_plan(catalog()).tasks[0].model_id=='local'
    reopened.close()
    assert path.stat().st_mode&0o777==0o600
    assert all(b'Secret task instructions' not in p.read_bytes() and b'alpha' not in p.read_bytes() for p in tmp_path.iterdir())


def test_stale_revision_cannot_overwrite_another_editor(tmp_path):
    path=tmp_path/'plans.db';a=AgentPlanStore(path,key=b'k'*32);b=AgentPlanStore(path,key=b'k'*32)
    first=a.save('alpha',plan(),catalog=catalog(),expected_revision=0)
    second=b.save('alpha',plan('Revised instructions'),catalog=catalog(),expected_revision=first.revision)
    with pytest.raises(PlanConflict):a.save('alpha',plan('Lost edit'),catalog=catalog(),expected_revision=first.revision)
    assert a.get('alpha','research')==second
    a.close();b.close()


def test_changed_model_configuration_requires_reviewed_resave(tmp_path):
    store=AgentPlanStore(tmp_path/'plans.db',key=b'k'*32)
    saved=store.save('alpha',plan(),catalog=catalog(),expected_revision=0)
    with pytest.raises(PlanConfigurationChanged):saved.checked_plan(catalog('2'))
    updated=store.save('alpha',plan(),catalog=catalog('2'),expected_revision=saved.revision)
    assert updated.checked_plan(catalog('2'))==updated.plan
    store.close()


def test_wrong_key_and_unrelated_database_are_refused(tmp_path):
    path=tmp_path/'plans.db';store=AgentPlanStore(path,key=b'k'*32);store.close()
    with pytest.raises(WorkflowError):AgentPlanStore(path,key=b'x'*32)
    import sqlite3
    other=tmp_path/'other.db';db=sqlite3.connect(other);db.execute('CREATE TABLE unrelated (name TEXT)');db.close();other.chmod(0o600)
    with pytest.raises(WorkflowError):AgentPlanStore(other,key=b'k'*32)


def test_paging_is_bounded_and_space_specific(tmp_path):
    store=AgentPlanStore(tmp_path/'plans.db',key=b'k'*32)
    for name in ('one','two','three'):
        store.save('alpha',plan().model_copy(update={'workflow_id':name}),catalog=catalog(),expected_revision=0)
    store.save('bravo',plan(),catalog=catalog(),expected_revision=0)
    first=store.list('alpha',limit=2);assert len(first.items)==2 and first.next_after is not None
    second=store.list('alpha',limit=2,after=first.next_after)
    assert len(second.items)==1 and second.next_after is None
    assert {item.plan.workflow_id for item in first.items+second.items}=={'one','two','three'}
    store.close()


def test_caps_allow_updates_but_reject_new_plans_and_invalid_paging(tmp_path):
    store=AgentPlanStore(tmp_path/'plans.db',key=b'k'*32,max_plans=1)
    first=store.save('alpha',plan(),catalog=catalog(),expected_revision=0)
    store.save('alpha',plan('Updated'),catalog=catalog(),expected_revision=first.revision)
    with pytest.raises(WorkflowError,match='plan_store_limit'):
        store.save('bravo',plan(),catalog=catalog(),expected_revision=0)
    for limit in (0,101,True):
        with pytest.raises(WorkflowError):store.list('alpha',limit=limit)
    store.close()
    with pytest.raises(WorkflowError,match='plan_store_closed'):store.get('alpha','research')


def test_cursor_cannot_cross_spaces(tmp_path):
    store=AgentPlanStore(tmp_path/'plans.db',key=b'k'*32)
    for name in ('one','two'):
        store.save('alpha',plan().model_copy(update={'workflow_id':name}),catalog=catalog(),expected_revision=0)
    cursor=store.list('alpha',limit=1).next_after
    assert cursor is not None
    with pytest.raises(WorkflowError,match='invalid_plan_cursor'):store.list('bravo',after=cursor)
    store.close()


def test_ciphertext_cannot_move_between_spaces_or_plans(tmp_path):
    import sqlite3
    path=tmp_path/'plans.db';store=AgentPlanStore(path,key=b'k'*32)
    store.save('alpha',plan(),catalog=catalog(),expected_revision=0)
    store.save('bravo',plan(),catalog=catalog(),expected_revision=0)
    store.close()
    db=sqlite3.connect(path)
    rows=db.execute('SELECT token,payload FROM agent_plans ORDER BY token').fetchall()
    db.execute('UPDATE agent_plans SET payload=? WHERE token=?',(rows[0][1],rows[1][0]));db.commit();db.close()
    store=AgentPlanStore(path,key=b'k'*32)
    failures=0
    for space in ('alpha','bravo'):
        try:store.get(space,'research')
        except WorkflowError as error:
            assert error.code=='plan_key_or_integrity';failures+=1
    assert failures==1
    store.close()


def test_model_factories_are_never_invoked_and_returned_values_do_not_mutate_storage(tmp_path):
    def forbidden():raise AssertionError('model construction while editing a plan')
    choices=AgentCatalog(models=[AgentModel('local','Local model','1',forbidden)],agents=[AgentDefinition(
        agent_id='a',instructions='Use evidence.',models=('local',),default_model='local')])
    store=AgentPlanStore(tmp_path/'plans.db',key=b'k'*32)
    saved=store.save('alpha',plan(),catalog=choices,expected_revision=0)
    assert saved.checked_plan(choices).tasks[0].model_id=='local'
    saved.bindings['find']='0'*64
    with pytest.raises(PlanConfigurationChanged):saved.checked_plan(choices)
    reloaded=store.get('alpha','research');assert reloaded is not None
    assert reloaded.checked_plan(choices).tasks[0].model_id=='local'
    store.close()


def test_private_file_policy_refuses_symlinks_and_public_files(tmp_path):
    target=tmp_path/'target';target.touch(mode=0o600)
    link=tmp_path/'link';link.symlink_to(target)
    with pytest.raises(WorkflowError):AgentPlanStore(link,key=b'k'*32)
    target.chmod(0o644)
    with pytest.raises(WorkflowError,match='private_file_required'):AgentPlanStore(target,key=b'k'*32)


def test_concurrent_editors_have_exactly_one_winner(tmp_path):
    from concurrent.futures import ThreadPoolExecutor
    from threading import Barrier
    path=tmp_path/'plans.db';store=AgentPlanStore(path,key=b'k'*32)
    store.save('alpha',plan(),catalog=catalog(),expected_revision=0);store.close()
    ready=Barrier(2)
    def edit(prompt):
        editor=AgentPlanStore(path,key=b'k'*32)
        try:
            ready.wait(timeout=5)
            try:return editor.save('alpha',plan(prompt),catalog=catalog(),expected_revision=1)
            except PlanConflict:return None
        finally:editor.close()
    with ThreadPoolExecutor(max_workers=2) as pool:
        results=list(pool.map(edit,('First edit','Second edit')))
    winners=[result for result in results if result is not None]
    assert len(winners)==1 and winners[0].revision==2
    reopened=AgentPlanStore(path,key=b'k'*32)
    assert reopened.get('alpha','research')==winners[0]
    reopened.close()


def test_replaced_path_is_refused_before_sqlite_can_initialize_target(tmp_path,monkeypatch):
    import sqlite3
    from scone_memory.agents import _encrypted_store as plan_store
    target=tmp_path/'unrelated.db';target.touch(mode=0o644)
    path=tmp_path/'plans.db';original=plan_store._private_file
    def swap(name):
        descriptor=original(name);name.unlink();name.symlink_to(target);return descriptor
    monkeypatch.setattr(plan_store,'_private_file',swap)
    with pytest.raises(WorkflowError):AgentPlanStore(path,key=b'k'*32)
    db=sqlite3.connect(target)
    assert db.execute('SELECT name FROM sqlite_master').fetchall()==[]
    db.close()


def test_shared_writable_parent_is_refused(tmp_path):
    shared=tmp_path/'shared';shared.mkdir(mode=0o777);shared.chmod(0o777)
    with pytest.raises(WorkflowError,match='private_directory_required'):
        AgentPlanStore(shared/'plans.db',key=b'k'*32)
    assert not (shared/'plans.db').exists()
