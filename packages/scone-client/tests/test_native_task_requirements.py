"""Python3.9 client -> native HTTP -> encrypted authored contracts and results."""
import json
import time

import pytest
from scone import ModelTask, TaskPlan, TaskAnswerRequirements, SconeError
from native_server import native_server

SCHEMA = {'$defs':{'summary':{'type':'string'}},'type':'object',
          'properties':{'summary':{'$ref':'#/$defs/summary'}},'required':['summary'],'additionalProperties':False}


def settled(agents):
    deadline=time.monotonic()+8
    while time.monotonic()<deadline:
        status=agents.status('run')
        if not status.active_local and status.status not in ('registered','created','running'):
            return status
        time.sleep(.01)
    pytest.fail('native contract run did not settle')


@pytest.mark.integration
@pytest.mark.parametrize('model', ['valid','invalid'])
def test_authored_contract_survives_native_restart_and_gates_publication(tmp_path,model):
    requirements=TaskAnswerRequirements(instructions='Return only the object.',format='json_object',
                                       max_bytes=128,max_lines=2,output_schema=SCHEMA)
    plan=TaskPlan('contract',(ModelTask('answer','worker',model,'Summarize.',answer_requirements=requirements),))
    with native_server(tmp_path,'task_contract_server.py') as client:
        agents=client.agents(expected_space='alpha')
        saved=agents.save_plan(plan,expected_revision=0)
        assert saved.plan==plan and agents.plan('contract').plan.to_json()==plan.to_json()
        agents.start('run',plan=saved,question='Keep the answer bounded.')
        status=settled(agents)
        if model=='valid':
            assert status.status=='completed'
            assert agents.result('run').results['answer'].text=='{"summary":"kept"}\n'
        else:
            assert status.status!='completed'
            with pytest.raises(SconeError): agents.result('run')
        calls=[json.loads(line) for line in (tmp_path/'calls.jsonl').read_text().splitlines()]
        assert calls==[{'model':model,'instructions':'Return only the object.','max_bytes':128,
                       'max_lines':2,'format':'json_object','has_schema':True}]
    with native_server(tmp_path,'task_contract_server.py') as client:
        agents=client.agents(expected_space='alpha')
        assert agents.plan('contract').plan.to_json()==plan.to_json()
        assert agents.request('run').plan.plan==plan
        if model=='valid': assert agents.result('run').results['answer'].text=='{"summary":"kept"}\n'
        else:
            with pytest.raises(SconeError): agents.result('run')
        assert len((tmp_path/'calls.jsonl').read_text().splitlines())==1
