"""Standalone SDK and native HTTP agree on final-only authored contracts."""
import json

import pytest
from scone import HandoffAgent, HandoffPlan, TaskAnswerRequirements, SconeError
from native_server import native_server
from test_native_task_requirements import SCHEMA, settled


@pytest.mark.integration
@pytest.mark.parametrize('model',['valid','invalid'])
def test_handoff_contract_survives_native_restart_and_checks_final_only(tmp_path,model):
    requirements=TaskAnswerRequirements(instructions='Return the summary object.',format='json_object',
                                       max_bytes=128,max_lines=2,output_schema=SCHEMA)
    plan=HandoffPlan('contract','router',(
        HandoffAgent('router','router',('writer',)),HandoffAgent('writer',model)),1,
        answer_requirements=requirements)
    with native_server(tmp_path,'handoff_contract_server.py') as client:
        agents=client.agents(expected_space='alpha')
        saved=agents.save_plan(plan,expected_revision=0)
        assert saved.plan==plan and agents.plan('contract').plan.to_json()==plan.to_json()
        agents.start('run',plan=saved,question='Keep the final answer bounded.')
        status=settled(agents)
        if model=='valid':
            assert status.status=='completed'
            result=agents.result('run')
            assert result.hops[0].output.text=='Delegate to the selected writer.'
            assert result.hops[0].handoff_to=='writer'
            assert isinstance(result.final.text,str)
            assert json.loads(result.final.text)=={'summary':'kept'}
            assert result.final==result.hops[-1].output
            assert len(result.hops)==2
        else:
            assert status.status!='completed'
            with pytest.raises(SconeError): agents.result('run')
        calls=[json.loads(line) for line in (tmp_path/'calls.jsonl').read_text().splitlines()]
        assert calls==[{'model':'router','format':'json_object'},{'model':model,'format':'json_object'}]
    with native_server(tmp_path,'handoff_contract_server.py') as client:
        agents=client.agents(expected_space='alpha')
        assert agents.plan('contract').plan.to_json()==plan.to_json()
        assert agents.request('run').plan.plan==plan
        if model=='valid': assert json.loads(agents.result('run').final.text)=={'summary':'kept'}
        else:
            with pytest.raises(SconeError): agents.result('run')
        assert len((tmp_path/'calls.jsonl').read_text().splitlines())==2
