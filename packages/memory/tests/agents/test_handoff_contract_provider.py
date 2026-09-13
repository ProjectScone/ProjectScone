"""The local structured adapter receives distinct final and delegation schemas."""
import json
import httpx
import pytest
from jsonschema import Draft202012Validator
from scone_memory.agents.catalog import AgentCatalog, AgentDefinition, AgentModel
from scone_memory.agents.workflow import WorkflowError
from scone_memory.providers.structured_tool_chat import SelfHostedStructuredToolChat
from .test_handoff_requirements import policy
from .test_handoff_workflow import memory, workflow


@pytest.mark.parametrize('valid', [True, False])
async def test_structured_adapter_embeds_final_contract_and_keeps_original_tokens(tmp_path, memory, valid):
    requests = []
    value = '0.12345678901234567890123456789' if valid else '"bad"'
    async def serve(request):
        body = json.loads(request.content)
        requests.append(body)
        envelope = ('{"answer":"Research notes","handoff_to":"write"}' if len(requests) == 1
                    else '{"answer":{"value":'+value+'},"handoff_to":null}')
        return httpx.Response(200, json={'choices': [{'finish_reason': 'stop', 'message': {
            'role': 'assistant', 'content': '{"action":"answer","answer":'+envelope+'}'}}]})
    def factory():
        return SelfHostedStructuredToolChat('http://127.0.0.1:11434/v1', 'fixture',
                                            transport=httpx.MockTransport(serve))
    agents = AgentCatalog(models=[AgentModel(name, name, '1', factory) for name in ('careful', 'fast')],
        agents=[AgentDefinition(agent_id=name, instructions='Work', models=('careful', 'fast'),
                                default_model='careful', initial_search=False) for name in ('research', 'write')])
    work = workflow(tmp_path/'run', memory, agents, policy({'format': 'json_object',
        'output_schema': {'required': ['value'], 'properties': {'value': {'type': 'number'}},
                          'additionalProperties': False}}))
    try:
        if valid:
            assert (await work.run('r', 'Question')).final.text == '{"value":'+value+'}'
        else:
            with pytest.raises(WorkflowError, match='step_failed'):
                await work.run('r', 'Question')
            assert work.progress('r', 'Question').completed_steps == ('hop-01',)
        assert len(requests) == 2
        for index, body in enumerate(requests):
            validator = Draft202012Validator(body['response_format']['json_schema']['schema'])
            target = 'write' if index == 0 else 'research'
            assert validator.is_valid({'action': 'answer', 'answer': {'answer': 'notes', 'handoff_to': target}})
            assert validator.is_valid({'action': 'answer', 'answer': {'answer': {'value': 1}, 'handoff_to': None}})
            assert not validator.is_valid({'action': 'answer', 'answer': {'answer': 'notes', 'handoff_to': None}})
            assert not validator.is_valid({'action': 'answer', 'answer': {'answer': {'value': 1}, 'handoff_to': target}})
    finally:
        work.close()
