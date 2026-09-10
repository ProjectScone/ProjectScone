import json

import pytest

from scone_memory.agents.evidence_loop import EvidenceToolLoop, ToolCall, ToolLoopLimits, ToolStep
from ..agents.test_compute_memory_tool import box, request, source
from ..agents.test_evidence_tool_loop import Script, search
from ..agents.test_evidence_answer import passage
from scone_memory.retrieval.computation import ComputeMemoryArgs, evaluate_computation
from ..agents.test_structured_tool_chat import model, retained_messages


async def test_search_compute_answer_keeps_evidence_and_call_budgets(engine):
    _, chunk = await source(engine)
    calculate = ToolStep(calls=(ToolCall(id='calculate',name='compute_memory',arguments=request(chunk)),))
    scripted = Script(search(query='Value'),calculate,ToolStep(content='12'))
    result = await EvidenceToolLoop(scripted,box(engine),limits=ToolLoopLimits(max_tool_calls=2)).run([
        {'role':'user','content':'What is the value?'}])
    assert [row['function']['name'] for row in scripted.requests[0][1]] == ['search_memory']
    assert 'compute_memory' in [row['function']['name'] for row in scripted.requests[1][1]]
    assert scripted.requests[2][1] == []
    assert result.text == '12' and result.tool_calls == 2
    assert result.tool_outcomes[-1].name == 'compute_memory'
    assert json.loads(result.evidence_packets[-1])['computation']['value'] == '12'
    assert await result.validate() and result.verified_accuracy is False


async def test_compute_cannot_probe_unknown_chunk(engine,monkeypatch):
    _, chunk = await source(engine)
    tools = box(engine)
    async def no_read(*args): pytest.fail('undiscovered chunk reached store')
    monkeypatch.setattr(tools,'prepare',no_read)
    scripted = Script(ToolStep(calls=(ToolCall(id='probe',name='compute_memory',arguments=request(chunk)),)),
        ToolStep(content='Need to search first.'))
    result = await EvidenceToolLoop(scripted,tools).run([{'role':'user','content':'Compute?'}])
    assert result.tool_outcomes[0].error == 'search_for_chunk_first'


async def test_compute_cannot_escape_final_source_revalidation(engine):
    added, chunk = await source(engine)
    async def deleted():
        await engine.forget('alpha',added.episode_id)
        return ToolStep(content='12')
    scripted = Script(search(query='Value'),
        ToolStep(calls=(ToolCall(id='compute',name='compute_memory',arguments=request(chunk)),)),deleted)
    with pytest.raises(RuntimeError,match='evidence changed'):
        await EvidenceToolLoop(scripted,box(engine)).run([{'role':'user','content':'Compute value'}])


@pytest.mark.parametrize('enabled,known',[(True,True),(True,False),(False,True)])
async def test_structured_compute_schema_and_dispatch_require_offered_chunks(enabled,known):
    tools = box(None).openai()
    if not enabled: tools = tools[:-1]
    messages = retained_messages({'status':'prepared','items':[{'chunk_id':7}]}) if known else [{'role':'user','content':'Compare values'}]
    action = {'action':'compute_memory','operation':'sum','left':[{'chunk_id':7,'quote':'12'}],'right':[]}
    requests=[]
    provider=model(action,requests)
    if not enabled or not known:
        with pytest.raises(RuntimeError,match='tool model unavailable'):
            await provider.complete(messages,tools)
    else:
        step=await provider.complete(messages,tools)
        assert step.calls[0].name == 'compute_memory'
        assert step.calls[0].arguments == {key:value for key,value in action.items() if key != 'action'}
    branches=requests[0]['response_format']['json_schema']['schema']['anyOf']
    compute=[row for row in branches if row['properties']['action']['const']=='compute_memory']
    assert bool(compute) == (enabled and known)
    if compute:
        assert compute[0]['properties']['left']['items']['properties']['chunk_id']['enum'] == [7]


async def test_structured_compute_history_reaches_final_synthesis():
    source_row=passage(7,'Value 12')
    calculation=evaluate_computation(ComputeMemoryArgs(operation='sum',left=[{'chunk_id':7,'quote':'12'}]),{7:'Value 12'})
    messages=retained_messages({'status':'prepared','items':[source_row]})
    messages.extend([
        {'role':'assistant','content':None,'tool_calls':[{'id':'calc','type':'function','function':{
            'name':'compute_memory','arguments':json.dumps({'operation':'sum','left':[{'chunk_id':7,'quote':'12'}]})}}]},
        {'role':'tool','tool_call_id':'calc','content':json.dumps({'status':'prepared','items':[source_row],
            'computation':calculation})},
    ])
    requests=[]
    step=await model({'action':'answer','answer':'12'},requests).complete(messages,[])
    assert step.content=='12' and not step.calls
    assert 'computation' in json.dumps(requests[0]['messages'])


async def test_structured_compute_rejects_unoffered_or_malformed_input():
    messages=retained_messages({'status':'prepared','items':[{'chunk_id':7}]})
    for chunk in (99,True,'7'):
        with pytest.raises(RuntimeError,match='tool model unavailable'):
            await model({'action':'compute_memory','operation':'sum','left':[{'chunk_id':chunk,'quote':'12'}]}).complete(messages,box(None).openai())
