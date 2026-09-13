"""Real local host for final contracts across an explicit two-agent handoff."""
import asyncio
import json
from pathlib import Path
import sys

import uvicorn
from scone_memory import HashEmbedder, InMemoryDocumentStore, InMemoryVectorIndex, MemoryEngine
from scone_memory.agents.catalog import AgentCatalog, AgentDefinition, AgentModel
from scone_memory.agents.evidence_loop import ToolStep
from scone_memory.agents.plan_store import AgentPlanStore
from scone_memory.agents.run_service import AgentRunService
from scone_memory.api.app import create_app
from scone_memory.retrieval.recall_scope import RecallScope


async def run():
    state, port = Path(sys.argv[1]), int(sys.argv[2])
    memory = await MemoryEngine(InMemoryDocumentStore(), InMemoryVectorIndex(), HashEmbedder()).open()
    class Model:
        def __init__(self, name): self.name = name
        async def complete(self, messages, tools):
            raise AssertionError('handoff requirements were dropped')
        async def complete_with_requirements(self, messages, tools, requirements):
            with (state/'calls.jsonl').open('a') as output:
                output.write(json.dumps({'model':self.name, 'format':requirements.format})+'\n')
            if self.name == 'router':
                return ToolStep(content=json.dumps({'answer':'Delegate to the selected writer.', 'handoff_to':'writer'}))
            return ToolStep(content=json.dumps({'answer':{'summary':'kept' if self.name=='valid' else 7}, 'handoff_to':None}))
    models = ('router','valid','invalid')
    catalog = AgentCatalog(models=[AgentModel(name,name,'1',lambda name=name:Model(name)) for name in models],
        agents=[AgentDefinition(agent_id='router',instructions='Hand off to writer.',models=('router',),default_model='router',initial_search=False),
                AgentDefinition(agent_id='writer',instructions='Return the final answer.',models=('valid','invalid'),default_model='valid',initial_search=False)])
    plans = AgentPlanStore(state/'plans.sqlite',key=b'k'*32)
    service = AgentRunService(state/'runs',key=b'k'*32,catalog=catalog,plans=plans,memory=memory,
                              scope_for=lambda _:RecallScope.validated())
    app = create_app(memory,{'agent-fixture':'alpha'},agent_catalog=catalog,agent_plan_store=plans,agent_run_service=service)
    server = uvicorn.Server(uvicorn.Config(app,host='127.0.0.1',port=port,log_level='warning'))
    async def commands():
        await asyncio.to_thread(sys.stdin.readline)
        server.should_exit = True
    command = asyncio.create_task(commands())
    try: await server.serve()
    finally:
        command.cancel()
        await asyncio.gather(command,return_exceptions=True)
        await service.aclose()
        plans.close()
        await memory.close()

asyncio.run(run())
