"""Real local agent HTTP fixture with persistent journals and scripted models."""
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
        def __init__(self, name):
            self.name = name
        async def complete(self, messages, tools):
            with (state / 'calls.jsonl').open('a') as output:
                output.write(json.dumps({'model': self.name, 'messages': messages}) + '\n')
            return ToolStep(content=self.name + ' completed the task.')
    catalog = AgentCatalog(models=[AgentModel(name, name.title() + ' local', '1', lambda name=name: Model(name))
                                   for name in ('fast', 'careful')], agents=[AgentDefinition(
        agent_id='worker', instructions='Use the provided direction.', models=('fast', 'careful'),
        default_model='fast', initial_search=False)])
    plans = AgentPlanStore(state / 'plans.sqlite', key=b'k' * 32)
    service = AgentRunService(state / 'runs', key=b'k' * 32, catalog=catalog, plans=plans, memory=memory,
                              scope_for=lambda space: RecallScope.validated())
    app = create_app(memory, {'agent-fixture': 'alpha'}, agent_catalog=catalog,
                     agent_plan_store=plans, agent_run_service=service)
    server = uvicorn.Server(uvicorn.Config(app, host='127.0.0.1', port=port, log_level='warning'))
    async def commands():
        await asyncio.to_thread(sys.stdin.readline)
        server.should_exit = True
    commands_task = asyncio.create_task(commands())
    try:
        await server.serve()
    finally:
        commands_task.cancel()
        await asyncio.gather(commands_task, return_exceptions=True)
        await service.aclose()
        plans.close()
        await memory.close()


asyncio.run(run())
