"""Actual scoped source verification and bounded handoff receipts, with local scripts."""
import asyncio
import json
from pathlib import Path
import sys

import uvicorn
from scone_memory import HashEmbedder, MemoryEngine
from scone_memory.backends.sqlite import SqliteDocumentStore, SqliteVectorIndex
from scone_memory.agents.catalog import AgentCatalog, AgentDefinition, AgentModel
from scone_memory.agents.evidence_loop import ToolStep
from scone_memory.agents.plan_store import AgentPlanStore
from scone_memory.agents.run_service import AgentRunService
from scone_memory.api.app import create_app
from scone_memory.retrieval.recall_scope import RecallScope


async def run():
    state, port = Path(sys.argv[1]), int(sys.argv[2])
    memory = await MemoryEngine(SqliteDocumentStore(state / 'catalog.db'),
                               SqliteVectorIndex(state / 'catalog.db'), HashEmbedder()).open()

    class Model:
        def __init__(self, name):
            self.name = name

        async def complete(self, messages, tools):
            with (state / 'calls.jsonl').open('a') as output:
                output.write(json.dumps({'model': self.name, 'messages': messages}) + '\n')
            if self.name == 'relay':
                return ToolStep(content=json.dumps({'answer': 'Ask the finisher.', 'handoff_to': 'finisher'}))
            if self.name == 'finish':
                return ToolStep(content=json.dumps({'answer': 'Completed answer.', 'handoff_to': None}))
            return ToolStep(content='Ada studies stars.')

    catalog = AgentCatalog(models=[AgentModel(name, name.title(), '1', lambda name=name: Model(name))
                                   for name in ('careful', 'relay', 'finish')], agents=[
        AgentDefinition(agent_id='researcher', instructions='Use retained evidence.', models=('careful',),
                        default_model='careful', initial_search=True),
        AgentDefinition(agent_id='relay', instructions='Delegate to finisher.', models=('relay',),
                        default_model='relay', initial_search=False),
        AgentDefinition(agent_id='finisher', instructions='Finish.', models=('finish',),
                        default_model='finish', initial_search=False)])
    plans = AgentPlanStore(state / 'plans.sqlite', key=b'k'*32)
    service = AgentRunService(state / 'runs', key=b'k'*32, catalog=catalog, plans=plans,
                              memory=memory, scope_for=lambda _: RecallScope.validated())
    app = create_app(memory, {'agent-fixture': 'alpha'}, agent_catalog=catalog,
                     agent_plan_store=plans, agent_run_service=service)
    server = uvicorn.Server(uvicorn.Config(app, host='127.0.0.1', port=port, log_level='warning'))

    async def commands():
        await asyncio.to_thread(sys.stdin.readline)
        server.should_exit = True

    task = asyncio.create_task(commands())
    try:
        await server.serve()
    finally:
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)
        await service.aclose()
        plans.close()
        await memory.close()


asyncio.run(run())
