"""Actual local native catalogue with shared tools and no inference on reads."""
import asyncio
from pathlib import Path
import sys

import uvicorn
from scone_memory import HashEmbedder, InMemoryDocumentStore, InMemoryVectorIndex, MemoryEngine
from scone_memory.agents.catalog import AgentCatalog, AgentDefinition, AgentModel
from scone_memory.agents.custom_tools import AgentTool
from scone_memory.agents.plan_store import AgentPlanStore
from scone_memory.api.app import create_app


async def run():
    state, port = Path(sys.argv[1]), int(sys.argv[2])
    memory = await MemoryEngine(InMemoryDocumentStore(), InMemoryVectorIndex(), HashEmbedder()).open()

    def forbidden(*args):
        (state / 'invoked').write_text('Unexpected callback')
        raise AssertionError('Catalogue reads must not create models or invoke tools')

    shared = AgentTool('inventory', '😀' * 1000, 'stock:1', {'type': 'object'}, forbidden)
    unused = AgentTool('private_unused', 'Unselected private tool', '1', {'type': 'object'}, forbidden)
    catalog = AgentCatalog(models=[AgentModel('local', 'Local', '1', forbidden)], tools=[shared, unused],
        agents=[AgentDefinition(agent_id=name, instructions='Answer', models=('local',), default_model='local',
                                tools=('inventory',) if name != 'empty' else ())
                for name in ('first', 'second', 'empty')])
    plans = AgentPlanStore(state / 'plans.sqlite', key=b'k' * 32)
    app = create_app(memory, {'agent-fixture': 'alpha'}, agent_catalog=catalog, agent_plan_store=plans)
    server = uvicorn.Server(uvicorn.Config(app, host='127.0.0.1', port=port, log_level='warning'))

    async def stop():
        await asyncio.to_thread(sys.stdin.readline)
        server.should_exit = True

    stopping = asyncio.create_task(stop())
    try:
        await server.serve()
    finally:
        stopping.cancel()
        await asyncio.gather(stopping, return_exceptions=True)
        plans.close()
        await memory.close()


asyncio.run(run())
