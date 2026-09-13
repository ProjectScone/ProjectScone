"""Local real HTTP approvals with persistent journals and counted model/effects."""

import asyncio
import json
from pathlib import Path
import sys

import uvicorn
from scone_memory import HashEmbedder, InMemoryDocumentStore, InMemoryVectorIndex, MemoryEngine
from scone_memory.agents.catalog import AgentCatalog, AgentDefinition, AgentModel
from scone_memory.agents.custom_tools import AgentTool
from scone_memory.agents.evidence_loop import ToolCall, ToolStep
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
            with (state / 'models.jsonl').open('a') as output:
                output.write(json.dumps({'model': self.name}) + '\n')
            if any(row.get('role') == 'tool' for row in messages):
                handoff = any(
                    'handoff_to' in str(row.get('content', '')) for row in messages if row['role'] == 'system'
                )
                return ToolStep(content='{"answer":"Done","handoff_to":null}' if handoff else 'Done')
            return ToolStep(calls=(ToolCall(id='send', name='send_note', arguments={'message': 'Hello'}),))

    async def send(arguments, context):
        with (state / 'effects.jsonl').open('a') as output:
            output.write(json.dumps({'space': context.space, 'arguments': arguments}) + '\n')
        return {'sent': True}

    tool = AgentTool(
        'send_note',
        'Send the literal message.',
        '1',
        {
            'type': 'object',
            'properties': {'message': {'type': 'string'}},
            'required': ['message'],
            'additionalProperties': False,
        },
        send,
        requires_approval=True,
    )
    catalog = AgentCatalog(
        models=[AgentModel(name, name, '1', lambda name=name: Model(name)) for name in ('fast', 'careful')],
        tools=[tool],
        agents=[
            AgentDefinition(
                agent_id='worker',
                instructions='Send a note.',
                models=('fast', 'careful'),
                default_model='fast',
                initial_search=False,
                tools=('send_note',),
            )
        ],
    )
    plans = AgentPlanStore(state / 'plans.sqlite', key=b'k' * 32)
    service = AgentRunService(
        state / 'runs',
        key=b'k' * 32,
        catalog=catalog,
        plans=plans,
        memory=memory,
        scope_for=lambda _: RecallScope.validated(),
    )
    app = create_app(
        memory,
        {'agent-fixture': 'alpha'},
        agent_catalog=catalog,
        agent_plan_store=plans,
        agent_run_service=service,
    )
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
        await service.aclose()
        plans.close()
        await memory.close()


asyncio.run(run())
