"""Exercise Jev + Gemma through Scone's durable conversation API on downloaded data."""
from __future__ import annotations

import argparse
import asyncio
import json
import os
from pathlib import Path
import sqlite3

import httpx

from scone_memory import HashEmbedder, MemoryEngine
from scone_memory.api.__main__ import build_app
from scone_memory.backends import SqliteDocumentStore, SqliteVectorIndex
from scone_memory.providers.jev import create_reranker
from scone_memory.runtime.config import Settings
from scone_memory.runtime.model_connections import ModelConnection, ModelConnectionStore
from scone_memory.testing.jev_public_qa import SPACE, Observer


async def run(dataset: Path, retrieval: Path, output: Path) -> None:
    completion = json.loads((retrieval/'completion.json').read_text())
    if not completion['terminal'] or not completion['code_and_inputs_unchanged']:
        raise ValueError('requires completed retrieval evaluation')
    output.mkdir(parents=True,exist_ok=False)
    with sqlite3.connect(f'file:{retrieval / "memory.db"}?mode=ro',uri=True) as source:
        with sqlite3.connect(output/'memory.db') as destination:
            source.backup(destination)
    question = json.loads((dataset/'reserved-queries.jsonl').read_text().splitlines()[0])
    key_env = os.environ.get('SCONE_JEV_API_KEY_ENV','OPENROUTER_API_KEY')
    ModelConnectionStore(output/'models.json',{}).replace('chat',ModelConnection(
        provider='openrouter',base_url='https://openrouter.ai/api/v1/',model='google/gemma-4-31b-it',
        api_key_env=key_env,timeout_s=60),expected_revision=0)
    settings = Settings.from_env({'SCONE_API_KEYS':f'fixture-key:{SPACE}',
        'SCONE_MODEL_CONNECTIONS':str(output/'models.json'),
        'SCONE_CONVERSATIONS_JOURNAL':str(output/'sessions.db'),
        'SCONE_CONVERSATIONS_TOOL_MODE':'native','SCONE_CONVERSATIONS_TOOL_INITIAL_SEARCH':'1',
        'SCONE_CONVERSATIONS_TOOL_TIMEOUT':'60','SCONE_CHAT_THINK':'false'})
    results = {}
    path = ''
    receipt = None
    for phase in ('generation','restart'):
        observer = Observer(create_reranker())
        engine = await MemoryEngine(SqliteDocumentStore(output/'memory.db'),SqliteVectorIndex(output/'memory.db'),
            HashEmbedder(),reranker=observer,rerank_timeout=10,candidate_limit=64).open()
        app = build_app(settings,engine)
        try:
            async with app.router.lifespan_context(app):
                async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app),base_url='http://127.0.0.1',
                                             headers={'Authorization':'Bearer fixture-key'}) as client:
                    if phase=='generation':
                        response = await client.post('/v1/conversations',json={'request_id':'jev-public-qa',
                            'capture':True,'recall_scope':{'kind':'file'}})
                        response.raise_for_status()
                        session = response.json()
                        path = '/v1/conversations/'+session['session_id']+'/turns/answer'
                        response = await client.post(path.rsplit('/',1)[0],json={'request_id':'answer',
                            'text':question['question'],'expected_revision':session['revision']})
                        response.raise_for_status()
                        async with asyncio.timeout(90):
                            while response.json()['status']=='pending':
                                await asyncio.sleep(.2)
                                response = await client.get(path)
                                response.raise_for_status()
                        receipt = response.json()
                        results['question'] = question
                        results['receipt_path'] = path
                        results['receipt'] = receipt
                        results['jev_applied'] = observer.last is not None
                        results['resolved_jev_model'] = observer.last.model if observer.last else None
                        assert receipt['status']=='completed', receipt
                        assert observer.last is not None
                        assert receipt['result']['memory_context']['tool_retrieval']['source_status']=='retained'
                    else:
                        response = await client.get(path)
                        response.raise_for_status()
                        restarted = response.json()
                        results['restart_receipt'] = restarted
                        assert receipt is not None
                        results['restart_receipt_matches'] = restarted == receipt
                        # The journal retains the answer episode, not transient
                        # history/provider/evidence diagnostics from the runtime.
                        results['restart_answer_matches'] = (
                            all(restarted.get(key) == receipt.get(key)
                                for key in ('request_id', 'status', 'result_state', 'error'))
                            and all(restarted['result'].get(key) == receipt['result'].get(key)
                                    for key in ('text', 'assistant_episode_id')))
                        results['restart_missing_result_fields'] = sorted(
                            receipt['result'].keys() - restarted['result'].keys())
                        results['restart_did_not_rerank'] = observer.last is None
                        assert results['restart_answer_matches'] and results['restart_did_not_rerank']
        finally:
            await engine.close()
            (output/'results.json').write_text(json.dumps(results,indent=2)+'\n')
    print(json.dumps({key: value for key, value in results.items()
                      if key not in ('receipt', 'restart_receipt')}, indent=2))


if __name__=='__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset',type=Path,required=True)
    parser.add_argument('--retrieval',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args = parser.parse_args()
    asyncio.run(run(args.dataset,args.retrieval,args.output))
