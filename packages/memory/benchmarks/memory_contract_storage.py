"""Exercise the contract against native SQLite, Qwen ingestion and direct Jev.

Creates synthetic records in a NEW local directory. Never opens the live store.
"""
from __future__ import annotations

import argparse
from dataclasses import replace
import json
import os
from pathlib import Path
import time

from scone_memory import MemoryEngine, Record
from scone_memory.backends.sqlite import SqliteDocumentStore, SqliteVectorIndex
from scone_memory.embedders.remote import RemoteEmbedder
from scone_memory.experimental.contract_judge import JevContractJudge
from scone_memory.experimental.memory_contracts import (
    ContractRequest, Evidence, MemoryContract, compile_contract, evidence_worlds,
)

SPACE = 'contract-experiment'


async def snapshot(memory: MemoryEngine) -> tuple[Evidence, ...]:
    # Entire tiny experimental space: prevents a new untracked source from being
    # mistaken for a withdrawal. Production needs an equivalent coherent snapshot.
    episodes = await memory.documents.recent_episodes(SPACE, 21)
    if len(episodes) > 20:
        raise ValueError('experiment snapshot is not bounded')
    if any(not item.source for item in episodes):
        raise ValueError('experiment source lacks origin provenance')
    return tuple(Evidence(str(item.episode_id), item.source or '', item.content) for item in episodes)


async def engine(path: Path) -> MemoryEngine:
    embedder = RemoteEmbedder(
        os.environ['SCONE_EMBED_URL'], os.environ['SCONE_EMBED_MODEL'],
        api_key=os.environ.get('SCONE_EMBED_API_KEY'), trust_env=False,
    )
    try:
        await embedder.embed(['dimension probe'])
        return await MemoryEngine(SqliteDocumentStore(path), SqliteVectorIndex(path), embedder).open()
    except BaseException:
        await embedder.close()
        raise


async def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--directory', type=Path, required=True)
    args = parser.parse_args()
    args.directory.mkdir(parents=True, exist_ok=False)
    path = args.directory / 'memory.db'
    artifact = args.directory / 'contract.json'
    judge = JevContractJudge(os.environ.get('TYPESAFE_BASE_URL', 'https://api.typesafe.ai'),
        os.environ.get('TYPESAFE_DEFAULT_MODEL', 'jev-latest'), api_key=os.environ['TYPESAFE_API_KEY'])
    memory = await engine(path)
    checks: list[dict[str, object]] = []
    try:
        await memory.remember_many(SPACE, (
            Record('Decision for Nova: Maya owns deployment.', source='meeting'),
            Record('Nova deployment responsibility: Maya.', source='roster'),
        ))
        evidence = await snapshot(memory)
        request = ContractRequest(SPACE + ':read-all:v1', 'Who owns Nova deployment?',
                                  'Maya owns Nova deployment.', evidence)
        assessed = await judge.assess(request, evidence_worlds(request))
        contract = compile_contract(request, assessed.judgments, model=assessed.model,
                                    expires_at=time.time() + 3600)
        artifact.write_text(contract.to_json())
        checks.append({'step': 'compiled', 'status': contract.evaluate(request, now=time.time()).status})
        await memory.forget(SPACE, int(evidence[0].source_id))
        current = replace(request, evidence=await snapshot(memory))
        checks.append({'step': 'one_independent_source_forgotten',
                       'status': contract.evaluate(current, now=time.time()).status})
    finally:
        await memory.close()
    memory = await engine(path)
    try:
        restored = MemoryContract.from_json(artifact.read_text())
        current = replace(request, evidence=await snapshot(memory))
        checks.append({'step': 'restart', 'status': restored.evaluate(current, now=time.time()).status})
        for item in current.evidence:
            await memory.forget(SPACE, int(item.source_id))
        current = replace(request, evidence=await snapshot(memory))
        checks.append({'step': 'all_support_forgotten',
                       'status': restored.evaluate(current, now=time.time()).status})
        await memory.remember_many(SPACE, (Record('Leo now owns Nova deployment, not Maya.', source='correction'),))
        current = replace(request, evidence=await snapshot(memory))
        checks.append({'step': 'new_correction', 'status': restored.evaluate(current, now=time.time()).status})
    finally:
        await memory.close()
    expected = ['supported', 'supported', 'supported', 'insufficient', 'recompile']
    passed = [item['status'] for item in checks] == expected
    report = {'passed': passed, 'checks': checks, 'judgment_api_requests': assessed.requests,
              'compiled_model': assessed.model, 'embedding_model': os.environ['SCONE_EMBED_MODEL']}
    (args.directory / 'result.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2))
    if not passed:
        raise SystemExit(1)


if __name__ == '__main__':
    import asyncio
    asyncio.run(main())
