"""Frozen public QA experiments. Labels are exported separately and never read here.

Run with ``python -m scone_memory.testing.public_qa_run {export,prepare,generate} DIR``.
Public source downloads are supplied by the operator; inference is self-managed.
"""
from __future__ import annotations

import argparse
import asyncio
from datetime import datetime, timezone
import fcntl
import hashlib
import json
from importlib.metadata import version
from pathlib import Path
import platform
import re
import time
from typing import Literal, cast

import httpx

from .public_qa import Document, FrozenRecord, Question, benchmark_messages

MODELS = ('gemma4-e4b-ctx8k:latest', 'llama3.2-ctx8k:latest', 'llama3.1-ctx8k:latest')
SPACE = 'public-qa-v1'
PACKAGE = Path(__file__).resolve().parent.parent
PROTOCOL = PACKAGE.parents[1] / 'benchmarks/public-qa-v1.protocol.md'


class Source(FrozenRecord):
    document_id: str
    text: str


class Prepared(FrozenRecord):
    question: Question
    request: list[dict[str, str]]
    request_sha256: str
    receipt: dict[str, object]
    sources: tuple[Source, ...]
    ranked_document_ids: tuple[str, ...]
    prepare_ms: float
    recall_ms: float
    recall_status: str


class Observation(FrozenRecord):
    id: str
    model: str
    block: int
    block_first: bool
    request_sha256: str
    status: str = 'unattempted'
    completed: bool = False
    answer_text: str = ''
    total_ms: float | None = None
    first_token_ms: float | None = None
    started_at: str | None = None
    details: dict[str, object] = {}


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def request_digest(messages: list[dict[str, str]]) -> str:
    return hashlib.sha256(json.dumps(messages, ensure_ascii=False, sort_keys=True).encode()).hexdigest()


def save(path: Path, value: object) -> None:
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')
    temporary.replace(path)


def code_hashes() -> dict[str, str]:
    return {**{str(p.relative_to(PACKAGE)): digest(p) for p in sorted(PACKAGE.rglob('*.py'))},
            'benchmark_protocol': digest(PROTOCOL)}


def frozen_files(root: Path) -> dict[str, str]:
    return {name: digest(root / name) for name in ('dataset.json', 'corpus.jsonl', 'queries.jsonl', 'prepared.jsonl', 'preparation.json')}


def validate_sample(root: Path) -> None:
    manifest = json.loads((root / 'dataset.json').read_text())
    for name in ('corpus.jsonl', 'queries.jsonl'):
        if manifest['files_sha256'][name] != digest(root / name):
            raise ValueError('sampled source or original questions changed')


def export(root: Path) -> None:
    from .public_qa import build_bundle
    targets = ('corpus.jsonl', 'queries.jsonl', 'reserved-queries.jsonl', 'gold.jsonl', 'dataset.json')
    if any((root / name).exists() for name in targets):
        raise FileExistsError('do not overwrite a sampled dataset')
    raw = root / 'raw'
    expected = {'hotpot_dev_distractor_v1.json': 'e3da074df24e8369009918aa5cdbdd254dadcde4c63f7569d36afd6f2268caa8',
                'squad_dev_v1.1.json': '95aa6a52d5d6a735563366753ca50492a658031da74f301ac5238b03966972c9'}
    if any(digest(raw / name) != sha for name, sha in expected.items()):
        raise ValueError('source bytes differ from the frozen v1 public datasets')
    bundle = build_bundle(raw / 'hotpot_dev_distractor_v1.json', raw / 'squad_dev_v1.1.json')
    for name, rows in zip(targets, (bundle.documents, bundle.queries, bundle.reserved, bundle.gold)):
        with (root / name).open('x') as stream:
            for row in rows:
                stream.write(row.model_dump_json() + '\n')
    save(root / 'dataset.json', {'raw_sha256': {p.name: digest(p) for p in sorted(raw.glob('*.json'))},
         'files_sha256': {name: digest(root / name) for name in targets[:-1]},
         'documents': len(bundle.documents), 'evaluation': len(bundle.queries), 'reserved': len(bundle.reserved)})


def extract_sources(request: list[dict[str, str]], episodes: dict[int, Document]) -> tuple[Source, ...]:
    from ..realtime.context import _PREFIX
    sources: list[Source] = []
    for message in request:
        content = message['content']
        if not content.startswith(_PREFIX):
            continue
        payload = json.loads(content[len(_PREFIX):])
        if payload.get('claims') or payload.get('relations') or payload.get('paths'):
            raise ValueError('raw-ingestion protocol must not inject seeded claims')
        for item in payload['sources']:
            document = episodes[item['episode_id']]
            if item['text'] not in document.content:
                raise ValueError('retained text is not verbatim source text')
            sources.append(Source(document_id=document.id, text=item['text']))
    return tuple(sources)


async def prepare(root: Path, qdrant_url: str, cache: Path) -> None:
    from ..backends import SqliteDocumentStore, QdrantVectorIndex
    from ..memory.engine import MemoryEngine, Record
    from ..providers.self_hosted import validate_self_hosted_endpoint
    from ..realtime.context import MemoryContext
    from .edge_retrieval_benchmark import _CachedBGE
    if any((root / name).exists() for name in ('prepared.jsonl', 'preparation.json', 'ledger.db')):
        raise FileExistsError('preparation requires a fresh run directory')
    validate_self_hosted_endpoint(qdrant_url)
    validate_sample(root)
    async with httpx.AsyncClient(timeout=10., trust_env=False) as client:
        response = await client.get(qdrant_url)
        response.raise_for_status()
        server = response.json()
        if server.get('version') != '1.19.1':
            raise ValueError('v1 protocol requires Qdrant 1.19.1')
    documents = [Document.model_validate_json(line) for line in (root / 'corpus.jsonl').read_text().splitlines()]
    queries = [Question.model_validate_json(line) for line in (root / 'queries.jsonl').read_text().splitlines()]
    before = code_hashes()
    vectors = QdrantVectorIndex(qdrant_url, collection='scone_public_qa_' + hashlib.sha256(str(root.resolve()).encode()).hexdigest()[:16])
    engine = await MemoryEngine(SqliteDocumentStore(str(root / 'ledger.db')), vectors, _CachedBGE(cache)).open()
    episodes: dict[int, Document] = {}
    try:
        for offset in range(0, len(documents), 50):
            batch = documents[offset:offset + 50]
            added = await engine.remember_many(SPACE, [Record(d.content, kind='file', source=d.source_url,
                metadata={'document_id': d.id}, created_at='2026-09-08') for d in batch])
            for document, result in zip(batch, added, strict=True):
                if result.episode_id in episodes:
                    raise ValueError('unexpected cross-document deduplication')
                episodes[result.episode_id] = document
            print(f'Indexed {len(episodes)}/{len(documents)} source paragraphs', flush=True)
        with (root / 'prepared.jsonl').open('x') as stream:
            for index, question in enumerate(queries):
                started = time.perf_counter()
                context = MemoryContext(engine, SPACE, question.id, limit=5, max_context_bytes=8000,
                                        recall_timeout=30., structured_paths=True)
                messages, receipt = await context.prepare([dict(m) for m in benchmark_messages(question)])
                prepare_ms = (time.perf_counter() - started) * 1000
                if any(not isinstance(m.get('content'), str) for m in messages):
                    raise ValueError('expected text-only request')
                request = cast(list[dict[str, str]], messages)
                if request[-1] != {'role': 'user', 'content': question.question}:
                    raise ValueError('original question was modified')
                sources = extract_sources(request, episodes)
                started = time.perf_counter()
                ranked: tuple[str, ...] = ()
                recall_status = 'completed'
                try:
                    async with asyncio.timeout(30.):
                        recalled = await engine.recall(SPACE, question.question, limit=10)
                    ranked = tuple(episodes[item.episode_id].id for item in recalled.items)
                except Exception as exc:
                    recall_status = type(exc).__name__
                row = Prepared(question=question, request=request, request_sha256=request_digest(request),
                    receipt=dict(receipt), sources=sources, ranked_document_ids=ranked, prepare_ms=prepare_ms,
                    recall_ms=(time.perf_counter() - started) * 1000, recall_status=recall_status)
                stream.write(row.model_dump_json() + '\n')
                stream.flush()
                print(f'Prepared {index + 1}/{len(queries)} unchanged questions', flush=True)
        if code_hashes() != before:
            raise RuntimeError('code changed during preparation')
        save(root / 'preparation.json', {'code_sha256': before, 'queries': len(queries), 'documents': len(documents),
             'prepared_sha256': digest(root / 'prepared.jsonl'), 'backend': 'SQLite + Qdrant',
             'embedder': 'bge-small-en-v1.5', 'qdrant_server': server, 'python': platform.python_version(),
             'dependencies': {name: version(name) for name in ('fastembed', 'onnxruntime', 'qdrant-client', 'httpx', 'pydantic')},
             'episode_documents': {str(k): v.id for k, v in episodes.items()}})
    finally:
        await engine.close()


def schedule(prepared: list[Prepared]) -> list[Observation]:
    if len({p.question.id for p in prepared}) != len(prepared):
        raise ValueError('duplicate prepared question ID')
    rows: list[Observation] = []
    for offset in range(0, len(prepared), 20):
        block = offset // 20
        rotation = block % len(MODELS)
        for model in (*MODELS[rotation:], *MODELS[:rotation]):
            for index, item in enumerate(prepared[offset:offset + 20]):
                rows.append(Observation(id=item.question.id, model=model, block=block,
                    block_first=index == 0, request_sha256=item.request_sha256))
    return rows


def load_prepared(root: Path) -> list[Prepared]:
    validate_sample(root)
    preparation = json.loads((root / 'preparation.json').read_text())
    if preparation['prepared_sha256'] != digest(root / 'prepared.jsonl'):
        raise ValueError('prepared request integrity failure')
    rows = [Prepared.model_validate_json(line) for line in (root / 'prepared.jsonl').read_text().splitlines()]
    queries = [Question.model_validate_json(line) for line in (root / 'queries.jsonl').read_text().splitlines()]
    if [row.question for row in rows] != queries:
        raise ValueError('prepared questions differ from original sampled queries')
    for row in rows:
        if (row.request_sha256 != request_digest(row.request)
            or row.request[-1] != {'role': 'user', 'content': row.question.question}):
            raise ValueError('request integrity failure')
    return rows


async def generate(root: Path, endpoint: str) -> None:
    from ..providers.llm import OpenAICompatibleTextModel
    from ..providers.self_hosted import validate_self_hosted_endpoint
    from .generation_ablation import capture_public_reply
    validate_self_hosted_endpoint(endpoint)
    prepared = load_prepared(root)
    if len(prepared) != 200:
        raise ValueError('v1 protocol requires all 200 evaluation questions')
    before = code_hashes()
    if json.loads((root / 'preparation.json').read_text())['code_sha256'] != before:
        raise ValueError('code changed since preparation; preserve this run and use a new protocol/run')
    files = frozen_files(root)
    planned = schedule(prepared)
    requests = {p.question.id: p.request for p in prepared}
    state_path = root / 'observations.json'
    manifest_path = root / 'manifest.json'
    async with httpx.AsyncClient(base_url=endpoint, timeout=30., trust_env=False) as client:
        response = await client.get('/api/tags')
        response.raise_for_status()
        installed = {m['name']: m for m in response.json()['models'] if m['name'] in MODELS}
        if set(installed) != set(MODELS):
            raise ValueError('all protocol models must already be installed')
        model_settings: dict[str, object] = {}
        for model in MODELS:
            response = await client.post('/api/show', json={'model': model})
            response.raise_for_status()
            detail = response.json()
            if re.search(r'(?m)^num_ctx\s+8192\s*$', detail.get('parameters', '')) is None:
                raise ValueError('protocol model must be configured with 8192 context tokens')
            model_settings[model] = detail
        manifest = {'code_sha256': before, 'files_sha256': files, 'models': installed, 'model_settings': model_settings,
                    'endpoint': endpoint, 'planned': [r.model_dump() for r in planned]}
        if manifest_path.exists():
            existing = json.loads(manifest_path.read_text())
            if any(existing[k] != manifest[k] for k in ('code_sha256', 'files_sha256', 'endpoint', 'planned', 'model_settings')):
                raise ValueError('frozen experiment manifest changed')
            if any(existing['models'][m]['digest'] != installed[m]['digest'] for m in MODELS):
                raise ValueError('installed model digest changed')
        else:
            save(manifest_path, manifest)
        if state_path.exists():
            rows = [Observation.model_validate(r) for r in json.loads(state_path.read_text())]
            identity = lambda r: (r.id, r.model, r.block, r.block_first, r.request_sha256)
            if list(map(identity, rows)) != list(map(identity, planned)):
                raise ValueError('observation schedule changed')
            rows = [r.model_copy(update={'status': 'interrupted', 'completed': False}) if r.status == 'running' else r for r in rows]
        else:
            rows = planned
        save(state_path, [r.model_dump() for r in rows])
        last_block: tuple[int, str] | None = None
        for index, row in enumerate(rows):
            if row.status != 'unattempted':
                continue
            if code_hashes() != before or frozen_files(root) != files:
                raise RuntimeError('frozen code or input changed during inference')
            # Resuming in a block also records its actual resident-model state.
            if last_block != (row.block, row.model):
                await verify_model_digests(client, {m: installed[m]['digest'] for m in MODELS})
                for other in MODELS:
                    if other == row.model:
                        continue
                    response = await client.post('/api/generate', json={'model': other, 'keep_alive': 0})
                    response.raise_for_status()
                await memory_snapshot(client, root, row, 'before')
                last_block = row.block, row.model
            rows[index] = row.model_copy(update={'status': 'running', 'started_at': datetime.now(timezone.utc).isoformat()})
            save(state_path, [r.model_dump() for r in rows])
            provider = OpenAICompatibleTextModel(endpoint + '/v1', row.model, think=False, temperature=0.,
                                                max_output_tokens=256, timeout=120., trust_env=False)
            result = await capture_public_reply(provider, requests[row.id], timeout=120.)
            rows[index] = Observation.model_validate({**rows[index].model_dump(),
                **{k: result[k] for k in ('status', 'completed', 'answer_text', 'total_ms', 'first_token_ms')},
                'details': result})
            save(state_path, [r.model_dump() for r in rows])
            print(f'{index + 1}/{len(rows)} {row.model} {rows[index].status} {rows[index].total_ms}ms', flush=True)
            if index + 1 == len(rows) or rows[index + 1].model != row.model or rows[index + 1].block != row.block:
                await memory_snapshot(client, root, row, 'after')
        await verify_model_digests(client, {m: installed[m]['digest'] for m in MODELS})
        unchanged = code_hashes() == before and frozen_files(root) == files
        save(root / 'completion.json', {'terminal': all(r.status not in ('running', 'unattempted') for r in rows),
             'code_and_inputs_unchanged': unchanged, 'manifest_sha256': digest(manifest_path),
             'observations_sha256': digest(state_path)})
        if not unchanged:
            raise RuntimeError('comparison invalidated by changed code or input')


async def verify_model_digests(client: httpx.AsyncClient, expected: dict[str, str]) -> None:
    response = await client.get('/api/tags')
    response.raise_for_status()
    current = {m['name']: m['digest'] for m in response.json()['models'] if m['name'] in expected}
    if current != expected:
        raise ValueError('model digests changed during the frozen experiment')


async def memory_snapshot(client: httpx.AsyncClient, root: Path, row: Observation, stage: Literal['before', 'after']) -> None:
    response = await client.get('/api/ps')
    response.raise_for_status()
    with (root / 'model-memory.jsonl').open('a') as stream:
        stream.write(json.dumps({'at': datetime.now(timezone.utc).isoformat(), 'model': row.model,
            'block': row.block, 'stage': stage, 'models': response.json()['models']}) + '\n')


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('phase', choices=('export', 'prepare', 'generate'))
    parser.add_argument('directory', type=Path)
    parser.add_argument('--qdrant-url', default='http://127.0.0.1:53410')
    parser.add_argument('--endpoint', default='http://127.0.0.1:11434')
    parser.add_argument('--embedding-cache', type=Path, default=Path.home() / '.scone-memory/fastembed')
    args = parser.parse_args()
    with (args.directory / '.experiment.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if args.phase == 'export':
            export(args.directory)
        elif args.phase == 'prepare':
            asyncio.run(prepare(args.directory, args.qdrant_url, args.embedding_cache))
        else:
            asyncio.run(generate(args.directory, args.endpoint.rstrip('/')))


if __name__ == '__main__':
    main()
