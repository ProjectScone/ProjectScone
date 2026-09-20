"""Paired answer evaluation through native Scone context and generation adapters.

The run command never reads gold labels. Score only complete, unchanged runs.
"""
from __future__ import annotations

import argparse
import asyncio
from collections import Counter
import json
import os
import re
import sqlite3
from pathlib import Path
import statistics
import time
import uuid
from urllib.parse import urlsplit
from typing import Literal, TYPE_CHECKING, cast

from .jev_public_qa import SPACE, Observer, code_digest, digest, save
from .public_qa import Document, FrozenRecord, Question, benchmark_messages
from .public_qa_run import Source, extract_sources, request_digest

if TYPE_CHECKING:
    from ..memory.engine import MemoryEngine
    from .public_qa import Gold

Arm = Literal['hybrid', 'hybrid_jev']
ARMS: tuple[Arm, ...] = ('hybrid', 'hybrid_jev')
MODEL = 'google/gemma-4-31b-it'
JEV_MODEL = 'typesafe/jev-1.13-20260917'
PROTOCOL = Path(__file__).resolve().parents[3] / 'benchmarks/jev-answers-v1.protocol.md'
QWEN_PROTOCOL = PROTOCOL.with_name('jev-qwen-answers-v1.protocol.md')
QWEN_MODEL = 'qwen/qwen3-embedding-8b'
QUERY_PREFIX = 'Instruct: Find passages that provide evidence to answer the question.\nQuery:'


def build_engine(output: Path, key: str, qdrant_url: str | None = None,
                 collection: str | None = None) -> MemoryEngine:
    from .. import HashEmbedder, MemoryEngine
    from ..backends import SqliteDocumentStore, SqliteVectorIndex
    from ..core.ports import Embedder, VectorIndex

    embedder: Embedder
    vectors: VectorIndex
    if qdrant_url is not None:
        parsed = urlsplit(qdrant_url)
        if (parsed.scheme != 'http' or parsed.hostname not in ('127.0.0.1', 'localhost', '::1')
                or parsed.username or parsed.password or parsed.query or parsed.fragment
                or parsed.path not in ('', '/')):
            raise ValueError('evaluation requires a local Qdrant URL without credentials')
        from ..backends.qdrant import QdrantVectorIndex
        from ..embedders.remote import RemoteEmbedder

        embedder = RemoteEmbedder('https://openrouter.ai/api/v1', QWEN_MODEL, api_key=key,
            dim=4096, query_prefix=QUERY_PREFIX, trust_env=False)
        vectors = QdrantVectorIndex(qdrant_url, collection or 'scone_qwen_answers_' + uuid.uuid4().hex)
    else:
        embedder = HashEmbedder()
        vectors = SqliteVectorIndex(output/'memory.db')
    return MemoryEngine(SqliteDocumentStore(output/'memory.db'), vectors, embedder,
        chunk_target=700, candidate_limit=64, rerank_limit=32, rerank_max_bytes=64000,
        rerank_timeout=10)


async def prepare_text_index(engine: MemoryEngine) -> dict[str, int]:
    from ..backends import SqliteDocumentStore
    store = cast(SqliteDocumentStore, engine.documents)
    indexed = 0
    while True:
        done, remaining = await store.sync_text_index(SPACE)
        indexed += done
        if remaining == 0:
            return {'indexed': indexed, 'remaining': 0}
        if done <= 0:
            raise ValueError('text index preparation made no progress')
        await asyncio.sleep(0)


async def prepare_query_vectors(engine: MemoryEngine, questions: list[Question]) -> dict[str, object]:
    from ..bench.comparative import CachedEmbedder
    from ..ingestion.embedding_cache import InMemoryEmbeddingCache
    cached = CachedEmbedder(engine.embedder, InMemoryEmbeddingCache(1000))
    engine.embedder = cached
    for offset in range(0, len(questions), 32):
        await cached.embed_queries([q.question for q in questions[offset:offset+32]])
    return cached.record()


async def indexed_documents(engine: MemoryEngine, corpus: list[Document]) -> dict[int, Document]:
    from ..backends import SqliteDocumentStore
    store = cast(SqliteDocumentStore, engine.documents)
    expected = {doc.id: doc for doc in corpus}
    documents: dict[int, Document] = {}
    seen: set[str] = set()
    before = None
    while True:
        episodes = await store.page_episodes(SPACE, before=before, limit=100, kind='file')
        if not episodes:
            break
        for episode in episodes:
            document_id = episode.metadata.get('document_id', '')
            doc = expected.get(document_id)
            if (doc is None or document_id in seen or episode.content != doc.content
                    or episode.source != doc.source_url):
                raise ValueError('reused index differs from the corpus')
            seen.add(document_id)
            documents[episode.episode_id] = doc
        before = episodes[-1].episode_id
    if seen != set(expected):
        raise ValueError('reused index does not contain the complete corpus')
    return documents


def reuse_collection(source: Path, output: Path, inputs: dict[str, str], qdrant_url: str | None) -> str:
    manifest = json.loads((source/'manifest.json').read_text())
    collection = manifest.get('qdrant_collection')
    if (not qdrant_url or manifest.get('qdrant_url') != qdrant_url or manifest.get('inputs') != inputs
            or manifest.get('embedding_model') != QWEN_MODEL or manifest.get('dimensions') != 4096
            or manifest.get('query_prefix') != QUERY_PREFIX or manifest.get('vector_backend') != 'qdrant'
            or not isinstance(collection, str) or not re.fullmatch(r'scone_qwen_answers_[0-9a-f]{32}', collection)):
        raise ValueError('reused index configuration or inputs differ')
    old = sqlite3.connect((source/'memory.db').resolve().as_uri() + '?mode=ro', uri=True)
    new = sqlite3.connect(output/'memory.db')
    try:
        old.backup(new)
    finally:
        old.close()
        new.close()
    return collection


class PreparedAnswer(FrozenRecord):
    question: Question
    arm: Arm
    request: list[dict[str, str]]
    request_sha256: str
    receipt: dict[str, object]
    sources: tuple[Source, ...]
    prepare_ms: float
    resolved_jev_model: str | None = None


class Answer(FrozenRecord):
    id: str
    arm: Arm
    request_sha256: str
    status: str
    completed: bool
    answer_text: str
    total_ms: float
    first_token_ms: float | None = None
    error_type: str | None = None
    cleanup_error_type: str | None = None
    output_bytes: int = 0
    truncated: bool = False


async def prepare_question(engine: MemoryEngine, question: Question, arm: Arm,
                           documents: dict[int, Document]) -> PreparedAnswer:
    from ..realtime.context import MemoryContext

    started = time.perf_counter()
    context = MemoryContext(engine, SPACE, question.id, kind='file', limit=5,
                            max_context_bytes=8000, recall_timeout=30)
    messages, receipt = await context.prepare([dict(m) for m in benchmark_messages(question)])
    if any(not isinstance(m.get('content'), str) for m in messages):
        raise ValueError('expected text-only request')
    request = cast(list[dict[str, str]], messages)
    if request[-1] != {'role': 'user', 'content': question.question}:
        raise ValueError('original question changed')
    return PreparedAnswer(question=question, arm=arm, request=request,
        request_sha256=request_digest(request), receipt=dict(receipt),
        sources=extract_sources(request, documents), prepare_ms=(time.perf_counter()-started)*1000)


def load_questions(dataset: Path) -> list[Question]:
    manifest = json.loads((dataset/'dataset.json').read_text())
    for name in ('corpus.jsonl', 'reserved-queries.jsonl'):
        if digest(dataset/name) != manifest['files_sha256'][name]:
            raise ValueError('downloaded inputs changed')
    questions = [Question.model_validate_json(line)
                 for line in (dataset/'reserved-queries.jsonl').read_text().splitlines()]
    if (len(questions) != 200 or len({q.id for q in questions}) != 200
            or Counter(q.dataset for q in questions) != {'hotpotqa': 100, 'squad': 100}):
        raise ValueError('requires all 200 reserved questions')
    return questions


async def run(dataset: Path, output: Path, qdrant_url: str | None = None,
              reuse_index: Path | None = None) -> None:
    from ..bench.comparative import CachedEmbedder
    from ..memory.engine import Record
    from ..providers.jev import JevReranker
    from ..providers.llm import OpenAICompatibleTextModel
    from .generation_ablation import capture_public_reply

    questions = load_questions(dataset)
    corpus = [Document.model_validate_json(line) for line in (dataset/'corpus.jsonl').read_text().splitlines()]
    if len(corpus) != 2176:
        raise ValueError('requires the complete 2176-paragraph corpus')
    key_name = os.environ.get('SCONE_JEV_API_KEY_ENV', 'OPENROUTER_API_KEY')
    key = os.environ.get(key_name)
    if not key:
        raise ValueError('configured server token is missing')
    observer = Observer(JevReranker(api_key=key, model=JEV_MODEL))
    output.mkdir(parents=True, exist_ok=False)
    inputs = {name: digest(dataset/name) for name in ('dataset.json', 'corpus.jsonl', 'reserved-queries.jsonl')}
    protocol_path = QWEN_PROTOCOL if qdrant_url else PROTOCOL
    code, protocol = code_digest(), digest(protocol_path)
    collection = reuse_collection(reuse_index, output, inputs, qdrant_url) if reuse_index else None
    engine = build_engine(output, key, qdrant_url, collection)
    save(output/'manifest.json', {'protocol': protocol_path.name.removesuffix('.protocol.md'), 'protocol_sha256': protocol,
        'code_sha256': code, 'inputs': inputs, 'questions': [q.model_dump() for q in questions],
        'gold_sha256': json.loads((dataset/'dataset.json').read_text())['files_sha256']['gold.jsonl'],
        'chat_model': MODEL, 'jev_model': JEV_MODEL, 'arms': ARMS, 'embedder': engine.embedder.id,
        'embedding_model': QWEN_MODEL if qdrant_url else 'HashEmbedder', 'dimensions': engine.embedder.dim,
        'query_prefix': QUERY_PREFIX if qdrant_url else '', 'vector_weight': engine.vector_weight,
        'vector_backend': engine.vectors.name, 'qdrant_url': qdrant_url,
        'qdrant_collection': getattr(engine.vectors, 'collection', None),
        'shared_query_vector_cache': bool(qdrant_url),
        'index_source': str(reuse_index) if reuse_index else None,
        'index_source_manifest_sha256': digest(reuse_index/'manifest.json') if reuse_index else None,
        'candidate_limit': 64, 'rerank_limit': 32, 'context_limit': 5, 'max_context_bytes': 8000,
        'temperature': 0, 'think': False, 'max_output_tokens': 256})
    await engine.open()
    documents: dict[int, Document] = {}
    try:
        if reuse_index:
            documents = await indexed_documents(engine, corpus)
            print(f'Validated {len(documents)} existing corpus documents', flush=True)
        else:
            for offset in range(0, len(corpus), 100):
                batch = corpus[offset:offset+100]
                episodes = await engine.remember_many(SPACE, [Record(doc.content, kind='file', source=doc.source_url,
                    metadata={'document_id': doc.id}, created_at='2026-09-08') for doc in batch])
                for episode, doc in zip(episodes, batch, strict=True):
                    if episode.episode_id in documents:
                        raise ValueError('unexpected corpus deduplication')
                    documents[episode.episode_id] = doc
                print(f'Indexed {len(documents)}/{len(corpus)}', flush=True)
        ready = await prepare_text_index(engine)
        if engine.vector_block is not None:
            raise ValueError('vector index is not ready')
        if qdrant_url:
            from ..backends import SqliteDocumentStore
            from ..backends.qdrant import QdrantVectorIndex
            store = cast(SqliteDocumentStore, engine.documents)
            chunks = store.conn.execute('SELECT count(*) FROM chunks WHERE space = ?', (SPACE,)).fetchone()[0]
            vector_index = cast(QdrantVectorIndex, engine.vectors)
            points = (await vector_index.client.count(vector_index.collection, exact=True)).count
            if points != chunks:
                raise ValueError('vector count differs from corpus chunks')
            ready.update(chunks=chunks, vector_points=points)
        save(output/'index-ready.json', ready)
        print('Text and vector indexes ready before generation', flush=True)
        if qdrant_url:
            query_vectors = await prepare_query_vectors(engine, questions)
            save(output/'query-vectors.json', query_vectors)
            print('Shared query vectors ready for both arms', flush=True)
        with (output/'prepared.jsonl').open('x') as prepared_stream, (output/'answers.jsonl').open('x') as answers_stream:
            for index, question in enumerate(questions):
                for arm in ARMS if index % 2 == 0 else tuple(reversed(ARMS)):
                    observer.last = None
                    engine.reranker = observer if arm == 'hybrid_jev' else None
                    prepared = await prepare_question(engine, question, arm, documents)
                    prepared = prepared.model_copy(update={
                        'resolved_jev_model': observer.last.model if observer.last else None})
                    prepared_stream.write(prepared.model_dump_json()+'\n')
                    prepared_stream.flush()
                    if prepared.receipt['status'] == 'failed':
                        reply = {'status': 'context_failed', 'completed': False, 'answer_text': '', 'total_ms': 0.0}
                    else:
                        model = OpenAICompatibleTextModel('https://openrouter.ai/api/v1', MODEL, api_key=key,
                            temperature=0, think=False, timeout=60, trust_env=False, max_output_tokens=256)
                        reply = await capture_public_reply(model, prepared.request, timeout=65)
                    answer = Answer.model_validate({'id': question.id, 'arm': arm,
                        'request_sha256': prepared.request_sha256, **reply})
                    answers_stream.write(answer.model_dump_json()+'\n')
                    answers_stream.flush()
                    print(f'{index+1}/200 {arm}: {answer.status}', flush=True)
    finally:
        await engine.close()
    unchanged = (code_digest() == code and digest(protocol_path) == protocol
                 and all(digest(dataset/name) == sha for name, sha in inputs.items()))
    save(output/'completion.json', {'terminal': True, 'code_and_inputs_unchanged': unchanged,
        'manifest_sha256': digest(output/'manifest.json'), 'prepared_sha256': digest(output/'prepared.jsonl'),
        'answers_sha256': digest(output/'answers.jsonl'), 'index_ready_sha256': digest(output/'index-ready.json'),
        'query_vectors_sha256': digest(output/'query-vectors.json') if qdrant_url else None,
        'embedding_cache': cast(CachedEmbedder, engine.embedder).record() if qdrant_url else None})
    if not unchanged:
        raise ValueError('code or inputs changed during generation')


def score_answers(questions: list[Question], rows: list[Answer], gold: dict[str, Gold]) -> dict:
    from .public_qa import answer_score
    from .public_qa_score import latency

    expected = {(q.id, arm) for q in questions for arm in ARMS}
    if len(rows) != len(expected) or {(r.id, r.arm) for r in rows} != expected:
        raise ValueError('answer observations differ from complete paired schedule')
    if any(r.completed != (r.status == 'completed') for r in rows):
        raise ValueError('inconsistent completion status')
    datasets = {q.id: q.dataset for q in questions}
    if any(gold[q.id].dataset != q.dataset for q in questions):
        raise ValueError('gold dataset mismatch')
    scored = [{**r.model_dump(), 'dataset': datasets[r.id],
        **answer_score(r.answer_text, gold[r.id].answers, datasets[r.id], r.completed),
        'abstained': r.completed and r.answer_text.strip() == 'INSUFFICIENT_EVIDENCE'} for r in rows]
    groups = []
    for dataset in ('all', 'hotpotqa', 'squad'):
        for arm in ARMS:
            selected = [r for r in scored if r['arm'] == arm and (dataset == 'all' or r['dataset'] == dataset)]
            if not selected:
                continue
            groups.append({'dataset': dataset, 'arm': arm, 'n': len(selected),
                **{m: statistics.mean(r[m] for r in selected) for m in ('em', 'f1')},
                'failures': sum(not r['completed'] for r in selected),
                'abstentions': sum(r['abstained'] for r in selected),
                'latency': latency([r['total_ms'] for r in selected])})
    by_id = {(r['id'], r['arm']): r for r in scored}
    paired = {}
    for metric in ('em', 'f1'):
        deltas = [by_id[q.id, 'hybrid_jev'][metric] - by_id[q.id, 'hybrid'][metric] for q in questions]
        paired[metric] = {'wins': sum(d > 0 for d in deltas), 'losses': sum(d < 0 for d in deltas),
                          'ties': sum(d == 0 for d in deltas), 'mean_delta': statistics.mean(deltas)}
    return {'groups': groups, 'paired': paired, 'per_question': scored}


def score(dataset: Path, output: Path) -> None:
    from .public_qa import Gold

    questions = load_questions(dataset)
    manifest = json.loads((output/'manifest.json').read_text())
    completion = json.loads((output/'completion.json').read_text())
    if not completion['terminal'] or not completion['code_and_inputs_unchanged']:
        raise ValueError('incomplete or changed experiment')
    for name in ('manifest', 'prepared', 'answers'):
        suffix = '.json' if name == 'manifest' else '.jsonl'
        if digest(output/(name+suffix)) != completion[name+'_sha256']:
            raise ValueError('experiment artifacts changed')
    if ('index_ready_sha256' in completion
            and digest(output/'index-ready.json') != completion['index_ready_sha256']):
        raise ValueError('index readiness artifact changed')
    if (completion.get('query_vectors_sha256') is not None
            and digest(output/'query-vectors.json') != completion['query_vectors_sha256']):
        raise ValueError('query vector artifact changed')
    if (any(digest(dataset/name) != sha for name, sha in manifest['inputs'].items())
            or digest(dataset/'gold.jsonl') != manifest['gold_sha256']):
        raise ValueError('dataset integrity failure')
    prepared = [PreparedAnswer.model_validate_json(line) for line in (output/'prepared.jsonl').read_text().splitlines()]
    rows = [Answer.model_validate_json(line) for line in (output/'answers.jsonl').read_text().splitlines()]
    requests = {(p.question.id, p.arm): p for p in prepared}
    expected = {(q.id, arm) for q in questions for arm in ARMS}
    if len(prepared) != len(expected) or set(requests) != expected:
        raise ValueError('incomplete prepared schedule')
    originals = {q.id: q for q in questions}
    for row in rows:
        p = requests[row.id, row.arm]
        if (p.question != originals[row.id] or p.request_sha256 != request_digest(p.request)
                or row.request_sha256 != p.request_sha256):
            raise ValueError('generation request integrity failure')
    gold = {g.id: g for line in (dataset/'gold.jsonl').read_text().splitlines()
            for g in (Gold.model_validate_json(line),)}
    report = score_answers(questions, rows, gold)
    report['jev_applied'] = sum(p.resolved_jev_model is not None for p in prepared if p.arm == 'hybrid_jev')
    from .jev_public_qa import rank_metrics
    from .public_qa_score import latency
    report['resolved_jev_models'] = dict(Counter(p.resolved_jev_model for p in prepared
                                               if p.resolved_jev_model is not None))
    report['context'] = []
    answers = {(r.id, r.arm): r for r in rows}
    for dataset_name in ('all', 'hotpotqa', 'squad'):
        for arm in ARMS:
            selected = [p for p in prepared if p.arm == arm
                        and (dataset_name == 'all' or p.question.dataset == dataset_name)]
            metrics = [rank_metrics([s.document_id for s in p.sources],
                                   set(gold[p.question.id].support_documents)) for p in selected]
            report['context'].append({'dataset': dataset_name, 'arm': arm, 'n': len(selected),
                'recall_at_5': statistics.mean(m['recall_at_5'] for m in metrics),
                'all_at_5': statistics.mean(m['all_at_5'] for m in metrics),
                'context_status': dict(Counter(p.receipt['status'] for p in selected)),
                'prepare_latency': latency([p.prepare_ms for p in selected]),
                'end_to_end_latency': latency([p.prepare_ms + answers[p.question.id, arm].total_ms
                                              for p in selected])})
    save(output/'scores.json', report)
    print(json.dumps({k: v for k, v in report.items() if k != 'per_question'}, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=('run', 'score'))
    parser.add_argument('--dataset', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--qdrant-url', help='Use Qwen 8B embeddings with this local Qdrant server')
    parser.add_argument('--reuse-index', type=Path, help='Validate and reuse the corpus index from a prior Qwen run')
    args = parser.parse_args()
    if args.command == 'run':
        asyncio.run(run(args.dataset, args.output, args.qdrant_url, args.reuse_index))
    else:
        score(args.dataset, args.output)


if __name__ == '__main__':
    main()
