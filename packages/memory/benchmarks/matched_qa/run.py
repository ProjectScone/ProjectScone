"""Complete matched-model evaluation with durable, non-retrying answer attempts."""
from __future__ import annotations

import argparse
import asyncio
from collections import Counter
import fcntl
import hashlib
from importlib.metadata import version
import json
import os
from pathlib import Path
import time
from typing import TextIO, cast

import httpx
from llama_index.core.schema import QueryBundle

from scone_memory import MemoryEngine
from scone_memory.backends import SqliteDocumentStore
from scone_memory.backends.qdrant import QdrantVectorIndex
from scone_memory.bench.comparative import CachedEmbedder
from scone_memory.embedders.remote import RemoteEmbedder
from scone_memory.ingestion.embedding_cache import SqliteEmbeddingCache
from scone_memory.testing.public_qa import Document, Question, _mapping

from .indexes import SPACE, Indices, build, validate_url, warm_vectors
from .pipeline import EMBED_MODEL, MODEL, QUERY_PREFIX, Passage, credential, error_name, generate, messages, pack_context, rank, rerank, unique_passages

ARMS = ('scone', 'llamaindex')


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save(path: Path, data: object) -> None:
    pending = path.with_suffix(path.suffix + '.pending')
    pending.write_text(json.dumps(data, indent=2, allow_nan=False) + '\n')
    pending.replace(path)


def append(stream: TextIO, data: object) -> None:
    stream.write(json.dumps(data, allow_nan=False) + '\n')
    stream.flush()
    os.fsync(stream.fileno())


def source_hashes() -> dict[str, str]:
    base = Path(__file__).resolve().parent
    return {str(path.relative_to(base.parent.parent)): digest(path)
        for path in sorted(list(base.glob('*.py')) + [base / 'PROTOCOL.md']
            + list((base.parent.parent / 'src/scone_memory').rglob('*.py')))}


def previous_rows(path: Path) -> dict[tuple[str, str], dict[str, object]]:
    rows: dict[tuple[str, str], dict[str, object]] = {}
    if not path.exists():
        return rows
    with path.open() as stream:
        for line in stream:
            row = _mapping(json.loads(line))
            if not isinstance(row.get('id'), str) or row.get('arm') not in ARMS:
                raise ValueError('invalid journal record')
            key = (cast(str, row['id']), cast(str, row['arm']))
            if key in rows:
                raise ValueError('duplicate journal record')
            rows[key] = row
    return rows


def repair_journal_tail(path: Path) -> None:
    """Preserve torn bytes before removing only an incomplete final record.

    Answer attempts are fsynced before any request. A discarded answer tail
    therefore becomes an interrupted failure, never a second answer request.
    """
    if not path.exists() or path.stat().st_size == 0:
        return
    with path.open('rb+') as stream:
        stream.seek(-1, os.SEEK_END)
        if stream.read(1) == b'\n':
            return
        stream.seek(0)
        offset = 0
        for line in stream:
            if line.endswith(b'\n'):
                offset += len(line)
                continue
            try:
                _mapping(json.loads(line))
            except (ValueError, UnicodeDecodeError):
                preserved = path.with_name(path.name + '.torn-' + hashlib.sha256(line).hexdigest()[:16])
                preserved.write_bytes(line)
                stream.truncate(offset)
            else:
                stream.seek(0, os.SEEK_END)
                stream.write(b'\n')
            stream.flush()
            os.fsync(stream.fileno())
            return


def failed_row(identifier: str, arm: str, error: str) -> dict[str, object]:
    return {'id': identifier, 'arm': arm, 'completed': False, 'answer': '', 'error': error,
            'retrieved_ids': [], 'context_ids': [], 'retrieval_ms': 0.0,
            'rerank_ms': 0.0, 'generation_ms': 0.0, 'total_ms': 0.0}


async def evaluate(indices: Indices, cached: CachedEmbedder, questions: list[Question],
                   output: Path, concurrency: int) -> None:
    for name in ('observations.jsonl', 'attempts.jsonl', 'judgments.jsonl'):
        repair_journal_tail(output / name)
    old = previous_rows(output / 'observations.jsonl')
    attempted = previous_rows(output / 'attempts.jsonl')
    expected = {(q.id, arm) for q in questions for arm in ARMS}
    if not set(old) <= set(attempted) <= expected:
        raise ValueError('journal differs from the planned schedule')
    with (output / 'observations.jsonl').open('a') as stream, (output / 'attempts.jsonl').open('a') as attempts, \
            (output / 'judgments.jsonl').open('a') as judgments:
        # An in-flight answer is never silently retried after interruption.
        for identifier, arm in sorted(set(attempted) - set(old)):
            row = failed_row(identifier, arm, 'interrupted_attempt')
            append(stream, row)
            old[identifier, arm] = row
        completed_count = len(old)
        async with httpx.AsyncClient(trust_env=False, follow_redirects=False) as client:
            async def question_run(position: int, question: Question) -> None:
                nonlocal completed_count
                arms = tuple(arm for arm in (ARMS if position % 2 == 0 else tuple(reversed(ARMS)))
                             if (question.id, arm) not in old)
                if not arms:
                    return
                for arm in arms:
                    append(attempts, {'id': question.id, 'arm': arm})
                candidates: dict[str, list[Passage]] = {}
                retrieval_times: dict[str, float] = {}
                errors: dict[str, str] = {}
                query_vector = (await cached.embed_queries([question.question.strip()]))[0]
                for arm in arms:
                    started = time.perf_counter()
                    try:
                        if arm == 'scone':
                            recalled = await indices.engine.recall(SPACE, question.question, limit=32,
                                                                   candidate_limit=64, rerank=False)
                            if recalled.degraded:
                                raise ValueError('degraded retrieval')
                            candidates[arm] = [indices.passages[str(item.chunk_id)] for item in recalled.items]
                        else:
                            found = await asyncio.to_thread(indices.reference.retrieve,
                                QueryBundle(query_str=question.question.strip(), embedding=query_vector))
                            candidates[arm] = [indices.passages[indices.reference_keys[node.node.node_id]] for node in found]
                    except Exception as error:
                        errors[arm] = type(error).__name__
                        candidates[arm] = []
                    retrieval_times[arm] = (time.perf_counter() - started) * 1000
                scores: dict[str, float] = {}
                rerank_ms = 0.0
                audit: dict[str, object]
                started = time.perf_counter()
                try:
                    scores, audit, rerank_ms = await rank(client, question.question, unique_passages(candidates))
                except Exception as error:
                    rerank_ms = (time.perf_counter() - started) * 1000
                    audit = {'error': error_name(error), 'question': question.question,
                             'candidates': [p.__dict__ for p in unique_passages(candidates)]}
                    errors.update({arm: 'rerank_' + error_name(error) for arm in arms})
                append(judgments, {'id': question.id, 'arms': list(arms), **audit})
                for arm in arms:
                    context_ids: list[str] = []
                    context = ''
                    reply: dict[str, object]
                    if arm in errors:
                        reply = {'completed': False, 'answer': '', 'error': errors[arm], 'generation_ms': 0.0}
                    else:
                        context, context_ids = pack_context(rerank(candidates[arm], scores))
                        reply = await generate(client, messages(question.question, context))
                    generation_ms = float(cast(float, reply['generation_ms']))
                    row = {'id': question.id, 'arm': arm, **reply,
                        'retrieved_ids': [p.document_id for p in candidates[arm]],
                        'retrieved_chunk_ids': [p.key for p in candidates[arm]],
                        'context_ids': context_ids, 'context_bytes': len(context.encode()),
                        'retrieval_ms': retrieval_times[arm], 'rerank_ms': rerank_ms,
                        'total_ms': retrieval_times[arm] + rerank_ms + generation_ms}
                    append(stream, row)
                    completed_count += 1
                    if completed_count % 20 == 0:
                        save(output / 'progress.json', {'phase': 'answers', 'terminal': completed_count,
                            'planned': len(expected), 'last_id': question.id})
                        print(f'Answers {completed_count}/{len(expected)}', flush=True)
            pending = [(i, q) for i, q in enumerate(questions) if any((q.id, arm) not in old for arm in ARMS)]
            for offset in range(0, len(pending), concurrency):
                async with asyncio.TaskGroup() as group:
                    for position, question in pending[offset:offset + concurrency]:
                        group.create_task(question_run(position, question))


async def run(dataset: Path, output: Path, url: str, concurrency: int, resume: bool) -> None:
    validate_url(url)
    if not 1 <= concurrency <= 8:
        raise ValueError('concurrency must be 1..8')
    credential('chat')
    credential('embed')
    for name in ('TYPESAFE_API_KEY',):
        if not os.environ.get(name):
            raise ValueError('required API credentials missing')
    if version('llama-index-core') != '0.14.24':
        raise ValueError('use the protocol-pinned LlamaIndex version')
    data = json.loads((dataset / 'dataset.json').read_text())
    inputs = {name: digest(dataset / name) for name in ('dataset.json', 'corpus.jsonl', 'questions.jsonl')}
    if any(inputs[name] != data['files'][name] for name in ('corpus.jsonl', 'questions.jsonl')):
        raise ValueError('input integrity failure')
    inputs['gold.jsonl'] = data['files']['gold.jsonl']
    documents = [Document.model_validate_json(line) for line in (dataset / 'corpus.jsonl').read_text().splitlines()]
    questions = [Question.model_validate_json(line) for line in (dataset / 'questions.jsonl').read_text().splitlines()]
    if not questions or len({q.id for q in questions}) != len(questions):
        raise ValueError('invalid question schedule')
    if Counter(q.dataset for q in questions) != {'hotpotqa': 7405, 'squad': 10570} or len(documents) != 68702:
        raise ValueError('the full runner requires every question and all 68702 corpus documents')
    output.mkdir(parents=True, exist_ok=resume)
    with (output / '.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        sources = source_hashes()
        manifest = {'protocol': 'matched-qa-full-v1', 'input_hashes': inputs, 'source_sha256': sources,
            'scheduled': [{'id': q.id, 'arm': arm} for q in questions for arm in ARMS],
            'versions': {name: version(name) for name in ('llama-index-core', 'llama-index-retrievers-bm25',
                'llama-index-vector-stores-qdrant', 'qdrant-client', 'httpx')},
            'chat_model': MODEL, 'embedding_model': EMBED_MODEL, 'dimensions': 4096,
            'jev_model': os.environ.get('TYPESAFE_DEFAULT_MODEL', 'jev-latest'), 'qdrant_url': url,
            'chunk_target': 700, 'lane_candidates': 64, 'fused_candidates': 32,
            'context_chunks': 5, 'context_bytes': 8000, 'warm_query_embeddings': True,
            'concurrency': concurrency,
            'scope': 'complete development splits, shared Scone chunking and answer/reranking stages'}
        if resume:
            if json.loads((output / 'manifest.json').read_text()) != manifest:
                raise ValueError('resume requires unchanged sources, inputs, versions and configuration')
        else:
            save(output / 'manifest.json', manifest)
        remote = RemoteEmbedder('https://openrouter.ai/api/v1', EMBED_MODEL,
            api_key=credential('embed'), dim=4096, query_prefix=QUERY_PREFIX, trust_env=False)
        async def embedding_started(request: httpx.Request) -> None:
            request.extensions['benchmark_started'] = time.perf_counter()

        async def embedding_finished(response: httpx.Response) -> None:
            await response.aread()
            packet: dict[str, object] = {}
            try:
                packet = _mapping(response.json())
            except ValueError:
                pass
            with (output / 'embedding-calls.jsonl').open('a') as log:
                append(log, {'status': response.status_code, 'model': packet.get('model'),
                    'usage': packet.get('usage'), 'elapsed_ms': (time.perf_counter()
                        - cast(float, response.request.extensions['benchmark_started'])) * 1000})

        remote._client = httpx.AsyncClient(timeout=remote.timeout, trust_env=False,
            event_hooks={'request': [embedding_started], 'response': [embedding_finished]})
        cache = SqliteEmbeddingCache(output / 'embeddings.db', max_entries=500000)
        cached = CachedEmbedder(remote, cache)
        engine = MemoryEngine(SqliteDocumentStore(output / 'memory.db'),
            QdrantVectorIndex(url, 'matched_scone'), cached, chunk_target=700, candidate_limit=64)
        complete = False
        failure: str | None = None
        indices: Indices | None = None
        try:
            started = time.perf_counter()
            save(output / 'progress.json', {'phase': 'embedding', 'questions': len(questions), 'documents': len(documents)})
            await warm_vectors(cached, documents, questions, concurrency)
            await engine.open()
            save(output / 'progress.json', {'phase': 'indexing', 'questions': len(questions), 'documents': len(documents)})
            indices = await build(engine, cached, documents, output, url)
            save(output / 'index.json', {'documents': len(documents), 'chunks': len(indices.passages),
                'preparation_ms_this_process': (time.perf_counter() - started) * 1000,
                'embedding_cache': cached.record(), 'scone_vector_weight': engine.vector_weight})
            await evaluate(indices, cached, questions, output, concurrency)
            complete = len(previous_rows(output / 'observations.jsonl')) == len(questions) * 2
        except BaseException as error:
            failure = type(error).__name__
            raise
        finally:
            unchanged = sources == source_hashes() and all(digest(dataset / name) == sha
                for name, sha in inputs.items() if name != 'gold.jsonl')
            save(output / 'completion.json', {'completed': complete, 'error': failure,
                'code_and_inputs_unchanged': unchanged, 'embedding_cache': cached.record(),
                'artifact_hashes': {name: digest(output / name)
                    for name in ('manifest.json', 'observations.jsonl', 'attempts.jsonl', 'judgments.jsonl', 'index.json', 'embedding-calls.jsonl')
                    if (output / name).exists()}})
            if indices is not None:
                indices.client.close()
            await engine.close()
            await remote.close()
            await cache.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--qdrant-url', default='http://127.0.0.1:16437')
    parser.add_argument('--concurrency', type=int, default=4)
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()
    asyncio.run(run(args.dataset, args.output, args.qdrant_url, args.concurrency, args.resume))
