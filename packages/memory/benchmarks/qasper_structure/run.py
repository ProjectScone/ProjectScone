"""Run the complete frozen QASPER test schedule with hosted models and local indices."""
from __future__ import annotations

import argparse
import asyncio
from dataclasses import asdict
import fcntl
from importlib.metadata import version
import json
from pathlib import Path
import time
from typing import Literal, Sequence, cast

import httpx
from pydantic import BaseModel, ConfigDict

from scone_memory.bench.comparative import CachedEmbedder
from scone_memory.embedders.remote import RemoteEmbedder
from scone_memory.ingestion.embedding_cache import SqliteEmbeddingCache
from scone_memory.providers.jev_sections import JevSectionChooser
from scone_memory.retrieval.section_routing import SectionRouter, SectionSnapshot
from scone_memory.retrieval.structured_document import Routing, StructuredDocumentIndex, StructuredResult
from matched_qa.pipeline import MODEL, OPENROUTER_JEV_MODEL, Passage, credential, generate, rank
from matched_qa.run import append, digest, repair_journal_tail, save
from .data import DEV_COUNTS, EXPECTED_COUNTS, Paper, Question
from .pipeline import context, messages, passage, reference, reference_candidates

ARMS = ('scone_flat', 'scone_structure', 'llamaindex')
DEV_ARMS = ('scone_flat', 'scone_structure', 'scone_vector_candidates', 'llamaindex')
EMBED_MODEL = 'nvidia/nemotron-3-embed-1b:free'


class PreparedArm(BaseModel):
    model_config = ConfigDict(extra='forbid', strict=True)
    context: str
    retrieval_ms: float
    route_ms: float
    fetch_ms: float
    rerank_ms: float
    effective_mode: str
    route_reason: str | None


class PreparedQuestion(BaseModel):
    model_config = ConfigDict(extra='forbid', strict=True)
    id: str
    paper_id: str
    arms: dict[str, PreparedArm]


class CaptureEmbedder:
    id = 'preparation-only'
    dim = 1
    def __init__(self) -> None:
        self.texts: list[str] = []
    async def embed(self, texts: Sequence[str]) -> list[list[float]]:
        self.texts.extend(texts)
        return [[1.0] for _ in texts]


def snapshot(paper: Paper, *, development: bool = False) -> SectionSnapshot:
    scope = 'qasper-dev-v0.3' if development else 'qasper-test-v0.3'
    return SectionSnapshot.from_markdown(scope, paper.id, paper.content)


def sources() -> dict[str, str]:
    base = Path(__file__).resolve().parents[2]
    paths = list((base / 'src/scone_memory').rglob('*.py'))
    paths += list(Path(__file__).parent.glob('*.py')) + [Path(__file__).with_name(name) for name in ('PROTOCOL.md', 'DEV_PROTOCOL.md')]
    paths += list((base / 'benchmarks/matched_qa').glob('*.py'))
    return {str(p.relative_to(base)): digest(p) for p in sorted(paths)}


def records(path: Path) -> list[dict[str, object]]:
    if not path.exists():
        return []
    repair_journal_tail(path)
    rows: list[dict[str, object]] = []
    for line in path.read_text().splitlines():
        value: object = json.loads(line)
        if not isinstance(value, dict):
            raise ValueError('invalid journal row')
        rows.append(value)
    return rows


async def warm(papers: list[Paper], questions: list[Question], cached: CachedEmbedder, *,
               development: bool = False) -> None:
    capture = CaptureEmbedder()
    for paper in papers:
        await StructuredDocumentIndex.build(snapshot(paper, development=development), capture)
    texts = list(dict.fromkeys(capture.texts))
    print(f'Preparing {len(texts)} distinct passages and {len(questions)} questions', flush=True)
    for name, values in (('passages', texts), ('queries', [q.question for q in questions])):
        for offset in range(0, len(values), 128):
            batch = values[offset:offset + 128]
            if name == 'queries':
                await cached.embed_queries(batch)
            else:
                await cached.embed(batch)
            if offset % 1024 == 0:
                print(f'Vectors {name} {min(offset + 128, len(values))}/{len(values)}', flush=True)


async def evaluate(papers: list[Paper], questions: list[Question], output: Path,
                   cached: CachedEmbedder, concurrency: int, *, development: bool = False) -> None:
    schedule_arms = DEV_ARMS if development else ARMS
    old = records(output / 'observations.jsonl')
    old_pairs = {(row['id'], row['arm']) for row in old}
    schedule = {(q.id, arm) for q in questions for arm in schedule_arms}
    if len(old_pairs) != len(old) or not old_pairs <= schedule:
        raise ValueError('observation journal differs from schedule')
    # Resume never replaces an answer that was already persisted, successful or failed.
    attempts = records(output / 'attempts.jsonl')
    attempted = {(row['id'], row['arm']) for row in attempts}
    if len(attempted) != len(attempts) or not old_pairs <= attempted <= schedule:
        raise ValueError('attempt journal differs from observations/schedule')
    if attempted != old_pairs:
        raise ValueError('unfinished provider attempt requires explicit recovery; no silent generation retry')
    prepared_rows = [PreparedQuestion.model_validate(item) for item in records(output / 'prepared.jsonl')]
    prepared = {row.id: row for row in prepared_rows}
    if len(prepared) != len(prepared_rows):
        raise ValueError('duplicate prepared question')
    records(output / 'judgments.jsonl')
    if not set(prepared) <= {q.id for q in questions}:
        raise ValueError('prepared context differs from schedule')
    question_papers = {q.id: q.paper_id for q in questions}
    if any(row.paper_id != question_papers[row.id] or set(row.arms) != set(schedule_arms)
           for row in prepared.values()):
        raise ValueError('prepared question identity/arms mismatch')
    count = len(old)
    ordinals = {question.id: ordinal for ordinal, question in enumerate(questions)}
    by_paper = {p.id: [q for q in questions if q.paper_id == p.id] for p in papers}
    with (output / 'observations.jsonl').open('a') as observations, \
            (output / 'attempts.jsonl').open('a') as attempt_log, \
            (output / 'judgments.jsonl').open('a') as judgments, \
            (output / 'prepared.jsonl').open('a') as prepared_log:
        async with httpx.AsyncClient(trust_env=False, follow_redirects=False) as client, \
                JevSectionChooser(api_key=credential('chat')) as chooser:
            for position, paper in enumerate(papers):
                pending = [q for q in by_paper[paper.id] if any((q.id, a) not in old_pairs for a in schedule_arms)]
                if not pending:
                    continue
                current = snapshot(paper, development=development)
                index = await StructuredDocumentIndex.build(current, cached)
                ref = await reference(index, cached)
                # Isolated queries measure cold route inference; no cache advantage across arms.
                router = SectionRouter(chooser, cache_size=0)
                async def one(question: Question, ordinal: int) -> None:
                    nonlocal count
                    prepared_question = prepared.get(question.id)
                    if prepared_question is None:
                        vector = (await cached.embed_queries([question.question]))[0]
                        started = time.perf_counter()
                        flat = await index.retrieve(question.question, current, router, chooser,
                                                    mode='flat_vector', limit=32, max_bytes=8000)
                        flat_ms = (time.perf_counter() - started) * 1000
                        policies: list[tuple[str, Routing]] = [('scone_structure', 'hierarchy')]
                        if development:
                            policies.append(('scone_vector_candidates', 'vector_candidates'))
                            if ordinal % 2:
                                policies.reverse()
                        routed: dict[str, StructuredResult] = {}
                        route_times: dict[str, float] = {}
                        for arm, policy in policies:
                            started = time.perf_counter()
                            routed[arm] = await index.retrieve(question.question, current, router, chooser,
                                mode='auto', limit=32, max_bytes=8000, routing=policy)
                            route_times[arm] = (time.perf_counter() - started) * 1000
                        li, li_ms = await asyncio.to_thread(reference_candidates, ref, index, question.question, vector)
                        candidates = {
                            'scone_flat': [passage(item, paper.id) for item in flat.evidence],
                            **{arm: [passage(item, paper.id) for item in result.evidence]
                               for arm, result in routed.items()},
                            'llamaindex': [passage(item, paper.id) for item in li]}
                        unique: dict[str, Passage] = {p.key: p for values in candidates.values() for p in values}
                        ordered = [unique[key] for key in sorted(unique)]
                        scores: dict[str, float] = {}
                        rerank_ms = 0.0
                        for offset in range(0, len(ordered), 64):
                            new_scores, receipt, elapsed = await rank(client, question.question,
                                ordered[offset:offset + 64], provider='openrouter')
                            scores.update(new_scores)
                            rerank_ms += elapsed
                            append(judgments, {'id': question.id, 'stage': 'shared_rerank', **receipt})
                        for arm, result in routed.items():
                            append(judgments, {'id': question.id, 'stage': 'structure', 'arm': arm,
                                'route': asdict(result.route) if result.route else None,
                                'fetch': asdict(result.fetch) if result.fetch else None,
                                'mode': result.mode, 'reason': result.reason})
                        times = {'scone_flat': flat_ms, **route_times, 'llamaindex': li_ms}
                        prepared_arms: dict[str, PreparedArm] = {}
                        for arm in schedule_arms:
                            selected = routed.get(arm)
                            prepared_arms[arm] = PreparedArm(
                                context=context(candidates[arm], scores), retrieval_ms=times[arm],
                                route_ms=selected.route.elapsed_ms if selected and selected.route else 0.0,
                                fetch_ms=selected.fetch_decision_ms if selected else 0.0,
                                rerank_ms=rerank_ms, effective_mode=selected.mode if selected else arm,
                                route_reason=selected.reason if selected else None)
                        prepared_question = PreparedQuestion(id=question.id, paper_id=paper.id,
                                                             arms=prepared_arms)
                        append(prepared_log, prepared_question.model_dump())
                        prepared[question.id] = prepared_question
                    if prepared_question.paper_id != paper.id or set(prepared_question.arms) != set(schedule_arms):
                        raise ValueError('prepared question identity/arms mismatch')
                    arms = schedule_arms[ordinal % len(schedule_arms):] + schedule_arms[:ordinal % len(schedule_arms)]
                    for arm in arms:
                        if (question.id, arm) in old_pairs:
                            continue
                        chosen = prepared_question.arms[arm]
                        packed = chosen.context
                        append(attempt_log, {'id': question.id, 'paper_id': paper.id, 'arm': arm})
                        reply = await generate(client, messages(question.question, packed))
                        generation_ms = float(cast(float, reply['generation_ms']))
                        row = {'id': question.id, 'paper_id': paper.id, 'arm': arm, **reply,
                            'context_text': packed, 'context_bytes': len(packed.encode()),
                            'evidence_paragraphs': list(dict.fromkeys(p.text for p in paper.paragraphs
                                if p.text and p.text in packed)),
                            'retrieval_ms': chosen.retrieval_ms, 'route_ms': chosen.route_ms,
                            'fetch_ms': chosen.fetch_ms, 'rerank_ms': chosen.rerank_ms,
                            'total_ms': chosen.retrieval_ms + chosen.rerank_ms + generation_ms,
                            'effective_mode': chosen.effective_mode, 'route_reason': chosen.route_reason}
                        append(observations, row)
                        old_pairs.add((question.id, arm))
                        count += 1
                        if count % 30 == 0:
                            print(f'Answers {count}/{len(schedule)}', flush=True)
                        if reply['completed'] is not True and reply['error'] != 'incomplete_generation':
                            raise RuntimeError('answer provider failure; completed observations retained')
                for start in range(0, len(pending), concurrency):
                    outcomes = await asyncio.gather(*(one(q, ordinals[q.id] if development else start + i) for i, q in enumerate(
                        pending[start:start + concurrency])), return_exceptions=True)
                    failures = [v for v in outcomes if isinstance(v, BaseException)]
                    if failures:
                        raise RuntimeError('evaluation paused: ' + type(failures[0]).__name__) from failures[0]
                save(output / 'progress.json', {'phase': 'answers', 'papers': position + 1,
                                                'observations': count, 'planned': len(schedule)})
                print(f'Papers {position + 1}/{len(papers)}; answers {count}/{len(schedule)}', flush=True)
    if old_pairs != schedule:
        raise ValueError('complete schedule was not executed')


async def run(dataset: Path, output: Path, concurrency: int, resume: bool, *,
              split: Literal['test', 'dev'] = 'test') -> None:
    if split not in ('test', 'dev'):
        raise ValueError('unsupported QASPER split')
    development = split == 'dev'
    schedule_arms = DEV_ARMS if development else ARMS
    expected_papers, expected_questions = DEV_COUNTS if development else EXPECTED_COUNTS
    if not 1 <= concurrency <= 8:
        raise ValueError('concurrency must be 1..8')
    papers = [Paper.model_validate_json(line) for line in (dataset / 'corpus.jsonl').read_text().splitlines()]
    questions = [Question.model_validate_json(line) for line in (dataset / 'questions.jsonl').read_text().splitlines()]
    if len(papers) != expected_papers or len(questions) != expected_questions:
        raise ValueError('the complete official QASPER ' + split + ' split is required')
    paper_ids, question_ids = {p.id for p in papers}, {q.id for q in questions}
    if len(paper_ids) != expected_papers or len(question_ids) != expected_questions or any(q.paper_id not in paper_ids for q in questions):
        raise ValueError('invalid dataset identities')
    inputs = {name: digest(dataset / name) for name in ('dataset.json', 'corpus.jsonl', 'questions.jsonl')}
    metadata = json.loads((dataset / 'dataset.json').read_text())
    if metadata.get('split') != split:
        raise ValueError('dataset split does not match requested protocol')
    for name in ('corpus.jsonl', 'questions.jsonl'):
        if metadata['files'][name]['sha256'] != inputs[name]:
            raise ValueError('dataset export digest mismatch')
    frozen_sources = sources()
    manifest = {'protocol': 'qasper-vector-development-v1' if development else 'qasper-structure-full-v1',
        'split': split, 'input_sha256': inputs, 'source_sha256': frozen_sources,
        'questions': len(questions), 'papers': len(papers), 'arms': schedule_arms,
        'scheduled': [{'id': q.id, 'paper_id': q.paper_id, 'arm': arm} for q in questions for arm in schedule_arms],
        'embedding_model': EMBED_MODEL, 'dimensions': 2048, 'query_prefix': 'query: ',
        'document_prefix': 'passage: ', 'jev_model': OPENROUTER_JEV_MODEL, 'answer_model': MODEL,
        'candidate_limit': 32, 'candidate_bytes': 8000, 'context_limit': 5, 'context_bytes': 8000,
        'concurrency': concurrency, 'query_vectors': 'precomputed shared; exclude preparation from query timings',
        'versions': {name: version(name) for name in ('llama-index-core', 'llama-index-retrievers-bm25',
            'httpx', 'pydantic', 'bm25s', 'numpy')}}
    # JSON's list representation must compare identically during resume.
    manifest = json.loads(json.dumps(manifest))
    if output.exists() and any(output.iterdir()) and not resume:
        raise ValueError('output is not empty; use resume only with the unchanged frozen run')
    output.mkdir(parents=True, exist_ok=True)
    with (output / 'run.lock').open('w') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if resume:
            if json.loads((output / 'manifest.json').read_text()) != manifest:
                raise ValueError('resume source/input/configuration mismatch')
        else:
            save(output / 'manifest.json', manifest)
        records(output / 'embedding-calls.jsonl')
        records(output / 'preparation.jsonl')
        remote = RemoteEmbedder('https://openrouter.ai/api/v1', EMBED_MODEL, api_key=credential('embed'),
            dim=2048, query_prefix='query: ', document_prefix='passage: ', trust_env=False)
        last_request = 0.0
        gate = asyncio.Lock()
        async def before(request: httpx.Request) -> None:
            nonlocal last_request
            async with gate:
                await asyncio.sleep(max(0., last_request + 3.2 - time.perf_counter()))
                last_request = time.perf_counter()
            request.extensions['start'] = time.perf_counter()
        async def after(response: httpx.Response) -> None:
            await response.aread()
            packet: object = response.json()
            if not isinstance(packet, dict):
                raise ValueError('invalid embedding packet')
            with (output / 'embedding-calls.jsonl').open('a') as stream:
                append(stream, {'status': response.status_code, 'model': packet.get('model'),
                    'usage': packet.get('usage'), 'elapsed_ms': (time.perf_counter() -
                    cast(float, response.request.extensions['start'])) * 1000})
        remote._client = httpx.AsyncClient(timeout=60, trust_env=False,
            event_hooks={'request': [before], 'response': [after]})
        cache = SqliteEmbeddingCache(output / 'embeddings.db', max_entries=100000)
        cached = CachedEmbedder(remote, cache)
        try:
            save(output / 'progress.json', {'phase': 'vectors', 'planned': len(questions) * len(schedule_arms)})
            started = time.perf_counter()
            await warm(papers, questions, cached, development=development)
            with (output / 'preparation.jsonl').open('a') as preparation_log:
                append(preparation_log, {'elapsed_ms': (time.perf_counter() - started) * 1000,
                                        'embedding_cache': cached.record(), 'resume': resume})
            await evaluate(papers, questions, output, cached, concurrency, development=development)
            if sources() != frozen_sources or any(digest(dataset / name) != expected for name, expected in inputs.items()):
                raise ValueError('sources or inputs changed during evaluation')
            save(output / 'completion.json', {'completed': True, 'questions': len(questions),
                'observations': len(questions) * len(schedule_arms), 'code_and_inputs_unchanged': True,
                'artifact_sha256': {name: digest(output / name) for name in ('manifest.json',
                    'observations.jsonl', 'attempts.jsonl', 'prepared.jsonl', 'judgments.jsonl', 'embedding-calls.jsonl', 'preparation.jsonl')}})
            save(output / 'progress.json', {'phase': 'complete', 'observations': len(questions) * len(schedule_arms)})
        except BaseException as error:
            save(output / 'progress.json', {'phase': 'paused', 'error_type': type(error).__name__,
                'observations': len(records(output / 'observations.jsonl'))})
            raise
        finally:
            await remote.close()
            await cache.close()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--concurrency', type=int, default=4)
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--split', choices=('test', 'dev'), default='test')
    args = parser.parse_args()
    asyncio.run(run(args.dataset, args.output, args.concurrency, args.resume, split=args.split))


if __name__ == '__main__':
    main()
