"""Frozen standalone Jev comparison of section and paragraph candidates; no gold."""
from __future__ import annotations

import argparse
import asyncio
import fcntl
from importlib.metadata import version
import json
from pathlib import Path
import platform
import sqlite3
import time

import httpx

from matched_qa.pipeline import OPENROUTER_JEV_MODEL, rank
from matched_qa.run import append, digest, save
from qasper_structure.data import Paper, Question
from qasper_structure.pipeline import passage, context
from qasper_structure.run import EMBED_MODEL, PreparedArm, snapshot, sources
from scone_memory.bench.comparative import CachedEmbedder
from scone_memory.ingestion.embedding_cache import SqliteEmbeddingCache
from scone_memory.retrieval.section_routing import ChoiceBatch, FetchDecision, RouteMenu, SectionRouter
from scone_memory.retrieval.structured_document import Mode, StructuredDocumentIndex

from .run import CachedOnly, Row

POLICIES: tuple[tuple[str, Mode], ...] = (
    ('structure_jev', 'local_structure'),
    ('hybrid_jev', 'local_hybrid'),
)


class NoDecisions:
    definition = 'local-only-no-model-decisions'

    async def choose(self, query: str, menus: tuple[RouteMenu, ...]) -> ChoiceBatch:
        raise AssertionError('local benchmark attempted model routing')

    async def choose_fetch(self, query: str, sections: tuple[tuple[str, int], ...],
                           max_bytes: int) -> FetchDecision:
        raise AssertionError('local benchmark attempted model fetch selection')


def source_hashes() -> dict[str, str]:
    root = Path(__file__).parents[2]
    extra = [Path(__file__), Path(__file__).with_name('COMPACT_PROTOCOL.md'),
             Path(__file__).with_name('run.py'), root / 'src/scone_memory/retrieval/lexical.py']
    return {**sources(), **{str(p.relative_to(root)): digest(p) for p in extra}}


async def paper_questions(paper: Paper, questions: list[Question], cached: CachedEmbedder,
                          ordinal: int, client: httpx.AsyncClient) -> list[Row]:
    started = time.perf_counter()
    current = snapshot(paper, development=True)
    index = await StructuredDocumentIndex.build(current, cached)
    preparation_ms = (time.perf_counter() - started) * 1000
    chooser = NoDecisions()
    router = SectionRouter(chooser, cache_size=0)

    async def one(question: Question, position: int) -> Row:
        shift = position % len(POLICIES)
        arms: dict[str, PreparedArm] = {}
        receipts: list[dict[str, object]] = []
        error = None
        for name, mode in POLICIES[shift:] + POLICIES[:shift]:
            started = time.perf_counter()
            result = await index.retrieve(question.question, current, router, chooser,
                                          mode=mode, limit=32, max_bytes=8000)
            candidates = list({p.key: p for p in (passage(p, paper.id) for p in result.evidence)}.values())
            retrieval_ms = (time.perf_counter() - started) * 1000
            started = time.perf_counter()
            packed = ''
            try:
                scores, receipt, _ = await rank(client, question.question, candidates, provider='openrouter')
                packed = context(candidates, scores)
                receipts.append({'arm': name, **receipt})
            except (httpx.HTTPError, TimeoutError, ValueError) as failure:
                error = type(failure).__name__
            elapsed = (time.perf_counter() - started) * 1000
            arms[name] = PreparedArm(context=packed, retrieval_ms=retrieval_ms,
                route_ms=0., fetch_ms=0., rerank_ms=elapsed, effective_mode=result.mode,
                route_reason=result.reason)
        return Row(id=question.id, paper_id=paper.id, completed=error is None, error=error,
                   arms=arms, receipts=receipts, route=None, paper_preparation_ms=preparation_ms)

    rows: list[Row] = []
    for offset in range(0, len(questions), 8):
        rows.extend(await asyncio.gather(*(one(q, ordinal + offset + i)
                    for i, q in enumerate(questions[offset:offset + 8]))))
    return rows


async def run(dataset: Path, previous: Path, output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.with_suffix('.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        await _run(dataset, previous, output)


async def _run(dataset: Path, previous: Path, output: Path) -> None:
    papers = [Paper.model_validate_json(x) for x in (dataset / 'corpus.jsonl').read_text().splitlines()]
    questions = [Question.model_validate_json(x) for x in (dataset / 'questions.jsonl').read_text().splitlines()]
    metadata = json.loads((dataset / 'dataset.json').read_text())
    paper_ids = {p.id for p in papers}
    if (len(papers) != 281 or len(paper_ids) != 281 or len(questions) != 1005
            or len({q.id for q in questions}) != 1005 or metadata['split'] != 'dev'
            or any(q.paper_id not in paper_ids for q in questions)):
        raise ValueError('complete unique development split required')
    inputs = {name: digest(dataset / name) for name in ('corpus.jsonl', 'questions.jsonl', 'dataset.json')}
    if any(inputs[name] != metadata['files'][name]['sha256'] for name in ('corpus.jsonl', 'questions.jsonl')):
        raise ValueError('export digest mismatch')
    if json.loads((previous / 'manifest.json').read_text())['inputs'] != inputs:
        raise ValueError('cache reference dataset mismatch')
    frozen = source_hashes()
    manifest = {'protocol': 'compact-paragraph-jev-dev-v1', 'source_sha256': frozen,
        'inputs': inputs, 'arms': [name for name, _ in POLICIES], 'embedding_model': EMBED_MODEL,
        'python': platform.python_version(),
        'versions': {name: version(name) for name in ('numpy', 'pydantic', 'httpx')},
        'embedding_cache_sha256': digest(previous / 'embeddings.db'),
        'previous_observations_sha256': digest(previous / 'observations.jsonl'),
        'context_items': 5, 'context_bytes': 8000, 'candidate_items': 32,
        'candidate_bytes': 8000, 'jev_model': OPENROUTER_JEV_MODEL,
        'scheduled': [{'id': q.id, 'paper_id': q.paper_id} for q in questions]}
    if output.exists() and any(output.iterdir()):
        raise ValueError('output not empty')
    output.mkdir(parents=True, exist_ok=True)
    save(output / 'manifest.json', manifest)
    with sqlite3.connect('file:' + str(previous / 'embeddings.db') + '?mode=ro', uri=True) as source:
        with sqlite3.connect(output / 'embeddings.db') as target:
            source.backup(target)
    remote = CachedOnly('https://openrouter.ai/api/v1', EMBED_MODEL, dim=2048,
                        query_prefix='query: ', document_prefix='passage: ')
    cache = SqliteEmbeddingCache(output / 'embeddings.db', max_entries=100000)
    cached = CachedEmbedder(remote, cache)
    count = 0
    try:
        async with httpx.AsyncClient(trust_env=False, follow_redirects=False) as client:
            with (output / 'observations.jsonl').open('a') as stream:
                for number, paper in enumerate(papers, 1):
                    selected = [q for q in questions if q.paper_id == paper.id]
                    for row in await paper_questions(paper, selected, cached, count, client):
                        append(stream, row.model_dump())
                        count += 1
                    print(f'Papers {number}/281; questions {count}/1005', flush=True)
        if source_hashes() != frozen or any(digest(dataset / name) != value for name, value in inputs.items()):
            raise ValueError('source or inputs changed during inference')
        if (digest(previous / 'embeddings.db') != manifest['embedding_cache_sha256']
                or digest(previous / 'observations.jsonl') != manifest['previous_observations_sha256']):
            raise ValueError('prior reference changed during inference')
        save(output / 'completion.json', {'complete': count == 1005, 'questions': count,
            'manifest_sha256': digest(output / 'manifest.json'),
            'observations_sha256': digest(output / 'observations.jsonl'), 'embedding_cache': cached.record()})
    finally:
        await cache.close()
        await remote.close()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', type=Path, required=True)
    parser.add_argument('--previous', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    asyncio.run(run(args.dataset, args.previous, args.output))


if __name__ == '__main__':
    main()
