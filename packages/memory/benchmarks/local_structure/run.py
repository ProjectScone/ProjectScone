"""Frozen retrieval/rerank development comparison; no gold or answer generation."""
from __future__ import annotations

import argparse
import asyncio
import fcntl
from dataclasses import asdict
from contextlib import nullcontext
from importlib.metadata import version
import json
from pathlib import Path
import sqlite3
import time
from typing import Protocol, Sequence

import httpx
from pydantic import BaseModel, ConfigDict

from matched_qa.pipeline import OPENROUTER_JEV_MODEL, Passage, credential, rank
from matched_qa.run import append, digest, save
from qasper_structure.data import Paper, Question
from qasper_structure.pipeline import context, passage, reference, reference_candidates
from qasper_structure.run import EMBED_MODEL, PreparedArm, snapshot, sources
from scone_memory.bench.comparative import CachedEmbedder
from scone_memory.embedders.remote import RemoteEmbedder
from scone_memory.ingestion.embedding_cache import SqliteEmbeddingCache
from scone_memory.providers.jev_sections import JevSectionChooser
from scone_memory.retrieval.section_routing import FetchChooser, SectionChooser, SectionRouter
from scone_memory.retrieval.structured_document import Mode, StructuredDocumentIndex

ARMS = ('flat', 'vector_guided', 'local_structure', 'llamaindex')


class Chooser(SectionChooser, FetchChooser, Protocol):
    pass


class CachedOnly(RemoteEmbedder):
    async def embed(self, texts: Sequence[str]) -> list[list[float]]:
        raise ValueError('reference embedding cache is incomplete')

    async def embed_queries(self, texts: Sequence[str]) -> list[list[float]]:
        raise ValueError('reference query cache is incomplete')


class Row(BaseModel):
    model_config = ConfigDict(extra='forbid')
    id: str
    paper_id: str
    completed: bool
    error: str | None
    arms: dict[str, PreparedArm]
    receipts: list[dict[str, object]]
    route: dict[str, object] | None
    paper_preparation_ms: float


async def paper_questions(paper: Paper, questions: list[Question], cached: CachedEmbedder,
                          chooser: Chooser, ordinal: int, *,
                          completed: frozenset[str] = frozenset(),
                          client: httpx.AsyncClient | None = None) -> list[dict[str, object]]:
    preparation_started = time.perf_counter()
    current = snapshot(paper, development=True)
    index = await StructuredDocumentIndex.build(current, cached)
    ref = await reference(index, cached)
    preparation_ms = (time.perf_counter() - preparation_started) * 1000
    router = SectionRouter(chooser, cache_size=0)
    async with (httpx.AsyncClient(trust_env=False, follow_redirects=False)
                if client is None else nullcontext(client)) as client:
        async def one(question: Question, position: int) -> dict[str, object]:
            candidates: dict[str, list[Passage]] = {}
            arms: dict[str, PreparedArm] = {}
            receipts: list[dict[str, object]] = []
            route_receipt: dict[str, object] | None = None
            policies: list[tuple[str, Mode]] = [('flat', 'flat_vector'),
                ('vector_guided', 'auto'), ('local_structure', 'local_structure')]
            shift = position % len(policies)
            policies = policies[shift:] + policies[:shift]
            for arm, mode in policies:
                started = time.perf_counter()
                result = await index.retrieve(question.question, current, router, chooser,
                    mode=mode, routing='vector_candidates', limit=32, max_bytes=8000)
                elapsed = (time.perf_counter() - started) * 1000
                candidates[arm] = [passage(item, paper.id) for item in result.evidence]
                arms[arm] = PreparedArm(context='', retrieval_ms=elapsed,
                    route_ms=result.route.elapsed_ms if result.route else 0.,
                    fetch_ms=result.fetch_decision_ms, rerank_ms=0.,
                    effective_mode=result.mode, route_reason=result.reason)
                if arm == 'vector_guided':
                    route_receipt = {'route': asdict(result.route) if result.route else None,
                                     'fetch': asdict(result.fetch) if result.fetch else None}
            vector = (await cached.embed_queries([question.question]))[0]
            li, elapsed = await asyncio.to_thread(reference_candidates, ref, index, question.question, vector)
            candidates['llamaindex'] = [passage(item, paper.id) for item in li]
            arms['llamaindex'] = PreparedArm(context='', retrieval_ms=elapsed, route_ms=0.,
                fetch_ms=0., rerank_ms=0., effective_mode='llamaindex', route_reason=None)
            unique = {p.key: p for items in candidates.values() for p in items}
            ordered = [unique[key] for key in sorted(unique)]
            scores: dict[str, float] = {}
            rerank_started = time.perf_counter()
            error = None
            try:
                for offset in range(0, len(ordered), 64):
                    batch, receipt, elapsed = await rank(client, question.question,
                        ordered[offset:offset + 64], provider='openrouter')
                    scores.update(batch)
                    receipts.append(receipt)
                for arm in ARMS:
                    arms[arm].context = context(candidates[arm], scores)
            except (httpx.HTTPError, TimeoutError, ValueError) as failure:
                error = type(failure).__name__
            rerank_ms = (time.perf_counter() - rerank_started) * 1000
            for arm in ARMS:
                arms[arm].rerank_ms = rerank_ms
            return Row(id=question.id, paper_id=paper.id, completed=error is None, error=error,
                       arms=arms, receipts=receipts, route=route_receipt,
                       paper_preparation_ms=preparation_ms).model_dump()
        result: list[dict[str, object]] = []
        pending = [(q, ordinal + i) for i, q in enumerate(questions) if q.id not in completed]
        for offset in range(0, len(pending), 8):
            result.extend(await asyncio.gather(*(one(q, position)
                for q, position in pending[offset:offset + 8])))
        return result


def journal(path: Path) -> list[Row]:
    if not path.exists():
        return []
    return [Row.model_validate_json(line) for line in path.read_text().splitlines()]


async def _run(dataset: Path, previous: Path, output: Path, resume: bool) -> None:
    papers = [Paper.model_validate_json(line) for line in (dataset / 'corpus.jsonl').read_text().splitlines()]
    questions = [Question.model_validate_json(line) for line in (dataset / 'questions.jsonl').read_text().splitlines()]
    if (len(papers) != 281 or len(questions) != 1005
            or len({p.id for p in papers}) != 281 or len({q.id for q in questions}) != 1005
            or any(q.paper_id not in {p.id for p in papers} for q in questions)):
        raise ValueError('complete development split required')
    metadata = json.loads((dataset / 'dataset.json').read_text())
    if metadata['split'] != 'dev':
        raise ValueError('development split required')
    for name in ('corpus.jsonl', 'questions.jsonl'):
        if digest(dataset / name) != metadata['files'][name]['sha256']:
            raise ValueError('export digest mismatch')
    frozen = {**sources(), 'local_structure/run.py': digest(Path(__file__)),
              'local_structure/PROTOCOL.md': digest(Path(__file__).with_name('PROTOCOL.md'))}
    inputs = {name: digest(dataset / name) for name in ('corpus.jsonl', 'questions.jsonl', 'dataset.json')}
    if json.loads((previous / 'manifest.json').read_text())['input_sha256'] != inputs:
        raise ValueError('previous vector cache belongs to a different dataset export')
    manifest = {'protocol': 'local-structure-dev-v1', 'source_sha256': frozen,
        'inputs': inputs, 'arms': ARMS, 'papers': 281, 'questions': 1005,
        'embedding_model': EMBED_MODEL, 'dimensions': 2048, 'jev_model': OPENROUTER_JEV_MODEL,
        'embedding_cache_sha256': digest(previous / 'embeddings.db'),
        'versions': {name: version(name) for name in ('llama-index-core', 'llama-index-retrievers-bm25', 'httpx', 'pydantic', 'numpy')}, 'candidate_bytes': 8000,
        'candidate_limit': 32, 'context_bytes': 8000, 'context_items': 5,
        'scheduled': [{'id': q.id, 'paper_id': q.paper_id} for q in questions],
        'prior_manifest_sha256': digest(previous / 'manifest.json')}
    manifest = json.loads(json.dumps(manifest))
    if output.exists() and any(output.iterdir()) and not resume:
        raise ValueError('output not empty')
    output.mkdir(parents=True, exist_ok=True)
    if resume:
        if json.loads((output / 'manifest.json').read_text()) != manifest:
            raise ValueError('resume manifest mismatch')
    else:
        save(output / 'manifest.json', manifest)
        with sqlite3.connect('file:' + str(previous / 'embeddings.db') + '?mode=ro', uri=True) as source:
            with sqlite3.connect(output / 'embeddings.db') as target:
                source.backup(target)
    old = journal(output / 'observations.jsonl')
    identities = {q.id: q.paper_id for q in questions}
    if len({r.id for r in old}) != len(old) or any(identities.get(r.id) != r.paper_id or set(r.arms) != set(ARMS) for r in old):
        raise ValueError('journal schedule mismatch')
    completed = {r.id for r in old}
    remote = CachedOnly('https://openrouter.ai/api/v1', EMBED_MODEL, dim=2048,
                        query_prefix='query: ', document_prefix='passage: ')
    cache = SqliteEmbeddingCache(output / 'embeddings.db', max_entries=100000)
    cached = CachedEmbedder(remote, cache)
    try:
        async with httpx.AsyncClient(trust_env=False, follow_redirects=False) as client, \
                JevSectionChooser(api_key=credential('chat')) as chooser:
            ordinal = 0
            with (output / 'observations.jsonl').open('a') as stream:
                for number, paper in enumerate(papers, 1):
                    all_questions = [q for q in questions if q.paper_id == paper.id]
                    pending = [q for q in all_questions if q.id not in completed]
                    if pending:
                        for row in await paper_questions(paper, all_questions, cached, chooser, ordinal,
                                                         completed=frozenset(completed), client=client):
                            append(stream, row)
                        completed.update(q.id for q in pending)
                        print(f'Papers {number}/281; questions {len(completed)}/1005', flush=True)
                    ordinal += len(all_questions)
        if {**sources(), 'local_structure/run.py': digest(Path(__file__)),
            'local_structure/PROTOCOL.md': digest(Path(__file__).with_name('PROTOCOL.md'))} != frozen:
            raise ValueError('sources changed during run')
        if any(digest(dataset / name) != value for name, value in inputs.items()):
            raise ValueError('inputs changed during run')
        save(output / 'completion.json', {'questions': len(completed), 'complete': len(completed) == 1005,
            'observations_sha256': digest(output / 'observations.jsonl'), 'manifest_sha256': digest(output / 'manifest.json'),
            'embedding_cache': cached.record()})
    finally:
        await cache.close()
        await remote.close()


async def run(dataset: Path, previous: Path, output: Path, resume: bool) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.with_suffix('.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        await _run(dataset, previous, output, resume)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', type=Path, required=True)
    parser.add_argument('--previous', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()
    asyncio.run(run(args.dataset, args.previous, args.output, args.resume))


if __name__ == '__main__':
    main()
