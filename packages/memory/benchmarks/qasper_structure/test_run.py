from __future__ import annotations

import json
from pathlib import Path
from typing import Sequence

import httpx
import pytest

from matched_qa.pipeline import Passage
from qasper_structure import run
from qasper_structure.data import Paper, Paragraph, Question
from scone_memory.bench.comparative import CachedEmbedder
from scone_memory.ingestion.embedding_cache import SqliteEmbeddingCache
from scone_memory.retrieval.section_routing import ChoiceBatch, RouteMenu


class Embeddings:
    id = 'offline-fixture'
    dim = 2

    async def embed(self, texts: Sequence[str]) -> list[list[float]]:
        return [[1., .2] if 'cats' in text.lower() else [.2, 1.] for text in texts]


class NoRoute:
    definition = 'offline-fixture'

    def __init__(self, *, api_key: str) -> None:
        pass

    async def __aenter__(self) -> NoRoute:
        return self

    async def __aexit__(self, *args: object) -> None:
        pass

    async def choose(self, query: str, menus: tuple[RouteMenu, ...]) -> ChoiceBatch:
        return ChoiceBatch(tuple(tuple(0. for _ in menu.options) + (1.,) for menu in menus), 'fixture')


async def test_real_three_arm_pipeline_resumes_identical_contexts(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    text = '# Animals\n## Cats\nCats purr.\n## Dogs\nDogs bark.'
    start = text.index('Cats purr.')
    paper = Paper(id='paper', content=text, paragraphs=(Paragraph(text='Cats purr.', start=start, end=start+10),))
    questions = [Question(id='question', paper_id='paper', question='What do cats do?')]
    monkeypatch.setattr(run, 'JevSectionChooser', NoRoute)
    monkeypatch.setattr(run, 'credential', lambda kind: 'fixture-not-a-secret')
    rank_calls = 0
    generate_calls = 0

    async def rank(client: httpx.AsyncClient, query: str, candidates: list[Passage], *, provider: str
                   ) -> tuple[dict[str, float], dict[str, object], float]:
        nonlocal rank_calls
        rank_calls += 1
        return {p.key: .9 for p in candidates}, {}, 1.

    async def generate(client: httpx.AsyncClient, messages: list[dict[str, str]]) -> dict[str, object]:
        nonlocal generate_calls
        generate_calls += 1
        success = generate_calls != 1
        return {'completed': success, 'answer': 'purr' if success else '',
                'error': None if success else 'http_503', 'generation_ms': 1.}

    monkeypatch.setattr(run, 'rank', rank)
    monkeypatch.setattr(run, 'generate', generate)
    cache = SqliteEmbeddingCache(tmp_path / 'cache.db')
    cached = CachedEmbedder(Embeddings(), cache)
    try:
        with pytest.raises(RuntimeError, match='evaluation paused'):
            await run.evaluate([paper], questions, tmp_path, cached, 1)
        prepared = (tmp_path / 'prepared.jsonl').read_bytes()
        await run.evaluate([paper], questions, tmp_path, cached, 1)
        assert (tmp_path / 'prepared.jsonl').read_bytes() == prepared
        rows = run.records(tmp_path / 'observations.jsonl')
        assert len(rows) == generate_calls == 3
        assert rank_calls == 1
        assert sum(row['completed'] is True for row in rows) == 2
        assert {row['arm'] for row in rows} == set(run.ARMS)
        frozen = json.loads(prepared)['arms']
        for row in rows:
            assert row['context_text'] == frozen[row['arm']]['context']
            assert isinstance(row['context_bytes'], int)
            assert row['context_bytes'] <= 8000
        await run.evaluate([paper], questions, tmp_path, cached, 1)
        assert generate_calls == 3
    finally:
        await cache.close()
