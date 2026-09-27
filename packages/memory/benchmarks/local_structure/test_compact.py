import asyncio

import httpx
import pytest

from qasper_structure.data import Paper, Paragraph, Question
from qasper_structure.test_run import Embeddings
from scone_memory.bench.comparative import CachedEmbedder
from scone_memory.ingestion.embedding_cache import InMemoryEmbeddingCache
from local_structure import compact


async def test_compact_comparison_measures_each_arm_separately(monkeypatch: pytest.MonkeyPatch) -> None:
    paper = Paper(id='paper', content='# Animals\nDogs bark.\n\nCats need taurine.', paragraphs=())
    question = Question(id='q', paper_id='paper', question='taurine')
    async def rank(client, query, candidates, *, provider):
        await asyncio.sleep(.005)
        scores = {p.key: float('taurine' in p.text) for p in candidates}
        return scores, {'response': {'usage': {}}}, 5.
    monkeypatch.setattr(compact, 'rank', rank)
    async with httpx.AsyncClient() as client:
        rows = await compact.paper_questions(paper, [question],
            CachedEmbedder(Embeddings(), InMemoryEmbeddingCache()), 0, client)
    assert len(rows) == 1 and rows[0].completed
    assert {r['arm'] for r in rows[0].receipts} == {'structure_jev', 'hybrid_jev'}
    assert rows[0].arms['hybrid_jev'].context.startswith('[Source 1]\nCats need taurine.')
    assert all(a.rerank_ms >= 5 and a.route_ms == a.fetch_ms == 0 for a in rows[0].arms.values())


async def test_compact_failed_rerank_retains_time_and_marks_pair_failed(monkeypatch: pytest.MonkeyPatch) -> None:
    async def rank(*args, **kwargs):
        await asyncio.sleep(.005)
        raise TimeoutError('fixture')
    monkeypatch.setattr(compact, 'rank', rank)
    paper = Paper(id='paper', content='# Cats\nCats purr.', paragraphs=())
    async with httpx.AsyncClient() as client:
        rows = await compact.paper_questions(paper, [Question(id='q', paper_id='paper', question='cats')],
            CachedEmbedder(Embeddings(), InMemoryEmbeddingCache()), 0, client)
    assert not rows[0].completed and rows[0].error == 'TimeoutError'
    assert all(a.rerank_ms >= 5 for a in rows[0].arms.values())
