from pathlib import Path

import pytest

from qasper_structure.test_run import Embeddings, NoRoute
from qasper_structure.data import Paper, Paragraph, Question
from local_structure import run
from scone_memory.bench.comparative import CachedEmbedder
from scone_memory.ingestion.embedding_cache import InMemoryEmbeddingCache


async def test_four_arm_retrieval_shares_rerank_and_retains_complete_source(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    text = '# Cats\nCats purr.\n# Dogs\nDogs bark.'
    paper = Paper(id='paper', content=text, paragraphs=(Paragraph(text='Cats purr.', start=7, end=17),))
    question = Question(id='q', paper_id='paper', question='cats')
    calls = []
    async def rank(client, query, candidates, *, provider):
        calls.append([p.key for p in candidates])
        return {p.key: .9 for p in candidates}, {'response': {'usage': {}}}, 1.
    monkeypatch.setattr(run, 'rank', rank)
    rows = await run.paper_questions(paper, [question], CachedEmbedder(Embeddings(), InMemoryEmbeddingCache()), NoRoute(api_key='fixture'), 0)
    assert len(rows) == 1
    assert rows[0]['completed'] is True
    arms = rows[0]['arms']
    assert set(arms) == {'flat', 'vector_guided', 'local_structure', 'llamaindex'}
    assert 'Cats purr.' in arms['local_structure']['context']
    assert arms['local_structure']['route_ms'] == 0
    assert arms['local_structure']['fetch_ms'] == 0
    assert arms['local_structure']['rerank_ms'] == arms['flat']['rerank_ms']
    assert arms['flat']['rerank_ms'] > 0
    assert len(calls) == 1
    assert len(calls[0]) == len(set(calls[0]))
    assert all(len(a['context'].encode()) <= 8000 for a in arms.values())


async def test_failed_rerank_retains_wait_time_and_index_preparation(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import asyncio
    text = '# Cats\nCats purr.'
    paper = Paper(id='paper', content=text, paragraphs=(Paragraph(text='Cats purr.', start=7, end=17),))
    question = Question(id='q', paper_id='paper', question='cats')
    async def rank(*args, **kwargs):
        await asyncio.sleep(.01)
        raise TimeoutError('fixture timeout')
    monkeypatch.setattr(run, 'rank', rank)
    rows = await run.paper_questions(paper, [question], CachedEmbedder(Embeddings(), InMemoryEmbeddingCache()), NoRoute(api_key='fixture'), 0)
    assert rows[0]['completed'] is False
    assert rows[0]['error'] == 'TimeoutError'
    assert all(a['rerank_ms'] >= 9 for a in rows[0]['arms'].values())
    assert rows[0]['paper_preparation_ms'] > 0
