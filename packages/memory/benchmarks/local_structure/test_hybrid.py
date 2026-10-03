from qasper_structure.data import Paper, Paragraph, Question
from qasper_structure.test_run import Embeddings
from scone_memory.bench.comparative import CachedEmbedder
from scone_memory.ingestion.embedding_cache import InMemoryEmbeddingCache

from local_structure.hybrid import paper_questions


async def test_local_comparison_retains_three_complete_contexts_without_hosted_receipts() -> None:
    text = '# Animals\nDogs bark.\n\nCats need taurine.\n\nBirds fly.'
    paper = Paper(id='paper', content=text, paragraphs=(Paragraph(text='Dogs bark.', start=10, end=20),))
    question = Question(id='q', paper_id='paper', question='taurine')
    cached = CachedEmbedder(Embeddings(), InMemoryEmbeddingCache())
    rows = await paper_questions(paper, [question], cached, 0)
    assert len(rows) == 1 and rows[0].completed
    assert set(rows[0].arms) == {'flat_no_rerank', 'structure_no_rerank', 'hybrid_no_rerank'}
    assert rows[0].receipts == [] and rows[0].route is None
    for arm in rows[0].arms.values():
        assert arm.route_ms == arm.fetch_ms == arm.rerank_ms == 0
        assert 0 < arm.retrieval_ms
        assert len(arm.context.encode()) <= 8000
    assert rows[0].arms['hybrid_no_rerank'].context.startswith('[Source 1]\nCats need taurine.')
