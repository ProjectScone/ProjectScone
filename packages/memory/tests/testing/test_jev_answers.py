import pytest

from scone_memory.testing.public_qa import Gold, Question


def test_paired_answer_scores_keep_failures_and_regressions_in_denominator():
    from scone_memory.testing.jev_answers import Answer, score_answers

    questions = [Question(id='one', dataset='squad', question='Who?'),
                 Question(id='two', dataset='hotpotqa', question='Where?')]
    gold = {'one': Gold(id='one', dataset='squad', answers=('Morgan',), support_documents=('a',)),
            'two': Gold(id='two', dataset='hotpotqa', answers=('New York',), support_documents=('b',))}
    def answer(id, arm, text, status='completed'):
        return Answer(id=id, arm=arm, request_sha256='hash', status=status,
                      completed=status == 'completed', answer_text=text, total_ms=1)
    rows = [answer('one', 'hybrid', 'INSUFFICIENT_EVIDENCE'), answer('one', 'hybrid_jev', 'Morgan'),
            answer('two', 'hybrid', 'New York'), answer('two', 'hybrid_jev', 'New York', 'timeout')]
    result = score_answers(questions, rows, gold)
    groups = {row['arm']: row for row in result['groups'] if row['dataset'] == 'all'}
    assert groups['hybrid']['em'] == groups['hybrid_jev']['em'] == 0.5
    assert groups['hybrid']['abstentions'] == 1
    assert groups['hybrid_jev']['failures'] == 1
    assert result['paired']['em'] == {'wins': 1, 'losses': 1, 'ties': 0, 'mean_delta': 0.0}
    with pytest.raises(ValueError, match='schedule'):
        score_answers(questions, rows[:-1], gold)
    with pytest.raises(ValueError, match='schedule'):
        score_answers(questions, rows + rows[:1], gold)


async def test_answer_preparation_uses_native_retained_context_with_reranking():
    from scone_memory import HashEmbedder, InMemoryDocumentStore, InMemoryVectorIndex, MemoryEngine
    from scone_memory.retrieval.reranking import RerankScore
    from scone_memory.testing.jev_answers import prepare_question
    from scone_memory.testing.jev_public_qa import SPACE
    from scone_memory.testing.public_qa import Document

    calls = []
    class Ranker:
        async def rerank(self, query, candidates):
            calls.append(query)
            return [RerankScore(c.chunk_id, 1 if 'Morgan' in c.text else 0) for c in candidates]
    engine = await MemoryEngine(InMemoryDocumentStore(), InMemoryVectorIndex(), HashEmbedder(),
                                reranker=Ranker()).open()
    try:
        documents = {}
        for i, text in enumerate([f'Cedar maintainer overview {i}.' for i in range(6)] +
                                 ['Morgan is the maintainer of Cedar.']):
            doc = Document(id=str(i), title='Cedar', text=text, source_url=f'local:{i}')
            row = await engine.remember(SPACE, doc.content, kind='file', metadata={'document_id': doc.id})
            documents[row.episode_id] = doc
        question = Question(id='q', dataset='squad', question='Who maintains Cedar?')
        prepared = await prepare_question(engine, question, 'hybrid_jev', documents)
        assert calls
        assert prepared.request[-1] == {'role': 'user', 'content': question.question}
        assert prepared.receipt['status'] == 'prepared'
        assert 0 < len(prepared.sources) <= 5
        assert 'Morgan' in prepared.sources[0].text
        assert any('Morgan' in message['content'] for message in prepared.request)
    finally:
        await engine.close()
