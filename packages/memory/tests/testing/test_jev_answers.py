import pytest
import json
import sqlite3

from scone_memory.testing.public_qa import Gold, Question


async def test_paired_query_vectors_are_computed_once_and_reused(tmp_path):
    from scone_memory import InMemoryDocumentStore, InMemoryVectorIndex, MemoryEngine
    from scone_memory.core.embedding import embed_queries
    from scone_memory.testing.jev_answers import prepare_query_vectors
    calls = []
    class Embedder:
        id = 'test-query-space'
        dim = 2
        async def embed(self, texts):
            return [[1.0, 0.0] for _ in texts]
        async def embed_queries(self, texts):
            calls.append(list(texts))
            return [[0.0, 1.0] for _ in texts]
    engine = await MemoryEngine(InMemoryDocumentStore(), InMemoryVectorIndex(), Embedder()).open()
    try:
        questions = [Question(id='q', dataset='squad', question='Who maintains Cedar?')]
        summary = await prepare_query_vectors(engine, questions)
        assert summary['embedded'] == 1
        assert await embed_queries(engine.embedder, [questions[0].question]) == [[0.0, 1.0]]
        assert await embed_queries(engine.embedder, [questions[0].question]) == [[0.0, 1.0]]
        assert calls == [['Who maintains Cedar?']]
        assert engine.embedder.id == 'test-query-space'
    finally:
        await engine.close()


def test_index_reuse_requires_matching_embedding_space_and_input_hashes(tmp_path):
    from scone_memory.testing.jev_answers import reuse_collection, QUERY_PREFIX, QWEN_MODEL
    source, output = tmp_path/'source', tmp_path/'output'
    source.mkdir()
    output.mkdir()
    with sqlite3.connect(source/'memory.db') as conn:
        conn.execute('CREATE TABLE marker(value TEXT)')
        conn.execute("INSERT INTO marker VALUES ('preserved')")
    manifest = {'qdrant_collection': 'scone_qwen_answers_' + 'a'*32,
                'qdrant_url': 'http://127.0.0.1:64076', 'inputs': {'corpus': 'hash'},
                'embedding_model': QWEN_MODEL, 'dimensions': 4096,
                'query_prefix': QUERY_PREFIX, 'vector_backend': 'qdrant'}
    (source/'manifest.json').write_text(json.dumps(manifest))
    assert reuse_collection(source, output, manifest['inputs'], manifest['qdrant_url']) == manifest['qdrant_collection']
    with sqlite3.connect(output/'memory.db') as conn:
        assert conn.execute('SELECT value FROM marker').fetchone()[0] == 'preserved'
    for change in [{'dimensions': 768}, {'query_prefix': ''}, {'embedding_model': 'another'},
                   {'inputs': {'corpus': 'different'}}, {'qdrant_collection': 'production'}]:
        (source/'manifest.json').write_text(json.dumps({**manifest, **change}))
        with pytest.raises(ValueError, match='configuration or inputs'):
            reuse_collection(source, output, manifest['inputs'], manifest['qdrant_url'])


async def test_index_readiness_finishes_all_migration_batches(tmp_path, monkeypatch):
    from scone_memory import HashEmbedder, InMemoryVectorIndex, MemoryEngine
    from scone_memory.backends import SqliteDocumentStore, sqlite_lexical
    from scone_memory.testing.jev_answers import prepare_text_index
    from scone_memory.testing.jev_public_qa import SPACE
    store = SqliteDocumentStore(tmp_path/'index.db')
    engine = await MemoryEngine(store, InMemoryVectorIndex(), HashEmbedder()).open()
    try:
        for i in range(7):
            await engine.remember(SPACE, f'Corpus passage {i}.')
        monkeypatch.setattr(sqlite_lexical, 'MAX_SYNC_ROWS', 2)
        monkeypatch.setattr(sqlite_lexical, '_VERSION', sqlite_lexical._VERSION + ';test-readiness')
        sqlite_lexical.initialize_lexical(store.conn)
        assert await prepare_text_index(engine) == {'indexed': 7, 'remaining': 0}
        assert store.conn.execute('SELECT count(*) FROM chunk_lexical_dirty').fetchone()[0] == 0
        assert await prepare_text_index(engine) == {'indexed': 0, 'remaining': 0}
    finally:
        await engine.close()


async def test_reused_index_must_match_every_corpus_document(tmp_path):
    from scone_memory import HashEmbedder, InMemoryVectorIndex, MemoryEngine
    from scone_memory.backends import SqliteDocumentStore
    from scone_memory.testing.jev_answers import indexed_documents
    from scone_memory.testing.jev_public_qa import SPACE
    from scone_memory.testing.public_qa import Document
    doc = Document(id='one', title='Cedar', text='Morgan maintains Cedar.', source_url='local:one')
    engine = await MemoryEngine(SqliteDocumentStore(tmp_path/'reuse.db'), InMemoryVectorIndex(), HashEmbedder()).open()
    try:
        saved = await engine.remember(SPACE, doc.content, kind='file', source=doc.source_url,
                                      metadata={'document_id': doc.id})
        assert await indexed_documents(engine, [doc]) == {saved.episode_id: doc}
        with pytest.raises(ValueError, match='corpus'):
            await indexed_documents(engine, [doc.model_copy(update={'text': 'Changed content'})])
        with pytest.raises(ValueError, match='corpus'):
            await indexed_documents(engine, [])
    finally:
        await engine.close()


async def test_neural_evaluation_uses_qwen_and_fresh_local_qdrant(tmp_path, monkeypatch):
    qdrant = pytest.importorskip('qdrant_client')
    client = qdrant.AsyncQdrantClient
    monkeypatch.setattr(qdrant, 'AsyncQdrantClient', lambda **kwargs: client(':memory:'))
    from scone_memory.testing.jev_answers import build_engine
    one = build_engine(tmp_path, 'not-a-real-key', 'http://127.0.0.1:64076')
    two = build_engine(tmp_path, 'not-a-real-key', 'http://127.0.0.1:64076')
    try:
        assert one.embedder.model == 'qwen/qwen3-embedding-8b'
        assert one.embedder.dim == 4096
        assert one.embedder.query_prefix.startswith('Instruct:')
        assert one.vectors.name == 'qdrant'
        assert one.vectors.collection != two.vectors.collection
        assert one.vector_weight == 1.0
        assert one.candidate_limit == 64
    finally:
        await one.vectors.close()
        await two.vectors.close()


@pytest.mark.parametrize('url', ['https://hosted.example', 'http://user:secret@localhost:6333',
                               'http://127.0.0.1:6333/?api_key=secret'])
def test_neural_evaluation_rejects_nonlocal_or_credential_urls(tmp_path, url):
    from scone_memory.testing.jev_answers import build_engine
    with pytest.raises(ValueError, match='local Qdrant'):
        build_engine(tmp_path, 'not-a-real-key', url)


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
