"""Preparing a source update must not revoke the still-usable old source."""
from __future__ import annotations

import asyncio
from dataclasses import replace

import pytest

from scone_memory import HashEmbedder, InMemoryDocumentStore, InMemoryVectorIndex, MemoryEngine, Record
from scone_memory.core.errors import InvalidInput
from scone_memory.observability.events import InMemoryEventLog
from scone_memory.testing import Clock


class ControlledEmbedder(HashEmbedder):
    def __init__(self):
        super().__init__()
        self.failure = None
        self.entered = asyncio.Event()
        self.release = asyncio.Event()
        self.wait = False
        self.calls = 0

    async def embed(self, texts):
        self.calls += 1
        if self.wait:
            self.entered.set()
            await self.release.wait()
        if self.failure == 'provider':
            raise RuntimeError('embedding provider unavailable')
        if self.failure == 'dimension':
            return [[1.0] for _ in texts]
        if self.failure == 'finite':
            return [[float('nan')] * self.dim for _ in texts]
        return await super().embed(texts)


@pytest.fixture(params=['memory', 'sqlite'])
async def source(request, tmp_path):
    if request.param == 'sqlite':
        from scone_memory.backends import SqliteDocumentStore, SqliteVectorIndex
        documents, vectors = SqliteDocumentStore(tmp_path / 'data.db'), SqliteVectorIndex(tmp_path / 'data.db')
    else:
        documents, vectors = InMemoryDocumentStore(), InMemoryVectorIndex()
    embedder = ControlledEmbedder()
    engine = await MemoryEngine(documents, vectors, embedder, clock=Clock(), events=InMemoryEventLog()).open()
    attachment = await engine.attach('alpha', b'original source bytes', 'text/plain', filename='observatory.txt')
    original = await engine.remember('alpha', 'The observatory is in Lisbon.', dedup_key='doc:observatory',
                                     attachment_ids=(attachment.attachment_id,))
    revision = await documents.revision('alpha')
    yield engine, embedder, original, revision
    await engine.close()


async def assert_preserved(source):
    engine, _, original, revision = source
    assert await engine.tombstone('alpha', original.episode_id) is None
    episode = await engine.episode('alpha', original.episode_id)
    assert episode.content == 'The observatory is in Lisbon.'
    assert len(episode.attachments) == 1
    assert (await engine.attachment('alpha', episode.attachments[0].attachment_id))[1] == b'original source bytes'
    assert await engine.documents.revision('alpha') == revision
    recalled = await engine.recall('alpha', 'observatory Lisbon', rerank=False)
    assert original.episode_id in {item.episode_id for item in recalled.items}


@pytest.mark.parametrize('fields', [
    {'content': ''}, {'kind': 'invalid'}, {'tags': ['x' * 65]}, {'metadata': {'secret': 42}},
    {'created_at': 'not-a-date'}, {'content_hash': 'caller-controlled-hash'},
])
async def test_invalid_replacement_preserves_old_source(source, fields):
    engine, _, _, _ = source
    record = replace(Record('The observatory moved to Porto.', dedup_key='doc:observatory'), **fields)
    with pytest.raises(InvalidInput):
        await engine.replace('alpha', record)
    await assert_preserved(source)


async def test_duplicate_replacement_still_validates_its_record(source):
    engine, _, _, _ = source
    with pytest.raises(InvalidInput):
        await engine.replace('alpha', Record('The observatory is in Lisbon.', kind='invalid', dedup_key='doc:observatory'))
    await assert_preserved(source)


@pytest.mark.parametrize('failure', ['provider', 'dimension', 'finite'])
async def test_embedding_failure_preserves_old_source(source, failure):
    engine, embedder, _, _ = source
    embedder.failure = failure
    with pytest.raises((RuntimeError, ValueError)):
        await engine.replace('alpha', Record('The observatory moved to Porto.', dedup_key='doc:observatory'))
    embedder.failure = None
    await assert_preserved(source)


async def test_cancellation_during_preparation_preserves_old_source(source):
    engine, embedder, _, _ = source
    embedder.wait = True
    pending = asyncio.create_task(engine.replace('alpha', Record('The observatory moved to Porto.', dedup_key='doc:observatory')))
    await asyncio.wait_for(embedder.entered.wait(), 2)
    pending.cancel()
    with pytest.raises(asyncio.CancelledError):
        await pending
    embedder.wait = False
    await assert_preserved(source)


async def test_a_source_forgotten_during_preparation_is_not_resurrected(source):
    engine, embedder, original, _ = source
    embedder.wait = True
    pending = asyncio.create_task(engine.replace('alpha', Record('The observatory moved to Porto.', dedup_key='doc:observatory')))
    await asyncio.wait_for(embedder.entered.wait(), 2)
    await engine.forget('alpha', original.episode_id)
    embedder.release.set()
    with pytest.raises(InvalidInput, match='changed.*prepar|prepar.*changed'):
        await pending
    assert (await engine.status('alpha')).episodes == 0


async def test_success_prepares_once_then_returns_replacement_receipt(source):
    engine, embedder, original, _ = source
    before = embedder.calls
    updated = await engine.replace('alpha', Record('The observatory moved to Porto.', dedup_key='doc:observatory'))
    assert embedder.calls == before + 1
    assert updated.outcome == 'updated' and updated.replaced.episode_id == original.episode_id
    assert (await engine.episode('alpha', updated.added.episode_id)).content == 'The observatory moved to Porto.'
    assert await engine.tombstone('alpha', original.episode_id) is not None
    duplicate = await engine.replace('alpha', Record('The observatory moved to Porto.', dedup_key='doc:observatory'))
    assert duplicate.outcome == 'duplicate' and duplicate.added.episode_id == updated.added.episode_id
    assert embedder.calls == before + 1


async def test_invalid_source_value_is_rejected_before_forgetting(source):
    engine, _, _, _ = source
    with pytest.raises(InvalidInput):
        await engine.replace('alpha', Record('The observatory moved to Porto.', source={'bad': 'source'}, dedup_key='doc:observatory'))
    await assert_preserved(source)


async def test_failed_preparation_records_failure_without_a_forget_event(source):
    engine, embedder, _, _ = source
    embedder.failure = 'provider'
    with pytest.raises(RuntimeError):
        await engine.replace('alpha', Record('The observatory moved to Porto.', dedup_key='doc:observatory'))
    events = await engine.events.query('alpha', kind='remember', limit=10)
    assert 'error' in events[0].payload
    assert await engine.events.query('alpha', kind='forget', limit=10) == []


async def test_failed_revision_receipt_does_not_claim_the_written_key_is_empty(source, monkeypatch):
    engine, _, _, _ = source
    original_bump = engine.documents.bump_revision
    calls = 0
    async def fail_second_bump(space):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise RuntimeError('revision write unavailable')
        return await original_bump(space)
    monkeypatch.setattr(engine.documents, 'bump_revision', fail_second_bump)
    with pytest.raises(InvalidInput) as failure:
        await engine.replace('alpha', Record('The observatory moved to Porto.', dedup_key='doc:observatory'))
    assert 'names nothing' not in str(failure.value)
    assert (await engine.status('alpha')).episodes == 1
    from scone_memory.ingestion.records import content_hash
    current = await engine.documents.episode_by_hash('alpha', content_hash('alpha', '', 'doc:observatory'))
    assert current.content == 'The observatory moved to Porto.'


@pytest.mark.parametrize('prior_forget', [False, True])
async def test_absent_key_created_then_forgotten_during_preparation_stays_forgotten(source, prior_forget):
    engine, embedder, _, _ = source
    key = 'doc:another'
    if prior_forget:
        old = await engine.remember('alpha', 'A prior edition', dedup_key=key)
        await engine.forget('alpha', old.episode_id)
    embedder.wait = True
    pending = asyncio.create_task(engine.replace('alpha', Record('Prepared edition', dedup_key=key)))
    await asyncio.wait_for(embedder.entered.wait(), 2)
    embedder.wait = False
    intermediate = await engine.remember('alpha', 'Intermediate edition', dedup_key=key)
    await engine.forget('alpha', intermediate.episode_id)
    embedder.release.set()
    with pytest.raises(InvalidInput, match='changed.*prepar|prepar.*changed'):
        await pending
    from scone_memory.ingestion.records import content_hash
    assert await engine.documents.episode_by_hash('alpha', content_hash('alpha', '', key)) is None
    # A fresh, intentional attempt after the forget remains possible.
    accepted = await engine.replace('alpha', Record('Intentionally restored edition', dedup_key=key))
    assert accepted.outcome == 'accepted'


@pytest.mark.parametrize('setting', ['embedder', 'contextual_embeddings', 'chunk_target'])
async def test_changed_preparation_configuration_keeps_old_source(source, setting):
    engine, embedder, _, _ = source
    before = getattr(engine, setting)
    embedder.wait = True
    pending = asyncio.create_task(engine.replace('alpha', Record('The observatory moved to Porto.', dedup_key='doc:observatory')))
    await asyncio.wait_for(embedder.entered.wait(), 2)
    changed = HashEmbedder() if setting == 'embedder' else not before if setting == 'contextual_embeddings' else before + 1
    setattr(engine, setting, changed)
    embedder.release.set()
    with pytest.raises(InvalidInput, match='configuration.*changed|changed.*configuration'):
        await pending
    setattr(engine, setting, before)
    embedder.wait = False
    await assert_preserved(source)


@pytest.mark.parametrize('fields', [{'source': '\ud800'}, {'tags': ['\ud800']}, {'metadata': {'label': '\ud800'}}])
async def test_invalid_utf8_metadata_is_rejected_before_forgetting(source, fields):
    engine, _, _, _ = source
    record = replace(Record('The observatory moved to Porto.', dedup_key='doc:observatory'), **fields)
    with pytest.raises(InvalidInput, match='UTF-8'):
        await engine.replace('alpha', record)
    await assert_preserved(source)


async def test_http_provider_failure_keeps_original_readable(source):
    from httpx import ASGITransport, AsyncClient
    from scone_memory.api import create_app
    engine, embedder, original, _ = source
    embedder.failure = 'provider'
    headers = {'Authorization': 'Bearer fixture-key'}
    transport = ASGITransport(app=create_app(engine, {'fixture-key': 'alpha'}), raise_app_exceptions=False)
    async with AsyncClient(transport=transport, base_url='http://fixture') as client:
        response = await client.post('/v1/episodes', headers=headers,
            json={'content': 'The observatory moved to Porto.', 'dedup_key': 'doc:observatory', 'replace': True})
        assert response.status_code == 500
        old = await client.get(f'/v1/episodes/{original.episode_id}', headers=headers)
        assert old.status_code == 200 and old.json()['content'] == 'The observatory is in Lisbon.'
    embedder.failure = None
    await assert_preserved(source)


async def test_write_failure_after_forget_reports_stage_and_can_be_retried(source, monkeypatch):
    engine, _, original, _ = source
    normal_upsert = engine.vectors.upsert
    async def unavailable(points):
        raise RuntimeError('vector store unavailable')
    monkeypatch.setattr(engine.vectors, 'upsert', unavailable)
    with pytest.raises(InvalidInput, match='storage or receipt stage failed') as failure:
        await engine.replace('alpha', Record('The observatory moved to Porto.', dedup_key='doc:observatory'))
    assert str(original.episode_id) in str(failure.value)
    assert (await engine.status('alpha')).episodes == 0
    assert await engine.documents.inflight() == []
    monkeypatch.setattr(engine.vectors, 'upsert', normal_upsert)
    retried = await engine.replace('alpha', Record('The observatory moved to Porto.', dedup_key='doc:observatory'))
    assert retried.outcome == 'accepted'
    assert (await engine.episode('alpha', retried.added.episode_id)).content == 'The observatory moved to Porto.'
