"""Jobs, evidence retention and whole-space deletion compose across stores."""
import pytest

from scone_memory.core.errors import InvalidInput, NotFound
from scone_memory.ingestion.records import Record
from scone_memory.integrations.scoped_tools import ScopedMemoryTools
from scone_memory.retrieval.recall_scope import RecallScope


@pytest.fixture
def job_engine(engine):
    methods = ('create_job', 'update_job', 'get_job', 'list_jobs')
    if not all(callable(getattr(engine.documents, method, None)) for method in methods):
        pytest.skip('document store does not support the ingest-job lifecycle')
    return engine


async def test_cancelled_job_remains_searchable_until_retention_and_space_deletion(job_engine):
    engine = job_engine
    engine.clock = lambda: '2026-01-01T00:00:00Z'
    job = await engine.ingest_batch('alpha', [Record('Juniper uses Polaris.'), Record('Juniper ships in April.')],
        request_id='batch-1')
    first, second = [item.episode_id for item in job.items]
    attachment = await engine.attach('alpha', b'original source bytes', 'text/plain')
    await engine.blobs.link('alpha', attachment.attachment_id, first)
    same_bytes = await engine.attach('beta', b'original source bytes', 'text/plain')
    neighbor = await engine.remember('beta', 'A separate source.', attachment_ids=[same_bytes.attachment_id])
    claim = await engine.assert_fact('alpha', 'juniper', 'uses', 'polaris', source_episode_id=first, quote='Polaris')
    await engine.note_consolidated('alpha', [first])
    await engine.note_failed('alpha', second, 'model unavailable')
    cancelled = await engine.cancel_job('alpha', job.job_id)
    assert [item.state for item in cancelled.items] == ['consolidated', 'cancelled']
    assert cancelled.searchable == 2 and cancelled.items[1].attempts == 1
    assert (await engine.ingest_batch('alpha', [Record('must not ingest a retry')], request_id='batch-1')).job_id == job.job_id
    assert (await engine.status('alpha')).episodes == 2

    tools = ScopedMemoryTools(engine, 'alpha', scope=RecallScope.validated())
    prepared = await tools.prepare('search_memory', {'query':'Juniper Polaris', 'limit':5})
    assert await prepared.validate()
    engine.clock = lambda: '2026-03-01T00:00:00Z'
    expired = await engine.expire('alpha', {'note':30}, limit=1)
    assert expired.forgotten == [first] and expired.remaining == 1
    assert not await prepared.validate()
    assert (await engine.fact('alpha', claim.fact_id)).status == 'active'
    assert (await engine.doctor('alpha')).facts_citing_forgotten == [claim.fact_id]
    assert (await engine.job('alpha', job.job_id)).state == 'cancelled'
    assert (await engine.attachment('beta', attachment.attachment_id))[1] == b'original source bytes'

    preview = await engine.space_impact('alpha')
    assert (preview.episodes, preview.facts, preview.tombstones) == (1, 1, 1)
    deleted = await engine.delete_space('alpha')
    assert (deleted.episodes, deleted.facts, deleted.tombstones) == (1, 1, 1)
    assert deleted.events == preview.events and deleted.deleted_at == engine.clock()
    assert await engine.events.query('alpha') == []
    assert (await engine.doctor('alpha')).healthy
    assert await engine.jobs('alpha') == []
    with pytest.raises(NotFound, match='deleted'):
        await engine.ingest_batch('alpha', [Record('cannot resurrect')], request_id='batch-1')
    assert (await engine.episode('beta', neighbor.episode_id)).content == 'A separate source.'
    assert (await engine.attachment('beta', attachment.attachment_id))[1] == b'original source bytes'


async def test_expiry_honors_host_forget_guard_and_leaves_protected_evidence(engine, monkeypatch):
    source = await engine.remember('alpha', 'A protected source.', created_at='2026-01-01T00:00:00Z')
    engine.clock = lambda: '2026-03-01T00:00:00Z'
    async def guarded_forget(space, episode_id):
        raise InvalidInput('source is on hold')
    monkeypatch.setattr(engine, 'forget', guarded_forget)
    with pytest.raises(InvalidInput, match='on hold'):
        await engine.expire('alpha', {'note':30})
    assert (await engine.episode('alpha', source.episode_id)).content == 'A protected source.'
    assert await engine.tombstone('alpha', source.episode_id) is None
    assert await engine.events.query('alpha', kind='expire') == []


async def test_job_page_limit_uses_current_host_configuration(job_engine):
    engine = job_engine
    engine.MAX_JOBS_PAGE = 2
    with pytest.raises(InvalidInput, match='1 through 2'):
        await engine.jobs('alpha', limit=3)
    engine.MAX_JOBS_PAGE = 3
    assert await engine.jobs('alpha', limit=3) == []
