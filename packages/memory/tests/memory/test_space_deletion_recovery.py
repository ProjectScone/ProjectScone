"""Whole-space cleanup remains finishable after records have disappeared."""
from __future__ import annotations

import pytest

from scone_memory.core.errors import InvalidInput, NotFound
from .test_forget_recovery import make_engine, NOW


@pytest.mark.parametrize('backend', ['memory', 'sqlite'])
@pytest.mark.parametrize('stage', ['blobs', 'rows', 'vectors', 'events'])
@pytest.mark.parametrize('after_write', [False, True])
async def test_interrupted_space_cleanup_resumes(tmp_path, monkeypatch, backend, stage, after_write):
    memory = await make_engine(tmp_path, backend)
    unique = await memory.attach('alpha', b'unique', 'text/plain')
    shared = await memory.attach('alpha', b'shared', 'text/plain')
    await memory.attach('bravo', b'shared', 'text/plain')
    source = await memory.remember('alpha', 'delete this source', attachment_ids=[unique.attachment_id, shared.attachment_id])
    other = await memory.remember('bravo', 'keep this source')
    preview = await memory.space_impact('alpha')
    chunk_ids = {c.chunk_id for c in await memory.documents.chunks_of('alpha', source.episode_id)}
    targets = {'blobs': (memory.blobs, 'release_space'), 'rows': (memory.documents, 'delete_space'),
               'vectors': (memory.vectors, 'delete_space'), 'events': (memory.events, 'purge')}
    target, name = targets[stage]
    original = getattr(target, name)

    async def interrupted(*args, **kwargs):
        if kwargs.get('preview'):
            return await original(*args, **kwargs)
        if after_write:
            await original(*args, **kwargs)
        raise RuntimeError('cleanup interruption')

    monkeypatch.setattr(target, name, interrupted)
    with pytest.raises(RuntimeError, match='cleanup interruption'):
        await memory.delete_space('alpha')
    monkeypatch.setattr(target, name, original)
    with pytest.raises(NotFound):
        await memory.remember('alpha', 'must not resurrect')
    if backend == 'sqlite':
        await memory.close()
        memory = await make_engine(tmp_path, backend)
    else:
        receipt = await memory.delete_space('alpha')
        assert receipt == preview.model_copy(update={'deleted_at': NOW})
    try:
        assert await memory.documents.space_deleted('alpha') == NOW
        assert (await memory.documents.counts('alpha')).episodes == 0
        assert chunk_ids.isdisjoint(await memory.vectors.ids('alpha'))
        assert await memory.events.query('alpha') == []
        assert (await memory.attachment('bravo', shared.attachment_id))[1] == b'shared'
        assert (await memory.episode('bravo', other.episode_id)).content == 'keep this source'
        assert (await memory.recover()).spaces_deleted == 0
    finally:
        await memory.close()


async def test_missing_space_catalog_refuses_before_releasing_attachments(tmp_path):
    memory = await make_engine(tmp_path, 'memory')
    blob = await memory.attach('alpha', b'keep this', 'text/plain')
    memory.documents.record_space_deletion = None
    with pytest.raises(InvalidInput, match='space deletion'):
        await memory.delete_space('alpha')
    assert (await memory.attachment('alpha', blob.attachment_id))[1] == b'keep this'
    await memory.close()


@pytest.mark.parametrize('stage', ['intent', 'ack'])
@pytest.mark.parametrize('after_write', [False, True])
async def test_intent_and_ack_failures(tmp_path, monkeypatch, stage, after_write):
    memory = await make_engine(tmp_path, 'sqlite')
    await memory.remember('alpha', 'source')
    method = 'record_space_deletion' if stage == 'intent' else 'clear_space_deletion'
    original = getattr(memory.documents, method)

    async def interrupted(*args):
        if after_write:
            await original(*args)
        raise RuntimeError('interruption')

    monkeypatch.setattr(memory.documents, method, interrupted)
    with pytest.raises(RuntimeError):
        await memory.delete_space('alpha')
    monkeypatch.setattr(memory.documents, method, original)
    if stage == 'intent' and not after_write:
        assert await memory.documents.space_deletion('alpha') is None
        assert (await memory.status('alpha')).episodes == 1
        await memory.delete_space('alpha')
    else:
        await memory.close()
        memory = await make_engine(tmp_path, 'sqlite')
    assert await memory.documents.space_deletion('alpha') is None
    assert (await memory.documents.counts('alpha')).episodes == 0
    await memory.close()


async def test_fallback_vector_adapter_recovers_original_chunk_ids(tmp_path, monkeypatch):
    memory = await make_engine(tmp_path, 'memory')
    await memory.remember('alpha', 'remove vectors after documents')
    await memory.remember('bravo', 'preserve neighbouring vectors')
    other_ids = await memory.vectors.ids('bravo')
    monkeypatch.setattr(memory.vectors, 'delete_space', None)
    original = memory.vectors.delete

    async def interrupted(*args):
        raise RuntimeError('interruption')

    monkeypatch.setattr(memory.vectors, 'delete', interrupted)
    with pytest.raises(RuntimeError):
        await memory.delete_space('alpha')
    assert (await memory.documents.counts('alpha')).chunks == 0
    assert await memory.vectors.ids('alpha')
    monkeypatch.setattr(memory.vectors, 'delete', original)
    report = await memory.recover()
    assert report.spaces_deleted == 1
    assert await memory.vectors.ids('alpha') == []
    assert await memory.vectors.ids('bravo') == other_ids
    await memory.close()


async def test_bounded_space_cleanup_precedes_source_and_ingestion_repair(tmp_path, monkeypatch):
    memory = await make_engine(tmp_path, 'memory')
    original = memory.documents.delete_space

    async def interrupted(*args):
        raise RuntimeError('interruption')

    for space in ['alpha', 'bravo', 'charlie']:
        source = await memory.remember(space, 'source')
        episode = await memory.episode(space, source.episode_id)
        await memory.documents.mark_inflight(space, episode.content_hash)
        monkeypatch.setattr(memory.documents, 'delete_space', interrupted)
        with pytest.raises(RuntimeError):
            await memory.delete_space(space)
        monkeypatch.setattr(memory.documents, 'delete_space', original)
    # Invalid budgets refuse before any pending deletion is advanced.
    for kwargs in [{'space_deletion_limit': True}, {'space_deletion_limit': 0}, {'retirement_limit': 0}]:
        with pytest.raises(InvalidInput):
            await memory.recover(**kwargs)
    assert len(await memory.documents.page_space_deletions(None, 100)) == 3
    report = await memory.recover(space_deletion_limit=2)
    assert report.spaces_deleted == 2 and report.space_deletions_pending and report.completed == 0
    assert len(await memory.documents.inflight()) == 1
    report = await memory.recover(space_deletion_limit=2)
    assert report.spaces_deleted == 1 and not report.space_deletions_pending and report.completed == 0
    assert await memory.documents.inflight() == []
    await memory.close()


async def test_open_refuses_remaining_space_cleanup_backlog(tmp_path):
    from scone_memory.core.models import SpaceReceipt
    from scone_memory.core.space_deletion import SpaceDeletion

    memory = await make_engine(tmp_path, 'memory')
    for number in range(101):
        space = f'space-{number}'
        await memory.documents.record_space_deletion(SpaceDeletion(
            space=space, requested_at=NOW, chunk_ids=(), receipt=SpaceReceipt(
                space=space, episodes=0, chunks=0, facts=0, links=0, tombstones=0, events=0,
            ),
        ))
    with pytest.raises(InvalidInput, match='space cleanup remains'):
        await memory.open()
    assert (await memory.recover()).spaces_deleted == 1
    assert await memory.open() is memory
    await memory.close()


@pytest.mark.parametrize('stage', ['read', 'write'])
async def test_foreign_catalog_response_cannot_redirect_deletion(tmp_path, monkeypatch, stage):
    from scone_memory.core.space_deletion import SpaceDeletion

    memory = await make_engine(tmp_path, 'memory')
    await memory.remember('alpha', 'alpha source')
    await memory.remember('bravo', 'bravo source')
    foreign = SpaceDeletion(space='bravo', requested_at=NOW,
                            chunk_ids=tuple(c for c, _ in await memory.documents.chunk_index('bravo')),
                            receipt=await memory.space_impact('bravo'))

    async def wrong(*args):
        return foreign

    monkeypatch.setattr(memory.documents, 'space_deletion' if stage == 'read' else 'record_space_deletion', wrong)
    with pytest.raises(InvalidInput, match='identity'):
        await memory.delete_space('alpha')
    assert (await memory.documents.counts('alpha')).episodes == 1
    assert (await memory.documents.counts('bravo')).episodes == 1
    await memory.close()


async def test_file_blob_byte_failure_does_not_lose_cleanup_targets(tmp_path, monkeypatch):
    from pathlib import Path

    memory = await make_engine(tmp_path, 'sqlite')
    unique = await memory.attach('alpha', b'private bytes must disappear', 'text/plain')
    shared = await memory.attach('alpha', b'shared retained bytes', 'text/plain')
    await memory.attach('bravo', b'shared retained bytes', 'text/plain')
    blob = memory.blobs._blob(unique.attachment_id)
    original = Path.unlink

    def interrupted(path, *args, **kwargs):
        if path == blob:
            raise OSError('interrupted unlink')
        return original(path, *args, **kwargs)

    monkeypatch.setattr(Path, 'unlink', interrupted)
    with pytest.raises(OSError, match='interrupted unlink'):
        await memory.delete_space('alpha')
    monkeypatch.setattr(Path, 'unlink', original)
    await memory.close()
    memory = await make_engine(tmp_path, 'sqlite')
    assert not blob.exists()
    assert await memory.blobs.held('alpha') == []
    assert (await memory.attachment('bravo', shared.attachment_id))[1] == b'shared retained bytes'
    assert await memory.documents.space_deletion('alpha') is None
    await memory.close()


async def test_open_cleans_deleted_text_before_automatic_reembedding(tmp_path, monkeypatch):
    from scone_memory import HashEmbedder, MemoryEngine

    memory = await make_engine(tmp_path, 'memory')
    await memory.remember('alpha', 'text accepted for deletion')
    original = memory.documents.delete_space

    async def interrupted(*args):
        raise OSError('interrupted rows')

    monkeypatch.setattr(memory.documents, 'delete_space', interrupted)
    with pytest.raises(OSError):
        await memory.delete_space('alpha')
    monkeypatch.setattr(memory.documents, 'delete_space', original)

    class ChangedHash(HashEmbedder):
        def __init__(self):
            super().__init__()
            self.id += '-new-revision'
            self.seen = []

        async def embed(self, texts):
            self.seen.extend(texts)
            return await super().embed(texts)

    embedder = ChangedHash()
    reopened = await MemoryEngine(memory.documents, memory.vectors, embedder,
                                  blobs=memory.blobs, events=memory.events).open()
    assert embedder.seen == []
    assert await memory.documents.space_deletion('alpha') is None
    await reopened.close()


async def test_space_deletion_keeps_pending_source_vector_targets(tmp_path, monkeypatch):
    memory = await make_engine(tmp_path, 'memory')
    source = await memory.remember('alpha', 'previously interrupted source cleanup')
    monkeypatch.setattr(memory.vectors, 'delete_space', None)
    original = memory.vectors.delete

    async def interrupted(*args):
        raise OSError('interrupted vectors')

    monkeypatch.setattr(memory.vectors, 'delete', interrupted)
    with pytest.raises(OSError):
        await memory.forget('alpha', source.episode_id)
    assert (await memory.documents.counts('alpha')).chunks == 0
    assert await memory.vectors.ids('alpha')
    # Whole-space cleanup must retain the IDs even when its first attempt fails.
    with pytest.raises(OSError):
        await memory.delete_space('alpha')
    monkeypatch.setattr(memory.vectors, 'delete', original)
    report = await memory.recover()
    assert report.spaces_deleted == 1 and report.retired == 0
    assert await memory.vectors.ids('alpha') == []
    assert await memory.documents.page_retirements(None, 1) == []
    await memory.close()


async def test_pending_space_is_refused_by_http_while_neighbour_remains_accessible(tmp_path, monkeypatch):
    import httpx
    from scone_memory.api import create_app

    memory = await make_engine(tmp_path, 'memory')
    await memory.remember('alpha', 'closed source')
    await memory.remember('bravo', 'open source')
    app = create_app(memory, {'alpha-key': 'alpha', 'bravo-key': 'bravo'})

    async def interrupted(*args):
        raise OSError('storage unavailable')

    monkeypatch.setattr(memory.documents, 'delete_space', interrupted)
    with pytest.raises(OSError):
        await memory.delete_space('alpha')
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url='http://test') as client:
        for path in ['/v1/status', '/v1/episodes', '/v1/spaces/alpha/impact']:
            response = await client.get(path, headers={'Authorization': 'Bearer alpha-key'})
            assert response.status_code == 404
        assert (await client.get('/v1/status', headers={'Authorization': 'Bearer bravo-key'})).status_code == 200
    await memory.close()


async def test_recovery_finishes_merge_source_cleanup_without_touching_destination(tmp_path, monkeypatch):
    memory = await make_engine(tmp_path, 'sqlite')
    blob = await memory.attach('alpha', b'moved retained evidence', 'text/plain')
    await memory.remember('alpha', 'move this source', attachment_ids=[blob.attachment_id])
    original = memory.vectors.delete_space

    async def interrupted(*args):
        raise OSError('source vector interruption')

    monkeypatch.setattr(memory.vectors, 'delete_space', interrupted)
    with pytest.raises(OSError):
        await memory.merge_space('alpha', into='bravo', confirm='alpha')
    assert await memory.documents.space_deletion('alpha') is not None
    assert (await memory.documents.counts('bravo')).episodes == 1
    monkeypatch.setattr(memory.vectors, 'delete_space', original)
    await memory.close()
    memory = await make_engine(tmp_path, 'sqlite')
    assert await memory.documents.space_deletion('alpha') is None
    assert (await memory.documents.counts('alpha')).episodes == 0
    assert (await memory.documents.counts('bravo')).episodes == 1
    assert (await memory.attachment('bravo', blob.attachment_id))[1] == b'moved retained evidence'
    await memory.close()
