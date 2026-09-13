"""A space closes only after its retained evidence reaches the destination."""
from __future__ import annotations

import pytest

from scone_memory import HashEmbedder, InMemoryDocumentStore, InMemoryVectorIndex, MemoryEngine
from scone_memory.backends.blobs import FileBlobStore
from scone_memory.backends.sqlite import SqliteDocumentStore, SqliteVectorIndex
from scone_memory.core.errors import InvalidInput, NotFound
from scone_memory.testing import Clock

pytestmark = pytest.mark.asyncio


@pytest.fixture(params=['memory', 'sqlite'])
async def engine(request, tmp_path):
    options = {'clock': Clock('2025-06-01T00:00:00.000Z')}
    if request.param == 'sqlite':
        path = tmp_path / 'merge.db'
        value = MemoryEngine(SqliteDocumentStore(path), SqliteVectorIndex(path), HashEmbedder(),
                             blobs=FileBlobStore(tmp_path / 'blobs'), **options)
    else:
        value = MemoryEngine(InMemoryDocumentStore(), InMemoryVectorIndex(), HashEmbedder(), **options)
    await value.open()
    yield value
    await value.close()


async def sources(engine):
    linked = await engine.attach('old', b'linked original', media_type='text/plain', filename='source.txt')
    unlinked = await engine.attach('old', b'unlinked original', media_type='text/plain', filename='draft.txt')
    note = await engine.remember('old', 'Source evidence', attachment_ids=[linked.attachment_id])
    await engine.remember('new', 'Existing destination source')
    return linked, unlinked, note


async def test_merge_moves_linked_and_unlinked_bytes_and_reports_preview(engine):
    linked, unlinked, note = await sources(engine)
    preview = await engine.merge_space('old', into='new', preview=True)
    assert preview.attachments == 2 and preview.unlinked_attachments == 1
    assert preview.attachment_bytes == len(b'linked originalunlinked original')
    assert not preview.moved and await engine.blobs.held('new') == []
    receipt = await engine.merge_space('old', into='new', confirm='old')
    assert receipt.moved and receipt.attachments == 2
    assert await engine.space_deleted('old') is not None
    assert await engine.attachment('new', linked.attachment_id) == (linked, b'linked original')
    assert await engine.attachment('new', unlinked.attachment_id) == (unlinked, b'unlinked original')
    restored = next(e for e in await engine.documents.recent_episodes('new', 10) if e.content == 'Source evidence')
    assert restored.episode_id != note.episode_id
    assert await engine.blobs.for_episode('new', restored.episode_id) == [linked]
    assert unlinked.attachment_id not in await engine.blobs.linked('new')
    with pytest.raises(NotFound):
        await engine.attachment('old', linked.attachment_id)


async def test_target_tombstone_skips_linked_bytes_but_keeps_unlinked_holds(engine):
    linked, unlinked, _ = await sources(engine)
    forgotten = await engine.remember('new', 'Source evidence')
    await engine.forget('new', forgotten.episode_id)
    preview = await engine.merge_space('old', into='new', preview=True)
    assert preview.tombstoned == 1 and preview.attachments_skipped == 1
    assert preview.attachments == 1
    receipt = await engine.merge_space('old', into='new', confirm='old')
    assert receipt.tombstoned == 1 and receipt.attachments_skipped == 1
    assert await engine.blobs.held('new') == [unlinked.attachment_id]
    assert await engine.space_deleted('old') is not None


@pytest.mark.parametrize('failure', ['put', 'link'])
async def test_partial_transfer_does_not_close_source_and_retry_finishes(engine, monkeypatch, failure):
    linked, unlinked, _ = await sources(engine)
    original = getattr(engine.blobs, failure)
    async def interrupt(space, *args, **kwargs):
        if space == 'new':
            raise OSError('transfer interrupted')
        return await original(space, *args, **kwargs)
    monkeypatch.setattr(engine.blobs, failure, interrupt)
    with pytest.raises(OSError, match='transfer interrupted'):
        await engine.merge_space('old', into='new', confirm='old')
    assert await engine.space_deleted('old') is None
    assert (await engine.documents.counts('old')).episodes == 1
    assert await engine.attachment('old', linked.attachment_id) == (linked, b'linked original')
    monkeypatch.setattr(engine.blobs, failure, original)
    receipt = await engine.merge_space('old', into='new', confirm='old')
    assert receipt.moved and receipt.attachments == 2
    assert (await engine.documents.counts('new')).episodes == 2


async def test_target_metadata_collision_preserves_original_space(engine):
    linked, unlinked, _ = await sources(engine)
    await engine.attach('new', b'unlinked original', media_type='application/octet-stream', filename='different.bin')
    with pytest.raises(InvalidInput, match='attachment'):
        await engine.merge_space('old', into='new', confirm='old')
    assert await engine.space_deleted('old') is None
    assert (await engine.documents.counts('new')).episodes == 1


async def test_source_change_during_copy_refuses_closure(engine, monkeypatch):
    await sources(engine)
    put = engine.blobs.put
    changed = False
    async def mutate(space, *args, **kwargs):
        nonlocal changed
        result = await put(space, *args, **kwargs)
        if space == 'new' and not changed:
            changed = True
            await engine.remember('old', 'Arrived during transfer')
        return result
    monkeypatch.setattr(engine.blobs, 'put', mutate)
    with pytest.raises(InvalidInput, match='source.*changed'):
        await engine.merge_space('old', into='new', confirm='old')
    assert changed and await engine.space_deleted('old') is None
    assert (await engine.documents.counts('old')).episodes == 2


async def test_destination_evidence_removed_after_import_preserves_source(engine, monkeypatch):
    linked, _, _ = await sources(engine)
    imported = engine.import_records
    async def remove(space, *args, **kwargs):
        result = await imported(space, *args, **kwargs)
        for episode in await engine.documents.recent_episodes(space, 10):
            if episode.content == 'Source evidence':
                await engine.blobs.unlink(space, episode.episode_id)
        return result
    monkeypatch.setattr(engine, 'import_records', remove)
    with pytest.raises(InvalidInput, match='destination.*evidence'):
        await engine.merge_space('old', into='new', confirm='old')
    assert await engine.space_deleted('old') is None
    assert await engine.attachment('old', linked.attachment_id) == (linked, b'linked original')


async def test_shared_blob_is_kept_when_only_one_source_is_tombstoned(engine):
    linked, _, _ = await sources(engine)
    await engine.remember('old', 'Second source using the same evidence', attachment_ids=[linked.attachment_id])
    forgotten = await engine.remember('new', 'Source evidence')
    await engine.forget('new', forgotten.episode_id)
    result = await engine.merge_space('old', into='new', confirm='old')
    assert result.tombstoned == 1 and result.attachments_skipped == 0 and result.attachments == 2
    assert await engine.attachment('new', linked.attachment_id) == (linked, b'linked original')


async def test_cli_reports_attachment_counts_and_target_tombstone_skips(engine):
    import io
    from scone_memory.runtime.cli import build_parser, run
    await sources(engine)
    forgotten = await engine.remember('new', 'Source evidence')
    await engine.forget('new', forgotten.episode_id)
    parser = build_parser()
    preview = io.StringIO()
    assert await run(parser.parse_args(['merge-space', '--space', 'old', '--into', 'new', '--dry-run']),
                     engine, io.StringIO(), preview) == 0
    assert '1 attachment(s)' in preview.getvalue() and '1 forgotten source(s) skipped' in preview.getvalue()
    moved = io.StringIO()
    assert await run(parser.parse_args(['merge-space', '--space', 'old', '--into', 'new', '--confirm', 'old']),
                     engine, io.StringIO(), moved) == 0
    assert '1 attachment(s)' in moved.getvalue() and '1 forgotten source(s) skipped' in moved.getvalue()


async def test_persistent_destination_evidence_survives_reopen(tmp_path):
    path = tmp_path / 'persistent.db'
    async def reopen():
        return await MemoryEngine(SqliteDocumentStore(path), SqliteVectorIndex(path), HashEmbedder(),
                                  blobs=FileBlobStore(tmp_path / 'blobs')).open()
    engine = await reopen()
    linked, unlinked, _ = await sources(engine)
    await engine.merge_space('old', into='new', confirm='old')
    await engine.close()
    engine = await reopen()
    try:
        assert await engine.space_deleted('old') is not None
        assert await engine.attachment('new', linked.attachment_id) == (linked, b'linked original')
        assert await engine.attachment('new', unlinked.attachment_id) == (unlinked, b'unlinked original')
        restored = next(e for e in await engine.documents.recent_episodes('new', 10) if e.content == 'Source evidence')
        assert await engine.blobs.for_episode('new', restored.episode_id) == [linked]
    finally:
        await engine.close()


async def test_retained_claim_from_forgotten_source_still_merges_with_disclosure(engine):
    await sources(engine)
    old_note = await engine.remember('old', 'Alice joined Acme')
    await engine.assert_fact('old', 'alice', 'works_at', 'acme', valid_from='2024-01-01T00:00:00Z',
                             source_episode_id=old_note.episode_id, quote='Alice joined Acme')
    await engine.forget('old', old_note.episode_id)
    preview = await engine.merge_space('old', into='new', preview=True)
    assert preview.forgotten_source_references == 1
    result = await engine.merge_space('old', into='new', confirm='old')
    assert result.moved and result.forgotten_source_references == 1
    facts = await engine.documents.list_facts('new', include_closed=True)
    assert len(facts) == 1 and facts[0].quote == 'Alice joined Acme'
    assert facts[0].source_episode_id is None


async def test_source_attachment_relink_during_copy_refuses_closure(engine, monkeypatch):
    linked, _, first = await sources(engine)
    second = await engine.remember('old', 'Second source')
    put = engine.blobs.put
    changed = False
    async def relink(space, *args, **kwargs):
        nonlocal changed
        result = await put(space, *args, **kwargs)
        if space == 'new' and not changed:
            changed = True
            await engine.blobs.link('old', linked.attachment_id, second.episode_id)
            await engine.blobs.unlink('old', first.episode_id)
        return result
    monkeypatch.setattr(engine.blobs, 'put', relink)
    with pytest.raises(InvalidInput, match='source.*changed'):
        await engine.merge_space('old', into='new', confirm='old')
    assert changed and await engine.space_deleted('old') is None
