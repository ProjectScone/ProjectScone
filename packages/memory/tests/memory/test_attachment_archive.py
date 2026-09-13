"""Portable attachment bytes remain bound to their remapped source episodes."""
from __future__ import annotations

import copy

import pytest

from scone_memory import HashEmbedder, InMemoryDocumentStore, InMemoryVectorIndex, MemoryEngine
from scone_memory.core.errors import InvalidInput
from scone_memory.testing import Clock

pytestmark = pytest.mark.asyncio


async def memory() -> MemoryEngine:
    return await MemoryEngine(InMemoryDocumentStore(), InMemoryVectorIndex(), HashEmbedder(),
                              clock=Clock('2025-06-01T00:00:00.000Z')).open()


async def source_archive():
    source = await memory()
    blob = await source.attach('alpha', b'retained evidence', media_type='text/plain', filename='evidence.txt')
    first = await source.remember('alpha', 'First source', attachment_ids=[blob.attachment_id])
    await source.remember('alpha', 'Second source', attachment_ids=[blob.attachment_id])
    await source.assert_fact('alpha', 'alice', 'knows', 'bob', valid_from='2024-01-01T00:00:00Z',
                             source_episode_id=first.episode_id, quote='First source')
    rows = [row async for row in source.export('alpha', include_attachments=True)]
    return source, blob, rows


async def test_bytes_links_and_fact_provenance_round_trip_with_remapped_ids():
    source, blob, rows = await source_archive()
    target = await memory()
    await target.remember('beta', 'Unrelated target source')
    assert rows[0]['profile'] == 'scone.archive/2'
    assert rows[0]['carries'][-1] == 'attachments'
    assert len([r for r in rows if r['type'] == 'attachment']) == 1
    result = await target.import_records('beta', rows)
    assert result.profile == 'scone.archive/2' and result.episodes == 2
    assert result.attachments == 1 and result.attachment_links == 2
    for episode in await target.documents.recent_episodes('beta', 10):
        linked = await target.blobs.for_episode('beta', episode.episode_id)
        assert linked == ([] if episode.content == 'Unrelated target source' else [blob])
    fact = (await target.documents.list_facts('beta', include_closed=True))[0]
    assert (await target.episode('beta', fact.source_episode_id)).content == 'First source'
    assert await target.attachment('beta', blob.attachment_id) == (blob, b'retained evidence')
    repeated = await target.import_records('beta', rows)
    assert repeated.episodes == 0 and repeated.deduplicated == 2 and repeated.facts == 0
    assert len(await target.blobs.linked('beta')) == 1
    await source.close()
    await target.close()


@pytest.mark.parametrize('corruption', ['hash', 'base64', 'length', 'missing_blob', 'duplicate_blob',
    'dangling_link', 'duplicate_link', 'duplicate_episode', 'unknown_field', 'wrong_space', 'extra_header'])
async def test_invalid_attachment_archive_refuses_before_any_target_write(corruption):
    source, blob, original = await source_archive()
    rows = copy.deepcopy(original)
    attachment = next(r for r in rows if r['type'] == 'attachment')
    episode = next(r for r in rows if r['type'] == 'episode')
    if corruption == 'hash':
        attachment['attachment_id'] = '0' * 64
    elif corruption == 'base64':
        attachment['data_base64'] = '*'
    elif corruption == 'length':
        attachment['bytes'] += 1
    elif corruption == 'missing_blob':
        rows.remove(attachment)
    elif corruption == 'duplicate_blob':
        rows.append(copy.deepcopy(attachment))
    elif corruption == 'dangling_link':
        episode['attachment_ids'] = ['0' * 64]
    elif corruption == 'duplicate_link':
        episode['attachment_ids'] *= 2
    elif corruption == 'duplicate_episode':
        rows.append(copy.deepcopy(episode))
    elif corruption == 'unknown_field':
        attachment['future'] = True
    elif corruption == 'wrong_space':
        episode['space'] = 'other'
    elif corruption == 'extra_header':
        rows.append(copy.deepcopy(rows[0]))
    target = await memory()
    with pytest.raises(InvalidInput):
        await target.import_records('beta', rows)
    assert (await target.documents.counts('beta')).episodes == 0
    assert await target.blobs.held('beta') == []
    await source.close()
    await target.close()


async def test_link_failure_is_reported_and_retry_repairs_without_duplicate_sources(monkeypatch):
    source, blob, rows = await source_archive()
    target = await memory()
    link = target.blobs.link
    async def fail(*args):
        raise OSError('interrupted attachment link')
    monkeypatch.setattr(target.blobs, 'link', fail)
    with pytest.raises(OSError, match='interrupted attachment link'):
        await target.import_records('beta', rows)
    assert (await target.documents.counts('beta')).episodes == 2
    monkeypatch.setattr(target.blobs, 'link', link)
    result = await target.import_records('beta', rows)
    assert result.episodes == 0 and result.deduplicated == 2
    for episode in await target.documents.recent_episodes('beta', 10):
        assert await target.blobs.for_episode('beta', episode.episode_id) == [blob]
    await source.close()
    await target.close()


async def test_corrupt_source_blob_refuses_export_before_header(monkeypatch):
    source, blob, _ = await source_archive()
    async def corrupt(*args):
        return blob, b'changed bytes'
    monkeypatch.setattr(source.blobs, 'get', corrupt)
    with pytest.raises(InvalidInput, match='attachment'):
        await anext(source.export('alpha', include_attachments=True))
    await source.close()


async def test_import_refuses_target_metadata_collision_before_episode_write():
    source, blob, rows = await source_archive()
    target = await memory()
    await target.attach('beta', b'retained evidence', media_type='application/octet-stream', filename='other.bin')
    with pytest.raises(InvalidInput, match='attachment'):
        await target.import_records('beta', rows)
    assert (await target.documents.counts('beta')).episodes == 0
    assert (await target.attachment('beta', blob.attachment_id))[0].filename == 'other.bin'
    await source.close()
    await target.close()


async def test_tombstoned_sources_do_not_resurrect_attachment_bytes():
    source = await memory()
    blob = await source.attach('alpha', b'forgotten bytes', media_type='text/plain')
    await source.remember('alpha', 'Forget this', attachment_ids=[blob.attachment_id])
    rows = [row async for row in source.export('alpha', include_attachments=True)]
    target = await memory()
    note = await target.remember('beta', 'Forget this')
    await target.forget('beta', note.episode_id)
    result = await target.import_records('beta', rows)
    assert result.tombstoned == 1 and result.attachments == 0
    assert await target.blobs.held('beta') == []
    restored = await target.import_records('beta', rows, resurrect=True)
    assert restored.episodes == 1 and restored.attachments == 1
    await source.close()
    await target.close()


@pytest.mark.parametrize('limit', ['per_blob', 'total', 'count', 'rows'])
async def test_archive_resource_limits_refuse_before_target_write(monkeypatch, limit):
    from scone_memory.memory import attachment_archive
    source, blob, rows = await source_archive()
    target = await memory()
    if limit == 'per_blob':
        target.max_attachment_bytes = 1
    elif limit == 'total':
        monkeypatch.setattr(attachment_archive, 'MAX_TOTAL_BYTES', 1)
    elif limit == 'count':
        monkeypatch.setattr(attachment_archive, 'MAX_ATTACHMENTS', 0)
    else:
        monkeypatch.setattr(attachment_archive, 'MAX_ROWS', 2)
    with pytest.raises(InvalidInput, match='limit'):
        await target.import_records('beta', iter(rows))
    assert (await target.documents.counts('beta')).episodes == 0
    assert await target.blobs.held('beta') == []
    await source.close()
    await target.close()


async def test_persistent_round_trip_keeps_bytes_links_and_retry_after_reopen(tmp_path):
    from scone_memory.backends.blobs import FileBlobStore
    from scone_memory.backends.sqlite import SqliteDocumentStore, SqliteVectorIndex
    source, blob, rows = await source_archive()
    path = tmp_path / 'archive.db'
    async def reopen():
        return await MemoryEngine(SqliteDocumentStore(path), SqliteVectorIndex(path), HashEmbedder(),
                                  blobs=FileBlobStore(tmp_path / 'blobs')).open()
    target = await reopen()
    await target.import_records('beta', rows)
    await target.close()
    target = await reopen()
    assert await target.attachment('beta', blob.attachment_id) == (blob, b'retained evidence')
    result = await target.import_records('beta', rows)
    assert result.deduplicated == 2 and result.attachment_links == 2
    for episode in await target.documents.recent_episodes('beta', 10):
        assert await target.blobs.for_episode('beta', episode.episode_id) == [blob]
    await target.close()
    await source.close()


async def test_unlinked_holds_are_disclosed_and_not_transferred():
    source, _, rows = await source_archive()
    held = await source.attach('alpha', b'not linked', media_type='text/plain')
    rows = [row async for row in source.export('alpha', include_attachments=True)]
    assert rows[0]['not_carried'] == {'unlinked_attachments': 1}
    assert held.attachment_id not in {r.get('attachment_id') for r in rows}
    await source.close()


async def test_attachment_dedup_must_not_bind_evidence_to_different_target_source():
    source, _, rows = await source_archive()
    episode = next(r for r in rows if r['type'] == 'episode')
    target = await memory()
    await target.remember('beta', episode['content'], source='different source')
    with pytest.raises(InvalidInput, match='episode.*conflict'):
        await target.import_records('beta', rows)
    assert await target.blobs.held('beta') == []
    assert (await target.documents.counts('beta')).episodes == 1
    await target.close()
    await source.close()


async def test_source_deleted_while_exporting_refuses_before_emitting_archive(monkeypatch):
    source, blob, _ = await source_archive()
    get = source.blobs.get
    async def delete_source(space, attachment_id):
        result = await get(space, attachment_id)
        for episode in await source.documents.recent_episodes(space, 10):
            await source.documents.delete_episode(space, episode.episode_id)
        return result
    monkeypatch.setattr(source.blobs, 'get', delete_source)
    with pytest.raises(InvalidInput, match='source.*changed'):
        await anext(source.export('alpha', include_attachments=True))
    await source.close()


@pytest.mark.parametrize('corruption', ['source_reference', 'foreign_fact', 'duplicate_fact'])
async def test_archive_two_refuses_ambiguous_ledger_references_before_write(corruption):
    source, _, rows = await source_archive()
    fact = next(row for row in rows if row['type'] == 'fact')
    if corruption == 'source_reference':
        fact['source_episode_id'] = 999999
    elif corruption == 'foreign_fact':
        fact['space'] = 'foreign'
    else:
        rows.append({**fact, 'object': 'someone else'})
    target = await memory()
    with pytest.raises(InvalidInput, match='archive'):
        await target.import_records('beta', rows)
    assert await target.blobs.held('beta') == []
    assert (await target.documents.counts('beta')).episodes == 0
    await target.close()
    await source.close()


async def test_caller_mutation_during_staging_cannot_change_validated_evidence(monkeypatch):
    source, blob, rows = await source_archive()
    episode_row = next(row for row in rows if row['type'] == 'episode')
    target = await memory()
    put = target.blobs.put
    async def mutate(*args, **kwargs):
        episode_row['metadata']['document_original'] = 'f' * 64
        return await put(*args, **kwargs)
    monkeypatch.setattr(target.blobs, 'put', mutate)
    await target.import_records('beta', rows)
    for episode in await target.documents.recent_episodes('beta', 10):
        assert episode.metadata == {}
        assert await target.blobs.for_episode('beta', episode.episode_id) == [blob]
    await target.close()
    await source.close()
