"""Real source changes, deletion, suppression and interruption recovery."""
import pytest

from scone_memory import HashEmbedder, InMemoryDocumentStore, InMemoryVectorIndex, MemoryEngine
from scone_memory.core.errors import Gone, InvalidInput
from scone_memory.ingestion.files import document_provenance
from scone_memory.ingestion.formats.types import DocumentSegment, ParsedDocument


class CountingEmbedder(HashEmbedder):
    def __init__(self):
        super().__init__()
        self.calls = 0

    async def embed(self, texts):
        self.calls += 1
        return await super().embed(texts)


class TextParser:
    def __init__(self):
        self.calls = 0

    async def parse(self, data, filename, limits):
        self.calls += 1
        return ParsedDocument(format='text', parser='test-local',
                             segments=(DocumentSegment(text=data.decode(), locator='line:1'),))


@pytest.fixture(params=['memory', 'sqlite'])
async def env(request, tmp_path):
    from scone_memory.backends import SqliteDocumentStore, SqliteVectorIndex
    from scone_memory.backends.blobs import FileBlobStore
    root = tmp_path / 'sources'
    root.mkdir()
    if request.param == 'sqlite':
        documents, vectors = SqliteDocumentStore(tmp_path / 'store.db'), SqliteVectorIndex(tmp_path / 'store.db')
    else:
        documents, vectors = InMemoryDocumentStore(), InMemoryVectorIndex()
    engine = await MemoryEngine(documents, vectors, CountingEmbedder(), blobs=FileBlobStore(tmp_path / 'blobs')).open()
    yield engine, root, tmp_path
    await engine.close()


def runner(env, **changes):
    from scone_memory.ingestion.directory_sync import DirectorySync
    memory, root, tmp_path = env
    options = dict(space='alpha', journal=tmp_path / 'sync.journal', key=b'k' * 32,
                   store_id='fixture-catalog', parser_revision='v1', parser=TextParser())
    return DirectorySync(memory, root, **(options | changes))


def receipts(result):
    return {item.path: item for item in result.receipts}


@pytest.mark.parametrize('failure', ['vectors', 'blobs', 'stone'])
async def test_directory_update_resumes_its_catalog_retirement(env, monkeypatch, failure):
    memory, root, _ = env
    (root / 'report.txt').write_text('Old observatory schedule')
    sync = runner(env)
    original = receipts(await sync.synchronize())['report.txt'].episode_id
    (root / 'report.txt').write_text('New observatory schedule')
    target, method = {
        'vectors': (memory.vectors, 'delete'), 'blobs': (memory.blobs, 'unlink'),
        'stone': (memory.documents, 'record_tombstone'),
    }[failure]
    operation = getattr(target, method)

    async def interrupt(*args, **kwargs):
        raise RuntimeError('cleanup interrupted')

    monkeypatch.setattr(target, method, interrupt)
    first = await sync.synchronize()
    assert not first.complete
    assert await memory.documents.retirement('alpha', original) is not None
    monkeypatch.setattr(target, method, operation)
    repaired = await sync.synchronize()
    assert repaired.complete
    assert receipts(repaired)['report.txt'].status == 'updated'
    assert await memory.documents.retirement('alpha', original) is None
    with pytest.raises(Gone):
        await memory.episode('alpha', original)


async def test_three_file_cycle_updates_deletes_and_reuses_unchanged(env):
    memory, root, _ = env
    for path, content in [('edit.txt', 'Old observatory schedule'), ('delete.txt', 'Retired telescope instructions'),
                          ('keep.txt', 'Stable calibration report')]:
        (root / path).write_text(content)
    sync = runner(env)
    first = await sync.synchronize()
    assert first.complete and {item.status for item in first.receipts} == {'added'}
    before = receipts(first)
    count = memory.embedder.calls
    parse_count = sync.parser.calls
    (root / 'edit.txt').write_text('New observatory schedule')
    (root / 'delete.txt').unlink()
    updated = await sync.synchronize(delete_missing=True)
    assert updated.complete
    after = receipts(updated)
    assert {path: item.status for path, item in after.items()} == {'edit.txt': 'updated', 'delete.txt': 'deleted', 'keep.txt': 'unchanged'}
    assert memory.embedder.calls == count + 1 and sync.parser.calls == parse_count + 1
    assert after['keep.txt'].episode_id == before['keep.txt'].episode_id
    for path in ('edit.txt', 'delete.txt'):
        with pytest.raises(Gone):
            await memory.episode('alpha', before[path].episode_id)
    provenance = await document_provenance(memory, 'alpha', after['edit.txt'].episode_id)
    assert provenance.segments[0].text == 'New observatory schedule'
    recalled = await memory.recall('alpha', 'observatory telescope calibration', limit=10)
    assert before['edit.txt'].episode_id not in {item.episode_id for item in recalled.items}
    assert before['delete.txt'].episode_id not in {item.episode_id for item in recalled.items}
    assert after['edit.txt'].episode_id in {item.episode_id for item in recalled.items}
    assert (await runner(env).synchronize(delete_missing=True)).complete


async def test_two_identical_paths_and_unmanaged_ingestion_remain_independent(env):
    from scone_memory.ingestion.files import ingest_document
    memory, root, _ = env
    for name in ('one.txt', 'two.txt'):
        (root / name).write_bytes(b'Shared source evidence')
    unmanaged = await ingest_document(memory, 'alpha', b'Shared source evidence', filename='one.txt', parser=TextParser())
    first = receipts(await runner(env).synchronize())
    assert first['one.txt'].episode_id != first['two.txt'].episode_id
    (root / 'one.txt').unlink()
    assert (await runner(env).synchronize(delete_missing=True)).complete
    assert (await document_provenance(memory, 'alpha', first['two.txt'].episode_id)).segments
    assert (await document_provenance(memory, 'alpha', unmanaged.added.episode_id)).segments


async def test_retired_bytes_can_return_and_deleted_file_can_reappear(env):
    memory, root, _ = env
    path = root / 'report.txt'
    ids = []
    for text in ('version A', 'version B', 'version A'):
        path.write_text(text)
        result = await runner(env).synchronize()
        assert result.complete
        ids.append(result.receipts[0].episode_id)
    assert len(set(ids)) == 3
    path.unlink()
    assert (await runner(env).synchronize(delete_missing=True)).complete
    path.write_text('version A')
    returned = await runner(env).synchronize()
    assert returned.complete and returned.receipts[0].episode_id not in ids


async def test_external_forget_stays_suppressed_even_after_file_changes(env):
    memory, root, _ = env
    (root / 'report.txt').write_text('forget this source')
    first = await runner(env).synchronize()
    await memory.forget('alpha', first.receipts[0].episode_id)
    calls = memory.embedder.calls
    for text in ('forget this source', 'changed but still forgotten'):
        (root / 'report.txt').write_text(text)
        result = await runner(env).synchronize(delete_missing=True)
        assert result.complete and result.receipts[0].status == 'suppressed'
    assert memory.embedder.calls == calls


@pytest.mark.parametrize('incomplete', ['symlink', 'limit', 'second_scan_changed'])
async def test_missing_deletion_requires_explicit_complete_stable_inventory(env, incomplete, monkeypatch):
    from scone_memory.ingestion.source_scan import ScanLimits
    memory, root, _ = env
    (root / 'delete.txt').write_text('preserve until confirmed missing')
    first = await runner(env).synchronize()
    episode_id = first.receipts[0].episode_id
    (root / 'delete.txt').unlink()
    assert (await runner(env).synchronize()).complete
    assert (await memory.episode('alpha', episode_id)).content
    if incomplete == 'symlink':
        (root / 'link').symlink_to(root / 'missing')
        sync = runner(env)
    elif incomplete == 'limit':
        (root / 'huge.txt').write_text('too large')
        sync = runner(env, scan_limits=ScanLimits(max_file_bytes=1))
    else:
        sync = runner(env)
        scan = sync.scanner.scan
        calls = 0
        def changing_scan():
            nonlocal calls
            calls += 1
            if calls == 2:
                (root / 'new.txt').write_text('arrived after inventory')
            return scan()
        monkeypatch.setattr(sync.scanner, 'scan', changing_scan)
    result = await sync.synchronize(delete_missing=True)
    assert not result.complete and result.issues
    assert (await memory.episode('alpha', episode_id)).content


@pytest.mark.parametrize('crash', ['after_index', 'before_retire_save', 'before_forget', 'after_forget', 'before_final_save'])
async def test_interrupted_replacement_recovers_exact_revision(env, crash, monkeypatch):
    import scone_memory.ingestion.directory_sync as module
    memory, root, _ = env
    path = root / 'report.txt'
    path.write_text('old source')
    old = (await runner(env).synchronize()).receipts[0].episode_id
    path.write_text('new source')
    sync = runner(env)
    fired = False
    save = sync.journal.save
    forget = memory.forget
    store = module.store_document
    def interrupted_save(state):
        nonlocal fired
        phase = state.entries['report.txt'].state
        if not fired and ((crash == 'before_retire_save' and phase == 'retire') or (crash == 'before_final_save' and phase == 'active' and state.entries['report.txt'].current.generation > 0)):
            fired = True
            raise OSError('simulated power loss')
        return save(state)
    async def interrupted_forget(*args, **kwargs):
        nonlocal fired
        if not fired and crash == 'before_forget':
            fired = True
            raise OSError('simulated power loss')
        result = await forget(*args, **kwargs)
        if not fired and crash == 'after_forget':
            fired = True
            raise OSError('simulated power loss')
        return result
    async def interrupted_store(*args, **kwargs):
        nonlocal fired
        result = await store(*args, **kwargs)
        if not fired and crash == 'after_index':
            fired = True
            raise OSError('simulated power loss')
        return result
    monkeypatch.setattr(sync.journal, 'save', interrupted_save)
    monkeypatch.setattr(memory, 'forget', interrupted_forget)
    monkeypatch.setattr(module, 'store_document', interrupted_store)
    failed = await sync.synchronize()
    assert fired and not failed.complete
    if crash not in ('after_forget', 'before_final_save'):
        assert (await memory.episode('alpha', old)).content == 'old source'
    calls = memory.embedder.calls
    resumed = await runner(env).synchronize()
    assert resumed.complete
    assert memory.embedder.calls == calls
    current = receipts(resumed)['report.txt'].episode_id
    assert (await document_provenance(memory, 'alpha', current)).segments[0].text == 'new source'
    with pytest.raises(Gone):
        await memory.episode('alpha', old)


async def test_parser_failure_keeps_prior_episode_and_other_sources_progress(env):
    memory, root, _ = env
    path = root / 'report.txt'
    path.write_text('old source')
    old = (await runner(env).synchronize()).receipts[0].episode_id
    path.write_text('invalid format')
    (root / 'other.txt').write_text('good format')
    class FailingParser(TextParser):
        async def parse(self, data, filename, limits):
            if filename == 'report.txt':
                raise InvalidInput('private parser detail must not be in receipt')
            return await super().parse(data, filename, limits)
    result = await runner(env, parser=FailingParser()).synchronize()
    assert not result.complete
    assert receipts(result)['other.txt'].status == 'added'
    assert receipts(result)['report.txt'].status == 'failed'
    assert 'private parser detail' not in repr(result)
    assert (await memory.episode('alpha', old)).content == 'old source'


async def test_journal_inside_source_root_is_refused(env):
    _, root, _ = env
    with pytest.raises(InvalidInput):
        runner(env, journal=root / 'state.journal')


async def test_pending_update_recovers_even_if_file_is_now_missing_without_deletion(env, monkeypatch):
    import scone_memory.ingestion.directory_sync as module
    memory, root, _ = env
    path = root / 'report.txt'
    path.write_text('old source')
    old = (await runner(env).synchronize()).receipts[0].episode_id
    path.write_text('new source')
    store = module.store_document
    async def fail_after_index(*args, **kwargs):
        await store(*args, **kwargs)
        raise OSError('interrupt between stages')
    with monkeypatch.context() as patch:
        patch.setattr(module, 'store_document', fail_after_index)
        assert not (await runner(env).synchronize()).complete
    path.unlink()
    result = await runner(env).synchronize()
    assert result.complete and result.receipts[0].status == 'updated'
    assert (await document_provenance(memory, 'alpha', result.receipts[0].episode_id)).segments[0].text == 'new source'
    with pytest.raises(Gone):
        await memory.episode('alpha', old)


async def test_reappearing_file_is_not_forgotten_after_deletion_intent(env, monkeypatch):
    memory, root, _ = env
    path = root / 'report.txt'
    path.write_text('retained source')
    old = (await runner(env).synchronize()).receipts[0].episode_id
    path.unlink()
    sync = runner(env)
    save = sync.journal.save
    def reappear(state):
        save(state)
        if state.entries['report.txt'].state == 'delete':
            path.write_text('returned before deletion')
    monkeypatch.setattr(sync.journal, 'save', reappear)
    result = await sync.synchronize(delete_missing=True)
    assert not result.complete
    assert (await memory.episode('alpha', old)).content == 'retained source'
    resumed = await runner(env).synchronize(delete_missing=True)
    assert resumed.complete
    assert (await memory.episode('alpha', resumed.receipts[0].episode_id)).content == 'returned before deletion'


@pytest.mark.parametrize('damage', ['inflight', 'chunks'])
async def test_incomplete_new_index_does_not_retire_current_source(env, monkeypatch, damage):
    import scone_memory.ingestion.directory_sync as module
    memory, root, _ = env
    path = root / 'report.txt'
    path.write_text('old source')
    old = (await runner(env).synchronize()).receipts[0].episode_id
    path.write_text('new source')
    store = module.store_document
    chunks_of = memory.documents.chunks_of
    pending_id = None
    async def index_but_damage(*args, **kwargs):
        nonlocal pending_id
        result = await store(*args, **kwargs)
        pending_id = result.added.episode_id
        episode = await memory.episode('alpha', pending_id)
        if damage == 'inflight':
            await memory.documents.mark_inflight('alpha', episode.content_hash)
        return result
    async def missing_chunks(space, episode_id):
        return [] if episode_id == pending_id else await chunks_of(space, episode_id)
    monkeypatch.setattr(module, 'store_document', index_but_damage)
    if damage == 'chunks':
        monkeypatch.setattr(memory.documents, 'chunks_of', missing_chunks)
    result = await runner(env).synchronize()
    assert not result.complete
    assert (await memory.episode('alpha', old)).content == 'old source'


@pytest.mark.parametrize('forgotten', ['old', 'pending'])
async def test_external_forget_after_interruption_retires_both_owned_revisions(env, monkeypatch, forgotten):
    import scone_memory.ingestion.directory_sync as module
    memory, root, _ = env
    path = root / 'report.txt'
    path.write_text('old source')
    old = (await runner(env).synchronize()).receipts[0].episode_id
    path.write_text('new source')
    store = module.store_document
    pending_id = None
    async def fail_after_index(*args, **kwargs):
        nonlocal pending_id
        result = await store(*args, **kwargs)
        pending_id = result.added.episode_id
        raise OSError('interrupted')
    with monkeypatch.context() as patch:
        patch.setattr(module, 'store_document', fail_after_index)
        assert not (await runner(env).synchronize()).complete
    await memory.forget('alpha', old if forgotten == 'old' else pending_id)
    resumed = await runner(env).synchronize()
    assert resumed.complete and resumed.receipts[0].status == 'suppressed'
    for episode_id in (old, pending_id):
        with pytest.raises(Gone):
            await memory.episode('alpha', episode_id)
    assert (await runner(env).synchronize()).receipts[0].status == 'suppressed'


async def test_parser_revision_reindexes_unchanged_bytes(env):
    memory, root, _ = env
    (root / 'report.txt').write_text('same source bytes')
    first = await runner(env).synchronize()
    calls = memory.embedder.calls
    second = await runner(env, parser_revision='v2').synchronize()
    assert second.complete and second.receipts[0].status == 'updated'
    assert memory.embedder.calls == calls + 1
    assert second.receipts[0].episode_id != first.receipts[0].episode_id


@pytest.mark.parametrize('changed', ['store', 'suffixes', 'root'])
async def test_journal_binding_changes_refuse_before_any_source_writes(env, changed):
    memory, root, tmp_path = env
    (root / 'report.txt').write_text('owned source')
    first = await runner(env).synchronize()
    calls = memory.embedder.calls
    journal_before = (tmp_path / 'sync.journal').read_bytes()
    if changed == 'root':
        root.rename(tmp_path / 'previous-root')
        root.mkdir()
        sync = runner(env)
    else:
        sync = runner(env, **({'store_id': 'foreign-store'} if changed == 'store' else {'extensions': frozenset({'.txt'})}))
    with pytest.raises(InvalidInput):
        await sync.synchronize(delete_missing=True)
    assert memory.embedder.calls == calls
    assert (tmp_path / 'sync.journal').read_bytes() == journal_before
    assert (await memory.episode('alpha', first.receipts[0].episode_id)).content == 'owned source'


async def test_foreign_empty_catalog_with_reused_store_label_cannot_apply_owned_journal(env):
    memory, root, tmp_path = env
    (root / 'report.txt').write_text('owned source')
    await runner(env).synchronize()
    (root / 'another.txt').write_text('must not be inserted into foreign catalog')
    foreign = await MemoryEngine(InMemoryDocumentStore(), InMemoryVectorIndex(), CountingEmbedder()).open()
    try:
        with pytest.raises(InvalidInput):
            await runner((foreign, root, tmp_path)).synchronize()
        assert foreign.embedder.calls == 0
    finally:
        await foreign.close()


@pytest.mark.parametrize('after', [False, True])
async def test_interrupted_missing_deletion_retries_without_touching_other_sources(env, monkeypatch, after):
    memory, root, _ = env
    (root / 'delete.txt').write_text('managed missing source')
    old = (await runner(env).synchronize()).receipts[0].episode_id
    other = await memory.remember('alpha', 'unrelated original')
    (root / 'delete.txt').unlink()
    forget = memory.forget
    async def interrupt(*args, **kwargs):
        if after:
            await forget(*args, **kwargs)
        raise OSError('interrupted deletion')
    with monkeypatch.context() as patch:
        patch.setattr(memory, 'forget', interrupt)
        assert not (await runner(env).synchronize(delete_missing=True)).complete
    resumed = await runner(env).synchronize(delete_missing=True)
    assert resumed.complete and resumed.receipts[0].status == 'deleted'
    with pytest.raises(Gone):
        await memory.episode('alpha', old)
    assert (await memory.episode('alpha', other.episode_id)).content == 'unrelated original'


async def test_pending_deletion_is_disclosed_when_retry_omits_delete_flag(env, monkeypatch):
    memory, root, _ = env
    (root / 'delete.txt').write_text('managed missing source')
    old = (await runner(env).synchronize()).receipts[0].episode_id
    (root / 'delete.txt').unlink()
    async def interrupt(*args, **kwargs):
        raise OSError('interrupted deletion')
    with monkeypatch.context() as patch:
        patch.setattr(memory, 'forget', interrupt)
        assert not (await runner(env).synchronize(delete_missing=True)).complete
    deferred = await runner(env).synchronize()
    assert not deferred.complete and deferred.receipts[0].code == 'deletion_pending'
    assert (await memory.episode('alpha', old)).content == 'managed missing source'
    assert (await runner(env).synchronize(delete_missing=True)).complete
