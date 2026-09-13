"""Deletion metadata survives erased data, restart and removed page cursors."""
import pytest
from pydantic import ValidationError

from scone_memory.backends.sqlite import SqliteDocumentStore
from scone_memory.core.errors import InvalidInput
from scone_memory.core.models import SpaceReceipt
from scone_memory.core.space_deletion import SpaceDeletion, SpaceDeletionStore

NOW = '2026-09-12T00:00:00Z'


def intent(space='alpha'):
    return SpaceDeletion(space=space, requested_at=NOW, chunk_ids=(1,), receipt=SpaceReceipt(
        space=space, episodes=1, chunks=1, facts=0, links=0, tombstones=0, events=1,
        attachments_released=['held-id'],
    ))


async def test_first_value_is_detached_and_survives_space_deletion(engine):
    store = engine.documents
    assert isinstance(store, SpaceDeletionStore)
    original = intent()
    saved = await store.record_space_deletion(original)
    original.receipt.attachments_released.append('caller-mutation')
    saved.receipt.attachments_released.clear()
    assert await store.record_space_deletion(intent().model_copy(update={'requested_at': '2026-09-13T00:00:00Z'})) == intent()
    await store.delete_space('alpha', NOW)
    assert await store.space_deletion('alpha') == intent()
    [paged] = await store.page_space_deletions(None, 1)
    paged.receipt.attachments_released.clear()
    assert await store.space_deletion('alpha') == intent()
    assert await store.space_deletion('foreign') is None
    await store.clear_space_deletion('foreign')
    assert await store.space_deletion('alpha') == intent()
    await store.clear_space_deletion('alpha')
    await store.clear_space_deletion('alpha')
    assert await store.space_deletion('alpha') is None


async def test_keyset_pages_are_bounded_after_removed_cursor(engine):
    store = engine.documents
    for name in ['charlie', 'alpha', 'bravo']:
        await store.record_space_deletion(intent(name))
    assert await store.page_space_deletions(None, 2) == [intent('alpha'), intent('bravo')]
    await store.clear_space_deletion('bravo')
    assert await store.page_space_deletions('bravo', 1) == [intent('charlie')]
    assert await store.page_space_deletions('charlie', 1) == []


@pytest.mark.parametrize('limit', [True, 0, -1, 102, 1.5])
async def test_invalid_limits_refuse(engine, limit):
    with pytest.raises(InvalidInput):
        await engine.documents.page_space_deletions(None, limit)


@pytest.mark.parametrize('changes', [
    {'space': 'foreign\n'}, {'space': 'foreign'}, {'chunk_ids': (True,)},
    {'chunk_ids': (1, 1)}, {'chunk_ids': (-1,)}, {'requested_at': 'yesterday'},
    {'receipt': intent().receipt.model_copy(update={'deleted_at': NOW})},
    {'receipt': intent().receipt.model_copy(update={'episodes': -1})},
])
async def test_unchecked_copies_cannot_enter_catalog(engine, changes):
    with pytest.raises((InvalidInput, ValidationError)):
        await engine.documents.record_space_deletion(intent().model_copy(update=changes))
    assert await engine.documents.page_space_deletions(None, 1) == []


async def test_sqlite_catalog_additive_upgrade_and_restart(tmp_path):
    path = tmp_path / 'deletions.db'
    documents = SqliteDocumentStore(path)
    documents.conn.execute('DROP TABLE space_deletions')
    documents.conn.commit()
    await documents.close()
    documents = SqliteDocumentStore(path)
    await documents.record_space_deletion(intent())
    await documents.delete_space('alpha', NOW)
    await documents.close()
    documents = SqliteDocumentStore(path)
    try:
        assert await documents.space_deletion('alpha') == intent()
        await documents.clear_space_deletion('alpha')
    finally:
        await documents.close()
    documents = SqliteDocumentStore(path)
    try:
        assert await documents.page_space_deletions(None, 1) == []
    finally:
        await documents.close()


@pytest.mark.parametrize('point', [False, True])
async def test_catalog_key_cannot_redirect_cleanup(tmp_path, point):
    documents = SqliteDocumentStore(tmp_path / 'deletions.db')
    try:
        await documents.record_space_deletion(intent())
        documents.conn.execute("UPDATE space_deletions SET space = 'foreign'")
        documents.conn.commit()
        with pytest.raises(InvalidInput, match='identity'):
            if point:
                await documents.space_deletion('foreign')
            else:
                await documents.page_space_deletions(None, 1)
    finally:
        await documents.close()
