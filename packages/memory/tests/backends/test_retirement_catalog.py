"""The catalog retains an immutable, paged cleanup intent until acknowledged."""
from __future__ import annotations

import pytest
from pydantic import ValidationError

from scone_memory.backends.sqlite import SqliteDocumentStore
from scone_memory.core.errors import InvalidInput
from scone_memory.core.models import ForgetReceipt
from scone_memory.core.retirement import Retirement, RetirementStore

NOW = "2026-09-11T00:00:00Z"


def intent(space="alpha", episode_id=1):
    return Retirement(space=space, episode_id=episode_id, content_hash=f"digest-{episode_id}",
                      requested_at=NOW, chunk_ids=(episode_id * 10,),
                      receipt=ForgetReceipt(episode_id=episode_id, chunks=1, facts_citing=[2]))


async def test_first_intent_wins_and_all_reads_are_detached(engine):
    store = engine.documents
    assert isinstance(store, RetirementStore)
    original = intent()
    saved = await store.record_retirement(original)
    original.receipt.facts_citing.append(999)
    saved.receipt.facts_citing.append(888)
    assert await store.retirement("alpha", 1) == intent()
    changed = intent().model_copy(update={"requested_at": "2026-09-12T00:00:00Z"})
    assert await store.record_retirement(changed) == intent()
    [paged] = await store.page_retirements(None, 1)
    paged.receipt.facts_citing.clear()
    assert await store.retirement("alpha", 1) == intent()
    assert await store.retirement("foreign", 1) is None
    await store.clear_retirement("foreign", 1)
    assert await store.retirement("alpha", 1) == intent()
    await store.clear_retirement("alpha", 1)
    await store.clear_retirement("alpha", 1)
    assert await store.retirement("alpha", 1) is None


async def test_bounded_keyset_pages_survive_removed_cursor_and_space_deletion(engine):
    store = engine.documents
    expected = [intent("alpha", 1), intent("alpha", 3), intent("bravo", 1), intent("bravo", 2)]
    for record in reversed(expected):
        await store.record_retirement(record)
    assert await store.page_retirements(None, 2) == expected[:2]
    await store.clear_retirement("alpha", 3)
    assert await store.page_retirements(("alpha", 3), 2) == expected[2:]
    assert await store.page_retirements(("bravo", 2), 1) == []
    await store.delete_space("alpha", NOW)
    assert await store.page_retirements(None, 101) == expected[2:]


@pytest.mark.parametrize("limit", [True, 0, -1, 102, 1.5])
async def test_invalid_page_limit_is_refused(engine, limit):
    with pytest.raises(InvalidInput):
        await engine.documents.page_retirements(None, limit)


@pytest.mark.parametrize("changes", [
    {"space": "foreign\n"}, {"episode_id": True}, {"chunk_ids": (True,)},
    {"chunk_ids": (1, 1)}, {"chunk_ids": (-1,)}, {"requested_at": "yesterday"},
    {"receipt": ForgetReceipt(episode_id=999, chunks=1)},
])
async def test_unchecked_copied_intent_cannot_enter_catalog(engine, changes):
    with pytest.raises((ValidationError, InvalidInput)):
        await engine.documents.record_retirement(intent().model_copy(update=changes))
    assert await engine.documents.page_retirements(None, 1) == []


async def test_sqlite_intent_survives_new_connection_and_additive_upgrade(tmp_path):
    path = tmp_path / "catalog.db"
    documents = SqliteDocumentStore(path)
    # An existing catalog produced before this feature has no retirement table.
    documents.conn.execute("DROP TABLE IF EXISTS retirements")
    documents.conn.commit()
    await documents.close()
    documents = SqliteDocumentStore(path)
    await documents.record_retirement(intent())
    await documents.close()
    documents = SqliteDocumentStore(path)
    try:
        assert await documents.retirement("alpha", 1) == intent()
        assert await documents.page_retirements(None, 100) == [intent()]
        await documents.clear_retirement("alpha", 1)
    finally:
        await documents.close()

    documents = SqliteDocumentStore(path)
    try:
        assert await documents.page_retirements(None, 100) == []
    finally:
        await documents.close()


@pytest.mark.parametrize("operation", ["point", "page"])
async def test_mismatched_persisted_key_cannot_redirect_cleanup(tmp_path, operation):
    documents = SqliteDocumentStore(tmp_path / "catalog.db")
    try:
        await documents.record_retirement(intent())
        documents.conn.execute("UPDATE retirements SET space = 'foreign'")
        documents.conn.commit()
        with pytest.raises(InvalidInput, match="identity"):
            if operation == "point":
                await documents.retirement("foreign", 1)
            else:
                await documents.page_retirements(None, 1)
    finally:
        await documents.close()
