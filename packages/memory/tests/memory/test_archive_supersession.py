"""Archive transfer preserves explicit replacement history across local IDs."""
import copy

import pytest

from scone_memory.core.errors import InvalidInput
from .test_forget_recovery import make_engine


async def source_archive(memory, attachments):
    first = await memory.assert_fact('alpha', 'Alice', 'works_at', 'Acme', valid_from='2026-01-01T00:00:00Z')
    second = await memory.assert_fact('alpha', 'Alice', 'works_at', 'Beta', valid_from='2026-02-01T00:00:00Z')
    third = await memory.assert_fact('alpha', 'Alice', 'works_at', 'Gamma', valid_from='2026-03-01T00:00:00Z')
    rows = [row async for row in memory.export('alpha', include_attachments=attachments)]
    assert (await memory.fact('alpha', first.fact_id)).superseded_by == second.fact_id
    assert (await memory.fact('alpha', second.fact_id)).superseded_by == third.fact_id
    return rows


@pytest.mark.parametrize('backend', ['memory', 'sqlite'])
@pytest.mark.parametrize('attachments', [False, True])
async def test_reordered_chain_remaps_and_retry_is_idempotent(tmp_path, backend, attachments):
    memory = await make_engine(tmp_path, backend)
    rows = await source_archive(memory, attachments)
    await memory.assert_fact('bravo', 'unrelated', 'keeps', 'existing ID')
    facts = [row for row in rows if row['type'] == 'fact']
    rows = [row for row in rows if row['type'] != 'fact'] + list(reversed(facts))
    receipt = await memory.import_records('bravo', rows)
    imported = {f.object: f for f in await memory.documents.list_facts('bravo', include_closed=True)}
    assert imported['Acme'].superseded_by == imported['Beta'].fact_id
    assert imported['Beta'].superseded_by == imported['Gamma'].fact_id
    assert imported['Gamma'].superseded_by is None
    assert receipt.supersessions == 2
    revision = await memory.documents.revision('bravo')
    again = await memory.import_records('bravo', rows)
    assert again.supersessions == 0 and again.facts == 0
    assert await memory.documents.revision('bravo') > revision
    await memory.close()


@pytest.mark.parametrize('problem', ['missing', 'self', 'cycle', 'bool', 'duplicate'])
async def test_malformed_chain_refuses_before_ingesting_any_records(tmp_path, problem):
    memory = await make_engine(tmp_path, 'memory')
    rows = await source_archive(memory, False)
    facts = [row for row in rows if row['type'] == 'fact']
    if problem == 'missing':
        facts[0]['superseded_by'] = 900000
    elif problem == 'self':
        facts[0]['superseded_by'] = facts[0]['fact_id']
    elif problem == 'cycle':
        facts[-1]['superseded_by'] = facts[0]['fact_id']
    elif problem == 'bool':
        facts[0]['superseded_by'] = True
    else:
        facts[-1]['fact_id'] = facts[0]['fact_id']
    rows.append({'type': 'episode', 'content': 'must not write on malformed chain'})
    with pytest.raises(InvalidInput, match='supersession'):
        await memory.import_records('bravo', rows)
    assert (await memory.documents.counts('bravo')).episodes == 0
    assert await memory.documents.list_facts('bravo', include_closed=True) == []
    await memory.close()


async def test_different_destination_successor_refuses_without_overwriting(tmp_path):
    memory = await make_engine(tmp_path, 'memory')
    rows = await source_archive(memory, False)
    without = copy.deepcopy(rows)
    for row in without:
        if row['type'] == 'fact':
            row['superseded_by'] = None
    await memory.import_records('bravo', without)
    facts = {f.object: f for f in await memory.documents.list_facts('bravo', include_closed=True)}
    await memory.documents.update_fact(facts['Acme'].model_copy(update={'superseded_by': facts['Gamma'].fact_id}))
    with pytest.raises(InvalidInput, match='supersession'):
        await memory.import_records('bravo', rows)
    assert (await memory.fact('bravo', facts['Acme'].fact_id)).superseded_by == facts['Gamma'].fact_id
    assert (await memory.fact('bravo', facts['Beta'].fact_id)).superseded_by is None
    await memory.close()


@pytest.mark.parametrize('after_write', [False, True])
async def test_retry_repairs_interrupted_edges_and_invalidates_before_lost_ack(tmp_path, monkeypatch, after_write):
    memory = await make_engine(tmp_path, 'sqlite')
    rows = await source_archive(memory, False)
    original = memory.documents.update_fact
    before = await memory.documents.revision('bravo')

    async def interrupted(fact):
        if after_write:
            await original(fact)
        raise OSError('interrupted edge')

    monkeypatch.setattr(memory.documents, 'update_fact', interrupted)
    with pytest.raises(OSError):
        await memory.import_records('bravo', rows)
    assert await memory.documents.revision('bravo') > before
    monkeypatch.setattr(memory.documents, 'update_fact', original)
    await memory.close()
    memory = await make_engine(tmp_path, 'sqlite')
    result = await memory.import_records('bravo', rows)
    assert result.facts == 0 and result.facts_skipped == 3
    assert result.supersessions == (1 if after_write else 2)
    facts = {f.object: f for f in await memory.documents.list_facts('bravo', include_closed=True)}
    assert facts['Acme'].superseded_by == facts['Beta'].fact_id
    assert facts['Beta'].superseded_by == facts['Gamma'].fact_id
    await memory.close()


async def test_missing_incoming_edge_preserves_destination_edge_and_exclusion(tmp_path):
    memory = await make_engine(tmp_path, 'memory')
    rows = await source_archive(memory, False)
    await memory.import_records('bravo', rows)
    facts = {f.object: f for f in await memory.documents.list_facts('bravo', include_closed=True)}
    await memory.documents.update_fact(facts['Acme'].model_copy(update={'excluded_reason': 'target review decision'}))
    for row in rows:
        if row['type'] == 'fact':
            row['superseded_by'] = None
    result = await memory.import_records('bravo', rows)
    assert result.facts == 0 and result.supersessions == 0
    kept = await memory.fact('bravo', facts['Acme'].fact_id)
    assert kept.superseded_by == facts['Beta'].fact_id and kept.excluded_reason == 'target review decision'
    await memory.close()


async def test_source_identity_collapse_cannot_create_self_reference(tmp_path):
    memory = await make_engine(tmp_path, 'memory')
    rows = await source_archive(memory, False)
    facts = [row for row in rows if row['type'] == 'fact']
    first, second = copy.deepcopy(facts[0]), copy.deepcopy(facts[0])
    first['fact_id'], first['superseded_by'] = 40, 41
    second['fact_id'], second['superseded_by'] = 41, None
    with pytest.raises(InvalidInput, match='supersession'):
        await memory.import_records('bravo', [rows[0], first, second])
    assert await memory.documents.list_facts('bravo', include_closed=True) == []
    await memory.close()


async def test_ambiguous_destination_identity_refuses_before_connecting_history(tmp_path):
    from scone_memory.core.ports import NewFact

    memory = await make_engine(tmp_path, 'memory')
    rows = await source_archive(memory, False)
    without = copy.deepcopy(rows)
    for row in without:
        if row['type'] == 'fact':
            row['superseded_by'] = None
    await memory.import_records('bravo', without)
    facts = {f.object: f for f in await memory.documents.list_facts('bravo', include_closed=True)}
    duplicate = NewFact(**facts['Acme'].model_dump(exclude={'fact_id'}))
    await memory.documents.insert_fact(duplicate)
    with pytest.raises(InvalidInput, match='supersession'):
        await memory.import_records('bravo', rows)
    assert all(f.superseded_by is None for f in await memory.documents.list_facts('bravo', include_closed=True))
    await memory.close()


@pytest.mark.parametrize('lose_ack', [False, True])
async def test_reader_revision_during_edge_write_is_invalidated_afterward(tmp_path, monkeypatch, lose_ack):
    memory = await make_engine(tmp_path, 'memory')
    rows = await source_archive(memory, False)
    without = copy.deepcopy(rows)
    for row in without:
        if row['type'] == 'fact':
            row['superseded_by'] = None
    await memory.import_records('bravo', without)
    observed = []
    original = memory.documents.update_fact

    async def observe(fact):
        observed.append(await memory.documents.revision('bravo'))
        assert (await memory.documents.get_fact('bravo', fact.fact_id)).superseded_by is None
        await original(fact)
        if lose_ack:
            raise OSError('lost edge acknowledgement')

    monkeypatch.setattr(memory.documents, 'update_fact', observe)
    if lose_ack:
        with pytest.raises(OSError):
            await memory.import_records('bravo', rows)
    else:
        await memory.import_records('bravo', rows)
    assert observed and await memory.documents.revision('bravo') > max(observed)
    monkeypatch.setattr(memory.documents, 'update_fact', original)
    await memory.close()


async def test_complete_paged_history_catches_cycle_hidden_by_capped_listing(tmp_path, monkeypatch):
    from scone_memory.core.ports import NewFact

    memory = await make_engine(tmp_path, 'memory')
    a, b, x = [await memory.documents.insert_fact(NewFact(
        space='bravo', subject='alice', predicate='works_at', object=value, valid_from='2026-01-01T00:00:00.000Z',
    )) for value in ['A', 'B', 'X']]
    await memory.documents.update_fact(a.model_copy(update={'superseded_by': x.fact_id}))
    await memory.documents.update_fact(x.model_copy(update={'superseded_by': b.fact_id}))
    rows = [{'type': 'fact', **a.model_dump(), 'superseded_by': None},
            {'type': 'fact', **b.model_dump(), 'superseded_by': a.fact_id}]
    original = memory.documents.list_facts
    pager = memory.documents.page_facts

    async def capped(space, include_closed):
        return (await original(space, include_closed))[:2]

    async def short_page(space, before_id, limit):
        return await pager(space, before_id, 1)

    monkeypatch.setattr(memory.documents, 'list_facts', capped)
    monkeypatch.setattr(memory.documents, 'page_facts', short_page)
    with pytest.raises(InvalidInput, match='supersession'):
        await memory.import_records('bravo', rows)
    assert (await memory.fact('bravo', b.fact_id)).superseded_by is None
    await memory.close()


async def test_supersession_inventory_limit_refuses_without_partial_fact_writes(tmp_path, monkeypatch):
    from scone_memory.memory import archive_supersession

    memory = await make_engine(tmp_path, 'memory')
    rows = await source_archive(memory, False)
    for number in range(3):
        await memory.assert_fact('bravo', f'person-{number}', 'keeps', 'value')
    monkeypatch.setattr(archive_supersession, 'MAX_DESTINATION_FACTS', 2, raising=False)
    with pytest.raises(InvalidInput, match='supersession'):
        await memory.import_records('bravo', rows)
    assert len(await memory.documents.list_facts('bravo', include_closed=True)) == 3
    await memory.close()


async def test_pending_backend_visibility_is_settled_before_inventory(tmp_path, monkeypatch):
    from scone_memory.memory.archive_supersession import inventory

    memory = await make_engine(tmp_path, 'memory')
    row = await memory.assert_fact('bravo', 'Alice', 'keeps', 'pending write')
    page = memory.documents.page_facts
    visible = False

    async def prepare(space):
        nonlocal visible
        assert space == 'bravo'
        visible = True

    async def pending_page(space, before, limit):
        return await page(space, before, limit) if visible else []

    monkeypatch.setattr(memory.documents, 'prepare_ledger_read', prepare, raising=False)
    monkeypatch.setattr(memory.documents, 'page_facts', pending_page)
    assert [fact.fact_id for fact in await inventory(memory.documents, 'bravo')] == [row.fact_id]
    await memory.close()


async def test_elastic_prepares_fact_visibility_without_enabling_bulk_refresh():
    from unittest.mock import AsyncMock
    pytest.importorskip('elasticsearch')
    from scone_memory.backends.elastic import ElasticsearchDocumentStore

    client = AsyncMock()
    store = ElasticsearchDocumentStore(client=client, refresh=False)
    await store.prepare_ledger_read('bravo')
    client.indices.refresh.assert_awaited_once_with(index=store._idx('facts'))
    assert store.shared.refresh is False
