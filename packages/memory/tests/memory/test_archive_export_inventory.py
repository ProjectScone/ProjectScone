"""Source exports must not inherit backend result-window truncation."""
import pytest

from scone_memory.core.errors import InvalidInput
from .test_forget_recovery import make_engine


@pytest.mark.parametrize('backend', ['memory', 'sqlite'])
@pytest.mark.parametrize('attachments', [False, True])
async def test_export_uses_complete_short_pages_not_capped_lists(tmp_path, monkeypatch, backend, attachments):
    memory = await make_engine(tmp_path, backend)
    for number in range(4):
        source = await memory.remember('alpha', f'evidence {number}')
        await memory.assert_fact('alpha', f'person-{number}', 'keeps', f'value-{number}', source_episode_id=source.episode_id)
    recent, listed = memory.documents.recent_episodes, memory.documents.list_facts
    episodes, facts = memory.documents.page_episodes, memory.documents.page_facts
    calls = []

    async def capped_recent(space, limit):
        return (await recent(space, limit))[:2]

    async def capped_facts(space, include_closed):
        return (await listed(space, include_closed))[:2]

    async def episode_page(space, before, limit, kind):
        calls.append(('episode', before))
        return await episodes(space, before, 1, kind)

    async def fact_page(space, before, limit):
        calls.append(('fact', before))
        return await facts(space, before, 1)

    monkeypatch.setattr(memory.documents, 'recent_episodes', capped_recent)
    monkeypatch.setattr(memory.documents, 'list_facts', capped_facts)
    monkeypatch.setattr(memory.documents, 'page_episodes', episode_page)
    monkeypatch.setattr(memory.documents, 'page_facts', fact_page)
    rows = [row async for row in memory.export('alpha', include_attachments=attachments)]
    assert len([row for row in rows if row['type'] == 'episode']) == 4
    assert len([row for row in rows if row['type'] == 'fact']) == 4
    assert len(calls) == 10, 'four single-row pages plus empty for each inventory'
    await memory.close()


@pytest.mark.parametrize('kind', ['episode', 'fact'])
@pytest.mark.parametrize('problem', ['foreign', 'repeat', 'boolean'])
async def test_export_refuses_invalid_pages(tmp_path, monkeypatch, kind, problem):
    memory = await make_engine(tmp_path, 'memory')
    source = await memory.remember('alpha', 'source')
    fact = await memory.assert_fact('alpha', 'Alice', 'keeps', 'evidence')
    row = await memory.episode('alpha', source.episode_id) if kind == 'episode' else await memory.fact('alpha', fact.fact_id)
    key = 'episode_id' if kind == 'episode' else 'fact_id'
    if problem == 'foreign':
        row = row.model_copy(update={'space': 'foreign'})
    elif problem == 'boolean':
        row = row.model_copy(update={key: True})

    async def invalid(*args):
        return [row]

    monkeypatch.setattr(memory.documents, 'page_episodes' if kind == 'episode' else 'page_facts', invalid)
    with pytest.raises(InvalidInput, match='archive'):
        _ = [row async for row in memory.export('alpha')]
    await memory.close()


async def test_export_prepares_visibility_before_counting_and_paging(tmp_path, monkeypatch):
    memory = await make_engine(tmp_path, 'memory')
    await memory.remember('alpha', 'pending source')
    calls = []
    counts = memory.documents.counts

    async def prepare(space):
        calls.append(('prepare', space))

    async def counted(space):
        calls.append(('count', space))
        return await counts(space)

    monkeypatch.setattr(memory.documents, 'prepare_archive_read', prepare, raising=False)
    monkeypatch.setattr(memory.documents, 'counts', counted)
    _ = [row async for row in memory.export('alpha')]
    assert calls[0] == ('prepare', 'alpha')
    await memory.close()


async def test_export_more_than_ten_thousand_sources_and_ledger_records(tmp_path):
    from scone_memory.core.affirmations import NewAffirmation
    from scone_memory.core.ports import NewEpisode, NewFact, NewFactLink
    from .test_forget_recovery import NOW

    memory = await make_engine(tmp_path, 'memory')
    size = 10_003
    for number in range(size):
        episode = await memory.documents.insert_episode(NewEpisode(
            space='alpha', kind='note', content=f'source {number}', content_hash=f'digest-{number}',
            created_at=NOW, ingested_at=NOW,
        ))
        fact = await memory.documents.insert_fact(NewFact(
            space='alpha', subject=f'person-{number}', predicate='keeps', object='evidence', valid_from=NOW,
            source_episode_id=episode.episode_id,
        ))
        await memory.documents.add_affirmation(NewAffirmation(
            space='alpha', fact_id=fact.fact_id, valid_from=NOW, recorded_at=NOW,
            source_episode_id=episode.episode_id,
        ))
        if number:
            await memory.documents.insert_fact_link(NewFactLink(
                space='alpha', from_fact=1, to_fact=fact.fact_id, kind='supports', created_at=NOW,
            ))
    found = {'episode': set(), 'fact': set(), 'fact_link': set(), 'affirmation': set()}
    keys = {'episode': 'episode_id', 'fact': 'fact_id', 'fact_link': 'link_id', 'affirmation': 'affirmation_id'}
    async for row in memory.export('alpha'):
        kind = row['type']
        if kind in found:
            assert row[keys[kind]] not in found[kind]
            found[kind].add(row[keys[kind]])
    assert {kind: len(values) for kind, values in found.items()} == {
        'episode': size, 'fact': size, 'fact_link': size - 1, 'affirmation': size,
    }
    await memory.close()


@pytest.mark.parametrize('operation', ['page_episodes', 'page_facts'])
async def test_custom_missing_pager_refuses_before_header(tmp_path, monkeypatch, operation):
    memory = await make_engine(tmp_path, 'memory')
    monkeypatch.setattr(memory.documents, operation, None)
    output = memory.export('alpha')
    with pytest.raises(InvalidInput, match='archive'):
        await anext(output)
    await memory.close()


async def test_missing_episode_page_refuses_count_mismatch(tmp_path, monkeypatch):
    memory = await make_engine(tmp_path, 'memory')
    await memory.remember('alpha', 'source cannot disappear')

    async def missing(*args):
        return []

    monkeypatch.setattr(memory.documents, 'page_episodes', missing)
    with pytest.raises(InvalidInput, match='source count'):
        _ = [row async for row in memory.export('alpha')]
    await memory.close()


@pytest.mark.parametrize('kind', ['fact_link', 'affirmation'])
@pytest.mark.parametrize('attachments', [False, True])
async def test_export_never_strips_foreign_scope_from_ledger_evidence(tmp_path, monkeypatch, kind, attachments):
    from scone_memory.core.affirmations import Affirmation
    from scone_memory.core.models import FactLink
    from .test_forget_recovery import NOW

    memory = await make_engine(tmp_path, 'memory')
    for name in ['Alice', 'Bob']:
        await memory.assert_fact('alpha', name, 'keeps', 'value')
    if kind == 'fact_link':
        wrong = FactLink(link_id=1, space='bravo', from_fact=1, to_fact=2, kind='supports',
                         created_at=NOW, quote='FOREIGN PRIVATE EVIDENCE')
    else:
        wrong = Affirmation(affirmation_id=1, space='bravo', fact_id=1, valid_from=NOW,
                            recorded_at=NOW, quote='FOREIGN PRIVATE EVIDENCE')

    async def foreign(*args):
        return [wrong]

    monkeypatch.setattr(memory.documents, 'space_fact_links' if kind == 'fact_link' else 'space_affirmations', foreign, raising=False)
    monkeypatch.setattr(memory.documents, 'fact_links', foreign if kind == 'fact_link' else memory.documents.fact_links)
    emitted = []
    with pytest.raises(InvalidInput, match='archive'):
        async for row in memory.export('alpha', include_attachments=attachments):
            emitted.append(row)
    assert all(row.get('quote') != 'FOREIGN PRIVATE EVIDENCE' for row in emitted)
    await memory.close()


async def test_space_link_inventory_keeps_links_with_missing_endpoints(tmp_path):
    from scone_memory.core.ports import NewFactLink
    from .test_forget_recovery import NOW

    memory = await make_engine(tmp_path, 'memory')
    link = await memory.documents.insert_fact_link(NewFactLink(
        space='alpha', from_fact=90, to_fact=91, kind='supports', created_at=NOW,
    ))
    rows = [row async for row in memory.export('alpha')]
    assert [row['link_id'] for row in rows if row['type'] == 'fact_link'] == [link.link_id]
    # Strict evidence archives refuse a dangling endpoint; they never omit it.
    with pytest.raises(InvalidInput, match='dangling'):
        _ = [row async for row in memory.export('alpha', include_attachments=True)]
    await memory.close()


@pytest.mark.parametrize('kind', ['link', 'affirmation'])
async def test_conflicting_duplicate_ledger_identity_refuses(tmp_path, monkeypatch, kind):
    from scone_memory.core.affirmations import Affirmation
    from scone_memory.core.models import FactLink
    from .test_forget_recovery import NOW

    memory = await make_engine(tmp_path, 'memory')
    if kind == 'link':
        first = FactLink(link_id=1, space='alpha', from_fact=90, to_fact=91, kind='supports', created_at=NOW)
    else:
        first = Affirmation(affirmation_id=1, space='alpha', fact_id=90, valid_from=NOW, recorded_at=NOW)

    async def duplicates(*args):
        return [first, first.model_copy(update={'quote': 'different evidence'})]

    monkeypatch.setattr(memory.documents, 'space_fact_links' if kind == 'link' else 'space_affirmations', duplicates)
    with pytest.raises(InvalidInput, match='archive'):
        _ = [row async for row in memory.export('alpha')]
    await memory.close()
