"""Passage filters must also constrain sourced claims and their history."""
import pytest
from scone_memory import MemoryEngine, HashEmbedder, InMemoryDocumentStore, InMemoryVectorIndex

@pytest.fixture
async def memory():
    engine = await MemoryEngine(InMemoryDocumentStore(), InMemoryVectorIndex(), HashEmbedder(),
                                clock=lambda: '2026-09-01T00:00:00Z').open()
    yield engine
    await engine.close()

async def stated(memory, subject, project, date='2026-01-01', value='local'):
    text = f'{subject} stores its journal {value}.'
    episode = await memory.remember('team', text, kind='file', source=f'docs/{project}/journal.md',
                                    tags=[project], metadata={'project': project}, created_at=date)
    return await memory.assert_fact('team', subject, 'journal', value, valid_from=date,
                                    source_episode_id=episode.episode_id, quote=text)

@pytest.mark.parametrize('filters', [
    {'where': {'project': 'edge'}}, {'tags': ['edge']}, {'source_prefix': 'docs/edge/'},
    {'conditions': {'field': 'project', 'is': 'edge'}},
])
async def test_scope_filters_claims_before_fact_limit(memory, filters):
    for number in range(12):
        await stated(memory, f'Beacon-{number}', 'other')
    wanted = await stated(memory, 'Beacon-edge', 'edge')
    result = await memory.recall('team', 'Beacon journal', **filters)
    assert [fact.fact_id for fact in result.facts] == [wanted.fact_id]

async def test_scoped_history_does_not_include_another_projects_source(memory):
    await stated(memory, 'Beacon', 'other', '2026-01-01', 'old')
    wanted = await stated(memory, 'Beacon', 'edge', '2026-02-01', 'new')
    result = await memory.recall('team', 'Beacon journal', history=True, where={'project': 'edge'})
    assert [fact.fact_id for fact in result.facts] == [wanted.fact_id]
    assert result.history == []

async def test_unsourced_claims_remain_available_only_without_source_filters(memory):
    fact = await memory.assert_fact('team', 'Beacon', 'journal', 'local')
    assert (await memory.recall('team', 'Beacon')).facts[0].fact_id == fact.fact_id
    assert (await memory.recall('team', 'Beacon', kind='file')).facts == []
