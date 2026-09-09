"""Reviewed archives remain scoped, graphable and usable by evidence tools."""
import json

from scone_memory.integrations.scoped_tools import ScopedMemoryTools
from scone_memory.retrieval.recall_scope import RecallScope


async def test_reviewed_archive_composes_with_graph_retrieval_and_computation(engine):
    text = 'Équipe Juniper delivered 12 red units and 18 blue units.'
    source = await engine.remember('original', text, source='delivery-note', metadata={'team': 'blue'})
    claims = []
    for color, count in [('red', '12'), ('blue', '18')]:
        claims.append(await engine.assert_fact('original', 'juniper', f'{color}_units', count,
            proposed=True, origin='extracted', source_episode_id=source.episode_id, quote=f'{count} {color} units'))
    decisions = await engine.decide('original', 'approve', [row.fact_id for row in reversed(claims)])
    assert decisions.applied == 2
    await engine.exclude('original', claims[0].fact_id, 'withheld from recall')
    await engine.close_fact('original', claims[1].fact_id, 'delivery complete')

    # Destination IDs and unrelated source content cannot become imported evidence.
    await engine.remember('restored', 'Juniper delivered 999 units.', metadata={'team': 'red'})
    archive = [record async for record in engine.export('original')]
    imported = await engine.import_records('restored', archive)
    assert (imported.episodes, imported.facts) == (1, 2)
    restored = await engine.facts('restored', include_closed=True, include_excluded=True)
    assert len(restored) == 2
    restored_source = restored[0].source_episode_id
    assert restored_source is not None and restored_source != source.episode_id
    assert {row.source_episode_id for row in restored} == {restored_source}
    assert {row.object: (row.status, row.excluded_reason, row.quote) for row in restored} == {
        '12': ('active', 'withheld from recall', '12 red units'),
        '18': ('closed', None, '18 blue units'),
    }
    assert await engine.facts('restored') == []

    bounded = await engine.graph('restored', episode_id=restored_source, fact_limit=1)
    assert bounded.fact_read_status == 'bounded' and bounded.facts_truncated
    assert len([node for node in bounded.nodes.values() if node.kind == 'claim']) == 1
    graph = await engine.graph('restored', episode_id=restored_source, fact_limit=2)
    assert not graph.facts_truncated and graph.provenance_missing == 0
    nodes = [node for node in graph.nodes.values() if node.kind == 'claim']
    assert {node.id for node in nodes} == {f'claim:{row.fact_id}' for row in restored}
    assert {node.data['source_episode_id'] for node in nodes} == {restored_source}

    tools = ScopedMemoryTools(engine, 'restored', scope=RecallScope.validated(where={'team': 'blue'}),
        enable_computation=True)
    search = await tools.prepare('search_memory', {'query': 'Juniper delivery', 'limit': 5})
    passages = json.loads(search.payload)['items']
    assert passages and {row['episode_id'] for row in passages} == {restored_source}
    chunk_id = passages[0]['chunk_id']
    computation = await tools.prepare('compute_memory', {'operation': 'sum', 'left': [
        {'chunk_id': chunk_id, 'quote': '12'}, {'chunk_id': chunk_id, 'quote': '18'}]})
    packet = json.loads(computation.payload)
    assert packet['computation']['value'] == '30' and packet['verified_accuracy'] is False
    assert {row['episode_id'] for row in packet['items']} == {restored_source}
    assert packet['items'][0]['text'] == text
    assert await search.validate() and await computation.validate()

    await engine.forget('restored', restored_source)
    assert not await search.validate() and not await computation.validate()
    # Forgetting the imported source cannot erase the original archive's evidence.
    assert (await engine.episode('original', source.episode_id)).content == text
