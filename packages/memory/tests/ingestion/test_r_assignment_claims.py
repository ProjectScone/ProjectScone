"""Only R assignments can introduce function entities into stored graphs."""

import pytest

from scone_memory import HashEmbedder, InMemoryDocumentStore, InMemoryVectorIndex, MemoryEngine
from scone_memory.ingestion.code import code_language
from scone_memory.ingestion.code_graph import code_claims
from scone_memory.ingestion.code_tree import available

pytestmark = pytest.mark.skipif(not available(), reason='the grammar pack is not installed')


@pytest.mark.parametrize('operator', ['==', '!=', '+', '-', '~', '|', '%in%'])
def test_non_assignment_expressions_do_not_declare_functions(operator):
    source = f'candidate {operator} function(x) {{ x }}\n'
    claims = code_claims(source, 'rules.R', language=code_language('rules.R'))
    assert not [claim for claim in claims if claim.predicate == 'defines']


@pytest.mark.parametrize('operator', ['<-', '<<-', '='])
def test_assignment_expressions_keep_function_definitions(operator):
    source = f'calculate {operator} function(x) {{ x }}\n'
    claims = code_claims(source, 'rules.R', language=code_language('rules.R'))
    assert [(claim.subject, claim.object) for claim in claims if claim.predicate == 'defines'] == [
        ('rules.R', 'rules.R:calculate')
    ]
    assert source.encode()[claims[0].start:claims[0].end].decode().strip() == claims[0].quote


async def test_comparison_does_not_create_a_phantom_entity_in_memory():
    engine = await MemoryEngine(
        InMemoryDocumentStore(), InMemoryVectorIndex(), HashEmbedder(), code_graph=True
    ).open()
    try:
        await engine.remember(
            'default',
            'candidate == function(x) { x }\ncalculate <- function(x) { x + 1 }\n',
            source='rules.r',
        )
        graph, _ = await engine.entities.projection('default', mode='all', when='2099-01-01T00:00:00Z')
        labels = {entity.entity_id: entity.label for entity in graph.entities}
        relations = [(labels[edge.subject_id], edge.predicate, labels[edge.object_id])
                     for edge in graph.relations]
        assert ('rules.r', 'defines', 'rules.r:calculate') in relations
        assert 'rules.r:candidate' not in labels.values()
    finally:
        await engine.close()
