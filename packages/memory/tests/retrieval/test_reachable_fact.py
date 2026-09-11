"""A requested attribute must have a complete, directed route from its anchor."""
import pytest

from scone_memory.retrieval.adaptive import EvidenceCandidate
from scone_memory.retrieval.structured_evidence import EvidenceRequirement, StructuredEvidenceAssessor


def fact(number, subject, predicate, obj):
    return EvidenceCandidate(id=f'fact:{number}', episode_id=number,
        text=f'{subject} {predicate} {obj}.', subject=subject, predicate=predicate, object=obj)


ROUTE = (fact(1, 'invoice', 'assigned to', 'team'),
         fact(2, 'team', 'managed by', 'office'),
         fact(3, 'office', 'located in', 'Oslo'))


def requirement(**updates):
    return EvidenceRequirement.model_validate(dict(kind='reachable_fact', subject='invoice',
        predicate='located in', via=('assigned to', 'managed by'), **updates))


async def assess(candidates=ROUTE, *, max_work=256, **updates):
    return await StructuredEvidenceAssessor('question', (requirement(**updates),),
        max_work=max_work).assess('question', candidates)


async def test_unknown_attribute_requires_the_complete_mixed_predicate_route():
    result = await assess()
    assert result.status == 'sufficient'
    assert result.selected_ids == ('fact:1', 'fact:2', 'fact:3')
    assert result.selected_groups == (('fact:1', 'fact:2', 'fact:3'),)
    assert result.followup_queries == ()


@pytest.mark.parametrize('candidates', [
    ROUTE[:2],
    (ROUTE[0], ROUTE[2]),
    (ROUTE[0], fact(2, 'office', 'managed by', 'team'), ROUTE[2]),
    (ROUTE[0], fact(2, 'team', 'audited by', 'office'), ROUTE[2]),
    (fact(1, 'other invoice', 'assigned to', 'team'), *ROUTE[1:]),
    (*ROUTE[:2], fact(3, 'office', 'founded in', 'Oslo')),
    (*ROUTE[:2], fact(3, 'other office', 'located in', 'Oslo')),
    (fact(1, 'invoice', 'located in', 'Oslo'),),
])
async def test_bridge_wrong_attribute_direction_or_unrelated_answer_cannot_satisfy(candidates):
    result = await assess(candidates)
    assert result.status == 'insufficient' and result.selected_ids == ()
    assert result.followup_queries == ('invoice assigned to managed by located in',)


async def test_object_filter_does_not_guess_a_value_and_keeps_competing_values():
    records = (*ROUTE, fact(4, 'office', 'located in', 'Bergen'))
    matching = await assess(records, object='Oslo')
    assert matching.status == 'sufficient'
    assert matching.selected_ids == ('fact:1', 'fact:2', 'fact:3', 'fact:4')
    missing = await assess(records, object='Rome')
    assert missing.status == 'insufficient' and missing.selected_ids == ()


async def test_each_reachable_answer_keeps_its_own_supporting_route():
    records = (*ROUTE, fact(4, 'team', 'managed by', 'branch'),
               fact(5, 'branch', 'located in', 'Bergen'))
    result = await assess(records)
    assert result.status == 'sufficient'
    assert set(result.selected_ids) == {f'fact:{number}' for number in range(1, 6)}
    assert len(result.selected_groups) == 1


async def test_shortest_route_still_searches_longer_branches_for_other_attributes():
    records = (*ROUTE, fact(4, 'team', 'located in', 'Bergen'))
    result = await assess(records)
    assert result.status == 'sufficient'
    assert set(result.selected_ids) == {'fact:1', 'fact:2', 'fact:3', 'fact:4'}


async def test_cycles_terminate_without_promoting_an_attribute_at_the_anchor():
    records = (ROUTE[0], fact(2, 'team', 'managed by', 'invoice'),
               fact(3, 'invoice', 'located in', 'Oslo'))
    result = await assess(records)
    assert result.status == 'insufficient' and result.selected_ids == ()


async def test_max_hops_counts_bridges_and_final_attribute():
    assert (await assess(max_hops=3)).status == 'sufficient'
    assert (await assess(max_hops=2)).status == 'insufficient'


async def test_work_exhaustion_never_claims_a_missing_or_complete_answer():
    result = await assess(max_work=2)
    assert result.status == 'uncertain' and result.selected_ids == ()
    records = (*ROUTE, fact(4, 'office', 'located in', 'Bergen'))
    partial = await assess(records, max_work=3)
    assert partial.status == 'uncertain'
    assert set(partial.selected_ids) == {'fact:1', 'fact:2', 'fact:3', 'fact:4'}


async def test_unstructured_chunks_do_not_become_relationships():
    chunk = ROUTE[1].model_copy(update={'id':'chunk:2'})
    result = await assess((ROUTE[0], chunk, ROUTE[2]))
    assert result.status == 'insufficient'


@pytest.mark.parametrize('updates', [
    {'subject':None}, {'via':()}, {'via':['assigned to']}, {'via':(' ',)},
    {'via':('assigned to', 'assigned to')}, {'via':('located in',)},
    {'via':tuple(str(i) for i in range(9))}, {'via':('a' * 129,)},
    {'max_hops':1}, {'max_hops':True}, {'via':(1,)},
])
def test_invalid_reachable_requirement_is_rejected(updates):
    values = dict(kind='reachable_fact', subject='invoice', predicate='located in',
                  via=('assigned to', 'managed by'))
    with pytest.raises(ValueError):
        EvidenceRequirement.model_validate({**values, **updates})


@pytest.mark.parametrize('kind', ['fact', 'path'])
def test_existing_requirements_reject_ignored_traversal_configuration(kind):
    with pytest.raises(ValueError):
        EvidenceRequirement(kind=kind, subject='invoice', predicate='located in', object='Oslo',
                            via=('assigned to',))


async def test_diamond_and_cycles_have_bounded_node_expansion():
    records = (fact(1, 'invoice', 'assigned to', 'left'),
        fact(2, 'invoice', 'assigned to', 'right'),
        fact(3, 'left', 'managed by', 'office'), fact(4, 'right', 'managed by', 'office'),
        fact(5, 'office', 'assigned to', 'left'), fact(6, 'office', 'located in', 'Oslo'))
    result = await assess(records, max_work=6)
    assert result.status == 'sufficient'
    assert 'fact:6' in result.selected_ids


async def test_selector_requires_all_route_cards_and_returns_original_quotes():
    from scone_memory.realtime.evidence_answer import EvidenceCard, EvidenceClaim
    from scone_memory.realtime.structured_selector import StructuredEvidenceSelector

    cards = tuple(EvidenceCard(id=f'card:{i}', kind='claim', evidence_ids=(f'fact:{i}',), text=row.text, claims=(EvidenceClaim(
        fact_id=i, subject=row.subject, predicate=row.predicate, object=row.object,
        source_episode_id=i, origin='stated'),)) for i, row in enumerate(ROUTE, 1))
    selector = StructuredEvidenceSelector('question', (requirement(),))
    full = await selector.select('question', cards)
    assert full.atomic and set(full.card_ids) == {'card:1', 'card:2', 'card:3'}
    missing = await selector.select('question', (cards[0], cards[2]))
    assert missing.atomic and missing.card_ids == ()


@pytest.mark.parametrize('bridge_state', ['retained', 'wrong_scope', 'foreign', 'deleted'])
async def test_reachable_witness_is_atomic_scoped_and_fresh(engine, monkeypatch, bridge_state):
    from scone_memory.core.models import RecallResult
    from scone_memory.core.ports import NewFact
    from scone_memory.retrieval.adaptive import AdaptiveLimits, AdaptiveRetriever
    from scone_memory.retrieval.recall_scope import RecallScope

    records = []
    for i, candidate in enumerate(ROUTE):
        space = 'other' if i == 1 and bridge_state == 'foreign' else 'alpha'
        team = 'red' if i == 1 and bridge_state == 'wrong_scope' else 'blue'
        episode = await engine.remember(space, candidate.text, metadata={'team':team})
        records.append(await engine.documents.insert_fact(NewFact(space=space,
            subject=candidate.subject, predicate=candidate.predicate, object=candidate.object,
            valid_from='2025-01-01T00:00:00Z', source_episode_id=episode.episode_id, quote=candidate.text)))

    async def recall(*args, **kwargs):
        # Overbroad candidates exercise the retention boundary independently of ranking.
        return RecallResult(facts=records)

    monkeypatch.setattr(engine, 'recall', recall)
    selector = StructuredEvidenceAssessor('question', (requirement(),))

    class SelectThenDelete:
        async def assess(self, question, candidates):
            decision = await selector.assess(question, candidates)
            if bridge_state == 'deleted':
                await engine.forget('alpha', records[1].source_episode_id)
            return decision

    result = await AdaptiveRetriever(engine, SelectThenDelete(), limits=AdaptiveLimits(max_rounds=1)).retrieve(
        'alpha', 'question', scope=RecallScope.validated(where={'team':'blue'}))
    if bridge_state == 'retained':
        assert result.status == 'sufficient'
        assert {row.fact_id for row in result.recall.facts} == {row.fact_id for row in records}
    else:
        assert result.status != 'sufficient'
        if bridge_state == 'deleted':
            assert result.recall.facts == []
        else:
            assert {row.subject for row in result.recall.facts} == {'invoice', 'office'}
            assert result.evidence_basis == 'unselected_candidates'
            assert result.model_selected_ids == ()


async def test_identity_matching_and_plan_revalidation():
    # Hops follow the ledger's identity rule: 'office' reaches claims about
    # 'Office', but not about a different name.
    assert (await assess((ROUTE[0], ROUTE[1], fact(3, 'Office', 'located in', 'Oslo')))).status == 'sufficient'
    assert (await assess((ROUTE[0], ROUTE[1], fact(3, 'Offices', 'located in', 'Oslo')))).status == 'insufficient'
    with pytest.raises(ValueError):
        StructuredEvidenceAssessor('question', (requirement().model_copy(update={'via':('located in',)}),))
    assert EvidenceRequirement.model_validate_json(requirement().model_dump_json()) == requirement()


@pytest.mark.parametrize('character', ['s', '界'])
async def test_long_valid_plan_keeps_whole_identities_in_bounded_followup(character):
    subject, predicate, obj = character * 128, 'p' * 128, 'o' * 128
    via = tuple(str(i) * 128 for i in range(8))
    plan = EvidenceRequirement(kind='reachable_fact', subject=subject, predicate=predicate,
                               object=obj, via=via)
    result = await StructuredEvidenceAssessor('question', (plan,)).assess('question', ())
    assert result.status == 'insufficient'
    assert len(result.followup_queries) == 1
    query = result.followup_queries[0]
    assert len(query) <= 1000
    terms = query.split(' ')
    assert terms[0] == subject and terms[-2:] == [predicate, obj]
    assert all(term in via for term in terms[1:-2])
