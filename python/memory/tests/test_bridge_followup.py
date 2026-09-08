"""Resolved graph bridges guide retrieval without becoming complete answers."""
import pytest

from scone_memory.retrieval.structured_evidence import StructuredEvidenceAssessor
from test_reachable_fact import ROUTE, fact, requirement
from test_structured_evidence import CHAIN, path


async def test_missing_attribute_targets_reached_entity_and_carries_its_route():
    assessor = StructuredEvidenceAssessor('question', (requirement(max_hops=3),), followup_strategy='bridge')
    report = await assessor.assess_with_coverage('question', ROUTE[:2])
    assert report.decision.status == 'insufficient'
    assert report.decision.followup_queries == ('office located in',)
    assert report.decision.selected_ids == ('fact:1', 'fact:2')
    assert report.decision.selected_groups == (('fact:1', 'fact:2'),)
    row = report.coverage[0]
    assert not row.witnessed and row.witness_ids == ()
    assert row.bridge_ids == ('fact:1', 'fact:2')
    assert row.requirement.subject == 'invoice'


async def test_missing_path_targets_bridge_without_changing_original_endpoint():
    assessor = StructuredEvidenceAssessor('question', (path(),), followup_strategy='bridge')
    decision = await assessor.assess('question', CHAIN[:2])
    assert decision.status == 'insufficient'
    assert decision.followup_queries == ('cedar depends on denver',)
    assert decision.selected_ids == ('fact:1', 'fact:2')
    final = await assessor.assess('question', CHAIN)
    assert final.status == 'sufficient' and final.followup_queries == ()


async def test_disconnected_and_reversed_edges_cannot_supply_query_anchor():
    assessor = StructuredEvidenceAssessor('question', (requirement(max_hops=3),), followup_strategy='bridge')
    records = (ROUTE[0], fact(2, 'office', 'managed by', 'team'),
               fact(3, 'private', 'managed by', 'unrelated'))
    result = await assessor.assess('question', records)
    assert result.followup_queries == ('team assigned to managed by located in',)
    assert result.selected_ids == ('fact:1',)


async def test_budget_exhaustion_keeps_only_examined_bridge_and_stays_uncertain():
    assessor = StructuredEvidenceAssessor('question', (requirement(),), max_work=1, followup_strategy='bridge')
    report = await assessor.assess_with_coverage('question', ROUTE)
    assert report.decision.status == 'uncertain'
    assert report.coverage[0].bridge_ids == ('fact:1',)
    assert report.work_used == 1
    assert report.decision.followup_queries[0].startswith('team ')


async def test_default_strategy_keeps_existing_selection_and_query():
    plan = requirement(max_hops=3)
    result = await StructuredEvidenceAssessor('question', (plan,)).assess('question', ROUTE[:2])
    assert result.selected_ids == ()
    assert result.followup_queries == (plan.search_query(),)


async def test_frontier_ties_are_stable_and_competing_edges_stay_visible():
    records = (ROUTE[0], fact(4, 'invoice', 'assigned to', 'alternate'),
               ROUTE[1], fact(5, 'alternate', 'managed by', 'branch'))
    report = await StructuredEvidenceAssessor('question', (requirement(max_hops=3),),
        followup_strategy='bridge').assess_with_coverage('question', records)
    assert report.coverage[0].bridge_ids == ('fact:1', 'fact:2')
    assert report.decision.selected_ids == ('fact:1', 'fact:4', 'fact:2')
    assert report.decision.followup_queries == ('office located in',)


async def test_overlong_entity_name_is_not_truncated_into_a_different_anchor():
    records = (fact(1, 'invoice', 'assigned to', 'x' * 129),)
    plan = requirement()
    report = await StructuredEvidenceAssessor('question', (plan,), followup_strategy='bridge').assess_with_coverage(
        'question', records)
    assert report.decision.followup_queries == (plan.search_query(),)
    assert report.coverage[0].bridge_ids == ()


async def test_hop_limit_reserves_space_for_the_missing_relation():
    report = await StructuredEvidenceAssessor('question', (path(max_hops=2),),
        followup_strategy='bridge').assess_with_coverage('question', CHAIN[:2])
    assert report.coverage[0].bridge_ids == ('fact:1',)
    assert report.decision.followup_queries == ('beacon depends on denver',)


@pytest.mark.parametrize('strategy', ['unknown', None, True])
def test_invalid_followup_strategy_is_rejected(strategy):
    with pytest.raises(ValueError):
        StructuredEvidenceAssessor('question', (requirement(),), followup_strategy=strategy)


@pytest.mark.parametrize('strategy,expected', [('requirement', 'insufficient'), ('bridge', 'sufficient')])
async def test_adaptive_round_resolves_missing_attribute_using_retained_bridge(engine, monkeypatch, strategy, expected):
    from scone_memory.core.models import RecallResult
    from scone_memory.core.ports import NewFact
    from scone_memory.retrieval.adaptive import AdaptiveLimits, AdaptiveRetriever
    from scone_memory.retrieval.recall_scope import RecallScope

    records = []
    for candidate in ROUTE:
        episode = await engine.remember('alpha', candidate.text, metadata={'team':'blue'})
        records.append(await engine.documents.insert_fact(NewFact(space='alpha', subject=candidate.subject,
            predicate=candidate.predicate, object=candidate.object, valid_from='2025-01-01T00:00:00Z',
            source_episode_id=episode.episode_id, quote=candidate.text)))
    queries = []

    async def recall(space, query, **kwargs):
        queries.append(query)
        assert space == 'alpha' and kwargs['where'] == {'team':'blue'}
        return RecallResult(facts=records[2:] if query == 'office located in' else records[:2])

    monkeypatch.setattr(engine, 'recall', recall)
    assessor = StructuredEvidenceAssessor('question', (requirement(max_hops=3),), followup_strategy=strategy)
    result = await AdaptiveRetriever(engine, assessor, limits=AdaptiveLimits(max_rounds=2)).retrieve(
        'alpha', 'question', scope=RecallScope.validated(where={'team':'blue'}))
    assert result.status == expected
    assert len(queries) == 2
    if strategy == 'bridge':
        assert queries == ['question', 'office located in']
        assert {row.fact_id for row in result.recall.facts} == {row.fact_id for row in records}


@pytest.mark.parametrize('bridge_state', ['foreign', 'wrong_scope', 'deleted'])
async def test_unavailable_bridge_cannot_redirect_search_or_complete_answer(engine, monkeypatch, bridge_state):
    from scone_memory.core.models import RecallResult
    from scone_memory.core.ports import NewFact
    from scone_memory.retrieval.adaptive import AdaptiveLimits, AdaptiveRetriever
    from scone_memory.retrieval.recall_scope import RecallScope

    records = []
    for i, candidate in enumerate(ROUTE):
        space = 'other' if i == 1 and bridge_state == 'foreign' else 'alpha'
        team = 'red' if i == 1 and bridge_state == 'wrong_scope' else 'blue'
        episode = await engine.remember(space, candidate.text, metadata={'team':team})
        records.append(await engine.documents.insert_fact(NewFact(space=space, subject=candidate.subject,
            predicate=candidate.predicate, object=candidate.object, valid_from='2025-01-01T00:00:00Z',
            source_episode_id=episode.episode_id, quote=candidate.text)))
    queries = []

    async def recall(space, query, **kwargs):
        queries.append(query)
        if len(queries) == 2 and bridge_state == 'deleted':
            await engine.forget('alpha', records[1].source_episode_id)
        return RecallResult(facts=records[:2] if len(queries) == 1 else records)

    monkeypatch.setattr(engine, 'recall', recall)
    assessor = StructuredEvidenceAssessor('question', (requirement(max_hops=3),), followup_strategy='bridge')
    result = await AdaptiveRetriever(engine, assessor, limits=AdaptiveLimits(max_rounds=2)).retrieve(
        'alpha', 'question', scope=RecallScope.validated(where={'team':'blue'}))
    assert result.status != 'sufficient'
    assert records[1].fact_id not in {row.fact_id for row in result.recall.facts}
    if bridge_state != 'deleted':
        assert queries[1].startswith('team ') and 'office' not in queries[1]
