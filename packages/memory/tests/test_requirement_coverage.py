"""Explicit requirement coverage separates witnesses, alternatives and limits."""
import asyncio

from scone_memory.retrieval.structured_evidence import EvidenceRequirement, StructuredEvidenceAssessor
from test_structured_evidence import CHAIN, fact, path


async def test_each_requirement_reports_its_own_witness_and_gap():
    requirements = (EvidenceRequirement(kind='fact', subject='cedar', predicate='depends on'),
                    EvidenceRequirement(kind='fact', subject='aster', predicate='manufactured by'))
    assessor = StructuredEvidenceAssessor('question', requirements)
    report = await assessor.assess_with_coverage('question', CHAIN)
    assert report.decision == await assessor.assess('question', CHAIN)
    assert report.decision.status == 'insufficient'
    found, missing = report.coverage
    assert found.requirement == requirements[0]
    assert found.witness_ids == found.selected_ids == ('fact:3',)
    assert found.witnessed and found.followup_query is None
    assert not missing.witnessed and missing.witness_ids == missing.selected_ids == ()
    assert missing.followup_query == 'aster manufactured by'
    assert not missing.work_exhausted
    assert report.work_used == 0


async def test_competing_value_is_retained_but_not_claimed_as_exact_value_witness():
    requirement = EvidenceRequirement(kind='fact', subject='aster', predicate='uses', object='blue')
    candidates = (fact(1, 'aster', 'uses', 'red'), fact(2, 'aster', 'uses', 'blue'))
    report = await StructuredEvidenceAssessor('question', (requirement,)).assess_with_coverage('question', candidates)
    assert report.coverage[0].witness_ids == ('fact:2',)
    assert report.coverage[0].selected_ids == report.decision.selected_ids == ('fact:1', 'fact:2')
    assert report.decision.selected_groups == (('fact:1', 'fact:2'),)


async def test_exhausted_path_is_not_reported_as_a_proven_absence():
    direct = EvidenceRequirement(kind='fact', subject='cedar', predicate='depends on')
    report = await StructuredEvidenceAssessor('question', (direct, path()), max_work=1).assess_with_coverage('question', CHAIN)
    assert report.decision.status == 'uncertain'
    assert report.coverage[0].witnessed and not report.coverage[0].work_exhausted
    assert not report.coverage[1].witnessed and report.coverage[1].work_exhausted
    assert report.coverage[1].work_used == report.work_used == report.work_limit == 1


async def test_reachable_attribute_keeps_anchor_and_final_fact_together():
    requirements = (EvidenceRequirement(kind='reachable_fact', subject='aster', predicate='uses',
                                       via=('forwards to',), max_hops=2),)
    candidates = (fact(1, 'aster', 'forwards to', 'beacon'), fact(2, 'beacon', 'uses', 'archive'),
                  fact(3, 'elsewhere', 'uses', 'private'))
    report = await StructuredEvidenceAssessor('question', requirements).assess_with_coverage('question', candidates)
    assert report.coverage[0].witness_ids == ('fact:1', 'fact:2')
    assert report.decision.selected_ids == ('fact:1', 'fact:2')
    assert report.work_used == 2


async def test_all_gaps_remain_visible_when_followup_queries_are_capped():
    requirements = tuple(EvidenceRequirement(kind='fact', subject=f'item-{i}', predicate='uses') for i in range(8))
    report = await StructuredEvidenceAssessor('question', requirements).assess_with_coverage('question', ())
    assert len(report.coverage) == 8
    assert len(report.decision.followup_queries) == 3
    assert tuple(row.followup_query for row in report.coverage) == tuple(row.search_query() for row in requirements)


async def test_positive_reachable_witness_does_not_hide_unfinished_traversal():
    requirement = EvidenceRequirement(kind='reachable_fact', subject='aster', predicate='uses',
                                      via=('forwards to',), max_hops=2)
    candidates = (fact(1, 'aster', 'forwards to', 'beacon'),
                  fact(2, 'beacon', 'uses', 'archive'), fact(3, 'beacon', 'uses', 'disk'))
    report = await StructuredEvidenceAssessor('question', (requirement,), max_work=2).assess_with_coverage(
        'question', candidates)
    row = report.coverage[0]
    assert row.witnessed and row.work_exhausted
    assert row.witness_ids == ('fact:1', 'fact:2')
    assert row.selected_ids == ('fact:1', 'fact:2', 'fact:3')
    assert row.followup_query is None
    assert report.decision.status == 'uncertain'
    assert report.work_used == report.work_limit == 2


async def test_reports_are_independent_and_do_not_retain_candidate_text():
    requirement = EvidenceRequirement(kind='fact', subject='aster', predicate='uses')
    assessor = StructuredEvidenceAssessor('question', (requirement,))
    source = fact(1, 'aster', 'uses', 'archive').model_copy(update={'text':'PRIVATE_SOURCE_BODY'})
    found, missing = await asyncio.gather(assessor.assess_with_coverage('question', (source,)),
                                          assessor.assess_with_coverage('question', ()))
    assert found.coverage[0].witnessed and not missing.coverage[0].witnessed
    assert 'PRIVATE_SOURCE_BODY' not in found.model_dump_json()
    assert 'archive' not in found.model_dump_json()
