"""Explicit positive requirements must have complete, directed witnesses."""
import pytest

from scone_memory.retrieval.adaptive import EvidenceCandidate
from scone_memory.retrieval.structured_evidence import EvidenceRequirement, StructuredEvidenceAssessor


def fact(number, subject, predicate, obj):
    return EvidenceCandidate(id=f'fact:{number}', episode_id=number,
        text=f'{subject} {predicate} {obj}.', subject=subject, predicate=predicate, object=obj)


CHAIN = (fact(1, 'aster', 'depends on', 'beacon'), fact(2, 'beacon', 'depends on', 'cedar'),
         fact(3, 'cedar', 'depends on', 'denver'))


def assessor(*requirements, max_work=256):
    return StructuredEvidenceAssessor('bound question', requirements, max_work=max_work)


def path(subject='aster', target='denver', **kwargs):
    return EvidenceRequirement(kind='path', subject=subject, predicate='depends on', object=target, **kwargs)


async def test_direct_lookup_selects_the_answering_fact_not_neighbor():
    decision = await assessor(EvidenceRequirement(kind='fact', subject='cedar', predicate='depends on')).assess('bound question', CHAIN)
    assert decision.status == 'sufficient'
    assert decision.selected_ids == ('fact:3',)
    assert decision.followup_queries == ()


async def test_inverse_fact_lookup_finds_recorded_subjects_without_reversing_predicate():
    requirement = EvidenceRequirement(kind='fact', predicate='uses', object='Polaris')
    candidates = (fact(1, 'Juniper', 'uses', 'Polaris'), fact(2, 'Cedar', 'uses', 'Polaris'),
                  fact(3, 'Polaris', 'uses', 'Beacon'), fact(4, 'Alder', 'manufactured by', 'Polaris'))
    decision = await assessor(requirement).assess('bound question', candidates)
    assert decision.status == 'sufficient'
    assert decision.selected_ids == ('fact:1', 'fact:2')
    assert decision.selected_groups == (('fact:1', 'fact:2'),)


async def test_inverse_requirement_preserves_competing_values_around_witnesses():
    requirement = EvidenceRequirement(kind='fact', predicate='uses', object='Polaris')
    candidates = (fact(1, 'Juniper', 'uses', 'Polaris'), fact(2, 'Juniper', 'uses', 'Alder'))
    decision = await assessor(requirement).assess('bound question', candidates)
    assert decision.status == 'sufficient' and decision.selected_ids == ('fact:1', 'fact:2')


async def test_missing_inverse_value_is_not_witnessed_by_the_forward_relation():
    requirement = EvidenceRequirement(kind='fact', predicate='uses', object='Polaris')
    decision = await assessor(requirement).assess('bound question', (fact(1, 'Polaris', 'uses', 'Juniper'),))
    assert decision.status == 'insufficient' and decision.selected_ids == ()
    assert decision.followup_queries == ('uses Polaris',)


@pytest.mark.parametrize('kind', ['fact', 'path'])
def test_requirement_cannot_leave_both_endpoints_unknown(kind):
    with pytest.raises(ValueError):
        EvidenceRequirement(kind=kind, predicate='uses')


def test_path_still_requires_a_named_subject():
    with pytest.raises(ValueError):
        EvidenceRequirement(kind='path', predicate='uses', object='Polaris')


async def test_path_requires_every_forward_predicate_matched_link():
    decision = await assessor(path()).assess('bound question', CHAIN)
    assert decision.status == 'sufficient'
    assert decision.selected_ids == ('fact:1', 'fact:2', 'fact:3')
    assert decision.selected_groups == (('fact:1', 'fact:2', 'fact:3'),)


@pytest.mark.parametrize('candidates,requirement', [
    ((CHAIN[0], CHAIN[2]), path()),
    (CHAIN, path('denver', 'aster')),
    ((CHAIN[0], fact(2, 'beacon', 'manufactured by', 'cedar'), CHAIN[2]), path()),
    (CHAIN, path(max_hops=2)),
])
async def test_broken_reversed_wrong_predicate_or_short_path_is_not_sufficient(candidates, requirement):
    decision = await assessor(requirement).assess('bound question', candidates)
    assert decision.status == 'insufficient'
    assert decision.selected_ids == ()
    assert decision.followup_queries


async def test_compound_missing_requirement_cannot_be_silently_dropped():
    dependency = EvidenceRequirement(kind='fact', subject='cedar', predicate='depends on')
    maker = EvidenceRequirement(kind='fact', subject='aster', predicate='manufactured by')
    partial = await assessor(dependency, maker).assess('bound question', CHAIN)
    assert partial.status == 'insufficient' and partial.selected_ids == ('fact:3',)
    complete = await assessor(dependency, maker).assess('bound question', (*CHAIN, fact(4, 'aster', 'manufactured by', 'Meridian')))
    assert complete.status == 'sufficient' and complete.selected_ids == ('fact:3', 'fact:4')


async def test_conflicting_direct_observations_are_retained_together():
    decision = await assessor(EvidenceRequirement(kind='fact', subject='aster', predicate='uses')).assess(
        'bound question', (fact(1, 'aster', 'uses', 'blue'), fact(2, 'aster', 'uses', 'red')))
    assert decision.selected_ids == ('fact:1', 'fact:2')
    assert decision.selected_groups == (('fact:1', 'fact:2'),)


async def test_passage_text_and_partial_identity_are_not_structured_witnesses():
    text = EvidenceCandidate(id='chunk:1', episode_id=1, text=CHAIN[0].text,
        subject='aster', predicate='depends on', object='beacon')
    decision = await assessor(EvidenceRequirement(kind='fact', subject='aster', predicate='depends on')).assess('bound question', (text,))
    assert decision.status == 'insufficient'


async def test_branch_search_finds_reachable_endpoint_and_does_not_loop():
    candidates = (fact(1, 'aster', 'depends on', 'loop'), fact(2, 'loop', 'depends on', 'aster'),
        fact(3, 'aster', 'depends on', 'bridge'), fact(4, 'bridge', 'depends on', 'denver'))
    result = await assessor(path()).assess('bound question', candidates)
    assert result.selected_ids == ('fact:1', 'fact:3', 'fact:4')
    assert 'fact:2' not in result.selected_ids  # Cycle is not a continuation.


async def test_budget_exhaustion_is_uncertain_not_proof_of_missing_path():
    result = await assessor(path(), max_work=1).assess('bound question', CHAIN)
    assert result.status == 'uncertain'
    assert result.selected_ids == ()


async def test_question_binding_prevents_reusing_the_wrong_plan():
    with pytest.raises(ValueError, match='question'):
        await assessor(path()).assess('different question', CHAIN)


@pytest.mark.parametrize('updates', [{'kind':'terminal'}, {'max_hops':True}, {'max_hops':7}, {'subject':' '}, {'predicate':''}, {'object':None}])
def test_invalid_path_requirements_rejected(updates):
    with pytest.raises(ValueError):
        EvidenceRequirement.model_validate({'kind':'path','subject':'aster','predicate':'depends on','object':'denver',**updates})


async def test_forged_candidates_are_revalidated_and_never_repaired():
    with pytest.raises(ValueError):
        await assessor(path()).assess('bound question', (CHAIN[0].model_copy(update={'id':'fact:0'}),))


async def test_identity_matching_follows_the_ledgers_rule_not_resemblance():
    # The ledger stores 'ASTER' and 'aster' as one subject, so a requirement
    # spelled either way finds it. A different spelling is still a different
    # name, and a value whose case carries meaning is never folded.
    folded = await assessor(EvidenceRequirement(kind='fact', subject='ASTER', predicate='depends on')).assess('bound question', CHAIN)
    assert folded.status == 'sufficient' and folded.selected_ids == ('fact:1',)
    other = await assessor(EvidenceRequirement(kind='fact', subject='asters', predicate='depends on')).assess('bound question', CHAIN)
    assert other.status == 'insufficient'
    units = (fact(1, 'disk', 'unit', 'MB'),)
    value = await assessor(EvidenceRequirement(kind='fact', subject='disk', predicate='unit', object='mb')).assess('bound question', units)
    assert value.status == 'insufficient'


@pytest.mark.parametrize('requirements', [(), [path()], (path(), path()), (path(),) * 9])
def test_plan_limits_are_validated(requirements):
    with pytest.raises(ValueError):
        StructuredEvidenceAssessor('question', requirements)


@pytest.mark.parametrize('max_work', [0, True, 2049, 1.5])
def test_work_budget_is_validated(max_work):
    with pytest.raises(ValueError):
        assessor(path(), max_work=max_work)


async def test_payload_and_duplicate_limits():
    selector = assessor(path())
    for candidates in ((CHAIN[0], CHAIN[0]), tuple(fact(i, 'a', 'p', 'b') for i in range(1, 102)),
                       (EvidenceCandidate(id='fact:1', episode_id=1, text='界' * 50000),)):
        with pytest.raises(ValueError):
            await selector.assess('bound question', candidates)


async def test_one_fact_path_and_overlapping_requirements_merge_atomic_witnesses():
    result = await assessor(path(target='beacon'), path()).assess('bound question', CHAIN)
    assert result.status == 'sufficient'
    assert result.selected_ids == ('fact:1', 'fact:2', 'fact:3')
    assert result.selected_groups == (('fact:1', 'fact:2', 'fact:3'),)


async def test_query_limit_does_not_drop_missing_requirements_from_verdict():
    requirements = tuple(EvidenceRequirement(kind='fact', subject=f'entity{i}', predicate='uses') for i in range(8))
    result = await assessor(*requirements).assess('bound question', ())
    assert result.status == 'insufficient' and len(result.followup_queries) == 3


async def test_native_retrieval_preserves_scope_and_rechecks_deletion(engine):
    from scone_memory.retrieval.adaptive import AdaptiveLimits, AdaptiveRetriever
    from scone_memory.core.ports import NewFact
    from scone_memory.retrieval.recall_scope import RecallScope
    for space, team, obj in [('alpha', 'blue', 'denver'), ('alpha', 'red', 'PRIVATE_WRONG_TEAM'),
                             ('other', 'blue', 'PRIVATE_WRONG_SPACE')]:
        quote = f'cedar depends on {obj}.'
        episode = await engine.remember(space, quote, kind='file', source='manuals/cedar', metadata={'team':team})
        await engine.documents.insert_fact(NewFact(space=space, subject='cedar', predicate='depends on', object=obj,
            valid_from='2025-01-01T00:00:00Z', source_episode_id=episode.episode_id, quote=quote))
        if space == 'alpha' and team == 'blue':
            retained_episode = episode.episode_id
    question = 'cedar depends on'
    selector = StructuredEvidenceAssessor(question, (EvidenceRequirement(kind='fact', subject='cedar', predicate='depends on'),))
    scope = RecallScope.validated(where={'team':'blue'}, kind='file', source_prefix='manuals/')
    result = await AdaptiveRetriever(engine, selector, limits=AdaptiveLimits(max_rounds=1)).retrieve('alpha', question, scope=scope)
    assert result.status == 'sufficient'
    assert [row.object for row in result.recall.facts] == ['denver']
    assert result.queries_used == 1 and result.errors == ()

    class DeleteAfterAssessment:
        async def assess(self, question, candidates):
            decision = await selector.assess(question, candidates)
            await engine.forget('alpha', retained_episode)
            return decision

    deleted = await AdaptiveRetriever(engine, DeleteAfterAssessment(), limits=AdaptiveLimits(max_rounds=1)).retrieve('alpha', question, scope=scope)
    assert deleted.status != 'sufficient' and deleted.recall.facts == []


@pytest.mark.parametrize('updates', [{'max_hops':True}, {'subject':None}, {'kind':'terminal'}])
def test_forged_requirement_is_revalidated(updates):
    with pytest.raises(ValueError):
        assessor(path().model_copy(update=updates))


async def test_partial_success_survives_work_exhaustion_without_sufficient_verdict():
    direct = EvidenceRequirement(kind='fact', subject='cedar', predicate='depends on')
    result = await assessor(direct, path(), max_work=1).assess('bound question', CHAIN)
    assert result.status == 'uncertain' and result.selected_ids == ('fact:3',)


async def test_exact_value_requirement_retains_competing_values():
    requirement = EvidenceRequirement(kind='fact', subject='aster', predicate='uses', object='blue')
    result = await assessor(requirement).assess('bound question',
        (fact(1, 'aster', 'uses', 'red'), fact(2, 'aster', 'uses', 'blue')))
    assert result.status == 'sufficient'
    assert result.selected_groups == (('fact:1', 'fact:2'),)


@pytest.mark.parametrize('bridge_state', ['retained', 'wrong_scope', 'foreign', 'deleted'])
async def test_native_path_witness_is_atomic_scoped_and_fresh(engine, monkeypatch, bridge_state):
    from scone_memory.core.models import RecallResult
    from scone_memory.core.ports import NewFact
    from scone_memory.retrieval.adaptive import AdaptiveLimits, AdaptiveRetriever
    from scone_memory.retrieval.recall_scope import RecallScope
    records = []
    for i, candidate in enumerate(CHAIN):
        space = 'other' if i == 1 and bridge_state == 'foreign' else 'alpha'
        team = 'red' if i == 1 and bridge_state == 'wrong_scope' else 'blue'
        episode = await engine.remember(space, candidate.text, metadata={'team':team})
        record = await engine.documents.insert_fact(NewFact(space=space, subject=candidate.subject,
            predicate=candidate.predicate, object=candidate.object, valid_from='2025-01-01T00:00:00Z',
            source_episode_id=episode.episode_id, quote=candidate.text))
        records.append(record)

    async def recall(*args, **kwargs):
        # Deliberately overbroad retrieval: the native boundary must reject it.
        return RecallResult(facts=records)

    monkeypatch.setattr(engine, 'recall', recall)
    selector = assessor(path())

    class SelectThenDelete:
        async def assess(self, question, candidates):
            result = await selector.assess(question, candidates)
            if bridge_state == 'deleted':
                await engine.forget('alpha', records[1].source_episode_id)
            return result

    result = await AdaptiveRetriever(engine, SelectThenDelete(), limits=AdaptiveLimits(max_rounds=1)).retrieve(
        'alpha', 'bound question', scope=RecallScope.validated(where={'team':'blue'}))
    if bridge_state == 'retained':
        assert result.status == 'sufficient'
        assert {f.fact_id for f in result.recall.facts} == {f.fact_id for f in records}
    else:
        assert result.status != 'sufficient'
        if bridge_state == 'deleted':
            assert result.recall.facts == []
        else:
            assert {row.subject for row in result.recall.facts} == {'aster', 'cedar'}
            assert result.evidence_basis == 'unselected_candidates'
            assert result.model_selected_ids == ()
