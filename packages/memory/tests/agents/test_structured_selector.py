"""Explicit question plans select complete quoted witnesses, not free prose."""

from ..paths import TESTS_ROOT
import json
from pathlib import Path

import pytest

from scone_memory.realtime.evidence_answer import EvidenceSelection, build_evidence_cards, construct_evidence_answer
from scone_memory.realtime.structured_selector import StructuredEvidenceSelector
from scone_memory.retrieval.structured_evidence import EvidenceRequirement
from ..agents.test_evidence_answer import claim, material, packet, passage, path


def selector(*requirements):
    return StructuredEvidenceSelector('question', requirements)


def route(subject='A', target='C', predicate='routes to'):
    return EvidenceRequirement(kind='path', subject=subject, predicate=predicate, object=target)


async def test_complete_plan_spanning_cards_is_selected_atomically():
    cards = build_evidence_cards(packet(claims=[claim(1, 'A', 'B'), claim(2, 'B', 'C')])).cards
    selection = await selector(route()).select('question', cards)
    assert selection.card_ids == ('card:1', 'card:2') and selection.atomic is True


@pytest.mark.parametrize('variant', ['missing', 'reverse', 'predicate'])
async def test_connected_or_related_records_cannot_witness_the_wrong_path(variant):
    claims = [claim(1, 'A', 'B'), claim(2, 'B', 'C')]
    requirement = route()
    if variant == 'missing':
        claims[1]['subject'] = 'D'
    elif variant == 'reverse':
        requirement = route('C', 'A')
    else:
        claims[0]['predicate'] = 'painted by'
    cards = build_evidence_cards(packet(claims=claims)).cards
    assert (await selector(requirement).select('question', cards)).card_ids == ()


async def test_compound_plan_and_competing_values_are_not_partially_selected():
    cards = build_evidence_cards(packet(claims=[claim(1, 'A', 'B'), claim(2, 'A', 'C')])).cards
    known = EvidenceRequirement(kind='fact', subject='A', predicate='routes to', object='B')
    missing = EvidenceRequirement(kind='fact', subject='A', predicate='manufactured by')
    assert (await selector(known).select('question', cards)).card_ids == ('card:1', 'card:2')
    assert (await selector(known, missing).select('question', cards)).card_ids == ()


async def test_selector_prefers_the_smallest_whole_cover_and_rejects_question_reuse():
    evidence = packet(claims=[claim(1, 'A', 'B'), claim(2, 'B', 'C')], paths=[path()])
    joined = build_evidence_cards(evidence).cards[0].model_copy(update={'id':'card:3'})
    separate = build_evidence_cards(packet(claims=[claim(1, 'A', 'B'), claim(2, 'B', 'C')])).cards
    planned = selector(route())
    assert (await planned.select('question', (*separate, joined))).card_ids == ('card:3',)
    with pytest.raises(ValueError, match='question'):
        await planned.select('other', separate)


async def test_more_than_three_required_cards_abstains_instead_of_dropping_a_bridge():
    cards = build_evidence_cards(packet(claims=[claim(n, str(n), str(n+1)) for n in range(1, 5)])).cards
    assert (await selector(route('1', '5')).select('question', cards)).card_ids == ()


async def test_citation_id_without_structured_claim_does_not_cover_a_requirement():
    from scone_memory.realtime.evidence_answer import EvidenceCard
    cards = build_evidence_cards(packet(claims=[claim(1, 'A', 'B')])).cards
    prose = EvidenceCard(id='card:2', kind='passage', text='No.', evidence_ids=('fact:1',))
    selected = await selector(route('A', 'B')).select('question', (*cards, prose))
    assert selected.card_ids == ('card:1',)


async def test_inverse_fact_selection_returns_subject_quotes_without_reversing_them():
    cards = build_evidence_cards(packet(claims=[claim(1, 'A', 'B'), claim(2, 'C', 'B'), claim(3, 'B', 'D')])).cards
    selected = await selector(EvidenceRequirement(kind='fact', predicate='routes to', object='B')).select('question', cards)
    assert selected.card_ids == ('card:1', 'card:2') and selected.atomic is True


@pytest.mark.parametrize('delete', [False, True])
async def test_inverse_lookup_uses_real_scoped_memory_and_checks_deletion(engine, delete):
    from scone_memory.core.ports import NewFact
    from scone_memory.realtime.text import TextConversation
    retained_id = None
    for space, team, subject in [('alpha', 'blue', 'Juniper'), ('alpha', 'red', 'PRIVATE_TEAM'),
                                 ('other', 'blue', 'PRIVATE_SPACE')]:
        quote = f'{subject} uses Polaris.'
        episode = await engine.remember(space, quote, metadata={'team':team})
        await engine.documents.insert_fact(NewFact(space=space, subject=subject, predicate='uses', object='Polaris',
            source_episode_id=episode.episode_id, quote=quote, valid_from='2025-01-01T00:00:00Z'))
        if space == 'alpha' and team == 'blue':
            retained_id = episode.episode_id
    question = 'Who uses Polaris?'
    planned = StructuredEvidenceSelector(question, (EvidenceRequirement(kind='fact', predicate='uses', object='Polaris'),))

    class SelectThenChange:
        async def select(self, question, cards):
            result = await planned.select(question, cards)
            if delete:
                await engine.forget('alpha', retained_id)
            return result

    conversation = TextConversation(engine, 'alpha', 'inverse', evidence_selector=SelectThenChange(),
        evidence_answer_policy='required', where={'team':'blue'})
    observed = []

    async def observe(text):
        observed.append(text)

    if delete:
        with pytest.raises(RuntimeError, match='evidence answer'):
            await conversation.reply(question, on_text=observe)
        assert observed == []
        assert [row.metadata['role'] for row in await engine.episodes('alpha', {'session_id':'inverse'})] == ['user']
    else:
        reply = await conversation.reply(question, on_text=observe)
        assert reply['evidence_answer']['status'] == 'selected'
        assert 'Juniper uses Polaris.' in reply['text'] and 'PRIVATE' not in reply['text']
        assert observed == [reply['text']]
    await conversation.close()


async def test_atomic_output_budget_never_publishes_a_partial_chain():
    evidence = packet(claims=[claim(1, 'A', 'B'), claim(2, 'B', 'C')])
    cards = build_evidence_cards(evidence).cards
    bound = max(64, len(cards[0].text.encode()))
    result = await construct_evidence_answer(selector(route()), 'question', material(evidence, ('fact:1', 'fact:2')),
                                             max_answer_bytes=bound)
    assert result.receipt['status'] == 'no_selection'
    assert result.receipt['evidence_ids'] == []
    assert result.receipt['omitted_card_count'] == 2
    assert result.receipt['atomic_selection'] is True
    assert 'A routes to B' not in result.answer and 'B routes to C' not in result.answer


async def test_exact_atomic_byte_boundary_preserves_quotes_and_source_ids():
    evidence = packet(claims=[claim(1, 'A', 'B'), claim(2, 'B', 'C')])
    expected = '\n\n'.join(card.text for card in build_evidence_cards(evidence).cards)
    result = await construct_evidence_answer(selector(route()), 'question', material(evidence, ('fact:1', 'fact:2')),
                                             max_answer_bytes=len(expected.encode()))
    assert result.answer == expected
    assert result.receipt['evidence_ids'] == ['fact:1', 'fact:2']
    assert result.receipt['verified_accuracy'] is False


@pytest.mark.parametrize('variant', ['duplicate_card', 'changed_identity', 'forged', 'oversize'])
async def test_invalid_card_sets_fail_closed(variant):
    card = build_evidence_cards(packet(claims=[claim(1, 'A', 'B')])).cards[0]
    cards = (card,)
    if variant == 'duplicate_card':
        cards = (card, card)
    elif variant == 'changed_identity':
        other = card.model_copy(update={'id':'card:2', 'claims':(card.claims[0].model_copy(update={'object':'C'}),)})
        cards = (card, other)
    elif variant == 'forged':
        cards = (card.model_copy(update={'claims':(card.claims[0].model_copy(update={'fact_id':True}),)}),)
    else:
        cards = (card.model_copy(update={'text':'界' * 60000}),)
    with pytest.raises(ValueError):
        await selector(route('A', 'B')).select('question', cards)


def test_atomic_selection_flag_is_strict():
    with pytest.raises(ValueError):
        EvidenceSelection(atomic='yes')


@pytest.mark.parametrize('case', json.loads((TESTS_ROOT / 'fixtures/tool_action_cases.json').read_text())['cases'],
                         ids=lambda case: case['id'])
async def test_development_failures_with_explicit_host_plans(case):
    # These are hand-authored plans, not an automatic natural-language score.
    plans = {
        'forward_chain': (route('Juniper', 'Beacon', 'depends on'),),
        'direct_fact': (EvidenceRequirement(kind='fact', subject='Vega', predicate='depends on'),),
        'missing_bridge': (route('Juniper', 'Beacon', 'depends on'),),
        'reverse_direction': (route('Beacon', 'Juniper', 'depends on'),),
        'unknown_attribute': (EvidenceRequirement(kind='fact', subject='Polaris', predicate='manufactured by'),),
        'competing_facts': (EvidenceRequirement(kind='fact', subject='Juniper', predicate='depends on'),),
        'wrong_relation_join': (route('Juniper', 'Beacon', 'depends on'),),
        'source_instruction': (EvidenceRequirement(kind='fact', subject='Juniper', predicate='depends on'),),
        'empty_store': (EvidenceRequirement(kind='fact', subject='Juniper', predicate='launch date'),),
    }
    claims = [dict(claim(i, subject, obj), predicate=predicate, quote=f'{subject} {predicate} {obj}.')
              for i, (subject, predicate, obj) in enumerate(case['facts'], 1)]
    sources = [passage(1, str(claims[0]['quote']) + ' ' + case['source_suffix'])] if case.get('source_suffix') else []
    evidence = packet(claims=claims, sources=sources)
    planned = StructuredEvidenceSelector(case['question'], plans[case['id']])
    result = await construct_evidence_answer(planned, case['question'],
        material(evidence, (('chunk:1',) if sources else ()) + tuple(f'fact:{i}' for i in range(1, len(claims)+1))))
    positive = case['id'] in {'forward_chain', 'direct_fact', 'competing_facts', 'source_instruction'}
    assert (result.receipt['status'] == 'selected') is positive
    if positive:
        assert all(row['quote'] in result.answer for row in claims)
    else:
        assert result.receipt['evidence_ids'] == []
    assert result.receipt['verified_accuracy'] is False
    assert 'ACCESS_GRANTED' not in result.answer
