from dataclasses import replace

import pytest

from scone_memory.experimental.memory_contracts import (
    ContractRequest, Evidence, Judgment, MemoryContract, compile_contract, evidence_worlds,
)


def request(evidence: tuple[Evidence, ...]) -> ContractRequest:
    return ContractRequest('project:alpha:permissions-v1', 'Who owns deployment?',
                           'Maya owns deployment.', evidence)


def compile_example(evidence: tuple[Evidence, ...], supported: set[int]):
    task = request(evidence)
    worlds = evidence_worlds(task)
    judgments = tuple(Judgment(.99 if w.mask in supported else .01, .01) for w in worlds)
    return compile_contract(task, judgments, model='test-v1', expires_at=200)


def test_independent_support_survives_one_withdrawal_but_copies_do_not():
    original = Evidence('a', 'meeting', 'Maya owns deployment.')
    copied = Evidence('b', 'meeting', 'Meeting recap: Maya owns deployment.')
    independent = Evidence('c', 'roster', 'Deployment owner: Maya.')
    contract = compile_example((original, copied, independent), {1, 2, 3})
    assert len(contract.judgments) == 4
    assert contract.evaluate(request((independent,)), now=100).status == 'supported'
    # A partial origin group is not an evaluated world, even if text looks equivalent.
    assert contract.evaluate(request((copied,)), now=100).status == 'recompile'
    assert contract.evaluate(request(()), now=100).status == 'insufficient'
    assert contract.minimal_withdrawals() == (('meeting', 'roster'),)


def test_multihop_requires_both_sources_and_reports_each_breaking_withdrawal():
    sources = (Evidence('a', 'rule', 'The release captain owns deployment.'),
               Evidence('b', 'roster', 'Maya is release captain.'))
    contract = compile_example(sources, {3})
    assert contract.evaluate(request(sources[:1]), now=100).status == 'insufficient'
    assert contract.minimal_withdrawals() == (('roster',), ('rule',))


def test_added_changed_context_and_expired_evidence_cannot_reuse_contract():
    source = Evidence('a', 'roster', 'Maya owns deployment.')
    contract = compile_example((source,), {1})
    changes = [
        request((replace(source, text='Leo owns deployment.'),)),
        request((source, Evidence('b', 'correction', 'Leo owns deployment.'))),
        replace(request((source,)), context_key='another-space'),
        replace(request((source,)), question='Who owns billing?'),
        replace(request((source,)), claim='Leo owns deployment.'),
    ]
    for change in changes:
        assert contract.evaluate(change, now=100).status == 'recompile'
    assert contract.evaluate(request((source,)), now=200).status == 'recompile'


def test_conflict_is_not_monotone_and_is_not_hidden_by_a_supporting_subset():
    sources = (Evidence('a', 'a', 'Maya owns deployment.'),
               Evidence('b', 'b', 'Leo owns deployment, not Maya.'))
    task = request(sources)
    contract = compile_contract(task, (
        Judgment(0, 0), Judgment(.99, .01), Judgment(.01, .99), Judgment(.99, .99),
    ), model='test', expires_at=200)
    assert contract.evaluate(task, now=100).status == 'conflict'
    assert contract.evaluate(request(sources[:1]), now=100).status == 'supported'
    assert contract.minimal_withdrawals() == ()


@pytest.mark.parametrize('value', [float('nan'), float('inf'), -1, 1.1, True])
def test_invalid_probabilities_fail_closed(value):
    with pytest.raises(ValueError):
        Judgment(value, .1)


def test_limits_incomplete_judgments_and_duplicate_source_ids_fail_closed():
    with pytest.raises(ValueError):
        evidence_worlds(request(tuple(Evidence(str(i), str(i), 'text') for i in range(6))))
    source = Evidence('a', 'a', 'Maya owns deployment.')
    with pytest.raises(ValueError):
        evidence_worlds(request((source, source)))
    with pytest.raises(ValueError):
        compile_contract(request((source,)), (Judgment(0, 0),), model='test', expires_at=200)


def test_contract_roundtrip_keeps_withdrawal_behavior_and_rejects_bad_artifacts():
    sources = (Evidence('a', 'a', 'Maya owns deployment.'),)
    original = compile_example(sources, {1})
    restored = MemoryContract.from_json(original.to_json())
    assert restored == original
    assert restored.evaluate(request(()), now=100).status == 'insufficient'
    for invalid in ('{}', original.to_json().replace('0.99', 'true'),
                    original.to_json().replace('test-v1', ''), ' ' * 128001):
        with pytest.raises(ValueError):
            MemoryContract.from_json(invalid)
