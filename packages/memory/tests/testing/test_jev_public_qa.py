import pytest


def test_document_coverage_does_not_count_duplicate_chunks_as_more_evidence():
    from scone_memory.testing.jev_public_qa import rank_metrics
    result = rank_metrics(['a','a','x','x','x','b'], {'a','b'})
    assert result['recall_at_5'] == 0.5
    assert result['all_at_5'] == 0
    assert result['recall_at_10'] == 1
    assert result['mrr_at_10'] == 1


def test_failed_retrieval_is_zero_not_removed_from_denominator():
    from scone_memory.testing.jev_public_qa import rank_metrics
    assert set(rank_metrics([], {'a'}).values()) == {0}
    with pytest.raises(ValueError):
        rank_metrics(['a'], set())


def test_first_relevant_rank_and_complete_evidence_are_different_metrics():
    from scone_memory.testing.jev_public_qa import rank_metrics
    result = rank_metrics(['x','a','y'], {'a','b'})
    assert result['mrr_at_10'] == 0.5
    assert result['recall_at_10'] == 0.5
    assert result['all_at_10'] == 0
