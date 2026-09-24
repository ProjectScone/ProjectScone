from __future__ import annotations

import pytest

from .pipeline import Passage, messages, pack_context, rerank, unique_passages


def test_byte_budget_preserves_unicode_and_limits_sources() -> None:
    passages = [Passage(str(i), str(i), 'é' * 100) for i in range(6)]
    context, ids = pack_context(passages, max_bytes=50)
    assert len(context.encode()) <= 50
    assert ids == ['0']
    assert '\ufffd' not in context
    assert len(pack_context(passages)[1]) == 5


def test_shared_scores_preserve_arm_tie_order() -> None:
    a, b = Passage('a', 'A', 'one'), Passage('b', 'B', 'two')
    assert unique_passages({'scone': [a, b], 'llamaindex': [b, a]}) == [a, b]
    assert rerank([b, a], {'a': .5, 'b': .5}) == [b, a]
    assert rerank([a, b], {'a': .1, 'b': .9}) == [b, a]
    with pytest.raises(ValueError, match='collision'):
        unique_passages({'scone': [a], 'llamaindex': [Passage('a', 'B', 'different')]})


def test_original_question_and_common_prompt() -> None:
    question = 'Which city?'
    result = messages(question, '[Source 1]\nLondon')
    assert result[-1] == {'role': 'user', 'content': question}
    assert 'INSUFFICIENT_EVIDENCE' in result[0]['content']
