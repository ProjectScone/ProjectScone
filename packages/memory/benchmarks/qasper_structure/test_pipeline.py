from __future__ import annotations

from matched_qa.pipeline import Passage
from scone_memory.retrieval.structured_document import SectionEvidence
from .pipeline import bounded, context, messages, passage


def test_candidate_bytes_do_not_split_utf8_and_original_offsets_match() -> None:
    text = 'é' * 6000
    item = SectionEvidence('section', text, 7, 7 + len(text.encode()))
    result = bounded([item])
    assert len(result[0].text.encode()) == 8000
    assert result[0].end == 8007
    assert len(bounded([item] * 40)) == 1


def test_shared_judgments_use_identical_text_identity() -> None:
    a = passage(SectionEvidence('one', 'same', 0, 4), 'paper')
    b = passage(SectionEvidence('two', 'same', 8, 12), 'paper')
    assert a.key == b.key


def test_final_context_has_same_item_and_byte_caps() -> None:
    items = [Passage(str(i), 'paper', f'{i} ' + 'x' * 2000) for i in range(10)]
    packed = context(items, {str(i): 1. for i in range(10)})
    assert len(packed.encode()) <= 8000
    assert packed.count('[Source ') <= 5
    assert 'Source 1' in packed
    assert messages('Question?', packed)[-1]['content'] == 'Question?'
    assert 'Unanswerable' in messages('Question?', packed)[0]['content']
