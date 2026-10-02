from __future__ import annotations

import json
from pathlib import Path

from .data import Sentence, bio_spans, fewnerd_to_scone, io_spans, to_scone
from .run import char_to_tokens, place_text, score


def _sentence(tokens: str, gold: set[tuple[int, int, str]], sid: str = 'ontonotes:0') -> Sentence:
    return Sentence(sid, tuple(tokens.split()), frozenset(gold))


def test_bio_and_io_spans_close_on_a_type_change() -> None:
    assert bio_spans(['B-ORG', 'I-ORG', 'O', 'B-GPE', 'I-ORG']) == {(0, 2, 'ORG'), (3, 4, 'GPE'), (4, 5, 'ORG')}
    assert io_spans(['a', 'a', 'O', 'b', 'a']) == {(0, 2, 'a'), (3, 4, 'b'), (4, 5, 'a')}


def test_types_map_onto_scones_and_unnamed_ones_drop() -> None:
    s = _sentence('Apple sold iPhones in France', {(0, 1, 'ORG'), (2, 3, 'PRODUCT'), (4, 5, 'GPE')})
    assert to_scone(s, 'ontonotes') == {(0, 1, 'organization'), (2, 3, 'product'), (4, 5, 'location')}
    assert fewnerd_to_scone('organization-company') == 'organization'
    assert fewnerd_to_scone('product-software') == 'product' and fewnerd_to_scone('other-disease') is None


def test_character_spans_must_align_with_token_edges() -> None:
    s = _sentence('the U.S. economy', set())
    assert char_to_tokens(s, 4, 8) == (1, 2)
    assert char_to_tokens(s, 4, 6) is None  # ends inside a token


def test_model_text_is_placed_on_the_first_free_matching_tokens() -> None:
    s = _sentence('Paris , Texas and Paris , France', set())
    taken: set[tuple[int, int]] = set()
    first = place_text(s, 'Paris', taken)
    assert first == (0, 1)
    taken.add(first)
    assert place_text(s, 'Paris', taken) == (4, 5)
    assert place_text(s, 'Paris, France', set()) == (4, 7)  # no space before the comma in the model's text
    assert place_text(s, 'London', set()) is None


def test_scoring_is_strict_on_span_and_type_and_ignores_unlabelled_types(tmp_path: Path) -> None:
    s = _sentence('Acme hired Bob in May', {(0, 1, 'ORG'), (2, 3, 'PERSON'), (4, 5, 'DATE')}, sid='fewnerd:1')
    # Few-NERD labels no dates, so a predicted date is neither right nor wrong there.
    gold_fewnerd = Sentence('fewnerd:1', s.tokens, frozenset({(0, 1, 'organization-company'), (2, 3, 'person-other')}))
    rows = [{'id': 'fewnerd:1', 'spans': [[0, 1, 'organization', 1], [2, 4, 'person', 1], [4, 5, 'date', 1]],
             'unplaced': 1, 'ms': 5}]
    path = tmp_path / 'p.jsonl'
    path.write_text(''.join(json.dumps(r) + '\n' for r in rows))
    result = score([gold_fewnerd], 'fewnerd', path)
    micro = result['micro']
    assert (micro['precision'], micro['recall']) == (0.5, 0.5)  # Bob's span is wrong: one hit, one miss, one false
    assert result['unplaced_predictions'] == 1 and result['missing'] == 0
