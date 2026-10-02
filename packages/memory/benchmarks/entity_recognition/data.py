"""Entity recognition test sets as sentences with gold typed spans, and the type map onto Scone's entity types.

Two public test sets, read from their Hugging Face parquet files:

- OntoNotes 5 (``tner/ontonotes5`` test, 8,262 sentences): BIO tags over 18 types, among them ORG, PRODUCT,
  NORP (nationalities and religious or political groups), GPE, LOC, FAC, PERSON, DATE, TIME, MONEY.
- Few-NERD supervised (``DFKI-SLT/few-nerd`` test, 37,648 sentences): IO tags over 66 fine types such as
  ``organization-company``, ``product-software`` and ``location-GPE``; consecutive tokens with one tag are one span.

A span is ``(start, end, type)`` over token indices, end exclusive. Sentences are also given as text, tokens joined
by single spaces, with each token's character offsets, so a model that reads text is scored on the same spans.
"""
from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

ONTONOTES_LABELS = {
    0: 'O', 1: 'B-CARDINAL', 2: 'B-DATE', 3: 'I-DATE', 4: 'B-PERSON', 5: 'I-PERSON', 6: 'B-NORP', 7: 'B-GPE',
    8: 'I-GPE', 9: 'B-LAW', 10: 'I-LAW', 11: 'B-ORG', 12: 'I-ORG', 13: 'B-PERCENT', 14: 'I-PERCENT', 15: 'B-ORDINAL',
    16: 'B-MONEY', 17: 'I-MONEY', 18: 'B-WORK_OF_ART', 19: 'I-WORK_OF_ART', 20: 'B-FAC', 21: 'B-TIME',
    22: 'I-CARDINAL', 23: 'B-LOC', 24: 'B-QUANTITY', 25: 'I-QUANTITY', 26: 'I-NORP', 27: 'I-LOC', 28: 'B-PRODUCT',
    29: 'I-TIME', 30: 'B-EVENT', 31: 'I-EVENT', 32: 'I-FAC', 33: 'B-LANGUAGE', 34: 'I-PRODUCT', 35: 'I-ORDINAL',
    36: 'I-LANGUAGE',
}  # the dataset card's label2id, inverted

#: Scone's entity types, what each test set's labels mean in them, and the description a model is given.
SCONE_TYPES = {
    'person': 'a person, by name',
    'organization': 'a company, institution, agency, team or other organization',
    'nationality': 'a nationality, religious or political group (e.g. American, Buddhist, Republicans)',
    'location': 'a country, city, state, region or other geographic place',
    'facility': 'a building, airport, road, bridge or other facility',
    'product': 'a product, tool, software, vehicle or other made object',
    'event': 'a named event such as a war, storm, election or sports event',
    'work_of_art': 'a title of a book, song, film or other work',
    'law': 'a named law or legal document',
    'language': 'a named language',
    'date': 'a date or period',
    'time': 'a time of day',
    'money': 'a monetary amount',
    'percent': 'a percentage',
    'quantity': 'a measurement with a unit',
    'ordinal': 'an ordinal such as first or 3rd',
    'cardinal': 'a number that is not another type',
}

ONTONOTES_TO_SCONE = {
    'PERSON': 'person', 'ORG': 'organization', 'NORP': 'nationality', 'GPE': 'location', 'LOC': 'location',
    'FAC': 'facility', 'PRODUCT': 'product', 'EVENT': 'event', 'WORK_OF_ART': 'work_of_art', 'LAW': 'law',
    'LANGUAGE': 'language', 'DATE': 'date', 'TIME': 'time', 'MONEY': 'money', 'PERCENT': 'percent',
    'QUANTITY': 'quantity', 'ORDINAL': 'ordinal', 'CARDINAL': 'cardinal',
}


def fewnerd_to_scone(fine: str) -> str | None:
    """Few-NERD's fine type in Scone's types; None for types Scone does not name (they are left out of scoring)."""
    coarse, _, detail = fine.partition('-')
    if coarse == 'person':
        return 'person'
    if coarse == 'organization':
        return 'organization'
    if coarse == 'location':
        return 'location'
    if coarse == 'building':
        return 'facility'
    if coarse == 'product':
        return 'product'
    if coarse == 'event':
        return 'event'
    if coarse == 'art':
        return 'work_of_art'
    if fine == 'other-law':
        return 'law'
    if fine == 'other-language':
        return 'language'
    if fine == 'other-currency':
        return None  # a currency name, not an amount
    return None


@dataclass(frozen=True)
class Sentence:
    id: str
    tokens: tuple[str, ...]
    gold: frozenset[tuple[int, int, str]]  # token spans in the dataset's own types

    @property
    def text(self) -> str:
        return ' '.join(self.tokens)

    def offsets(self) -> list[tuple[int, int]]:
        spans, position = [], 0
        for token in self.tokens:
            spans.append((position, position + len(token)))
            position += len(token) + 1
        return spans


def bio_spans(tags: Sequence[str]) -> set[tuple[int, int, str]]:
    spans: set[tuple[int, int, str]] = set()
    start, kind = None, None
    for index, tag in enumerate([*tags, 'O']):
        if tag == 'O' or tag.startswith('B-') or (tag.startswith('I-') and tag[2:] != kind):
            if start is not None and kind is not None:
                spans.add((start, index, kind))
            start, kind = (index, tag[2:]) if tag != 'O' else (None, None)
    return spans


def io_spans(tags: Sequence[str]) -> set[tuple[int, int, str]]:
    spans: set[tuple[int, int, str]] = set()
    start, kind = None, None
    for index, tag in enumerate([*tags, 'O']):
        if tag != kind:
            if kind not in (None, 'O') and start is not None:
                spans.add((start, index, kind))
            start, kind = index, tag
    return spans


def load_ontonotes(path: Path) -> list[Sentence]:
    import pyarrow.parquet as pq

    table = pq.read_table(path)
    out = []
    for row, (tokens, tags) in enumerate(zip(table.column('tokens').to_pylist(), table.column('tags').to_pylist())):
        out.append(Sentence(f'ontonotes:{row}', tuple(tokens), frozenset(bio_spans([ONTONOTES_LABELS[t] for t in tags]))))
    return out


def load_fewnerd(path: Path, labels_path: Path) -> list[Sentence]:
    import pyarrow.parquet as pq

    names = json.loads(labels_path.read_text(encoding='utf-8'))['fine']
    table = pq.read_table(path)
    out = []
    for sid, tokens, tags in zip(table.column('id').to_pylist(), table.column('tokens').to_pylist(),
                                 table.column('fine_ner_tags').to_pylist()):
        out.append(Sentence(f'fewnerd:{sid}', tuple(tokens), frozenset(io_spans([names[t] for t in tags]))))
    return out


def to_scone(sentence: Sentence, dataset: str) -> frozenset[tuple[int, int, str]]:
    """The gold spans in Scone's types; spans of types Scone does not name are dropped."""
    mapped = set()
    for start, end, kind in sentence.gold:
        scone = ONTONOTES_TO_SCONE.get(kind) if dataset == 'ontonotes' else fewnerd_to_scone(kind)
        if scone:
            mapped.add((start, end, scone))
    return frozenset(mapped)


def scored_types(dataset: str) -> tuple[str, ...]:
    """The Scone types a dataset's gold can contain; predictions of other types are not counted against it."""
    if dataset == 'ontonotes':
        return tuple(sorted(set(ONTONOTES_TO_SCONE.values())))
    return ('event', 'facility', 'language', 'law', 'location', 'organization', 'person', 'product', 'work_of_art')
