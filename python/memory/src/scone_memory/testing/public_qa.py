"""Public QA data preparation and standard answer metrics; no model or network calls."""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import re
import string
from typing import Literal, Sequence, cast
from urllib.parse import quote

from pydantic import BaseModel, ConfigDict

Dataset = Literal['hotpotqa', 'squad']
SEED = 'scone-public-qa-v1:20260908:'
FORMAT_INSTRUCTION = ('Answer the latest question with only the shortest complete answer (a name, '
    'date, number, phrase, or yes/no). Do not add explanation, citations, or source '
    'IDs. If the supplied memory does not contain sufficient evidence, answer '
    'exactly INSUFFICIENT_EVIDENCE.')


class FrozenRecord(BaseModel):
    model_config = ConfigDict(extra='forbid', frozen=True)


class Document(FrozenRecord):
    id: str
    title: str
    text: str
    source_url: str

    @property
    def content(self) -> str:
        return self.title + '\n\n' + self.text


class Question(FrozenRecord):
    id: str
    dataset: Dataset
    question: str


class Gold(FrozenRecord):
    id: str
    dataset: Dataset
    answers: tuple[str, ...]
    support_documents: tuple[str, ...]
    support_sentences: tuple[tuple[str, str], ...] = ()


@dataclass(frozen=True)
class Bundle:
    documents: tuple[Document, ...]
    queries: tuple[Question, ...]
    reserved: tuple[Question, ...]
    gold: tuple[Gold, ...]


def _mapping(value: object) -> dict[str, object]:
    if not isinstance(value, dict) or any(not isinstance(k, str) for k in value):
        raise ValueError('expected JSON object')
    return cast(dict[str, object], value)


def _array(value: object) -> list[object]:
    if not isinstance(value, list):
        raise ValueError('expected JSON array')
    return cast(list[object], value)


def _text(value: object) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError('expected nonempty text')
    return value


def _sentence(value: object) -> str:
    if not isinstance(value, str):
        raise ValueError('expected sentence text')
    return value


def _document(title: str, text: str) -> Document:
    digest = hashlib.sha256((title + '\0' + text).encode()).hexdigest()
    return Document(id=digest, title=title, text=text,
        source_url='https://en.wikipedia.org/wiki/' + quote(title.replace(' ', '_'), safe=''))


def _index(rows: list[dict[str, object]], key: str) -> dict[str, dict[str, object]]:
    indexed: dict[str, dict[str, object]] = {}
    for row in rows:
        identifier = _text(row[key])
        if identifier in indexed:
            raise ValueError('duplicate question ID')
        indexed[identifier] = row
    return indexed


def _selection(rows: dict[str, dict[str, object]], dataset: Dataset, count: int) -> list[str]:
    if type(count) is not int or count < 1 or count * 2 > len(rows):
        raise ValueError('insufficient questions for evaluation and reserve')
    return sorted(rows, key=lambda key: hashlib.sha256((SEED + dataset + ':' + key).encode()).hexdigest())[:count * 2]


def _hotpot(row: dict[str, object]) -> tuple[list[Document], Gold]:
    documents: list[Document] = []
    titles: dict[str, tuple[Document, list[str]]] = {}
    for raw in _array(row['context']):
        pair = _array(raw)
        if len(pair) != 2:
            raise ValueError('invalid context pair')
        title = _text(pair[0])
        sentences = [_sentence(s) for s in _array(pair[1])]
        if title in titles:
            raise ValueError('duplicate context title')
        document = _document(title, '\n'.join(sentences))
        documents.append(document)
        titles[title] = document, sentences
    supports: list[tuple[str, str]] = []
    for raw in _array(row['supporting_facts']):
        pair = _array(raw)
        if len(pair) != 2 or type(pair[1]) is not int:
            raise ValueError('invalid support annotation')
        title, index = _text(pair[0]), pair[1]
        if title not in titles or not 0 <= index < len(titles[title][1]):
            raise ValueError('support annotation outside context')
        document, sentences = titles[title]
        supports.append((document.id, sentences[index]))
    if not supports:
        raise ValueError('empty support annotations')
    return documents, Gold(id='hotpotqa:' + _text(row['_id']), dataset='hotpotqa',
        answers=(_text(row['answer']),), support_documents=tuple(sorted({s[0] for s in supports})),
        support_sentences=tuple(dict.fromkeys(supports)))


def _squad(row: dict[str, object]) -> tuple[list[Document], Gold]:
    document = _document(_text(row['title']), _text(row['context']))
    answers: list[str] = []
    for raw in _array(row['answers']):
        answer = _mapping(raw)
        text, start = _text(answer['text']), answer['answer_start']
        if type(start) is not int or start < 0 or document.text[start:start + len(text)] != text:
            raise ValueError('answer span outside source')
        answers.append(text)
    if not answers:
        raise ValueError('answerable dataset has no answer')
    return [document], Gold(id='squad:' + _text(row['id']), dataset='squad',
        answers=tuple(dict.fromkeys(answers)), support_documents=(document.id,))


def build_bundle(hotpot: Path, squad: Path, *, count: int = 100) -> Bundle:
    """Sample by ID alone, then separate labels from all model-visible data."""
    h_rows = [_mapping(row) for row in _array(json.loads(hotpot.read_bytes()))]
    s_rows: list[dict[str, object]] = []
    for raw_article in _array(_mapping(json.loads(squad.read_bytes()))['data']):
        article = _mapping(raw_article)
        for raw_paragraph in _array(article['paragraphs']):
            paragraph = _mapping(raw_paragraph)
            for raw_question in _array(paragraph['qas']):
                s_rows.append({**_mapping(raw_question), 'title': article['title'], 'context': paragraph['context']})
    documents: dict[str, Document] = {}
    queries: list[Question] = []
    reserved: list[Question] = []
    gold: list[Gold] = []
    for dataset, rows, key in [('hotpotqa', h_rows, '_id'), ('squad', s_rows, 'id')]:
        name = cast(Dataset, dataset)
        indexed = _index(rows, key)
        for position, identifier in enumerate(_selection(indexed, name, count)):
            row = indexed[identifier]
            sources, labels = _hotpot(row) if name == 'hotpotqa' else _squad(row)
            for source in sources:
                if source.id in documents and documents[source.id] != source:
                    raise ValueError('source digest collision')
                documents[source.id] = source
            question = Question(id=labels.id, dataset=name, question=_text(row['question']))
            (queries if position < count else reserved).append(question)
            gold.append(labels)
    order = lambda q: hashlib.sha256((SEED + q.id).encode()).hexdigest()
    return Bundle(tuple(documents[key] for key in sorted(documents)),
        tuple(sorted(queries, key=order)), tuple(sorted(reserved, key=order)), tuple(gold))


def benchmark_messages(question: Question) -> list[dict[str, str]]:
    from ..realtime.text import DEFAULT_SYSTEM_PROMPT
    return [{'role': 'system', 'content': DEFAULT_SYSTEM_PROMPT + '\n' + FORMAT_INSTRUCTION},
            {'role': 'user', 'content': question.question}]


def normalize_answer(value: str) -> str:
    clean = ''.join(character for character in value.lower() if character not in string.punctuation)
    return ' '.join(re.sub(r'\b(a|an|the)\b', ' ', clean).split())


def answer_score(prediction: str, answers: Sequence[str], dataset: Dataset, completed: bool) -> dict[str, float]:
    """Official normalization and token overlap; never repair generated answers."""
    if not completed or prediction.strip() == 'INSUFFICIENT_EVIDENCE':
        return {'em': 0., 'f1': 0.}
    normalized = normalize_answer(prediction)
    em, f1 = 0., 0.
    for answer in answers:
        expected = normalize_answer(answer)
        em = max(em, float(normalized == expected))
        if dataset == 'hotpotqa' and normalized != expected and (
            normalized in ('yes', 'no', 'noanswer') or expected in ('yes', 'no', 'noanswer')):
            continue
        left, right = normalized.split(), expected.split()
        overlap = sum((Counter(left) & Counter(right)).values())
        if overlap:
            f1 = max(f1, 2. * overlap / (len(left) + len(right)))
    return {'em': em, 'f1': f1}
