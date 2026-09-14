"""Which passage each sentence of an answer came from, found without a model.

An answer composed from recalled passages reads as sourced whether or not
it is. The leading reference asks the model to cite numbered sources while
it writes, and nothing checks the numbers afterwards. This takes an answer
already written, by a model or a person, and the passages it was given,
and aligns each sentence to them mechanically:

- **quoted**: the sentence shares a run of at least ``QUOTE_WORDS``
  consecutive words with a passage, compared case-folded with the
  punctuation between words ignored. The span of the passage it matched is
  given, so a reader can see the words, not a score.
- **overlapping**: no such run, but at least ``OVERLAP_SHARE`` of the
  sentence's content words (the lexical lane's tokens, stopwords left out)
  appear in one passage.
- **unattributed**: neither, for every passage.
- **too_short**: fewer than two content words, too little to align.

Among passages, a quote beats an overlap, a longer run beats a shorter one,
then more of the sentence's words, then the passage given first. Numbers
in a sentence (tokens with a digit) that its passage does not hold are
named on the sentence, attributed or not.

Word overlap is not support. A sentence can quote a passage and still
misstate it, and paraphrase can be faithful and unattributed. Neither
threshold is measured yet. The record states both and never claims the
answer is accurate. A run of words is found between words separated by
space or punctuation, so text in scripts written without spaces can
overlap but is rarely quoted.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal, Optional, Sequence

from ..core.errors import InvalidInput
from .lexical import tokenize
from .synthesis import Passage
from .window import sentence_spans

if TYPE_CHECKING:  # pragma: no cover - typing only
    from ..memory.engine import MemoryEngine

#: Consecutive words a sentence must share with a passage to be a quote. Not measured.
QUOTE_WORDS = 5
#: Share of a sentence's content words one passage must hold for an overlap. Not measured.
OVERLAP_SHARE = 0.6
#: Characters of answer taken; more is refused, not cut.
MAX_ANSWER_CHARS = 20_000
#: Passages taken; more are refused, not cut.
MAX_PASSAGES = 50

Status = Literal["quoted", "overlapping", "unattributed", "too_short"]
STATUSES: tuple[Status, ...] = ("quoted", "overlapping", "unattributed", "too_short")
_WORD = re.compile(r"\w+")


@dataclass(frozen=True)
class AttributedSentence:
    text: str
    #: Character offsets of the sentence in the answer.
    start: int
    end: int
    status: Status
    #: The passage's id; None when unattributed or too short.
    passage: Optional[str]
    #: Character offsets in that passage of the run of words shared, for a quote.
    quote: Optional[tuple[int, int]]
    #: Share of the sentence's content words the passage holds.
    coverage: float
    #: Content words of the sentence the passage holds.
    shared: int
    #: Numbers in the sentence the passage does not hold.
    numbers_missing: tuple[str, ...]


@dataclass(frozen=True)
class Attribution:
    sentences: tuple[AttributedSentence, ...]
    passages: tuple[Passage, ...]
    verified_accuracy: Literal[False] = False

    def counts(self) -> dict[str, int]:
        return {status: sum(1 for sentence in self.sentences if sentence.status == status) for status in STATUSES}

    def record(self) -> dict[str, object]:
        held = {passage.id: passage.text for passage in self.passages}
        return {
            "schema_version": 1,
            "sentences": [{
                "text": sentence.text, "start": sentence.start, "end": sentence.end, "status": sentence.status,
                "passage": sentence.passage, "coverage": round(sentence.coverage, 4), "shared": sentence.shared,
                "quote": None if sentence.quote is None or sentence.passage is None else {
                    "start": sentence.quote[0], "end": sentence.quote[1],
                    "text": held[sentence.passage][sentence.quote[0]:sentence.quote[1]]},
                "numbers_missing": list(sentence.numbers_missing)} for sentence in self.sentences],
            "counts": self.counts(),
            "rules": {"quote_words": QUOTE_WORDS, "overlap_share": OVERLAP_SHARE, "measured": False},
            "verified_accuracy": False,
        }


@dataclass(frozen=True)
class _Read:
    """A passage prepared once: its words with offsets, their runs of QUOTE_WORDS, its tokens."""

    passage: Passage
    words: tuple[str, ...]
    spans: tuple[tuple[int, int], ...]
    grams: dict[tuple[str, ...], tuple[int, ...]]
    tokens: frozenset[str]


def _read(passage: Passage) -> _Read:
    found = list(_WORD.finditer(passage.text))
    words = tuple(match.group().casefold() for match in found)
    grams: dict[tuple[str, ...], list[int]] = {}
    for index in range(len(words) - QUOTE_WORDS + 1):
        grams.setdefault(words[index:index + QUOTE_WORDS], []).append(index)
    return _Read(passage, words, tuple(match.span() for match in found),
                 {gram: tuple(at) for gram, at in grams.items()}, frozenset(tokenize(passage.text)))


def _longest_run(words: Sequence[str], read: _Read) -> tuple[int, int]:
    """The longest run of the sentence's words found in the passage, as
    (words, index of its first word in the passage); (0, 0) when none
    reaches QUOTE_WORDS, so a shorter run counts for nothing."""
    best = (0, 0)
    for index in range(len(words) - QUOTE_WORDS + 1):
        for at in read.grams.get(tuple(words[index:index + QUOTE_WORDS]), ()):
            length = QUOTE_WORDS
            while (index + length < len(words) and at + length < len(read.words)
                   and words[index + length] == read.words[at + length]):
                length += 1
            if length > best[0]:
                best = (length, at)
    return best


def _numbers(text: str) -> list[str]:
    return [word for word in (match.group().casefold() for match in _WORD.finditer(text)) if any(c.isdigit() for c in word)]


def attribute_answer(answer: str, passages: Sequence[Passage]) -> Attribution:
    """Each sentence of ``answer`` with the passage it was drawn from, if one bears it out."""
    if not isinstance(answer, str) or not answer.strip():
        raise InvalidInput("an answer must have something in it")
    if len(answer) > MAX_ANSWER_CHARS:
        raise InvalidInput(f"an answer of {len(answer)} characters is over the bound of {MAX_ANSWER_CHARS}; "
                           "refused rather than cut")
    given = tuple(passages)
    if len(given) > MAX_PASSAGES:
        raise InvalidInput(f"{len(given)} passages are over the bound of {MAX_PASSAGES}; refused rather than cut")
    if len({passage.id for passage in given}) != len(given):
        raise InvalidInput("each passage needs its own id")
    reads = [_read(passage) for passage in given]
    sentences = []
    for start, end in sentence_spans(answer):
        text = answer[start:end]
        words = [match.group().casefold() for match in _WORD.finditer(text)]
        tokens = set(tokenize(text))
        numbers = _numbers(text)
        if len(tokens) < 2:
            sentences.append(AttributedSentence(text, start, end, "too_short", None, None, 0.0, 0, ()))
            continue
        # A run is 0 unless it reaches QUOTE_WORDS, so when nothing is quoted
        # the passage holding most of the sentence's words ranks first.
        ranked = [(*_longest_run(words, read), len(tokens & read.tokens), -order, read)
                  for order, read in enumerate(reads)]
        run, at, shared, _, read = (max(ranked, key=lambda item: (item[0], item[2], item[3])) if ranked
                                    else (0, 0, 0, 0, None))
        coverage = shared / len(tokens)
        if read is not None and run >= QUOTE_WORDS:
            status: Status = "quoted"
            quote: Optional[tuple[int, int]] = (read.spans[at][0], read.spans[at + run - 1][1])
        elif read is not None and coverage >= OVERLAP_SHARE:
            status, quote = "overlapping", None
        else:
            status, quote = "unattributed", None
        held = set(_numbers(read.passage.text)) if read is not None and status != "unattributed" else set()
        missing = tuple(number for number in dict.fromkeys(numbers) if number not in held)
        passage = read.passage.id if read is not None and status != "unattributed" else None
        sentences.append(AttributedSentence(text, start, end, status, passage, quote,
                                            coverage if passage else 0.0, shared if passage else 0, missing))
    return Attribution(tuple(sentences), given)


async def attribute_to_chunks(engine: "MemoryEngine", space: str, answer: str,
                              chunk_ids: Sequence[int]) -> tuple[Attribution, tuple[int, ...], tuple[int, ...]]:
    """``answer`` attributed to the space's chunks named by id, the ids the
    space does not hold, and the ids it holds whose text is blank.

    The passages are read from the space, never taken from the caller, so an
    answer cannot be attributed to text the space does not hold. A blank
    chunk holds nothing to attribute to, and is named rather than dropped."""
    from ..memory.engine import check_space

    check_space(space)
    wanted = list(dict.fromkeys(chunk_ids))
    if not wanted:
        raise InvalidInput("name at least one chunk to attribute the answer to")
    if any(type(chunk_id) is not int or chunk_id < 1 for chunk_id in wanted):
        raise InvalidInput("chunk ids are whole numbers from 1")
    if len(wanted) > MAX_PASSAGES:
        raise InvalidInput(f"{len(wanted)} chunks are over the bound of {MAX_PASSAGES}; refused rather than cut")
    found = {chunk.chunk_id: chunk for chunk in await engine.documents.get_chunks(space, wanted)}
    passages = [Passage(f"chunk:{chunk_id}", found[chunk_id].text) for chunk_id in wanted
                if chunk_id in found and found[chunk_id].text.strip()]
    missing = tuple(chunk_id for chunk_id in wanted if chunk_id not in found)
    empty = tuple(chunk_id for chunk_id in wanted if chunk_id in found and not found[chunk_id].text.strip())
    return attribute_answer(answer, passages), missing, empty
