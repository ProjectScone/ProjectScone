"""Split a question that asks more than one thing, without asking a model.

A question with two halves searched as one query gets one blend of
passages, and the half with the rarer words tends to lose. The leading
frameworks solve this by having a model write sub-questions: a paid call
per question, and nothing a person can read when it comes out wrong.
Here the rule is written down, and every decomposition says what it did
and why.

**Nothing is paraphrased.** A part is a verbatim span of the question,
carrying its own offsets, so a receipt can quote exactly what was
searched.

The rule is deliberately shy, because the risk is one-sided. A question
wrongly left whole retrieves what it would have retrieved anyway; a
question wrongly split is searched as two queries that mean nothing
("where can I buy salt", "pepper") and the answer is worse than before.
So:

- A sentence-final ``?`` always ends a part: two questions in one message
  are two questions, and that needs no judgment.
- A ``;`` splits when **both sides stand up on their own** — each side
  names something of its own and carries an asking word.
- An "and" needs more than that, because "and" joins lists far more often
  than it joins questions. It splits only when the conjunction is
  **punctuated as a clause break** (a comma before it) or the right side
  **opens with its own interrogative**. "…about billing, and who was at
  the meeting" qualifies twice over; "of jogging and yoga did I do last
  week" qualifies neither way and is left alone.
- **Nothing inside quotation marks is a boundary.** A quoted span is
  verbatim text, so its punctuation belongs to the thing being named. A
  real question asks about the paper "To Adapt or Not to Adapt?
  Real-Time Adaptation for Semantic Segmentation", and its question mark
  ended a part until this rule stopped it.
- An "and" inside **"between … and …"** never splits. That construction
  *is* the question — two dates and the subtraction between them — and
  splitting it throws the subtraction away.

The last three rules were not guesses. An earlier version required only
an asking word on each side, and on real questions it split "how many
hours of jogging and yoga did I do last week" into "how many hours of
jogging" and "yoga did I do last week", because "did" satisfied the test.
The auxiliary was doing work the comma and the interrogative do properly.

This under-splits on purpose. "What did I decide about billing and about
the schedule?" does not split, because the second half borrows the first
half's verb and we cannot tell that from a list of nouns without a
parser. Whether splitting helps retrieval at all is a question for
measurement, not for this docstring.
"""

from __future__ import annotations

from dataclasses import dataclass
import re

from ..core.errors import InvalidInput
from .lexical import tokenize

#: Parts searched. A question asking more than this is answered from the
#: first few, and the decomposition says so rather than quietly dropping
#: the rest.
MAX_PARTS = 4
#: Splits attempted within one sentence, so a pathological message cannot
#: make this loop long.
MAX_SPLITS = 32

# A part ends after sentence-final punctuation followed by space, or after
# a full-width terminator (those scripts put no space after one).
_SENTENCE = re.compile(r"(?<=[?？؟])[\"'”’)\]」』]*\s+|(?<=[。！？])")
# Mid-sentence joins. The comma before a conjunction belongs to neither
# side, and whether it was there decides whether the join reads as a
# clause break or as a list.
_JOIN = re.compile(r"\s*;\s+|\s*,?\s+(?:and also|as well as|and|also|plus)\s+", re.IGNORECASE)
# "between X and Y" is one relation, so the "and" in it is not a join.
_SPANNING = re.compile(r"\bbetween\b", re.IGNORECASE)
# Quoted spans, whose punctuation is part of what is being named. A single
# quote only opens one away from a word character, so an apostrophe in
# "didn't" cannot start a quotation that swallows the rest of the line.
_QUOTED = re.compile(r"\"[^\"]{1,400}\"|“[^”]{1,400}”|(?<![^\W_])\'[^\']{1,400}\'(?![^\W_])"
                     r"|(?<![^\W_])‘[^’]{1,400}’(?![^\W_])", re.DOTALL)

# Every word, including the ones the lexical lane does not index. The
# asking test needs these: `tokenize` drops "who was" as stopwords, which
# is right for an index and useless for telling a question from a list.
_WORD = re.compile(r"[^\W\d_]+", re.UNICODE)
# An asking word. One on each side of a mid-sentence join is the weakest
# of the three tests: on its own it is not enough for a conjunction.
_ASKING = frozenset("""
what who whom whose which where why how when
do does did is are was were am be been have has had
can could will would shall should may might must
""".split())
# A question word, which starting a side is strong evidence that side is
# its own question rather than the tail of a list.
_INTERROGATIVE = frozenset("what who whom whose which where why how when".split())


@dataclass(frozen=True)
class Part:
    """One thing the question asks, quotable from the question."""

    text: str
    start: int
    end: int


@dataclass(frozen=True)
class Decomposition:
    """What the rule made of a question, and what it could not."""

    whole: str
    parts: tuple[Part, ...]
    #: Parts the rule found, which is more than ``parts`` when it found
    #: more than :data:`MAX_PARTS`.
    parts_found: int
    why: str

    @property
    def split(self) -> bool:
        return len(self.parts) > 1

    @property
    def capped(self) -> bool:
        return self.parts_found > len(self.parts)

    def queries(self) -> tuple[str, ...]:
        """The text to search, in order."""
        return tuple(part.text for part in self.parts)

    def record(self) -> dict[str, object]:
        return {"whole": self.whole, "split": self.split, "why": self.why,
                "parts_found": self.parts_found, "capped": self.capped,
                "parts": [{"text": p.text, "start": p.start, "end": p.end} for p in self.parts]}


def _names_something(text: str) -> bool:
    """Would the lexical lane index anything from this? A side that is all
    stopwords is not a question about anything."""
    return bool(tokenize(text))


def _stands_up(text: str) -> bool:
    """Could this side be asked on its own? It has to name something and
    it has to ask — a bare noun phrase does neither."""
    said = frozenset(word.lower() for word in _WORD.findall(text))
    return _names_something(text) and bool(said & _ASKING)


def _quoted(question: str) -> tuple[tuple[int, int], ...]:
    """Spans whose punctuation belongs to a name, not to the sentence."""
    return tuple((found.start(), found.end()) for found in _QUOTED.finditer(question))


def _within(spans: tuple[tuple[int, int], ...], start: int, end: int) -> bool:
    return any(span[0] < start and end <= span[1] for span in spans)


def _opens_asking(text: str) -> bool:
    found = _WORD.search(text)
    return found is not None and found.group().lower() in _INTERROGATIVE


def _reads_as_a_clause_break(joined: str, left: str, right: str) -> bool:
    """Is this join separating two questions, or joining a list?"""
    if ";" in joined:
        return True
    if _SPANNING.search(left):
        return False
    return "," in joined or _opens_asking(right)


def _trimmed(question: str, start: int, end: int) -> tuple[int, int]:
    while start < end and question[start].isspace():
        start += 1
    while end > start and question[end - 1].isspace():
        end -= 1
    return start, end


def _sentences(question: str, quoted: tuple[tuple[int, int], ...]) -> list[tuple[int, int]]:
    """Spans between sentence-final question marks, outside quotations."""
    spans: list[tuple[int, int]] = []
    at = 0
    for boundary in _SENTENCE.finditer(question):
        if _within(quoted, boundary.start(), boundary.end()):
            continue
        spans.append(_trimmed(question, at, boundary.end()))
        at = boundary.end()
    spans.append(_trimmed(question, at, len(question)))
    return [span for span in spans if span[1] > span[0]]


def _joined(question: str, span: tuple[int, int],
            quoted: tuple[tuple[int, int], ...]) -> list[tuple[int, int]]:
    """Split one sentence at joins whose both sides stand up."""
    found: list[tuple[int, int]] = []
    start, end = span
    for _ in range(MAX_SPLITS):
        cut = None
        for join in _JOIN.finditer(question, start, end):
            if _within(quoted, join.start(), join.end()):
                continue
            left = _trimmed(question, start, join.start())
            right = _trimmed(question, join.end(), end)
            said, asked = question[left[0]:left[1]], question[right[0]:right[1]]
            if left[1] > left[0] and right[1] > right[0] \
                    and _reads_as_a_clause_break(join.group(), said, asked) \
                    and _stands_up(said) and _stands_up(asked):
                cut = (left, right)
                break
        if cut is None:
            break
        found.append(cut[0])
        start, end = cut[1]
    found.append((start, end))
    return found


def decompose(question: str, *, limit: int = MAX_PARTS) -> Decomposition:
    """Read a question as the parts it asks, or as one thing if it asks one."""
    whole = question.strip()
    if not whole:
        raise InvalidInput("a question to decompose cannot be empty")
    if limit < 1:
        raise InvalidInput(f"at least one part has to be searched, not {limit}")

    quoted = _quoted(question)
    found: list[tuple[int, int]] = []
    for sentence in _sentences(question, quoted):
        found.extend(_joined(question, sentence, quoted))
    found = [span for span in found if _names_something(question[span[0]:span[1]])]

    if len(found) < 2:
        return Decomposition(whole=whole, parts=(Part(whole, *_trimmed(question, 0, len(question))),),
                             parts_found=1,
                             why="it asks one thing, so it was searched whole")
    kept = found[:limit]
    parts = tuple(Part(question[start:end], start, end) for start, end in kept)
    why = (f"it asks {len(found)} parts; only the first {len(kept)} are searched"
           if len(found) > len(kept)
           else f"it asks {len(found)} parts, and each is searched on its own")
    return Decomposition(whole=whole, parts=parts, parts_found=len(found), why=why)
