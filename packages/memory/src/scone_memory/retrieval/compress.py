"""Cut a widened passage back to the sentences that bear on the question.

A window of whole sentences around a hit (``window.widen``) holds the
answer more often than the hit alone and a good deal besides. This keeps
the sentences the passage was retrieved for, always, and at most a share
of the sentences widening added around them, chosen by how much of the
question each one names:

- **terms** (the default, no model): a sentence scores the weight of the
  question's words it names, each word weighted by how few of the
  passage's sentences name it, so a word the whole passage repeats
  counts for less than one it names once. A sentence naming none is
  never kept.
- **embedding**: a sentence scores its cosine to the question under the
  embedder given, all sentences in one call, up to
  ``MAX_EMBEDDED_SENTENCES``.

Between sentences of equal score the one nearer the hit is kept. Each run
of adjacent kept sentences is the episode's own bytes, listed in ``runs``
as offsets; runs that are not adjacent are joined by ``GAP`` so the text
never reads as one quote. Neither rule is measured: which share keeps the
answer is a bench question, and the record says so.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Literal, Optional, Sequence

from ..core.errors import InvalidInput
from ..core.models import RecallItem
from ..core.ports import Embedder
from .lexical import tokenize
from .window import byte_spans, sentence_spans

Scorer = Literal["terms", "embedding"]
SCORERS: tuple[Scorer, ...] = ("terms", "embedding")
#: Joins runs of kept sentences that were not adjacent in the episode.
GAP = " … "
#: Sentences one call may send to an embedder. Passages past it are left
#: whole and counted in ``unscored``, never cut on a partial score.
MAX_EMBEDDED_SENTENCES = 400
#: Words that say what form a question takes rather than what it is about.
#: The shared tokenizer keeps them, since a search for "how" is a search;
#: as a reason to keep a sentence they kept every "how many" in a passage.
_ASKING = frozenset({"how", "why", "whom", "whose"})
#: Counted as asking only straight after "how".
_AMOUNTS = frozenset({"many", "much"})


@dataclass(frozen=True)
class Compressed:
    """The passages after compression, and what was left out of them."""

    items: tuple[RecallItem, ...] = ()
    #: Passages given shorter text.
    compressed: int = 0
    sentences_dropped: int = 0
    bytes_before: int = 0
    bytes_after: int = 0
    #: Passages left whole because no retrieved span was found inside them.
    unpinned: int = 0
    #: Passages left whole because the embedding budget was spent first.
    unscored: int = 0
    sentences_embedded: int = 0
    #: Chunk id to the episode byte spans kept, one per run, in order.
    runs: dict[int, tuple[tuple[int, int], ...]] = field(default_factory=dict)
    scorer: Scorer = "terms"
    keep: float = 0.0
    why: str = ""

    def record(self) -> dict:
        return {"compressed": self.compressed, "sentences_dropped": self.sentences_dropped,
                "bytes_before": self.bytes_before, "bytes_after": self.bytes_after,
                "unpinned": self.unpinned, "unscored": self.unscored,
                "sentences_embedded": self.sentences_embedded,
                "runs": {str(chunk): [[start, end] for start, end in spans] for chunk, spans in self.runs.items()},
                "rules": {"scorer": self.scorer, "keep": self.keep, "gap": GAP, "measured": False},
                "why": self.why}


@dataclass
class _Plan:
    index: int
    item: RecallItem
    raw: bytes
    #: Sentence byte spans relative to the passage.
    sentences: list[tuple[int, int]]
    hits: set[int]
    #: The retrieved span relative to the passage.
    hit: tuple[int, int]
    scores: dict[int, float] = field(default_factory=dict)


def _check(query: object, keep: object, scorer: object, embedder: object) -> None:
    if not isinstance(query, str) or not query.strip():
        raise InvalidInput("compression scores sentences against a question; the question is blank")
    if isinstance(keep, bool) or not isinstance(keep, (int, float)) or not 0 <= keep <= 1:
        raise InvalidInput(f"keep is the share of sentences around a hit kept, from 0 to 1, not {keep!r}")
    if not isinstance(scorer, str) or scorer not in SCORERS:
        raise InvalidInput(f"sentences are scored by {' or '.join(SCORERS)}, not {scorer!r}")
    if scorer == "embedding" and embedder is None:
        raise InvalidInput("the embedding scorer needs an embedder")


def _plan(index: int, item: RecallItem, hit: RecallItem) -> Optional[_Plan]:
    raw = item.text.encode()
    start, end = hit.start - item.start, hit.end - item.start
    if len(raw) != item.end - item.start or start < 0 or end > len(raw) or start >= end:
        return None
    sentences = byte_spans(item.text, sentence_spans(item.text))
    hits = {number for number, (begin, stop) in enumerate(sentences) if begin < end and stop > start}
    if not hits and sentences:
        # The span is only the space between sentences: the sentence after it holds it.
        hits = {next((number for number, (begin, _) in enumerate(sentences) if begin >= end), len(sentences) - 1)}
    return _Plan(index, item, raw, sentences, hits, (start, end))


def _term_scores(plan: _Plan, question: set[str]) -> None:
    named = [set(tokenize(plan.raw[begin:stop].decode())) & question for begin, stop in plan.sentences]
    count = len(named)
    weight = {term: math.log(1 + count / sum(term in words for words in named))
              for term in set().union(*named)}
    for number, words in enumerate(named):
        if number not in plan.hits:
            plan.scores[number] = sum(weight[term] for term in words)


def _question_words(query: str) -> set[str]:
    words = tokenize(query)
    return {word for number, word in enumerate(words)
            if word not in _ASKING and not (word in _AMOUNTS and number and words[number - 1] == "how")}


def _cosine(left: Sequence[float], right: Sequence[float]) -> float:
    norm = math.sqrt(sum(x * x for x in left)) * math.sqrt(sum(y * y for y in right))
    return sum(x * y for x, y in zip(left, right)) / norm if norm else 0.0


def _cut(plan: _Plan, keep: float, scorer: Scorer) -> tuple[RecallItem, tuple[tuple[int, int], ...], int]:
    around = [number for number in range(len(plan.sentences)) if number not in plan.hits]
    ranked = sorted((number for number in around if scorer != "terms" or plan.scores[number] > 0),
                    key=lambda number: (-plan.scores[number], min(abs(number - h) for h in plan.hits), number))
    kept = plan.hits | set(ranked[:math.floor(keep * len(around) + 1e-9)])
    runs: list[list[int]] = []
    for number in sorted(kept):
        begin, stop = plan.sentences[number]
        if number in plan.hits:
            begin, stop = min(begin, plan.hit[0]), max(stop, plan.hit[1])
        if runs and number - 1 in kept:
            runs[-1][1] = max(runs[-1][1], stop)
        else:
            runs.append([begin, stop])
    text = GAP.join(plan.raw[begin:stop].decode() for begin, stop in runs)
    offset = plan.item.start
    spans = tuple((offset + begin, offset + stop) for begin, stop in runs)
    shorter = plan.item.model_copy(update={"text": text, "start": spans[0][0], "end": spans[-1][1],
                                           "first_line": None, "last_line": None})
    return shorter, spans, len(plan.sentences) - len(kept)


async def compress(items: Sequence[RecallItem], query: str, *, hits: Sequence[RecallItem], keep: float = 0.5,
                   scorer: Scorer = "terms", embedder: Optional[Embedder] = None) -> Compressed:
    """``items`` with the sentences around each retrieved span cut to at most
    ``keep`` of them, by ``scorer``. ``hits`` are the items as retrieved,
    before widening, matched by chunk id: their spans are never dropped."""
    _check(query, keep, scorer, embedder)
    retrieved = {hit.chunk_id: hit for hit in hits}
    out = list(items)
    before = sum(len(item.text.encode()) for item in items)
    plans: list[_Plan] = []
    unpinned = unscored = embedded = 0
    reasons: list[str] = []
    for index, item in enumerate(items):
        hit = retrieved.get(item.chunk_id)
        plan = _plan(index, item, hit) if hit is not None else None
        if plan is None:
            unpinned += 1
        elif len(plan.hits) < len(plan.sentences):
            plans.append(plan)
    if unpinned:
        reasons.append(f"{unpinned} passage(s) held no retrieved span to keep and were left whole")
    question = _question_words(query)
    if scorer == "terms" and not question and plans:
        reasons.append("the question names no word the terms scorer can weigh, so nothing was compressed")
        plans = []
    if scorer == "terms":
        for plan in plans:
            _term_scores(plan, question)
    else:
        assert embedder is not None
        scored: list[_Plan] = []
        for plan in plans:
            around = len(plan.sentences) - len(plan.hits)
            if embedded + around > MAX_EMBEDDED_SENTENCES:
                unscored += 1
                continue
            embedded += around
            scored.append(plan)
        if unscored:
            reasons.append(f"{unscored} passage(s) were past the budget of {MAX_EMBEDDED_SENTENCES} embedded "
                           "sentences and were left whole")
        plans = scored
        texts = [plan.raw[begin:stop].decode() for plan in plans
                 for number, (begin, stop) in enumerate(plan.sentences) if number not in plan.hits]
        if texts:
            from ..core.embedding import QueryEmbedder, embed_queries
            if isinstance(embedder, QueryEmbedder):
                [asked] = await embed_queries(embedder, [query])
                rest = iter(await embedder.embed(texts))
            else:
                vectors = await embedder.embed([query, *texts])
                asked, rest = vectors[0], iter(vectors[1:])
            for plan in plans:
                for number in range(len(plan.sentences)):
                    if number not in plan.hits:
                        plan.scores[number] = _cosine(asked, next(rest))
    compressed = dropped = 0
    runs: dict[int, tuple[tuple[int, int], ...]] = {}
    for plan in plans:
        shorter, spans, gone = _cut(plan, float(keep), scorer)
        if gone:
            out[plan.index] = shorter
            runs[plan.item.chunk_id] = spans
            compressed += 1
            dropped += gone
    return Compressed(items=tuple(out), compressed=compressed, sentences_dropped=dropped, bytes_before=before,
                      bytes_after=sum(len(item.text.encode()) for item in out), unpinned=unpinned,
                      unscored=unscored, sentences_embedded=embedded, runs=runs, scorer=scorer,
                      keep=float(keep), why="; ".join(reasons))
