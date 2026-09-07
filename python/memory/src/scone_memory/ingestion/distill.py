"""Consolidation: a model reads episodes and proposes dated facts.

The model only proposes. The engine's ``assert_fact`` enforces the
ledger's rules: a fact is true from when the episode happened, not from
when it was read; a newer object for the same subject and predicate
closes the older fact and keeps it; a stale fact arriving late is
stored already closed; a restatement changes nothing.

A failure is a typed error, never a quiet skip. ``distill_pending``
records failures per episode on the ``Distiller`` instance; an episode
that fails ``max_attempts`` times is parked and no longer sent to the
model. That bookkeeping lives in memory, so a fresh ``Distiller``
retries everything.
"""

from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass, field
from typing import Iterator, Optional, Sequence

from ..memory.engine import MemoryEngine, check_space, normalise_term
from ..core.errors import SconeError
from ..providers.llm import ChatModel
from ..core.models import Episode, Fact
from ..core.timeutil import parse_rfc3339

EXTRACTION_PROMPT = """\
You turn a piece of someone's memory into durable facts.

Read the text and list only durable observations it directly states about \
named things: people, places, projects, tools, organisations, dates. Do not \
turn instructions, questions, hypotheticals, uncertainty, negation, or tests \
that still need to be performed into positive observations.

Reply with a JSON array and nothing else. Each element is an object with \
exactly these keys:
- "subject": the entity the fact is about, named as the text names it
- "predicate": a short lowercase verb phrase in snake_case, such as \
lives_in, works_at, prefers, was_born_on
- "object": the value, in the text's own words
- "quote": an exact, contiguous substring of the text that directly supports \
the whole proposed triple and contains both the subject and object
- "statement_type": one of observation, instruction, hypothetical, question, \
or uncertain
- "confidence": a number from 0 to 1; 1.0 for a fact stated outright, \
lower when wording is less direct. Confidence never substitutes for evidence.

Rules:
- Record only direct observations. Do not guess, infer implied claims, add \
world knowledge, or fill gaps.
- A quote merely containing words from the triple is not enough. The quote \
must support the relationship expressed by the subject, predicate, and object.
- Use predicate wording whose meaningful action or status words occur in the \
quote. Do not replace them with a synonym or an inferred opposite.
- Classify instructions, plans, recommendations, conditions, possibilities, \
questions, uncertainty, and negated statements with their non-observation \
statement_type. They will not become proposals.
- Prefer facts that stay true for a while (where someone lives, what \
they use, what they decided) over passing events.
- Use the same subject spelling and predicate wording for the same kind \
of fact, so a repeated fact matches the earlier one.
- Do not record questions, opinions about the text, or facts about the \
text itself.
- If the text states no facts, reply with [].
- No prose, no explanation, no code fences: the JSON array only.\
"""

#: Confidence assumed when the model leaves it out or sends junk.
DEFAULT_CONFIDENCE = 0.5
#: Mirrors the fact persistence boundary. Longer evidence cannot be stored,
#: so the distiller withholds it instead of creating an ungrounded proposal.
MAX_QUOTE = 2000


class DistillError(SconeError):
    """The model's reply could not be used, or a batch had failures.

    ``failed`` counts the episodes that failed in the batch; ``outcomes``
    carries the per-episode results the batch produced before raising,
    so nothing the batch learned is lost with the error.
    """

    def __init__(self, message: str, failed: int = 1, outcomes: Sequence["DistillOutcome"] = ()) -> None:
        super().__init__(message)
        self.failed = failed
        self.outcomes = list(outcomes)


@dataclass(frozen=True)
class Extracted:
    subject: str
    predicate: str
    object: str
    confidence: float
    #: Optional only so the public legacy parser keeps accepting its old
    #: four-field shape. The Distiller requires both fields by default.
    quote: Optional[str] = None
    statement_type: Optional[str] = None


@dataclass(frozen=True)
class RejectedExtraction:
    """A model candidate withheld before it can touch the fact store."""

    extraction: Optional[Extracted]
    reason: str
    #: Original array entry when it could not be parsed as an extraction.
    raw: object = None


@dataclass
class DistillOutcome:
    #: None for text distilled without an episode.
    episode_id: Optional[int]
    #: Facts newly recorded, a late stale one (stored closed) included.
    added: list[Fact] = field(default_factory=list)
    #: Pre-existing active facts closed by what this text asserted.
    closed: int = 0
    #: Triples that restated a fact already on record.
    skipped: int = 0
    error: Optional[str] = None
    #: Candidates withheld by the source-grounding gate. These are audit
    #: results, not facts and not edits to previously approved history.
    rejected: list[RejectedExtraction] = field(default_factory=list)

    @property
    def failed(self) -> bool:
        return self.error is not None


@dataclass
class _Attempts:
    count: int = 0
    last_error: str = ""


# -- parsing ------------------------------------------------------------------


def parse_triples(text: str) -> list[Extracted]:
    """The model's reply as triples.

    Accepts a bare JSON array, one inside code fences, or one with prose
    around it. Entries that are not objects with string subject,
    predicate and object are dropped; confidence is clamped to 0..1 and
    whitespace inside every string is collapsed. Raises ``DistillError``
    when no JSON array can be found at all.
    """
    array = _find_array(text)
    if array is None:
        raise DistillError(f"model did not return a JSON array; got: {text.strip()[:120]!r}")
    triples = (_triple(entry) for entry in array)
    return [t for t in triples if t is not None]


def _find_array(text: str) -> Optional[list]:
    try:
        whole = json.loads(text)
    except ValueError:
        whole = None
    if isinstance(whole, list):
        return whole
    decoder = json.JSONDecoder()
    for start in _bracket_positions(text):
        try:
            value, _ = decoder.raw_decode(text, start)
        except ValueError:
            continue
        if isinstance(value, list):
            return value
    return None


def _bracket_positions(text: str) -> Iterator[int]:
    start = text.find("[")
    while start != -1:
        yield start
        start = text.find("[", start + 1)


def _triple(entry: object) -> Optional[Extracted]:
    if not isinstance(entry, dict):
        return None
    subject = _clean(entry.get("subject"))
    predicate = _clean(entry.get("predicate"))
    obj = _clean(entry.get("object"))
    if not (subject and predicate and obj):
        return None
    return Extracted(
        subject,
        predicate,
        obj,
        _confidence(entry.get("confidence")),
        quote=_literal(entry.get("quote")),
        statement_type=_clean(entry.get("statement_type")) or None,
    )


def _clean(value: object) -> str:
    return " ".join(value.split()) if isinstance(value, str) else ""


def _literal(value: object) -> Optional[str]:
    """Keep internal whitespace exact because quotes are source spans."""
    if not isinstance(value, str) or not value.strip():
        return None
    return value


def _confidence(value: object) -> float:
    # bool is an int in Python; "true" is not a confidence.
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        return DEFAULT_CONFIDENCE
    return min(1.0, max(0.0, float(value)))


# -- the distiller ------------------------------------------------------------


class Distiller:
    def __init__(
        self,
        engine: MemoryEngine,
        chat: ChatModel,
        max_attempts: int = 3,
        prompt: str = EXTRACTION_PROMPT,
        accept_at: Optional[float] = None,
        require_grounding: bool = True,
    ) -> None:
        self.engine = engine
        self.chat = chat
        self.max_attempts = max_attempts
        self.prompt = prompt
        #: Grounded extractions always enter as proposals. ``accept_at`` is
        #: retained only for the explicit ``require_grounding=False`` legacy
        #: path; there, None still means every extraction is proposed.
        self.accept_at = accept_at
        #: False is an explicit compatibility escape hatch for callers with
        #: pre-grounding canned responses. It preserves the old parser/apply
        #: contract and may auto-accept via ``accept_at``. New extraction must
        #: leave this at True.
        self.require_grounding = require_grounding
        # Both keyed by (space, episode_id); in memory only, so a new
        # Distiller starts with no history and retries parked episodes.
        self._failures: dict[tuple[str, int], _Attempts] = {}
        self._done: set[tuple[str, int]] = set()

    def parked(self, space: str) -> dict[int, str]:
        """Episodes this instance has given up on, with their last error."""
        return {
            episode_id: attempts.last_error
            for (owner, episode_id), attempts in self._failures.items()
            if owner == space and attempts.count >= self.max_attempts
        }

    async def distill_text(self, space: str, text: str, created_at: Optional[str] = None) -> list[Fact]:
        """Facts from text that is not stored as an episode. They date
        from ``created_at`` (the engine's clock when omitted) and carry
        no provenance."""
        check_space(space)
        triples, rejected = await self._extract(text)
        if self.require_grounding and triples:
            raise DistillError(
                "source-grounded claims require a stored episode so their quote can be persisted"
            )
        outcome = await self._apply(space, None, triples, created_at, rejected)
        return outcome.added

    async def distill_episode(self, space: str, episode_id: int) -> DistillOutcome:
        """Read one episode through the model and assert what it states.
        Raises ``ChatError`` or ``DistillError`` on failure."""
        episode = await self.engine.episode(space, episode_id)
        return await self._distill(episode)

    async def distill_pending(self, space: str, limit: int = 50) -> list[DistillOutcome]:
        """Distil up to ``limit`` episodes no fact refers to, oldest first.

        An episode whose facts were all restatements, or that stated no
        facts, is remembered as done on this instance so it is not sent
        again. A failure counts against the episode; at ``max_attempts``
        it parks and is reported as failed without another model call.
        When any episode fails in this batch a ``DistillError`` carrying
        every outcome is raised after the batch completes.
        """
        check_space(space)
        outcomes: list[DistillOutcome] = []
        errors: list[str] = []
        processed = 0
        for episode in await self._pending(space):
            attempts = self._failures.get((space, episode.episode_id))
            if attempts is not None and attempts.count >= self.max_attempts:
                outcomes.append(DistillOutcome(episode.episode_id, error=f"parked: {attempts.last_error}"))
                continue
            if processed >= limit:
                break
            processed += 1
            outcome = await self._attempt(episode)
            outcomes.append(outcome)
            if outcome.error is not None:
                errors.append(f"episode {episode.episode_id}: {outcome.error}")
        if errors:
            raise DistillError(
                f"{len(errors)} of {processed} episodes failed distillation in {space!r}: " + "; ".join(errors),
                failed=len(errors),
                outcomes=outcomes,
            )
        return outcomes

    # -- internals --------------------------------------------------------

    async def _pending(self, space: str) -> list[Episode]:
        documents = self.engine.documents
        counts = await documents.counts(space)
        episodes = await documents.recent_episodes(space, max(counts.episodes, 1))
        referenced = {f.source_episode_id for f in await documents.list_facts(space, include_closed=True)}
        fresh = [
            e for e in episodes if e.episode_id not in referenced and (space, e.episode_id) not in self._done
        ]
        return sorted(fresh, key=lambda e: (parse_rfc3339(e.created_at), e.episode_id))

    async def _attempt(self, episode: Episode) -> DistillOutcome:
        key = (episode.space, episode.episode_id)
        try:
            outcome = await self._distill(episode)
        except SconeError as e:
            attempts = self._failures.setdefault(key, _Attempts())
            attempts.count += 1
            attempts.last_error = f"{type(e).__name__}: {e}"
            return DistillOutcome(episode.episode_id, error=attempts.last_error)
        self._failures.pop(key, None)
        self._done.add(key)
        return outcome

    async def _distill(self, episode: Episode) -> DistillOutcome:
        triples, rejected = await self._extract(episode.content)
        return await self._apply(
            episode.space, episode.episode_id, triples, episode.created_at, rejected
        )

    async def _extract(self, text: str) -> tuple[list[Extracted], list[RejectedExtraction]]:
        reply = await self.chat.complete(self.prompt, text)
        if not self.require_grounding:
            return parse_triples(reply), []
        triples, malformed = _strict_triples(reply)
        grounded, rejected = _grounded(triples, text)
        return grounded, [*malformed, *rejected]

    async def _apply(
        self,
        space: str,
        episode_id: Optional[int],
        triples: Sequence[Extracted],
        valid_from: Optional[str],
        rejected: Sequence[RejectedExtraction] = (),
    ) -> DistillOutcome:
        outcome = DistillOutcome(episode_id, rejected=list(rejected))
        seen: set[int] = set()
        for triple in triples:
            before = await self._active_ids(space, triple.subject, triple.predicate)
            fact = await self.engine.assert_fact(
                space,
                triple.subject,
                triple.predicate,
                triple.object,
                valid_from=valid_from,
                confidence=triple.confidence,
                source_episode_id=episode_id,
                origin="extracted",
                quote=triple.quote if self.require_grounding else None,
                # Source-grounded model readings remain proposals regardless
                # of confidence. A person may approve them later; extraction
                # never rewrites the accepted ledger in this mode.
                proposed=self.require_grounding
                or self.accept_at is None
                or triple.confidence < self.accept_at,
            )
            after = await self._active_ids(space, triple.subject, triple.predicate)
            outcome.closed += len(before - after)
            if fact.fact_id in before or fact.fact_id in seen:
                outcome.skipped += 1
                continue
            seen.add(fact.fact_id)
            outcome.added.append(fact)
        return outcome

    async def _active_ids(self, space: str, subject: str, predicate: str) -> set[int]:
        # facts_for is keyed on the engine's normalised terms.
        found = await self.engine.documents.facts_for(
            space, normalise_term(subject, "subject"), normalise_term(predicate, "predicate")
        )
        return {f.fact_id for f in found if f.status == "active"}


# -- source grounding ---------------------------------------------------------


_NON_ASSERTED = re.compile(
    r"\b(?:if|unless|assuming|suppose|supposing|whether|might|may|could|would|"
    r"should|perhaps|maybe|possibly|potentially|unclear|unknown|not|never|no|"
    r"nothing|neither|unlikely|likely|doubt|doubts|doubted|plans?|planned|planning|"
    r"intends?|intended|intending|intents?|intentions?)\b|"
    r"\b(?:aren|can|couldn|didn|doesn|don|hadn|hasn|haven|isn|mightn|mustn|"
    r"needn|shan|shouldn|wasn|weren|won|wouldn)['’]t\b|\bcannot\b|"
    r"\bevidence\s+to\s+verify\b|\bto\s+verify\b|"
    r"\bfollowed\s+by\s+confirmation\b",
    re.IGNORECASE,
)
_IMPERATIVE = re.compile(
    r"^\s*(?:[-*]\s*)?(?:please\s+)?(?:verify|confirm|check|ensure|use|run|"
    r"install|create|open|record|remember|follow|test)\b",
    re.IGNORECASE,
)
_PREDICATE_GLUE = {
    "a",
    "an",
    "are",
    "at",
    "be",
    "been",
    "by",
    "did",
    "do",
    "does",
    "for",
    "from",
    "had",
    "has",
    "have",
    "in",
    "is",
    "of",
    "on",
    "the",
    "through",
    "to",
    "was",
    "were",
    "with",
}


def _strict_triples(text: str) -> tuple[list[Extracted], list[RejectedExtraction]]:
    """Parse every array entry for the strict path without silent loss.

    ``parse_triples`` intentionally retains its legacy behavior of dropping
    malformed entries. Source-grounded distillation instead reports one
    rejection for each entry that cannot form a complete triple.
    """
    array = _find_array(text)
    if array is None:
        parse_triples(text)  # raises the legacy typed error with its message
        raise AssertionError("parse_triples accepted a missing array")  # pragma: no cover
    triples: list[Extracted] = []
    rejected: list[RejectedExtraction] = []
    for entry in array:
        triple = _triple(entry)
        if triple is None:
            rejected.append(RejectedExtraction(None, "malformed_candidate", raw=entry))
        else:
            triples.append(triple)
    return triples, rejected


def _grounded(
    triples: Sequence[Extracted], source: str
) -> tuple[list[Extracted], list[RejectedExtraction]]:
    """Apply necessary, deliberately conservative source checks.

    These checks do not claim to prove semantic entailment. They establish a
    literal span, require the named endpoints of the relationship in that
    span, and reject common non-asserted contexts. The surviving candidate is
    still only a proposal for human review.
    """
    reasons: list[Optional[str]] = [_grounding_reason(t, source) for t in triples]
    groups: dict[tuple[str, str], list[int]] = {}
    for index, (triple, reason) in enumerate(zip(triples, reasons)):
        if reason is None:
            key = (_clean(triple.subject).casefold(), _clean(triple.predicate).casefold())
            groups.setdefault(key, []).append(index)
    for indexes in groups.values():
        objects = {_clean(triples[index].object).casefold() for index in indexes}
        if len(objects) > 1:
            for index in indexes:
                reasons[index] = "same_source_conflict"
    accepted = [triple for triple, reason in zip(triples, reasons) if reason is None]
    rejected = [
        RejectedExtraction(triple, reason)
        for triple, reason in zip(triples, reasons)
        if reason is not None
    ]
    return accepted, rejected


def _grounding_reason(triple: Extracted, source: str) -> Optional[str]:
    quote = triple.quote
    if quote is None:
        return "missing_quote"
    if len(quote) > MAX_QUOTE:
        return "quote_too_long"
    starts = list(_occurrences(source, quote))
    if not starts:
        return "quote_not_in_source"
    if triple.statement_type != "observation":
        return "not_an_observation"
    if not _mentions(quote, triple.subject):
        return "subject_not_in_quote"
    if not _mentions(quote, triple.object):
        return "object_not_in_quote"
    contexts = [_clause_around(source, start, start + len(quote)) for start in starts]
    # The extraction format carries a quote, not a byte offset. If the same
    # quote appears in both asserted and non-asserted contexts, accepting the
    # first occurrence would invent certainty the model did not provide.
    if any(_is_non_asserted(context) for context in contexts):
        return "context_not_asserted"
    if not _predicate_mentioned(quote, triple.predicate):
        return "predicate_not_in_quote"
    return None


def _mentions(quote: str, value: str) -> bool:
    words = value.split()
    if not words:
        return False
    phrase = r"\s+".join(re.escape(word) for word in words)
    return re.search(rf"(?<!\w){phrase}(?!\w)", quote, re.IGNORECASE) is not None


def _predicate_mentioned(quote: str, predicate: str) -> bool:
    terms = [
        term
        for term in re.findall(r"[^\W_]+", predicate.casefold())
        if term not in _PREDICATE_GLUE
    ]
    quote_words = re.findall(r"[^\W_]+", quote.casefold())
    quote_forms = {form for word in quote_words for form in _word_forms(word)}
    return bool(terms) and all(_word_forms(term) & quote_forms for term in terms)


def _word_forms(word: str) -> set[str]:
    forms = {word}
    if len(word) > 3 and word.endswith("s"):
        forms.add(word[:-1])
    if len(word) > 4 and word.endswith("ed"):
        forms.update((word[:-2], word[:-1]))
    if len(word) > 5 and word.endswith("ing"):
        forms.update((word[:-3], word[:-3] + "e"))
    return forms


def _occurrences(source: str, quote: str) -> Iterator[int]:
    start = source.find(quote)
    while start >= 0:
        yield start
        start = source.find(quote, start + 1)


def _clause_around(source: str, start: int, end: int) -> str:
    left = max(source.rfind(mark, 0, start) for mark in ".?!;\n") + 1
    boundaries = [position for mark in ".?!;\n" if (position := source.find(mark, end)) >= 0]
    right = min(boundaries) + 1 if boundaries else len(source)
    return source[left:right]


def _is_non_asserted(context: str) -> bool:
    return bool(
        _NON_ASSERTED.search(context)
        or _IMPERATIVE.search(context)
        or context.rstrip().endswith("?")
    )


__all__ = [
    "DEFAULT_CONFIDENCE",
    "EXTRACTION_PROMPT",
    "DistillError",
    "DistillOutcome",
    "Distiller",
    "Extracted",
    "RejectedExtraction",
    "parse_triples",
]
