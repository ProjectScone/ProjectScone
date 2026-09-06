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
from dataclasses import dataclass, field
from typing import Iterator, Optional, Sequence

from .engine import MemoryEngine, check_space, normalise_term
from .errors import SconeError
from .llm import ChatModel
from .models import Episode, Fact
from .timeutil import parse_rfc3339

EXTRACTION_PROMPT = """\
You turn a piece of someone's memory into durable facts.

Read the text and list the facts it states about named things: people, \
places, projects, tools, organisations, dates. Express each fact as a \
triple with a confidence.

Reply with a JSON array and nothing else. Each element is an object with \
exactly these keys:
- "subject": the entity the fact is about, named as the text names it
- "predicate": a short lowercase verb phrase in snake_case, such as \
lives_in, works_at, prefers, was_born_on
- "object": the value, in the text's own words
- "confidence": a number from 0 to 1; 1.0 for a fact stated outright, \
lower for one that is only implied

Rules:
- Record only what the text states or clearly implies. Do not guess, do \
not add world knowledge, do not fill gaps.
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
    return Extracted(subject, predicate, obj, _confidence(entry.get("confidence")))


def _clean(value: object) -> str:
    return " ".join(value.split()) if isinstance(value, str) else ""


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
    ) -> None:
        self.engine = engine
        self.chat = chat
        self.max_attempts = max_attempts
        self.prompt = prompt
        #: Extractions enter the ledger as proposals for a person to review
        #: unless their confidence reaches ``accept_at``. None means every
        #: extraction is proposed: a model's reading is never presented as
        #: an established fact by default.
        self.accept_at = accept_at
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
        triples = await self._extract(text)
        outcome = await self._apply(space, None, triples, created_at)
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
        triples = await self._extract(episode.content)
        return await self._apply(episode.space, episode.episode_id, triples, episode.created_at)

    async def _extract(self, text: str) -> list[Extracted]:
        reply = await self.chat.complete(self.prompt, text)
        return parse_triples(reply)

    async def _apply(
        self,
        space: str,
        episode_id: Optional[int],
        triples: Sequence[Extracted],
        valid_from: Optional[str],
    ) -> DistillOutcome:
        outcome = DistillOutcome(episode_id)
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
                proposed=self.accept_at is None or triple.confidence < self.accept_at,
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


__all__ = [
    "DEFAULT_CONFIDENCE",
    "EXTRACTION_PROMPT",
    "DistillError",
    "DistillOutcome",
    "Distiller",
    "Extracted",
    "parse_triples",
]
