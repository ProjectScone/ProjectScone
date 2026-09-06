"""Re-run the distiller's source checks over facts already in a ledger.

The checks arrived after the first extractions did, so a store can hold
claims that would not be accepted today: the live store has
``claude code / is_installed / nothing``, taken from a sentence that says
an authenticated API "says nothing about whether the Claude Code or Codex
hooks are actually installed".

This reads. It decides nothing and writes nothing: a flagged claim is a
claim whose own source cannot support it, which is a question for a
person, and the repair path is `exclude` with a reason, which is
reversible and leaves the record and its history in place.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Iterator, Optional, Sequence

from .errors import NotFound

# The checks below mirror the ones the distiller applies to a new
# extraction. They are stated here rather than imported because an audit
# that cannot run without the extractor's current patch is an audit that
# cannot run; when the distiller's source grounding lands, the two should
# share one definition of a non-asserted clause.
_NON_ASSERTED = re.compile(
    r"\b(?:if|unless|assuming|suppose|supposing|whether|might|may|could|would|"
    r"should|perhaps|maybe|possibly|potentially|unclear|unknown|not|never|no|"
    r"nothing|neither|unlikely|likely|doubt|doubts|doubted|plans?|planned|planning|"
    r"intends?|intended|intending|intents?|intentions?)\b|"
    r"\b(?:aren|can|couldn|didn|doesn|don|hadn|hasn|haven|isn|mightn|mustn|"
    r"needn|shan|shouldn|wasn|weren|won|wouldn)['\u2019]t\b|\bcannot\b|"
    r"\bevidence\s+to\s+verify\b|\bto\s+verify\b|"
    r"\bfollowed\s+by\s+confirmation\b",
    re.IGNORECASE,
)
_IMPERATIVE = re.compile(
    r"^\s*(?:[-*]\s*)?(?:please\s+)?(?:verify|confirm|check|ensure|use|run|"
    r"install|create|open|record|remember|follow|test)\b",
    re.IGNORECASE,
)

#: Verdicts that mean the source cannot support the claim.
FLAGGED = frozenset({
    "object_not_in_source",
    "object_only_in_non_asserted_context",
    "quote_not_in_source",
    "object_not_in_quote",
    "quote_context_not_asserted",
})


@dataclass(frozen=True)
class Finding:
    """One audited fact and what its own source says about it."""

    fact_id: int
    subject: str
    predicate: str
    object: str
    status: str
    source_episode_id: Optional[int]
    verdict: str
    flagged: bool
    quote: Optional[str] = None
    #: The clause that decided a flagged verdict, so a person can read the
    #: evidence without opening the episode.
    evidence: Optional[str] = None


async def audit_grounding(engine, space: str, statuses: Sequence[str] = ("active",)) -> list[Finding]:
    """Every extracted fact in ``space`` with one of ``statuses``, judged
    against the text it came from. Facts a person stated are left out:
    grounding is a question about extraction."""
    findings: list[Finding] = []
    sources: dict[int, Optional[str]] = {}
    for status in statuses:
        for fact in await engine.facts(space, status=status):
            if fact.origin != "extracted":
                continue
            findings.append(await _judge(engine, space, fact, sources))
    return findings


async def _judge(engine, space: str, fact, sources: dict) -> Finding:
    """One fact against its source. ``sources`` holds the episodes already
    read: one episode commonly backs many claims, and a ledger of hundreds
    would otherwise read the same text hundreds of times."""
    if fact.source_episode_id is None:
        return _finding(fact, "no_source")
    if fact.source_episode_id not in sources:
        try:
            sources[fact.source_episode_id] = (await engine.episode(space, fact.source_episode_id)).content
        except NotFound:
            sources[fact.source_episode_id] = None
    source = sources[fact.source_episode_id]
    if source is None:
        return _finding(fact, "source_missing")
    if fact.quote is not None:
        return _judge_quote(fact, source)
    spans = list(_spans(source, fact.object))
    if not spans:
        return _finding(fact, "object_not_in_source")
    clauses = [_clause_around(source, start, end) for start, end in spans]
    denials = [clause for clause in clauses if _is_non_asserted(clause)]
    if len(denials) == len(clauses):
        return _finding(fact, "object_only_in_non_asserted_context", evidence=denials[0])
    return _finding(fact, "unverifiable_without_a_quote")


def _judge_quote(fact, source: str) -> Finding:
    """A claim that stored its evidence is judged on that evidence: the
    quote has to be in the source, the object has to be in the quote, and
    the clause the quote sits in has to be asserting something."""
    quote_at = next(_spans(source, fact.quote), None)
    if quote_at is None:
        return _finding(fact, "quote_not_in_source")
    if next(_spans(fact.quote, fact.object), None) is None:
        return _finding(fact, "object_not_in_quote")
    clause = _clause_around(source, *quote_at)
    if _is_non_asserted(clause):
        return _finding(fact, "quote_context_not_asserted", evidence=clause)
    return _finding(fact, "grounded")


def _clause_around(source: str, start: int, end: int) -> str:
    left = max(source.rfind(mark, 0, start) for mark in ".?!;\n") + 1
    boundaries = [position for mark in ".?!;\n" if (position := source.find(mark, end)) >= 0]
    right = min(boundaries) + 1 if boundaries else len(source)
    return source[left:right]


def _is_non_asserted(clause: str) -> bool:
    return bool(
        _NON_ASSERTED.search(clause) or _IMPERATIVE.search(clause) or clause.rstrip().endswith("?")
    )


def _finding(fact, verdict: str, evidence: Optional[str] = None) -> Finding:
    return Finding(
        fact_id=fact.fact_id, subject=fact.subject, predicate=fact.predicate, object=fact.object,
        status=fact.status, source_episode_id=fact.source_episode_id,
        verdict=verdict, flagged=verdict in FLAGGED,
        quote=fact.quote, evidence=evidence,
    )


def _spans(source: str, value: str) -> Iterator[tuple[int, int]]:
    """Where ``value`` appears in ``source`` as whole words, ignoring case
    and how much whitespace separates the words."""
    words = value.split()
    if not words:
        return
    phrase = r"\s+".join(re.escape(word) for word in words)
    for found in re.finditer(rf"(?<!\w){phrase}(?!\w)", source, re.IGNORECASE):
        yield found.start(), found.end()


__all__ = ["Finding", "audit_grounding", "FLAGGED"]
