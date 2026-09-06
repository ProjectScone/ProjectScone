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

from .distill import Extracted, _clause_around, _grounding_reason, _is_non_asserted
from .errors import NotFound

#: Verdicts that mean the source cannot support the claim.
FLAGGED = frozenset({"object_not_in_source", "object_only_in_non_asserted_context"})


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
    for status in statuses:
        for fact in await engine.facts(space, status=status):
            if fact.origin != "extracted":
                continue
            findings.append(await _judge(engine, space, fact))
    return findings


async def _judge(engine, space: str, fact) -> Finding:
    if fact.source_episode_id is None:
        return _finding(fact, "no_source")
    try:
        source = (await engine.episode(space, fact.source_episode_id)).content
    except NotFound:
        return _finding(fact, "source_missing")
    if fact.quote is not None:
        reason = _grounding_reason(
            Extracted(
                subject=fact.subject, predicate=fact.predicate, object=fact.object,
                confidence=fact.confidence, quote=fact.quote, statement_type="observation",
            ),
            source,
        )
        return _finding(fact, "grounded" if reason is None else reason)
    spans = list(_spans(source, fact.object))
    if not spans:
        return _finding(fact, "object_not_in_source")
    clauses = [_clause_around(source, start, end) for start, end in spans]
    denials = [clause for clause in clauses if _is_non_asserted(clause)]
    if len(denials) == len(clauses):
        return _finding(fact, "object_only_in_non_asserted_context", evidence=denials[0])
    return _finding(fact, "unverifiable_without_a_quote")


def _finding(fact, verdict: str, evidence: Optional[str] = None) -> Finding:
    return Finding(
        fact_id=fact.fact_id, subject=fact.subject, predicate=fact.predicate, object=fact.object,
        status=fact.status, source_episode_id=fact.source_episode_id,
        verdict=verdict, flagged=verdict in FLAGGED or verdict.startswith("quote_") or verdict.endswith("_in_quote"),
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
