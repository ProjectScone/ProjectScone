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

# What the distiller refuses and what an audit may condemn are not the
# same list. The distiller refuses doubt as well as denial, because a
# false positive there costs one missed extraction. Applied to a ledger
# that already exists, that same list calls "every extraction would land
# as a proposal for Review" unsupported, when the source asserts it. So
# the audit knows only denial, and even then it condemns just the case
# with no other reading: the object IS the denial.
_DENIAL = re.compile(
    r"\b(?:no|not|never|nothing|none|neither|nor|cannot|without|unclear|unknown)\b|"
    r"\b(?:aren|can|couldn|didn|doesn|don|hadn|hasn|haven|isn|mightn|mustn|"
    r"needn|shan|shouldn|wasn|weren|won|wouldn)['\u2019]t\b",
    re.IGNORECASE,
)
#: An object made only of these is an extraction that swallowed the
#: negation it was reading, which is what put "is_installed / nothing" in
#: the live ledger.
_DENIAL_WORDS = frozenset({
    "no", "not", "never", "nothing", "none", "neither", "nor", "cannot",
    "unclear", "unknown", "false",
})

#: Verdicts that mean the source cannot support the claim.
FLAGGED = frozenset({
    "object_is_a_denial",
    "object_not_in_source",
    "quote_not_in_source",
    "object_not_in_quote",
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
        return _finding(fact, _absent_verdict(source, fact.object))
    clauses = [_clause_around(source, start, end) for start, end in spans]
    denials = [clause for clause in clauses if _DENIAL.search(clause)]
    if len(denials) < len(clauses):
        return _finding(fact, "unverifiable_without_a_quote")
    if _is_a_denial(fact.object):
        return _finding(fact, "object_is_a_denial", evidence=denials[0])
    # A real object inside a denied clause may or may not survive
    # reading. Reported with the clause, not condemned by it.
    return _finding(fact, "denial_in_the_same_clause", evidence=denials[0])


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
    if _DENIAL.search(clause) and _is_a_denial(fact.object):
        return _finding(fact, "object_is_a_denial", evidence=clause)
    return _finding(fact, "grounded")


#: Words that carry no evidence either way, so a claim made only of them
#: cannot be checked against a text by looking for them in it.
_EMPTY_WORDS = frozenset({
    "a", "an", "and", "are", "as", "at", "be", "been", "by", "did", "do", "does", "false",
    "for", "from", "had", "has", "have", "in", "is", "it", "its", "no", "none", "not", "of",
    "on", "or", "that", "the", "then", "there", "this", "to", "true", "was", "were", "with",
    "yes",
})
#: How much of an object's own vocabulary has to be missing from the
#: source before the claim is one a person should look at. Above this it
#: is a rewording, which the audit cannot settle and does not flag.
MIN_WORD_SHARE = 0.5


def _absent_verdict(source: str, object: str) -> str:
    """The object is not in the source as a phrase. That alone does not
    make it fabricated: the live store is full of claims that say their
    source in another word order, and calling those fabricated sends a
    person to reject good claims. So ask how much of the object's own
    vocabulary the source contains at all."""
    wanted = _content_words(object)
    if not wanted:
        return "unverifiable_without_a_quote"
    # One side is expanded, not both: expanding the object's words too
    # would count "fixtures" and "fixture" as two things to find.
    have = {form for word in _content_words(source) for form in _forms(word)}
    found = sum(1 for word in wanted if _forms(word) & have)
    return "object_not_in_source" if found / len(wanted) < MIN_WORD_SHARE else "unverifiable_without_a_quote"


def _content_words(text: str) -> set[str]:
    return {
        word for word in re.findall(r"[^\W_]+", text.casefold())
        if word not in _EMPTY_WORDS and len(word) > 1
    }


def _forms(word: str) -> set[str]:
    """A word and the stems a rewording is likely to reach for."""
    forms = {word}
    if len(word) > 3 and word.endswith("s"):
        forms.add(word[:-1])
    if len(word) > 4 and word.endswith("ed"):
        forms.update((word[:-2], word[:-1]))
    if len(word) > 5 and word.endswith("ing"):
        forms.update((word[:-3], word[:-3] + "e"))
    return forms


def _clause_around(source: str, start: int, end: int) -> str:
    left = max(source.rfind(mark, 0, start) for mark in ".?!;\n") + 1
    boundaries = [position for mark in ".?!;\n" if (position := source.find(mark, end)) >= 0]
    right = min(boundaries) + 1 if boundaries else len(source)
    return source[left:right]


def _is_a_denial(object: str) -> bool:
    words = set(re.findall(r"[^\W_]+", object.casefold()))
    return bool(words) and words <= _DENIAL_WORDS


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
