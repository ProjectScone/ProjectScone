"""What people said about a passage, as a small term on its fused score.

Lessons (``lessons.py``) put what people said beside a passage and leave the
order alone. This is the same record, read at query time and added to a
candidate's fused score the way recency is, when ``feedback_weight`` is set;
zero, the default, reads nothing and moves nothing.

Per candidate, the latest judgement of each question is weighed as a lesson
weighs one (+1 useful, -1 not, halving every ``half_life_days``). A question
is told apart by the hash of what its recall recorded, so asking one again
and judging again replaces the judgement rather than adding one. Four rules
sit on top:

- a useful judgement counts only once ``min_corroboration`` questions' do,
  so one question moves nothing, and an uncorroborated judgement offsets
  nothing. No identity is recorded: one caller who asks two questions, or
  one question spelled two ways, corroborates it;
- only the newest ``min_corroboration`` judgements each way count: those
  that pile up past what corroboration asks add nothing, so a popular
  passage (or one caller respelling a question) weighs what its newest two
  weigh, and ``held`` counts the candidates whose older ones were left out;
- a judgement against the passage outweighs every useful one older than it:
  those no longer count, and the passage must be corroborated again after it;
- a judgement counts only while the passage says what it said when it was
  judged. ``feedback`` records a fingerprint of the chunk's text and its
  episode's content; a candidate whose fingerprint now differs (a
  rebuilt store handing the same id to other text, to a span of an episode
  whose content changed, or to a span chunked differently) has those
  judgements dropped as ``stale``. One that cannot be checked is dropped as
  ``unverified`` -- recorded without a fingerprint or a question, or of a
  candidate whose episode cannot be read now -- since it is not trusted.

The term is ``feedback_weight`` times that score, cut at ``MAX_FEEDBACK_BOOST``
either way; ``capped`` counts the candidates it cut. The read takes the
newest ``lessons.MAX_FEEDBACK_EVENTS`` judgements of the last
``FEEDBACK_WINDOW_DAYS`` and says when that bound bit; the record names the
window, half-life and corroboration it read and folded with, as a lesson's does.

The term does not know the question. A passage judged useful rises for every
query it is a candidate for, so a recorded question hashed in the log (the
default) costs nothing here, and a question it was never judged for can
lose a place to it: ``benchmarks/feedback-replay-v1.results.md`` measures both.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field, replace
from datetime import timedelta
from typing import TYPE_CHECKING, Iterable, Mapping, Optional, Sequence

from ..core.errors import InvalidInput
from ..core.ports import Event
from ..core.timeutil import parse_rfc3339
from . import lessons
from .fusion import RRF_K

if TYPE_CHECKING:
    from ..core.models import Chunk
    from ..core.ports import DocumentStore, EventLog

FEEDBACK_PRIOR_VERSION = "feedback-prior-v1"
#: The largest weight a setting may give: a judgement's full weight times it is the term before its cut.
MAX_FEEDBACK_WEIGHT = 1.0
#: The most the term moves one candidate's fused score either way: what first place is worth
#: over second under rank fusion when both lanes agree at full voice. A lane at a lighter voice
#: makes a place worth less: at a hashed embedder's default vector voice (a hundredth) one place
#: where both lanes agree is worth 1.01 * (1/61 - 1/62), so this bound is about two places. It
#: bounds each term, not who a passage can pass: a leader sunk and a follower lifted close twice
#: it, and deeper ranks sit closer than first and second. Nor is it what keeps unrelated questions
#: whole -- the weight and the score's hold are: on the replay a term of 0.00027, half this bound,
#: already cost them (benchmarks/feedback-replay-v1.results.md).
MAX_FEEDBACK_BOOST = 2 * (1 / (RRF_K + 1) - 1 / (RRF_K + 2))
FEEDBACK_WINDOW_DAYS = 90
FEEDBACK_HALF_LIFE_DAYS = 30.0
MIN_CORROBORATION = 2


def validate_feedback_weight(weight: object) -> None:
    """A finite number from 0 to MAX_FEEDBACK_WEIGHT; 0 turns the prior off."""
    # NaN and the infinities fail the range check, so it is the only one they need.
    if isinstance(weight, bool) or not isinstance(weight, (int, float)) or not 0 <= weight <= MAX_FEEDBACK_WEIGHT:
        raise InvalidInput(f"feedback_weight must be a finite number from 0 to {MAX_FEEDBACK_WEIGHT}")


def fingerprint(content: str, text: str) -> str:
    """What a judged passage said: its text, and the content of the episode it is a span of.

    The content itself, not the episode's ``content_hash``: a keyed record's hash is its key,
    the same whatever the content says. Not the episode's id either: the same content stored
    again under another id still says the same thing."""
    return hashlib.sha256(f"{content}\x00{text}".encode()).hexdigest()


def question(recorded: object) -> str:
    """Which question a judged recall asked: a hash of the query its event recorded, hashed as
    the event log hashes a query by default (``feedback`` hashes one kept in the clear first),
    so two recalls of the same words are one question."""
    return hashlib.sha256(str(recorded).encode()).hexdigest()[:16]


@dataclass(frozen=True)
class PriorTerms:
    #: Candidate chunk id to the term added to its fused score; candidates with none are absent.
    terms: dict[int, float] = field(default_factory=dict)
    weight: float = 0.0
    max_boost: float = MAX_FEEDBACK_BOOST
    #: Candidates whose term was cut to ``max_boost``.
    capped: int = 0
    #: Judgements dropped because the passage no longer says what was judged.
    stale: int = 0
    #: Judgements dropped because they cannot be checked: no fingerprint or question, or no episode to check against.
    unverified: int = 0
    #: Candidates with useful judgements too few to count.
    tentative: int = 0
    #: False when there was no event log to read.
    read: bool = True
    events_read: int = 0
    events_cut: bool = False
    #: Candidates with more than ``min_corroboration`` counted judgements one way, the older left out.
    held: int = 0
    #: What the judgements were read over and folded with: none older than ``window_days`` is read.
    window_days: int = FEEDBACK_WINDOW_DAYS
    half_life_days: float = FEEDBACK_HALF_LIFE_DAYS
    min_corroboration: int = MIN_CORROBORATION

    def record(self, returned: Sequence[int]) -> dict[str, object]:
        return {"version": FEEDBACK_PRIOR_VERSION, "weight": self.weight, "max_boost": self.max_boost,
                "read": self.read, "window_days": self.window_days, "half_life_days": self.half_life_days,
                "min_corroboration": self.min_corroboration, "events_read": self.events_read, "events_cut": self.events_cut,
                "boosted": sum(term > 0 for term in self.terms.values()),
                "demoted": sum(term < 0 for term in self.terms.values()),
                "capped": self.capped, "held": self.held, "stale": self.stale, "unverified": self.unverified, "tentative": self.tentative,
                "returned_terms": {str(chunk): round(self.terms[chunk], 6) for chunk in returned if chunk in self.terms}}


def prior_terms(events: Iterable[Event], fingerprints: Mapping[int, Optional[str]], *, now: str, weight: float,
                half_life_days: float = FEEDBACK_HALF_LIFE_DAYS, min_corroboration: int = MIN_CORROBORATION,
                max_boost: float = MAX_FEEDBACK_BOOST) -> PriorTerms:
    """The term for each candidate in ``fingerprints`` (chunk id to what it says now, None when that
    cannot be read) from ``events``."""
    validate_feedback_weight(weight)
    lessons._check(half_life_days, min_corroboration)
    moment = parse_rfc3339(now)
    found: dict[int, float] = {}
    capped = held = stale = unverified = tentative = 0
    for chunk, judged in lessons.judgements_by_passage(events, moment).items():
        if chunk not in fingerprints:
            continue
        says = fingerprints[chunk]
        kept: list[Event] = []
        for event in judged:  # the latest judgement of each question, oldest first
            marked, asked = event.payload.get("fingerprint"), event.payload.get("question")
            if says is None or marked is None or asked is None:
                unverified += 1
            elif marked != says:
                stale += 1
            else:
                kept.append(event)
        last_against = max((index for index, event in enumerate(kept) if not event.payload["useful"]), default=-1)
        counted = [event for index, event in enumerate(kept) if not event.payload["useful"] or index > last_against]
        # Each way, newest (so heaviest) first: only the newest min_corroboration count.
        useful, against = ([abs(lessons.judgement_weight(event, moment, half_life_days))
                            for event in counted if event.payload["useful"] is kind] for kind in (True, False))
        useful.sort(reverse=True)
        against.sort(reverse=True)
        if len(useful) > min_corroboration or len(against) > min_corroboration:
            held += 1
        score = -sum(against[:min_corroboration])
        if len(useful) >= min_corroboration:
            score += sum(useful[:min_corroboration])
        elif useful:
            tentative += 1
        raw = weight * score
        if raw == 0:
            continue
        if abs(raw) > max_boost:
            capped += 1
        found[chunk] = max(-max_boost, min(max_boost, raw))
    return PriorTerms(found, float(weight), max_boost, capped, stale, unverified, tentative, held=held,
                      half_life_days=float(half_life_days), min_corroboration=min_corroboration)


async def read_prior(events: "Optional[EventLog]", documents: "DocumentStore", space: str,
                     candidates: Mapping[int, "Chunk"], *, now: str, weight: float) -> PriorTerms:
    """The terms for a recall's fused candidates, from the newest judgements of the window."""
    window = FEEDBACK_WINDOW_DAYS
    if events is None:
        return PriorTerms(weight=float(weight), read=False)
    bound = lessons.MAX_FEEDBACK_EVENTS
    since = (parse_rfc3339(now) - timedelta(days=window)).isoformat().replace("+00:00", "Z")
    read = await events.query(space, kind="feedback", since=since, limit=bound + 1)
    kept = read[:bound]  # newest first, so a cut drops the oldest
    judged = {int(event.payload["chunk_id"]) for event in kept} & set(candidates)  # type: ignore[call-overload]
    episodes = {}
    for episode_id in sorted({candidates[chunk].episode_id for chunk in judged}):
        episode = await documents.get_episode(space, episode_id)
        if episode is not None:
            episodes[episode_id] = episode
    # A candidate whose episode cannot be read has nothing to check its judgements against: None.
    fingerprints = {chunk: (fingerprint(episodes[candidates[chunk].episode_id].content, candidates[chunk].text)
                            if candidates[chunk].episode_id in episodes else None)
                    for chunk in judged}
    relevant = [event for event in kept if int(event.payload["chunk_id"]) in fingerprints]  # type: ignore[call-overload]
    terms = prior_terms(relevant, fingerprints, now=now, weight=weight, max_boost=MAX_FEEDBACK_BOOST)
    return replace(terms, events_read=len(kept), events_cut=len(read) > bound, window_days=window)
