"""What people said about a passage, as a small term on its fused score.

Lessons (``lessons.py``) put what people said beside a passage and leave the
order alone. This is the same record, read at query time and added to a
candidate's fused score the way recency is, when ``feedback_weight`` is set;
zero, the default, reads nothing and moves nothing.

Per candidate, the latest judgement of each recall is weighed as a lesson is
(+1 useful, -1 not, halving every ``half_life_days``), with three rules on
top:

- a useful judgement counts only once ``min_corroboration`` of them do, so
  one person's word moves nothing, and an uncorroborated one offsets nothing;
- a judgement against the passage outweighs every useful one older than it:
  those no longer count, and the passage must be corroborated again after it;
- a judgement counts only while the passage says what it said when it was
  judged. ``feedback`` records a fingerprint of the chunk's text and its
  episode's content hash; a candidate whose fingerprint now differs (a
  rebuilt store handing the same id to other text, to a span of an episode
  whose content changed, or to a span chunked differently) has those
  judgements dropped as ``stale``, and one recorded without a fingerprint is
  dropped as ``unverified`` -- it cannot be checked, so it is not trusted.

The term is ``feedback_weight`` times that score, cut at ``MAX_FEEDBACK_BOOST``
either way; ``capped`` counts the candidates it cut. The read takes the
newest ``lessons.MAX_FEEDBACK_EVENTS`` judgements of the last
``FEEDBACK_WINDOW_DAYS`` and says when that bound bit.

The term does not know the question. A passage judged useful rises for every
query it is a candidate for, so a recorded question hashed in the log (the
default) costs nothing here, and a question it was never judged for can
lose a place to it: ``benchmarks/feedback-replay-v1.results.md`` measures both.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
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
#: The most the term moves a fused score either way: what first place is worth over second
#: under rank fusion when both lanes agree, so however many people agree about a passage, it
#: cannot pass one both lanes put ahead of it. The bound is not what keeps unrelated questions
#: whole -- the weight is: on the replay a term of 0.0004, under this bound, already cost them
#: (benchmarks/feedback-replay-v1.results.md).
MAX_FEEDBACK_BOOST = 2 * (1 / (RRF_K + 1) - 1 / (RRF_K + 2))
FEEDBACK_WINDOW_DAYS = 90
FEEDBACK_HALF_LIFE_DAYS = 30.0
MIN_CORROBORATION = 2


def validate_feedback_weight(weight: object) -> None:
    """A finite number from 0 to MAX_FEEDBACK_WEIGHT; 0 turns the prior off."""
    # NaN and the infinities fail the range check, so it is the only one they need.
    if isinstance(weight, bool) or not isinstance(weight, (int, float)) or not 0 <= weight <= MAX_FEEDBACK_WEIGHT:
        raise InvalidInput(f"feedback_weight must be a finite number from 0 to {MAX_FEEDBACK_WEIGHT}")


def fingerprint(content_hash: str, text: str) -> str:
    """What a judged passage said: its text, and the content hash of the episode it is a span of.
    Not the episode's id: the same content stored again under another id still says the same thing."""
    return hashlib.sha256(f"{content_hash}\x00{text}".encode()).hexdigest()


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
    #: Judgements dropped because they carry no fingerprint to check.
    unverified: int = 0
    #: Candidates with useful judgements too few to count.
    tentative: int = 0
    #: False when there was no event log to read.
    read: bool = True
    events_read: int = 0
    events_cut: bool = False

    def record(self, returned: Sequence[int]) -> dict[str, object]:
        return {"version": FEEDBACK_PRIOR_VERSION, "weight": self.weight, "max_boost": self.max_boost,
                "read": self.read, "events_read": self.events_read, "events_cut": self.events_cut,
                "boosted": sum(term > 0 for term in self.terms.values()),
                "demoted": sum(term < 0 for term in self.terms.values()),
                "capped": self.capped, "stale": self.stale, "unverified": self.unverified, "tentative": self.tentative,
                "returned_terms": {str(chunk): round(self.terms[chunk], 6) for chunk in returned if chunk in self.terms}}


def prior_terms(events: Iterable[Event], fingerprints: Mapping[int, str], *, now: str, weight: float,
                half_life_days: float = FEEDBACK_HALF_LIFE_DAYS, min_corroboration: int = MIN_CORROBORATION,
                max_boost: float = MAX_FEEDBACK_BOOST) -> PriorTerms:
    """The term for each candidate in ``fingerprints`` (chunk id to what it says now) from ``events``."""
    validate_feedback_weight(weight)
    lessons._check(half_life_days, min_corroboration)
    moment = parse_rfc3339(now)
    found: dict[int, float] = {}
    capped = stale = unverified = tentative = 0
    for chunk, judged in lessons.judgements_by_passage(events, moment).items():
        if chunk not in fingerprints:
            continue
        kept = []
        for event in judged:
            marked = event.payload.get("fingerprint")
            if marked is None:
                unverified += 1
            elif marked != fingerprints[chunk]:
                stale += 1
            else:
                kept.append(event)
        last_against = max((index for index, event in enumerate(kept) if not event.payload["useful"]), default=-1)
        counted = [event for index, event in enumerate(kept) if not event.payload["useful"] or index > last_against]
        useful = [event for event in counted if event.payload["useful"]]
        score = sum(lessons.judgement_weight(event, moment, half_life_days)
                    for event in counted if not event.payload["useful"])
        if len(useful) >= min_corroboration:
            score += sum(lessons.judgement_weight(event, moment, half_life_days) for event in useful)
        elif useful:
            tentative += 1
        raw = weight * score
        if raw == 0:
            continue
        if abs(raw) > max_boost:
            capped += 1
        found[chunk] = max(-max_boost, min(max_boost, raw))
    return PriorTerms(found, float(weight), max_boost, capped, stale, unverified, tentative)


async def read_prior(events: "Optional[EventLog]", documents: "DocumentStore", space: str,
                     candidates: Mapping[int, "Chunk"], *, now: str, weight: float) -> PriorTerms:
    """The terms for a recall's fused candidates, from the newest judgements of the window."""
    if events is None:
        return PriorTerms(weight=float(weight), read=False)
    bound = lessons.MAX_FEEDBACK_EVENTS
    since = (parse_rfc3339(now) - timedelta(days=FEEDBACK_WINDOW_DAYS)).isoformat().replace("+00:00", "Z")
    read = await events.query(space, kind="feedback", since=since, limit=bound + 1)
    kept = read[:bound]  # newest first, so a cut drops the oldest
    judged = {int(event.payload["chunk_id"]) for event in kept} & set(candidates)  # type: ignore[call-overload]
    episodes = {}
    for episode_id in sorted({candidates[chunk].episode_id for chunk in judged}):
        episode = await documents.get_episode(space, episode_id)
        if episode is not None:
            episodes[episode_id] = episode
    fingerprints = {chunk: fingerprint(episodes[candidates[chunk].episode_id].content_hash,
                                       candidates[chunk].text)
                    for chunk in judged if candidates[chunk].episode_id in episodes}
    terms = prior_terms(kept, fingerprints, now=now, weight=weight, max_boost=MAX_FEEDBACK_BOOST)
    return PriorTerms(terms.terms, terms.weight, terms.max_boost, terms.capped, terms.stale, terms.unverified,
                      terms.tentative, True, len(kept), len(read) > bound)
