"""How much voice a lane has earned on this query.

Rank fusion gives each lane a weight, and that weight is one number for
every query a space will ever be asked. It is a bet that a lane is
equally worth listening to whatever the question was, and the bench says
it is not: the vector lane's weight decides the retrieval row, and the
setting that wins at rank 5 is not the one that wins on MRR, because the
lane is right on some questions and guessing on others.

Which of the two happened is written in the lane's own scores, before
anything is fused and without a model. A lane that places one candidate
far above the rest has found something. A lane whose twenty candidates
sit within a hair of each other has ranked near-identical neighbours,
and its order is close to arbitrary -- on that query it should argue
more quietly, not because the lane is bad but because this answer of it
carries no information.

The measure is the top's distance above the lane's **middle**, in units
of the lane's whole range:

    separation = (top - median) / (top - bottom)

Zero when nothing separates, one when the top stands alone above a flat
field, a half when the lane falls away evenly. The middle rather than
the mean: one candidate far below the rest drags a mean down and would
read as a confident top, where the middle does not move. This is the
vocabulary ``fusion.distribution_scores`` already uses to place a score
inside its lane, turned on the lane itself.

What it may do is bounded on both sides. The voice runs from the weight
the operator configured to the weight of the lane it argues against, and
no further: switching this on can only move a lane inside the band its
operator already accepted, never past it. Where there is no band, or too
little to judge, the rule says so and changes nothing -- a lane of one
candidate is its own top and bottom, which would read as perfect
separation and is the opposite of what one candidate shows.
"""

from __future__ import annotations

import math
from statistics import median
from typing import Sequence

#: Candidates a lane needs before its shape is read. Below this there is
#: no middle to speak of: two scores are a top and a bottom and nothing
#: in between, and every such lane would read as perfectly separated.
MIN_JUDGED = 3


def separation_of(scores: Sequence[float]) -> float:
    """How far the top sits above the middle, as a share of the lane's
    whole range. 0 when the lane does not spread, 1 when the top stands
    alone above a flat field."""
    top, bottom = max(scores), min(scores)
    whole = top - bottom
    if whole <= 0:
        return 0.0
    return min(1.0, max(0.0, (top - median(scores)) / whole))


def lane_voice(hits: Sequence[tuple[int, float]], *, floor: float,
               ceiling: float = 1.0) -> tuple[float, dict[str, object]]:
    """The weight this lane has earned on this query, and what was read
    to decide it.

    ``floor`` is the weight the lane has when nothing is known -- the
    configured one -- and ``ceiling`` the weight of the lane it argues
    against. The answer is always between them, and equal to ``floor``
    whenever the lane's scores say nothing: no candidates, fewer than
    ``MIN_JUDGED`` of them, or scores that are not there to read.

    The second value is for the recall event, and it names the reason
    whenever the rule declined to judge, so a reader never mistakes "the
    lane kept its weight because nothing could be read" for "the lane
    was judged and found flat".
    """
    if not hits:
        return floor, {"voice": floor, "reason": "no candidates"}
    if ceiling <= floor:
        # The operator already trusts this lane as much as the one it
        # argues with. There is nothing to move inside.
        return floor, {"voice": floor, "reason": "no band to move in"}
    if len(hits) < MIN_JUDGED:
        return floor, {"voice": floor, "reason": f"fewer than {MIN_JUDGED} candidates to judge"}
    scores = [score for _, score in hits]
    if any(not isinstance(score, (int, float)) or isinstance(score, bool) for score in scores):
        raise TypeError("a lane's scores must be numbers")
    if not all(math.isfinite(score) for score in scores):
        # A store that ranks without a score hands back NaN, and an
        # infinity has no distance to anything. Either way the order may
        # be perfect and nothing in it says so.
        return floor, {"voice": floor, "reason": "no scores to judge"}
    separation = separation_of(scores)
    voice = floor + (ceiling - floor) * separation
    judged: dict[str, object] = {"voice": voice, "separation": separation, "floor": floor, "ceiling": ceiling}
    if separation <= 0:
        # Judged, and the judgment is that nothing separates. Said out
        # loud, because a voice back at the floor otherwise reads the
        # same as one the rule never looked at.
        judged["reason"] = "no spread"
    return voice, judged

