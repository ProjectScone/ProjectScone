"""Pure atomic membership for bounded pre-assessment graph candidates.

Exact joins reuse the grouping adapter's rule. Temporary membership records
carry join keys only; neither they nor synthetic text enter model evidence.
"""
from __future__ import annotations

from ..core.models import Fact
from .adaptive import EvidenceCandidate
from .evidence_groups import _joins


def merge_groups(groups: tuple[tuple[str, ...], ...]) -> tuple[tuple[str, ...], ...]:
    """Union overlapping contracts without splitting any established group."""
    merged: list[list[str]] = []
    for group in groups:
        current = list(dict.fromkeys(group))
        untouched: list[list[str]] = []
        for existing in merged:
            if set(current).intersection(existing):
                current = list(dict.fromkeys([*existing, *current]))
            else:
                untouched.append(existing)
        merged = [*untouched, current]
    return tuple(tuple(group) for group in merged if len(group) >= 2)


def exact_fact_groups(facts: tuple[Fact, ...]) -> tuple[tuple[str, ...], ...]:
    """Return exact connected components independently of quote byte size.

    Input is capped by bounded recall and traversal windows. Full original
    records are separately verified and packed; membership-only objects allow
    an oversized quote's entire component to be omitted safely. The shared
    helper enforces its existing 2048 directed-edge ceiling.
    """
    candidates = tuple(EvidenceCandidate(id=f"fact:{fact.fact_id}", episode_id=fact.source_episode_id or 1,
        text="membership only", subject=fact.subject, object=fact.object) for fact in facts)
    return merge_groups(tuple((candidates[left].id, candidates[right].id) for left, right in _joins(candidates)))
