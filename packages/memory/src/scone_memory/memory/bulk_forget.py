"""Forget every source a filter selects, after seeing what that takes.

One episode can be forgotten with a receipt, a space deleted with an
impact preview, and a retention policy can expire by kind and age. This
is the bulk case between them: everything synced from one directory,
everything tagged for one client, forgotten as one operation.

A call previews unless told to apply. The preview names the episodes the
filter chose for this pass, what forgetting them would take with it, and
a digest of exactly that choice. Applying requires the digest and chooses
again; if the choice differs -- a source arrived that matches, one left --
it refuses and forgets nothing, because the preview a person agreed to is
no longer the one that would run. Each forget is the ordinary one, with
its retirement record, retry and receipt, and the claims policy of
``forget``.

The walk is bounded, a pass is bounded, and the report says when either
bit: ``selection_complete`` false means the walk stopped before reading
every source, and ``pass_limited`` means more matched than one pass takes.
"""
from __future__ import annotations

import hashlib
from typing import TYPE_CHECKING, Literal, Mapping, Optional, Sequence

from ..core.errors import Gone, InvalidInput, NotFound
from ..core.models import BulkForgetReport, Episode
from ..core.validation import check_space, normalise_tags

if TYPE_CHECKING:
    from .engine import MemoryEngine

#: Sources a selection reads before it stops and says so.
MAX_SELECTED = 10_000
#: Episodes one pass may forget, at most.
MAX_PASS = 1_000
#: Sources read per page of the walk.
_PAGE = 100


def _digest(space: str, episode_ids: Sequence[int]) -> str:
    return hashlib.sha256(("scone.forget_matching/1\n" + space + "\n" + ",".join(map(str, episode_ids))).encode()).hexdigest()


async def _select(engine: "MemoryEngine", space: str, *, source_prefix: Optional[str], tags: tuple[str, ...],
                  conditions: Mapping[str, object] | None, kind: Optional[str]) -> tuple[list[Episode], bool]:
    """Every retained source the filter matches, newest id first, and whether
    the walk read to the end."""
    matched: list[Episode] = []
    before: Optional[int] = None
    while True:
        page = await engine.source_page(space, before=before, limit=_PAGE, kind=kind, conditions=conditions)
        for episode in page.episodes:
            if source_prefix is not None and not (episode.source or "").startswith(source_prefix):
                continue
            if tags and not set(tags) <= set(episode.tags):
                continue
            matched.append(episode)
        if not page.has_more:
            return matched, True
        if len(matched) >= MAX_SELECTED or page.next_before is None:
            return matched, False
        before = page.next_before


async def forget_matching(engine: "MemoryEngine", space: str, *, source_prefix: Optional[str] = None,
                          tags: Sequence[str] = (), conditions: Mapping[str, object] | None = None,
                          kind: Optional[str] = None, limit: int = 100, apply: bool = False,
                          selection: Optional[str] = None,
                          with_claims: Literal["keep", "exclude"] = "keep") -> BulkForgetReport:
    check_space(space)
    if isinstance(limit, bool) or not isinstance(limit, int) or not 1 <= limit <= MAX_PASS:
        raise InvalidInput(f"limit must be an integer in 1..{MAX_PASS}")
    if source_prefix is not None and (not isinstance(source_prefix, str) or not source_prefix):
        raise InvalidInput("source_prefix must be non-empty text when given")
    clean_tags = tuple(normalise_tags(tags))
    if source_prefix is None and not clean_tags and conditions is None and kind is None:
        raise InvalidInput("forget_matching needs a filter: a source prefix, tags, conditions or a kind. "
                           "Deleting a whole space is its own operation, with its own impact preview.")
    if with_claims not in ("keep", "exclude"):
        raise InvalidInput(f"with_claims must be keep or exclude, not {with_claims!r}")
    matched, complete = await _select(engine, space, source_prefix=source_prefix, tags=clean_tags,
                                      conditions=conditions, kind=kind)
    # Oldest first, so repeated passes work through a selection in a stable order.
    chosen = sorted(episode.episode_id for episode in matched)[:limit]
    digest = _digest(space, chosen)
    report = BulkForgetReport(
        space=space, applied=apply, matched=len(matched), selection_complete=complete,
        pass_limited=len(matched) > len(chosen), episode_ids=chosen, selection=digest,
        filter={"source_prefix": source_prefix, "tags": list(clean_tags),
                "conditions": dict(conditions) if conditions is not None else None, "kind": kind},
        with_claims=with_claims)
    if not apply:
        for episode_id in chosen:
            impact = await engine.impact(space, episode_id)
            report.chunks += impact.chunks
            report.attachments_released += len(impact.attachments_released)
            report.facts_citing += len(impact.facts_citing)
            report.links_citing += len(impact.links_citing)
            report.affirmations_citing += len(impact.affirmations_citing)
        return report
    if selection is None:
        raise InvalidInput("applying needs the selection digest from a preview of the same filter, so what is "
                           "forgotten is what was shown")
    if selection != digest:
        raise InvalidInput("the selection changed since its preview: sources matching the filter arrived or left. "
                           "Nothing was forgotten; preview again")
    for episode_id in chosen:
        try:
            receipt = await engine.forget(space, episode_id, with_claims=with_claims)
        except (NotFound, Gone) as error:
            report.skipped.append({"episode_id": episode_id, "reason": type(error).__name__})
            continue
        report.forgotten.append(episode_id)
        report.receipts.append(receipt)
        report.chunks += receipt.chunks
        report.attachments_released += len(receipt.attachments_released)
        report.facts_citing += len(receipt.facts_citing)
        report.links_citing += len(receipt.links_citing)
        report.affirmations_citing += len(receipt.affirmations_citing)
    report.remaining = len(matched) - len(report.forgotten)
    return report
