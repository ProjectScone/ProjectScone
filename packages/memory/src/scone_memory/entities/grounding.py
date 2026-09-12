"""Re-read the facts behind a graph item and check each quote against its source now.

A projection is a snapshot; by the time a person or a model reads what it
cites, a fact may have been closed or excluded and its source changed. Every
surface that shows a fact's evidence re-reads the fact and checks its quote
against the episode it names, and only that episode: a store that returns
another space's or another id's episode gets ``quote_source_mismatch``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..core.ports import DocumentStore


async def checked_facts(documents: "DocumentStore", space: str, fact_ids: list[int]) -> list[dict[str, object]]:
    """Re-read each fact and check its quote against the retained source now."""
    episodes: dict[int, object] = {}
    checked = []
    for fact_id in fact_ids:
        fact = await documents.get_fact(space, fact_id)
        if fact is None or fact.space != space:
            continue
        if fact.source_episode_id is None:
            grounding = "stated"
        elif fact.quote is None:
            grounding = "source_unquoted"
        else:
            if fact.source_episode_id not in episodes:
                episodes[fact.source_episode_id] = await documents.get_episode(space, fact.source_episode_id)
            episode = episodes[fact.source_episode_id]
            content = getattr(episode, "content", None)
            own = (getattr(episode, "space", None) == space
                   and getattr(episode, "episode_id", None) == fact.source_episode_id)
            grounding = ("quote_source_missing" if episode is None or content is None
                         else "quote_source_mismatch" if not own
                         else "quote_verified" if fact.quote in content else "quote_not_found")
        checked.append({"fact_id": fact.fact_id, "subject": fact.subject, "predicate": fact.predicate,
                        "object": fact.object, "status": fact.status, "excluded": fact.excluded,
                        "origin": fact.origin, "valid_from": fact.valid_from, "valid_until": fact.valid_until,
                        "source_episode_id": fact.source_episode_id, "quote": fact.quote,
                        "superseded_by": fact.superseded_by, "grounding": grounding})
    return checked
