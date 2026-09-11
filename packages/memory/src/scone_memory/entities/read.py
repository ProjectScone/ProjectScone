"""Read a space's ledger into an entity projection, bounded and honest about it.

The whole-ledger read happens only for an explicit graph request, never on a
recall path. Reads are capped and every cap that bites is reported: a store
that truncates silently (Elasticsearch returns at most 10,000 rows) is named,
and past ``MAX_FACTS`` only the newest facts are projected.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from ..core.timeutil import parse_rfc3339
from ..core.validation import check_space
from .project import EntityProjection, project_entities

if TYPE_CHECKING:
    from ..memory.engine import MemoryEngine
    from .view import StatusMode

MAX_FACTS = 50_000
#: Stores whose whole-ledger read stops at a fixed row count without saying so.
_SILENT_CAPS = {"elasticsearch": 10_000}


async def load_projection(engine: "MemoryEngine", space: str, *, mode: "StatusMode" = "all",
                          as_of: str | None = None) -> tuple[EntityProjection, dict[str, object]]:
    """Project the facts that count in ``mode`` at ``as_of``.

    Facts are chosen before projecting, so classification and kind hints
    come only from what the view counts: an excluded claim or one that
    begins in 2030 cannot make a 2021 value into a thing.
    """
    from .view import counts

    check_space(space)
    facts = await engine.documents.list_facts(space, include_closed=True)
    reasons: list[str] = []
    cap = _SILENT_CAPS.get(engine.documents.name)
    if cap is not None and len(facts) >= cap:
        reasons.append("store_read_cap_reached")
    if len(facts) > MAX_FACTS:
        facts = sorted(facts, key=lambda fact: fact.fact_id)[-MAX_FACTS:]
        reasons.append("fact_limit")
    read = len(facts)
    when = parse_rfc3339(as_of if as_of is not None else engine.clock())
    facts = [fact for fact in facts if counts(fact.status, fact.excluded, fact.valid_from, fact.valid_until, mode, when)]
    projection = project_entities(space, facts, revision=await engine.revision(space))
    return projection, {"facts_read": read, "facts_counted": len(facts), "facts_limit": MAX_FACTS, "reasons": reasons}
