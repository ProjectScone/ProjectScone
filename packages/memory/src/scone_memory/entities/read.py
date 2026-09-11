"""Read a space's ledger into an entity projection, bounded and honest about it.

The whole-ledger read happens only for an explicit graph request, never on a
recall path. Reads are capped and every cap that bites is reported: a store
that truncates silently (Elasticsearch returns at most 10,000 rows) is named,
and past ``MAX_FACTS`` only the newest facts are projected.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from ..core.validation import check_space
from .project import EntityProjection, project_entities

if TYPE_CHECKING:
    from ..memory.engine import MemoryEngine

MAX_FACTS = 50_000
#: Stores whose whole-ledger read stops at a fixed row count without saying so.
_SILENT_CAPS = {"elasticsearch": 10_000}


async def load_projection(engine: "MemoryEngine", space: str) -> tuple[EntityProjection, dict[str, object]]:
    check_space(space)
    facts = await engine.documents.list_facts(space, include_closed=True)
    reasons: list[str] = []
    cap = _SILENT_CAPS.get(engine.documents.name)
    if cap is not None and len(facts) >= cap:
        reasons.append("store_read_cap_reached")
    if len(facts) > MAX_FACTS:
        facts = sorted(facts, key=lambda fact: fact.fact_id)[-MAX_FACTS:]
        reasons.append("fact_limit")
    projection = project_entities(space, facts, revision=await engine.revision(space))
    return projection, {"facts_read": len(facts), "facts_limit": MAX_FACTS, "reasons": reasons}
