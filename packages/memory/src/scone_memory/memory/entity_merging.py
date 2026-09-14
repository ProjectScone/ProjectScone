"""Record and undo the decision that two names are one entity.

``merge_entities`` is the only writer of the reserved predicate
(``entities.merges.SAME_ENTITY``): ordinary assertion refuses it, so a
document saying two names are one, or a model reading one, can propose a
claim to weigh but never merge two entities. The decision is a ledger fact
placed like any other -- one target per alias at a time, so pointing an
alias somewhere new closes the decision it replaces -- and the reason and
actor go on the event. ``unmerge_entities`` closes the decision in force.

Refused before anything is written: a name that cannot name one thing
(prose, a quotation, a pronoun), two names that already share a key, a
missing reason, and a merge that would loop back to its alias. The loop
check follows the target's chain of decisions in force one lookup at a
time, so it reads only the chain, never the whole ledger.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional

from ..core.errors import InvalidInput, NotFound
from ..core.models import Fact
from ..core.validation import check_space, normalise_term
from ..entities.classify import join_block_reason
from ..entities.merges import SAME_ENTITY
from . import fact_review

if TYPE_CHECKING:
    from .engine import MemoryEngine

__all__ = ["MAX_MERGE_CHAIN", "merge_entities", "unmerge_entities"]

#: Decisions followed from a target while checking for a loop. Past it the
#: merge is refused, since a loop could not be ruled out.
MAX_MERGE_CHAIN = 64
_UNJOINABLE = frozenset({"prose", "quoted_text", "pronoun"})


def _name(text: str, what: str) -> str:
    key = normalise_term(text, what)
    reason = join_block_reason(text)
    if reason in _UNJOINABLE:
        raise InvalidInput(f"{what} {text.strip()!r} cannot name one thing ({reason}); nothing was merged")
    return key


async def _in_force(engine: "MemoryEngine", space: str, key: str) -> Optional[Fact]:
    held = [fact for fact in await engine.documents.facts_for(space, key, SAME_ENTITY)
            if fact.status == "active" and fact.valid_until is None and not fact.excluded]
    return max(held, key=lambda fact: fact.fact_id) if held else None


async def merge_entities(engine: "MemoryEngine", space: str, alias: str, into: str, *, reason: str,
                         actor: Optional[str] = None) -> Fact:
    await engine._living(space)
    check_space(space)
    alias_key, into_key = _name(alias, "alias"), _name(into, "into")
    if alias_key == into_key:
        raise InvalidInput(f"{alias.strip()!r} and {into.strip()!r} are already one name; nothing was merged")
    reason = fact_review._reason(reason)
    step = into_key
    for _ in range(MAX_MERGE_CHAIN):
        decision = await _in_force(engine, space, step)
        if decision is None:
            break
        step = normalise_term(decision.object, "into")
        if step == alias_key:
            raise InvalidInput(f"{into.strip()!r} is already merged into {alias.strip()!r}; "
                               "merging back would make a cycle")
    else:
        raise InvalidInput(f"{into.strip()!r} starts a chain of more than {MAX_MERGE_CHAIN} merges; "
                           "merge into the end of the chain")
    fact = await engine._assert_placed(space, alias_key, SAME_ENTITY, into.strip(), origin="stated")
    await engine._emit(space, "entity_merge", {"fact_id": fact.fact_id, "alias": alias_key, "into": into_key,
                                               "reason": reason, "actor": actor})
    return fact


async def unmerge_entities(engine: "MemoryEngine", space: str, alias: str, *, reason: str,
                           actor: Optional[str] = None) -> Fact:
    await engine._living(space)
    check_space(space)
    alias_key = normalise_term(alias, "alias")
    decision = await _in_force(engine, space, alias_key)
    if decision is None:
        raise NotFound(f"{alias.strip()!r} is not merged into anything in {space!r}")
    return await fact_review.close_fact(engine._review_runtime(), space, decision.fact_id, reason, actor=actor,
                                        kind="entity_unmerge")
