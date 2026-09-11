"""Optional bounded fact inventory used by recorded activity graphs."""
from typing import Protocol, runtime_checkable

from .models import Fact

DEFAULT_GRAPH_FACTS = 400
MAX_GRAPH_FACTS = 2000


@runtime_checkable
class GraphFactReader(Protocol):
    async def facts_for_graph(self, space: str, source_episode_id: int | None, limit: int) -> list[Fact]:
        """Oldest fact IDs first, every status, filter exact space/source before limit.

        None selects all sources. At most 2001 records may be returned, including
        the caller's lookahead. This operation must not materialize the ledger.
        """
        ...


def graph_fact_read_limit(source_episode_id: int | None, limit: int) -> int:
    if source_episode_id is not None and (type(source_episode_id) is not int or not 1 <= source_episode_id < 2**63):
        raise ValueError('source_episode_id must be a positive 64-bit integer or None')
    if type(limit) is not int:
        raise ValueError('fact read limit must be an integer')
    return max(0, min(limit, MAX_GRAPH_FACTS + 1))


#: The most rows one ledger page may hold.
MAX_LEDGER_PAGE = 1000


@runtime_checkable
class LedgerPager(Protocol):
    async def page_facts(self, space: str, before_id: int | None, limit: int) -> list[Fact]:
        """One page of the space's ledger for a whole-space projection.

        Newest fact ids first, every status (proposed, closed and excluded
        facts included), only ``space``, only ids below ``before_id`` (None
        starts at the newest), at most ``limit`` rows and never more than
        ``MAX_LEDGER_PAGE``. Pages chain: the next cursor is the last id.
        """
        ...


def ledger_page_limit(limit: int) -> int:
    if type(limit) is not int:
        raise ValueError("ledger page limit must be an integer")
    return max(0, min(limit, MAX_LEDGER_PAGE))


def checked_ledger_page(rows: list[Fact], *, space: str, before_id: int | None, limit: int) -> str | None:
    """What is wrong with a page a store returned, or None. A reader refuses
    a page that breaks the contract rather than projecting from it."""
    if len(rows) > limit:
        return "too_long"
    if any(row.space != space for row in rows):
        return "other_space"
    ids = [row.fact_id for row in rows]
    if any(later >= earlier for earlier, later in zip(ids, ids[1:])):
        return "not_newest_first"
    if before_id is not None and ids and ids[0] >= before_id:
        return "not_before_cursor"
    return None
