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
