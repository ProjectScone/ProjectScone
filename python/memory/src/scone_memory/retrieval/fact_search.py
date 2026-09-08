"""Optional bounded lexical fact search; legacy document stores need not implement it."""

from typing import Protocol, runtime_checkable

from ..core.models import Fact
from ..core.ports import TextFilter


@runtime_checkable
class IndexedFactSearch(Protocol):
    """Return current scoped facts ranked by overlap, confidence, then fact ID.

    Implementations apply exact query token overlap, exclusion, ledger validity
    at ``when``, and source scope before ``limit``. The engine verifies returned
    rows and falls back to its scan when an optional index is unavailable.
    """

    async def search_facts(self, space: str, query: str, when: str, limit: int,
                           scope: TextFilter | None = None) -> list[Fact]: ...
