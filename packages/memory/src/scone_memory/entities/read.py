"""Read a space's ledger into an entity projection, bounded and honest about it.

The whole-ledger read happens only for an explicit graph request, never on a
recall path. ``read_ledger`` reads it:

- **Paged** when the store implements ``LedgerPager``: newest first, a page
  at a time, stopping once the newest ``MAX_FACTS`` are in hand. A page that
  breaks the pager contract is refused, and the read falls back to one
  whole-ledger list, reporting ``pager_rejected``.
- **Unpaged** otherwise: one ``list_facts``. A store that truncates that
  silently (Elasticsearch returns at most 10,000 rows) is named
  (``store_read_cap_reached``), and past ``MAX_FACTS`` only the newest facts
  are kept (``fact_limit``).
- **Fenced** by the space's revision, read before and after. A write that
  lands during the read makes it read again; if the space is still moving,
  the read says so (``ledger_changed_during_read``, ``consistent=False``).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal, cast

from ..core import graph_read
from ..core.models import Fact
from ..core.timeutil import parse_rfc3339
from ..core.validation import check_space
from .project import EntityProjection, project_entities

if TYPE_CHECKING:
    from ..core.ports import DocumentStore
    from ..memory.engine import MemoryEngine
    from .view import StatusMode

MAX_FACTS = 50_000
#: Stores whose whole-ledger read stops at a fixed row count without saying so.
_SILENT_CAPS = {"elasticsearch": 10_000}
#: Rows asked of a pager at a time; the port caps it at MAX_LEDGER_PAGE.
MAX_LEDGER_PAGE = graph_read.MAX_LEDGER_PAGE
_ATTEMPTS = 2


@dataclass(frozen=True)
class LedgerRead:
    space: str
    #: The space's revision when the rows were read.
    revision: int
    #: Oldest first: the newest ``MAX_FACTS`` rows in every status.
    facts: tuple[Fact, ...]
    reasons: tuple[str, ...]
    read_mode: Literal["paged", "unpaged"]
    #: False when the space changed during every attempt to read it.
    consistent: bool
    #: The most facts the read would keep.
    limit: int = MAX_FACTS


class _PageRefused(Exception):
    pass


async def _paged(pager: graph_read.LedgerPager, space: str, limit: int) -> tuple[list[Fact], bool]:
    rows: list[Fact] = []
    before_id: int | None = None
    while len(rows) <= limit:  # one row past the limit says whether it cut anything
        want = min(MAX_LEDGER_PAGE, limit + 1 - len(rows))
        page = await pager.page_facts(space, before_id, want)
        problem = graph_read.checked_ledger_page(page, space=space, before_id=before_id, limit=want)
        if problem is not None:
            raise _PageRefused(problem)
        rows.extend(page)
        if len(page) < want:
            break
        before_id = page[-1].fact_id
    return rows[:limit][::-1], len(rows) > limit


async def _rows(documents: "DocumentStore", space: str,
                limit: int) -> tuple[list[Fact], list[str], Literal["paged", "unpaged"]]:
    reasons: list[str] = []
    if callable(getattr(documents, "page_facts", None)):
        try:
            facts, cut = await _paged(cast(graph_read.LedgerPager, documents), space, limit)
            return facts, ["fact_limit"] if cut else [], "paged"
        except _PageRefused:
            reasons.append("pager_rejected")
    facts = await documents.list_facts(space, include_closed=True)
    cap = _SILENT_CAPS.get(documents.name)
    if cap is not None and len(facts) >= cap:
        reasons.append("store_read_cap_reached")
    facts = sorted(facts, key=lambda fact: fact.fact_id)
    if len(facts) > limit:
        facts = facts[-limit:]
        reasons.append("fact_limit")
    return facts, reasons, "unpaged"


async def read_ledger(engine: "MemoryEngine", space: str, *, max_facts: int | None = None) -> LedgerRead:
    """Every fact of the space, newest ``max_facts`` at most, read between two
    matching revisions when the space holds still long enough."""
    check_space(space)
    limit = max(1, MAX_FACTS if max_facts is None else max_facts)
    for _attempt in range(_ATTEMPTS):
        before = await engine.revision(space)
        facts, reasons, mode = await _rows(engine.documents, space, limit)
        if await engine.revision(space) == before:
            return LedgerRead(space, before, tuple(facts), tuple(reasons), mode, True, limit)
    return LedgerRead(space, before, tuple(facts), (*reasons, "ledger_changed_during_read"), mode, False, limit)


async def load_projection(engine: "MemoryEngine", space: str, *, mode: "StatusMode" = "all",
                          as_of: str | None = None) -> tuple[EntityProjection, dict[str, object]]:
    """Project the facts that count in ``mode`` at ``as_of``.

    Facts are chosen before projecting, so classification and kind hints
    come only from what the view counts: an excluded claim or one that
    begins in 2030 cannot make a 2021 value into a thing.
    """
    from .view import counts

    from . import service

    check_space(space)
    when = parse_rfc3339(as_of if as_of is not None else engine.clock())
    return await engine.entities.projection(space, mode=mode, when=when, timeout=service.BUILD_TIMEOUT)
