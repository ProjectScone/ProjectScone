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

A recognizer's mentions (``entities.mentions``) are read under a budget
of their own, ``MAX_MENTION_FACTS``, beside the ``MAX_FACTS`` budget for
every other fact. A space with many records names far more things than
anyone states about them, and under one shared budget its mentions would
push the claims out of every view. Each budget says when it bit:
``fact_limit`` for claims, ``mention_limit`` for mentions. A paged read
walks on until the claims' budget is full, however many mentions that
passes, so mentions never displace a claim, and stops there: the read
covers the time back to its oldest claim, ``fact_limit`` says it was cut,
and a mention older than that is as far outside it as an older claim. A
ledger with no mentions is read exactly as far as before.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal, cast

from ..core import graph_read
from ..core.models import Fact
from ..core.timeutil import parse_rfc3339
from ..core.validation import check_space
from .mentions import is_mention
from .project import EntityProjection, project_entities

if TYPE_CHECKING:
    from ..core.ports import DocumentStore
    from ..memory.engine import MemoryEngine
    from .view import StatusMode

MAX_FACTS = 50_000
#: Mention facts kept beside them, newest first.
MAX_MENTION_FACTS = 50_000
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
    #: The most facts the read would keep, mentions apart.
    limit: int = MAX_FACTS
    #: The most mention facts the read would keep.
    mention_limit: int = MAX_MENTION_FACTS


class _PageRefused(Exception):
    pass


class _Budgets:
    """Newest-first rows sorted into the two budgets, each keeping at most
    its limit and noting whether a row past it was seen."""

    def __init__(self, limit: int, mention_limit: int) -> None:
        self.limits = {False: limit, True: mention_limit}
        self.kept: dict[bool, list[Fact]] = {False: [], True: []}
        self.over = {False: False, True: False}

    def take(self, fact: Fact) -> None:
        which = is_mention(fact)
        if len(self.kept[which]) < self.limits[which]:
            self.kept[which].append(fact)
        else:
            self.over[which] = True

    def claims_wanted(self) -> int:
        """Claims still to read, one past the budget to say it cut something."""
        return self.limits[False] + 1 - len(self.kept[False])

    def done(self) -> bool:
        """Whether a walk may stop: once a claim past the claims' budget
        has been seen, and never before, whatever the mentions did."""
        return self.over[False]

    def result(self) -> tuple[list[Fact], list[str]]:
        rows = sorted((*self.kept[False], *self.kept[True]), key=lambda fact: fact.fact_id)
        reasons = (["fact_limit"] if self.over[False] else []) + (["mention_limit"] if self.over[True] else [])
        return rows, reasons


async def _paged(pager: graph_read.LedgerPager, space: str, budgets: _Budgets) -> None:
    before_id: int | None = None
    # Until a claim past its budget has been seen, which says it cut something.
    while not budgets.done():
        # Never more rows than could still be claims the budget wants: a
        # ledger without mentions is read exactly as far as it needs.
        want = min(MAX_LEDGER_PAGE, budgets.claims_wanted())
        page = await pager.page_facts(space, before_id, want)
        problem = graph_read.checked_ledger_page(page, space=space, before_id=before_id, limit=want)
        if problem is not None:
            raise _PageRefused(problem)
        if not page:  # the port promises at most the rows asked for; only an empty page ends the ledger
            break
        for fact in page:
            budgets.take(fact)
        before_id = page[-1].fact_id


async def _rows(documents: "DocumentStore", space: str, limit: int,
                mention_limit: int) -> tuple[list[Fact], list[str], Literal["paged", "unpaged"]]:
    reasons: list[str] = []
    if callable(getattr(documents, "page_facts", None)):
        budgets = _Budgets(limit, mention_limit)
        try:
            await _paged(cast(graph_read.LedgerPager, documents), space, budgets)
            facts, cut = budgets.result()
            return facts, cut, "paged"
        except _PageRefused:
            reasons.append("pager_rejected")
    facts = await documents.list_facts(space, include_closed=True)
    cap = _SILENT_CAPS.get(documents.name)
    if cap is not None and len(facts) >= cap:
        reasons.append("store_read_cap_reached")
    budgets = _Budgets(limit, mention_limit)
    for fact in sorted(facts, key=lambda fact: fact.fact_id, reverse=True):
        budgets.take(fact)
    facts, cut = budgets.result()
    return facts, reasons + cut, "unpaged"


async def read_ledger(engine: "MemoryEngine", space: str, *, max_facts: int | None = None,
                      max_mentions: int | None = None) -> LedgerRead:
    """Every fact of the space, the newest ``max_facts`` claims and newest
    ``max_mentions`` mentions at most, read between two matching revisions
    when the space holds still long enough. ``max_mentions=0`` keeps none,
    for a reader that has no use for them."""
    check_space(space)
    limit = max(1, MAX_FACTS if max_facts is None else max_facts)
    mention_limit = max(0, MAX_MENTION_FACTS if max_mentions is None else max_mentions)
    for _attempt in range(_ATTEMPTS):
        before = await engine.revision(space)
        facts, reasons, mode = await _rows(engine.documents, space, limit, mention_limit)
        if await engine.revision(space) == before:
            return LedgerRead(space, before, tuple(facts), tuple(reasons), mode, True, limit, mention_limit)
    return LedgerRead(space, before, tuple(facts), (*reasons, "ledger_changed_during_read"), mode, False, limit,
                      mention_limit)


def read_record(coverage: dict[str, object]) -> tuple[bool, dict[str, object]]:
    """Whether a read held every fact, and its coverage as answered. A
    capped read can show what it found, never that something is absent."""
    reasons = coverage.get("reasons")
    capped = isinstance(reasons, list) and bool(reasons)
    return not capped, {**coverage, "truncated": capped}


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
