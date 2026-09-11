"""Reading a space's whole ledger for a projection: paged, fenced, bounded.

A store that can page its ledger newest-first is read a page at a time and
never materialised twice; a page that breaks the contract is refused and
the read falls back, saying so. The read is fenced by the space's
revision: a write that lands during it makes the read go again, and a
space that will not hold still is reported, never cached as whole.
"""
from __future__ import annotations

import pytest

from scone_memory import HashEmbedder, InMemoryDocumentStore, InMemoryVectorIndex, MemoryEngine
from scone_memory.core.graph_read import MAX_LEDGER_PAGE, LedgerPager, checked_ledger_page
from scone_memory.entities import read


@pytest.fixture(params=["memory", "sqlite"])
async def engine(request, tmp_path):
    if request.param == "memory":
        documents, vectors = InMemoryDocumentStore(), InMemoryVectorIndex()
    else:
        from scone_memory.backends import SqliteDocumentStore, SqliteVectorIndex

        documents, vectors = SqliteDocumentStore(tmp_path / "m.db"), SqliteVectorIndex(tmp_path / "m.db")
    engine = await MemoryEngine(documents, vectors, HashEmbedder()).open()
    yield engine
    await engine.close()


async def ledger(engine: MemoryEngine) -> list[int]:
    """Five facts in alpha, one in beta, in every status the ledger has."""
    made = [await engine.assert_fact("alpha", f"person {n}", "based_in", "Lisbon") for n in range(3)]
    await engine.assert_fact("beta", "zed", "based_in", "Porto")
    await engine.close_fact("alpha", made[0].fact_id, "moved")
    await engine.exclude("alpha", made[1].fact_id, "private")
    proposed = await engine.assert_fact("alpha", "person 9", "based_in", "Faro", proposed=True)
    return [fact.fact_id for fact in made] + [proposed.fact_id]


async def test_a_store_pages_its_ledger_newest_first_in_every_status(engine):
    ids = await ledger(engine)
    assert isinstance(engine.documents, LedgerPager)
    first = await engine.documents.page_facts("alpha", None, 2)
    rest = await engine.documents.page_facts("alpha", first[-1].fact_id, 10)
    assert [fact.fact_id for fact in first + rest] == sorted(ids, reverse=True)
    assert {fact.space for fact in first + rest} == {"alpha"}
    assert {fact.status for fact in first + rest} >= {"active", "closed", "proposed"}
    assert any(fact.excluded for fact in first + rest)
    assert await engine.documents.page_facts("alpha", min(ids), 10) == []
    assert await engine.documents.page_facts("alpha", None, 0) == []


async def test_a_page_is_never_longer_than_the_largest_page(engine):
    for number in range(MAX_LEDGER_PAGE + 5):
        await engine.assert_fact("alpha", f"team {number}", "based_in", "Lisbon")
    rows = await engine.documents.page_facts("alpha", None, MAX_LEDGER_PAGE * 3)
    assert len(rows) == MAX_LEDGER_PAGE


class Counting(InMemoryDocumentStore):
    def __init__(self) -> None:
        super().__init__()
        self.pages = 0
        self.rows = 0
        self.listed = 0

    async def page_facts(self, space, before_id, limit):
        self.pages += 1
        page = await super().page_facts(space, before_id, limit)
        self.rows += len(page)
        return page

    async def list_facts(self, space, include_closed):
        self.listed += 1
        return await super().list_facts(space, include_closed)


async def many(store, count: int) -> MemoryEngine:
    engine = await MemoryEngine(store, InMemoryVectorIndex(), HashEmbedder()).open()
    for number in range(count):
        await engine.assert_fact("alpha", f"team {number}", "based_in", "Lisbon")
    return engine


async def test_a_pager_is_read_a_page_at_a_time_and_the_ledger_never_listed(monkeypatch):
    monkeypatch.setattr(read, "MAX_LEDGER_PAGE", 4)
    store = Counting()
    engine = await many(store, 10)
    store.listed = 0
    found = await read.read_ledger(engine, "alpha")
    # Pages of 4, 4 and 2, then the empty page that ends the ledger.
    assert found.read_mode == "paged" and store.listed == 0 and store.pages == 4
    assert [fact.fact_id for fact in found.facts] == list(range(1, 11)) and found.reasons == ()


async def test_a_capped_paged_read_stops_after_the_newest_facts(monkeypatch):
    monkeypatch.setattr(read, "MAX_LEDGER_PAGE", 4)
    store = Counting()
    engine = await many(store, 20)
    found = await read.read_ledger(engine, "alpha", max_facts=5)
    assert [fact.fact_id for fact in found.facts] == [16, 17, 18, 19, 20]
    # Five kept and one more to know the cap cut something: nothing past it is read.
    assert found.reasons == ("fact_limit",) and store.rows == 6


class Unpaged(InMemoryDocumentStore):
    page_facts = None


async def test_a_store_that_cannot_page_is_listed_once():
    engine = await many(Unpaged(), 3)
    found = await read.read_ledger(engine, "alpha")
    assert found.read_mode == "unpaged" and len(found.facts) == 3 and found.reasons == ()


class Faulty(InMemoryDocumentStore):
    fault = "space"

    async def page_facts(self, space, before_id, limit):
        rows = await super().page_facts(space, before_id, limit)
        if self.fault == "space":
            return [row.model_copy(update={"space": "beta"}) for row in rows]
        return list(reversed(rows))


@pytest.mark.parametrize("fault", ["space", "order"])
async def test_a_page_that_breaks_the_contract_is_refused_and_the_read_falls_back(fault):
    store = Faulty()
    store.fault = fault
    engine = await many(store, 3)
    found = await read.read_ledger(engine, "alpha")
    assert found.read_mode == "unpaged" and "pager_rejected" in found.reasons
    assert sorted(fact.fact_id for fact in found.facts) == [1, 2, 3] and {f.space for f in found.facts} == {"alpha"}


def test_the_page_check_names_each_violation():
    from scone_memory.core.models import Fact

    def row(fact_id: int, space: str = "alpha") -> Fact:
        return Fact(fact_id=fact_id, space=space, subject="a", predicate="p", object="b",
                    valid_from="2025-01-01T00:00:00Z")

    assert checked_ledger_page([row(5), row(3)], space="alpha", before_id=6, limit=2) is None
    assert checked_ledger_page([row(5), row(3), row(2)], space="alpha", before_id=None, limit=2) == "too_long"
    assert checked_ledger_page([row(5, "beta")], space="alpha", before_id=None, limit=2) == "other_space"
    assert checked_ledger_page([row(3), row(5)], space="alpha", before_id=None, limit=2) == "not_newest_first"
    assert checked_ledger_page([row(5), row(5)], space="alpha", before_id=None, limit=2) == "not_newest_first"
    assert checked_ledger_page([row(6)], space="alpha", before_id=6, limit=2) == "not_before_cursor"


class Moving(InMemoryDocumentStore):
    """Bumps the revision while the ledger is read, ``times`` times."""
    times = 0

    async def page_facts(self, space, before_id, limit):
        rows = await super().page_facts(space, before_id, limit)
        if self.times:
            self.times -= 1
            await self.bump_revision(space)
        return rows


async def test_a_write_during_the_read_makes_it_read_again():
    store = Moving()
    engine = await many(store, 3)
    store.times = 1
    found = await read.read_ledger(engine, "alpha")
    assert found.consistent is True and found.revision == await engine.revision("alpha")


async def test_a_space_that_will_not_hold_still_is_reported():
    store = Moving()
    engine = await many(store, 3)
    store.times = 100
    found = await read.read_ledger(engine, "alpha")
    assert found.consistent is False and "ledger_changed_during_read" in found.reasons
    assert found.revision < await engine.revision("alpha")


class Short(InMemoryDocumentStore):
    """A conforming pager that returns fewer rows than asked, mid-ledger."""

    async def page_facts(self, space, before_id, limit):
        return await super().page_facts(space, before_id, min(limit, 2))


async def test_a_short_page_is_not_the_end_of_the_ledger():
    """The port promises at most ``limit`` rows, not exactly that many; only
    an empty page ends the read."""
    engine = await many(Short(), 10)
    found = await read.read_ledger(engine, "alpha")
    assert [fact.fact_id for fact in found.facts] == list(range(1, 11))
    assert found.read_mode == "paged" and found.reasons == ()


class WrongShape(InMemoryDocumentStore):
    shape = "none"

    async def page_facts(self, space, before_id, limit):
        if self.shape == "none":
            return [None]
        if self.shape == "dict":
            return [{"fact_id": 1, "space": space}]
        real = await super().page_facts(space, before_id, limit)
        return (fact for fact in real)  # real facts, but a one-pass generator, not a list


@pytest.mark.parametrize("shape", ["none", "dict", "generator"])
async def test_a_page_of_things_that_are_not_facts_is_refused(shape):
    store = WrongShape()
    store.shape = shape
    engine = await many(store, 3)
    found = await read.read_ledger(engine, "alpha")
    assert found.read_mode == "unpaged" and "pager_rejected" in found.reasons and len(found.facts) == 3


class CountedFacts(dict):
    """Counts every fact a whole-ledger pass looks at."""
    visits = 0

    def values(self):
        for value in super().values():
            self.visits += 1
            yield value

    def items(self):
        for item in super().items():
            self.visits += 1
            yield item


async def test_paging_the_in_memory_ledger_never_scans_it():
    from scone_memory.core.ports import NewFact

    store = InMemoryDocumentStore()
    await MemoryEngine(store, InMemoryVectorIndex(), HashEmbedder()).open()
    for number in range(3000):
        await store.insert_fact(NewFact(space="alpha" if number % 3 else "beta", subject=f"person {number}",
                                        predicate="knows", object="Bob", valid_from="2025-01-01T00:00:00Z"))
    store._facts = CountedFacts(store._facts)
    first = await store.page_facts("alpha", None, 10)
    older = await store.page_facts("alpha", first[-1].fact_id, 10)
    assert store._facts.visits == 0 and len(first) == len(older) == 10
    assert [fact.fact_id for fact in first + older] == sorted((f.fact_id for f in first + older), reverse=True)
    assert {fact.space for fact in first + older} == {"alpha"}
