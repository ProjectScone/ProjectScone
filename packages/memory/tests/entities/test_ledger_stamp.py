"""A store's ledger stamp lets a moved revision skip the whole-ledger read.

Storing an episode moves the space's revision without touching a fact. A
store that can stamp its ledger (a value that changes whenever any fact
row changes) lets the projection cache restamp its views instead of
re-reading every fact. Whatever sequence of writes happens, the stamped
path must give exactly what a full re-read gives.
"""
from __future__ import annotations

import random

import pytest

from scone_memory import HashEmbedder, InMemoryDocumentStore, InMemoryVectorIndex, MemoryEngine
from scone_memory.core.graph_read import LedgerStamp
from scone_memory.core.timeutil import parse_rfc3339
from scone_memory.entities.project import project_entities
from scone_memory.entities.read import load_projection, read_ledger
from scone_memory.entities.view import counts
from scone_memory.testing import Clock

MOMENT = "2025-06-01T00:00:00.000Z"


@pytest.fixture(params=["memory", "sqlite"])
async def engine(request, tmp_path):
    if request.param == "memory":
        documents, vectors = InMemoryDocumentStore(), InMemoryVectorIndex()
    else:
        from scone_memory.backends import SqliteDocumentStore, SqliteVectorIndex

        documents, vectors = SqliteDocumentStore(tmp_path / "m.db"), SqliteVectorIndex(tmp_path / "m.db")
    engine = await MemoryEngine(documents, vectors, HashEmbedder(), clock=Clock(MOMENT)).open()
    yield engine
    await engine.close()


async def test_a_stamp_changes_with_every_fact_write_and_not_with_an_episode(engine):
    assert isinstance(engine.documents, LedgerStamp)
    first = await engine.documents.ledger_stamp("alpha")
    fact = await engine.assert_fact("alpha", "alice", "works_at", "Acme", valid_from="2024-01-01T00:00:00Z")
    added = await engine.documents.ledger_stamp("alpha")
    await engine.remember("alpha", "A note that asserts nothing.")
    assert await engine.documents.ledger_stamp("alpha") == added != first
    await engine.exclude("alpha", fact.fact_id, "private")
    assert await engine.documents.ledger_stamp("alpha") != added
    assert await engine.documents.ledger_stamp("beta") == await engine.documents.ledger_stamp("beta")


async def test_an_episode_bump_restamps_the_views_without_reading_the_ledger(engine):
    await engine.assert_fact("alpha", "alice", "works_at", "Acme", valid_from="2024-01-01T00:00:00Z")
    before, _ = await load_projection(engine, "alpha", mode="current")
    reads = []
    for name in ("page_facts", "list_facts"):
        real = getattr(engine.documents, name)

        async def spy(*args, _real=real, _name=name, **kwargs):
            reads.append(_name)
            return await _real(*args, **kwargs)

        setattr(engine.documents, name, spy)
    await engine.remember("alpha", "Another note that asserts nothing.")
    after, _ = await load_projection(engine, "alpha", mode="current")
    assert reads == [] and after.digest == before.digest and after.revision == await engine.revision("alpha")


async def full(engine, mode):
    ledger = await read_ledger(engine, "alpha")
    when = parse_rfc3339(MOMENT)
    return project_entities("alpha", [fact for fact in ledger.facts
                                      if counts(fact.status, fact.excluded, fact.valid_from, fact.valid_until,
                                                mode, when)], revision=ledger.revision).digest


@pytest.mark.parametrize("seed", range(25))
async def test_the_stamped_path_always_equals_a_full_reread(engine, seed):
    rng = random.Random(seed)
    names = ["alice", "bob", "carol", "acme", "globex"]
    facts = []
    for _step in range(8):
        choice = rng.choice(["assert", "propose", "approve", "close", "exclude", "include", "remember"])
        if choice in ("assert", "propose") or not facts:
            facts.append(await engine.assert_fact("alpha", rng.choice(names), rng.choice(["knows", "works_at"]),
                                                  rng.choice(names).title(), valid_from="2024-01-01T00:00:00Z",
                                                  proposed=choice == "propose"))
        elif choice == "remember":
            await engine.remember("alpha", f"note {rng.random()}")
        else:
            target = rng.choice(facts)
            try:
                if choice == "approve":
                    await engine.approve("alpha", target.fact_id)
                elif choice == "close":
                    await engine.close_fact("alpha", target.fact_id, "moved")
                elif choice == "exclude":
                    await engine.exclude("alpha", target.fact_id, "private")
                else:
                    await engine.include("alpha", target.fact_id)
            except Exception:  # noqa: BLE001 - an operation the ledger refuses changes nothing
                pass
        for mode in ("current", "all"):
            cached, _ = await load_projection(engine, "alpha", mode=mode, as_of=MOMENT)
            assert cached.digest == await full(engine, mode), (seed, choice, mode)
