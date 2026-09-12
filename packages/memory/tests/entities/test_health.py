"""What in the graph wants attention, counted and named.

A knowledge graph goes wrong quietly: claims that rest on nothing, kinds
that disagree or are missing, an entity nothing links to, a predicate
used once. Each of those is countable, so they are counted and
shown with examples, and nothing is changed: this reads.
"""

from __future__ import annotations

import pytest

from scone_memory import HashEmbedder, InMemoryDocumentStore, InMemoryVectorIndex, MemoryEngine
from scone_memory.entities.health import HealthError, graph_health
from scone_memory.testing import Clock

DAY = "2024-01-01T00:00:00Z"


async def engine_with(*triples, **options) -> MemoryEngine:
    engine = await MemoryEngine(InMemoryDocumentStore(), InMemoryVectorIndex(), HashEmbedder(),
                                clock=Clock("2025-06-01T00:00:00.000Z"), **options).open()
    for subject, predicate, obj in triples:
        await engine.assert_fact("alpha", subject, predicate, obj, valid_from=DAY)
    return engine


def concerns(found) -> dict[str, int]:
    return {concern["kind"]: concern["count"] for concern in found.concerns}


async def test_a_claim_resting_on_nothing_is_counted_and_named():
    """A claim from a source with no quote cannot be checked against it;
    one with no source at all rests on whoever wrote it."""
    engine = await engine_with(("alice chen", "works_at", "Acme Robotics"))
    added = await engine.remember("alpha", "Alice Chen works at Acme Robotics, everyone says.")
    await engine.assert_fact("alpha", "bob stone", "works_at", "Globex", valid_from=DAY,
                             source_episode_id=added.episode_id)
    found = await graph_health(engine, "alpha")
    assert concerns(found)["ungrounded"] == 2
    ungrounded = next(c for c in found.concerns if c["kind"] == "ungrounded")
    assert any("bob stone works_at Globex" in example["claim"] for example in ungrounded["examples"])
    assert "ungrounded: 2 claims" in found.text


async def test_entities_whose_kinds_disagree_are_named():
    engine = await engine_with(("jordan", "works_at", "Acme"), ("sam", "employed_by", "Jordan"))
    found = await graph_health(engine, "alpha")
    assert concerns(found)["contested_kind"] == 1
    contested = next(c for c in found.concerns if c["kind"] == "contested_kind")
    assert contested["examples"][0]["label"].casefold() == "jordan"


async def test_an_entity_nothing_links_to_is_named():
    engine = await engine_with(("alice chen", "works_at", "Acme Robotics"), ("project atlas", "status", "ready"))
    found = await graph_health(engine, "alpha")
    assert concerns(found)["unconnected"] == 1
    assert "project atlas" in found.text


async def test_a_predicate_used_once_is_named_as_thin():
    engine = await engine_with(("alice chen", "works_at", "Acme"), ("bob stone", "works_at", "Globex"),
                               ("alice chen", "reticulates", "Splines"))
    found = await graph_health(engine, "alpha")
    thin = next(c for c in found.concerns if c["kind"] == "thin_predicate")
    assert thin["count"] == 1 and thin["examples"][0]["predicate"] == "reticulates"


async def test_an_entity_nothing_says_the_kind_of_is_named():
    """A kind comes from what the claims around an entity imply; when
    nothing implies one, that is worth knowing."""
    engine = await engine_with(("project atlas", "status", "ready"), ("alice chen", "works_at", "Acme Robotics"))
    found = await graph_health(engine, "alpha")
    nameless = next(c for c in found.concerns if c["kind"] == "kind_unknown")
    assert nameless["count"] == 1 and nameless["examples"][0]["label"] == "project atlas"


async def test_pairs_that_may_be_one_thing_are_carried_in():
    engine = await engine_with(("alice chen", "works_at", "Acme Robotics"), ("dr. alice chen", "leads", "Robotics Lab"))
    found = await graph_health(engine, "alpha")
    assert concerns(found)["likely_duplicate"] == 1
    pair = next(c for c in found.concerns if c["kind"] == "likely_duplicate")
    assert {pair["examples"][0]["a"], pair["examples"][0]["b"]} == {"alice chen", "dr. alice chen"}


async def test_a_graph_with_nothing_to_fix_says_so():
    """Quoted claims, kinds the claims imply, everything linked, and no
    predicate used once: nothing to report, and it says that."""
    engine = await engine_with()
    said = await engine.remember("alpha", "Alice Chen works at Acme Robotics. Bob Stone works at Acme Robotics.")
    for subject, quote in (("alice chen", "Alice Chen works at Acme Robotics"),
                           ("bob stone", "Bob Stone works at Acme Robotics")):
        await engine.assert_fact("alpha", subject, "works_at", "Acme Robotics", valid_from=DAY,
                                 source_episode_id=said.episode_id, quote=quote)
    found = await graph_health(engine, "alpha")
    assert found.status == "clean" and found.concerns == ()
    assert "result: nothing to fix" in found.text


async def test_each_concern_shows_no_more_examples_than_asked_for():
    """The count is of everything found; the examples are as many as the
    caller asked to see."""
    engine = await engine_with(*[(f"project {name}", "status", "ready") for name in ("atlas", "borealis", "cygnus")])
    found = await graph_health(engine, "alpha", limit=1)
    alone = next(concern for concern in found.concerns if concern["kind"] == "unconnected")
    assert alone["count"] == 3 and len(alone["examples"]) == 1


async def test_every_concern_is_bounded_and_the_read_is_said(monkeypatch):
    from scone_memory.entities import read

    monkeypatch.setattr(read, "MAX_FACTS", 1)
    engine = await engine_with(("alice chen", "works_at", "Acme"), ("bob stone", "works_at", "Globex"))
    found = await graph_health(engine, "alpha", limit=1)
    assert found.coverage["read"]["truncated"] is True
    assert "coverage: limited: " in found.text
    assert all(len(concern["examples"]) <= 1 for concern in found.concerns)


@pytest.mark.parametrize("options, message", [
    ({"limit": 0}, "limit"), ({"limit": 101}, "limit"), ({"max_bytes": 10}, "max_bytes"),
])
async def test_bounds_are_refused_before_anything_is_read(options, message):
    engine = await engine_with(("alice chen", "works_at", "Acme"))
    with pytest.raises(HealthError, match=message):
        await graph_health(engine, "alpha", **options)


async def test_the_record_is_what_every_surface_gives():
    engine = await engine_with(("alice chen", "works_at", "Acme Robotics"), ("project atlas", "status", "ready"))
    record = (await graph_health(engine, "alpha")).record("alpha", status="current", as_of="2025-06-01T00:00:00.000Z")
    assert record["schema_version"] == 1 and record["space"] == "alpha"
    assert record["status"] in ("clean", "concerns") and isinstance(record["concerns"], list)
    assert record["totals"]["entities"] == 3 and "text" in record and "coverage" in record


async def test_a_claim_whose_source_was_forgotten_no_longer_reads_as_grounded():
    """The projection records what a claim said when it was written. A
    source can be forgotten since, and then the claim rests on nothing:
    grounding is checked against the sources kept now."""
    engine = await engine_with()
    source = await engine.remember("alpha", "Alice works at Acme. Bob works at Globex.")
    for name, firm in (("Alice", "Acme"), ("Bob", "Globex")):
        await engine.assert_fact("alpha", name, "works_at", firm, valid_from=DAY,
                                 source_episode_id=source.episode_id, quote=f"{name} works at {firm}.")
    assert (await graph_health(engine, "alpha")).status == "clean"
    await engine.forget("alpha", source.episode_id)
    found = await graph_health(engine, "alpha")
    ungrounded = next(c for c in found.concerns if c["kind"] == "ungrounded")
    assert ungrounded["count"] == 2
    assert all(example["grounding"] == "quote_source_missing" for example in ungrounded["examples"])
    assert found.status == "concerns"


async def test_a_ledger_that_moves_while_health_is_read_is_said(monkeypatch):
    """Everything in one answer is read at one revision: the totals, the
    grounding and the duplicate pairs. A ledger that keeps moving while
    that is done is said, not hidden."""
    from scone_memory.entities import health as health_module

    engine = await engine_with()
    original = health_module.likely_duplicates
    added = []

    async def meanwhile(*arguments, **options):
        if not added:
            added.append(1)
            await engine.assert_fact("alpha", "Acme Robotics", "based_in", "Lisbon", valid_from=DAY)
            await engine.assert_fact("alpha", "AcmeRobotics", "based_in", "Lisbon", valid_from=DAY)
        return await original(*arguments, **options)

    monkeypatch.setattr(health_module, "likely_duplicates", meanwhile)
    found = await graph_health(engine, "alpha")
    # The second read sees the new facts, so it settles and reports them.
    assert found.totals["entities"] == 3 and concerns(found).get("likely_duplicate") == 1
    assert found.coverage["revision"] == (await engine.documents.revision("alpha"))


async def test_a_ledger_that_never_settles_is_said(monkeypatch):
    from scone_memory.entities import health as health_module

    engine = await engine_with(("alice chen", "works_at", "Acme Robotics"))
    original = health_module.likely_duplicates
    written = []

    async def always(*arguments, **options):
        written.append(1)
        await engine.assert_fact("alpha", f"person {len(written)}", "knows", "Someone New", valid_from=DAY)
        return await original(*arguments, **options)

    monkeypatch.setattr(health_module, "likely_duplicates", always)
    found = await graph_health(engine, "alpha")
    assert "ledger_moved_during_read" in found.coverage["reasons"]
    assert "coverage: limited: " in found.text and "ledger_moved_during_read" in found.text


async def test_pairs_read_at_another_revision_do_not_settle_the_answer(monkeypatch):
    """The duplicate pairs are evidence like the rest, so they must come
    from the revision the rest was read at."""
    from dataclasses import replace as _replace

    from scone_memory.entities import health as health_module

    engine = await engine_with(("alice chen", "works_at", "Acme Robotics"))
    original = health_module.likely_duplicates

    async def elsewhere(*arguments, **options):
        found = await original(*arguments, **options)
        return _replace(found, coverage={**found.coverage, "revision": 999})

    monkeypatch.setattr(health_module, "likely_duplicates", elsewhere)
    found = await graph_health(engine, "alpha")
    assert "ledger_moved_during_read" in found.coverage["reasons"]


async def test_a_capped_grounding_check_is_said(monkeypatch):
    from scone_memory.entities import health as health_module

    monkeypatch.setattr(health_module, "MAX_CHECKED", 1)
    engine = await engine_with()
    source = await engine.remember("alpha", "Alice works at Acme. Bob works at Globex.")
    for name, firm in (("Alice", "Acme"), ("Bob", "Globex")):
        await engine.assert_fact("alpha", name, "works_at", firm, valid_from=DAY,
                                 source_episode_id=source.episode_id, quote=f"{name} works at {firm}.")
    found = await graph_health(engine, "alpha")
    assert "grounding_checked 1 of 2" in found.coverage["reasons"]
    assert found.coverage["grounding_checked"] == 1 and "coverage: limited: " in found.text


async def test_each_concern_says_where_to_see_the_whole_of_it():
    """A count is useful beside the place that shows all of it."""
    engine = await engine_with(("alice chen", "works_at", "Acme Robotics"), ("project atlas", "status", "ready"))
    found = await graph_health(engine, "alpha")
    where = {concern["kind"]: concern["where"] for concern in found.concerns}
    assert where["ungrounded"] == "scone audit-grounding --flagged-only"
    assert where["unconnected"].startswith("/v1/graph/knowledge")
    assert "see scone audit-grounding --flagged-only" in found.text


def test_every_concern_points_at_something_that_exists():
    """A concern that names a route or a command nobody has is worse than
    one that names none, so every one of them is checked against the app
    and the parser."""
    import asyncio

    from scone_memory.api import create_app
    from scone_memory.entities.health import WHERE
    from scone_memory.runtime.cli import build_parser

    async def built():
        return await MemoryEngine(InMemoryDocumentStore(), InMemoryVectorIndex(), HashEmbedder()).open()

    app = create_app(asyncio.run(built()), {"key": "alpha"})
    routes = {getattr(route, "path", "") for route in app.routes}
    for kind, where in WHERE.items():
        if where.startswith("/v1/"):
            assert where.split(",")[0].replace("{id}", "{entity_id}") in routes, kind
        else:
            said = where.split()
            assert said[0] == "scone", kind
            build_parser().parse_args(said[1:])
