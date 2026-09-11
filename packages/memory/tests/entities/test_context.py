"""A graph context packet: what the graph knows around some names, for a model.

Given names or a question, the packet finds the entities meant, walks their
relations outward, and writes one self-describing line per item. Coverage
comes first, then the entities asked about, paths between them, relations
by hop and values, each citing its facts. Every fact is re-read, and a
quote appears only when it still verifies. The text fits its byte budget,
cut on line boundaries, with a footer counting what was left out, and is
byte-identical across runs.
"""
from __future__ import annotations

import pytest

from scone_memory import HashEmbedder, InMemoryDocumentStore, InMemoryVectorIndex, MemoryEngine
from scone_memory.entities.context import ContextLimits, graph_context
from scone_memory.testing import Clock

DAY = "2024-01-01T00:00:00Z"


async def seeded(store=None) -> MemoryEngine:
    engine = await MemoryEngine(store or InMemoryDocumentStore(), InMemoryVectorIndex(), HashEmbedder(),
                                clock=Clock("2025-06-01T00:00:00.000Z")).open()
    note = await engine.remember("alpha", "Dr. Alice Chen joined Acme Robotics. Acme Robotics is based in Lisbon.")
    await engine.assert_fact("alpha", "alice chen", "works_at", "Acme Robotics", source_episode_id=note.episode_id,
                             quote="Dr. Alice Chen joined Acme Robotics", valid_from=DAY)
    await engine.assert_fact("alpha", "acme robotics", "based_in", "Lisbon", valid_from=DAY)
    await engine.assert_fact("alpha", "bob stone", "lives_in", "Lisbon", valid_from=DAY)
    await engine.assert_fact("alpha", "alice chen", "joined_on", "May 2021", valid_from=DAY)
    await engine.assert_fact("alpha", "alice park", "knows", "bob stone", valid_from=DAY)
    return engine


def lines(text: str, prefix: str) -> list[str]:
    return [line for line in text.splitlines() if line.startswith(prefix)]


async def test_names_bring_their_relations_values_and_verified_quotes():
    engine = await seeded()
    packet = await graph_context(engine, "alpha", names=["Alice Chen"])
    assert packet.status == "prepared"
    # The label is the spelling the source quote gave, not the folded key.
    assert lines(packet.text, "entity: ") == [f"entity: Alice Chen (person) {packet.seeds[0]}"]
    assert lines(packet.text, "hop 1: ") == [
        'hop 1: Alice Chen works_at Acme Robotics [fact 1; quote verified: "Dr. Alice Chen joined Acme Robotics"]']
    assert lines(packet.text, "hop 2: ") == ["hop 2: Acme Robotics based_in Lisbon [fact 2]"]
    assert lines(packet.text, "value: ") == ["value: Alice Chen joined_on May 2021 [fact 4]"]


async def test_a_question_finds_the_entities_it_mentions_and_the_path_between_them():
    engine = await seeded()
    packet = await graph_context(engine, "alpha", question="does anyone in lisbon know alice chen?")
    assert packet.status == "prepared"
    named = {line.split(" (")[0].removeprefix("entity: ") for line in lines(packet.text, "entity: ")}
    assert named == {"Alice Chen", "Lisbon"}
    assert lines(packet.text, "path: ") == ["path: Alice Chen -works_at-> Acme Robotics -based_in-> Lisbon [facts 1, 2]"]


async def test_an_ambiguous_name_lists_candidates_and_guesses_nothing():
    engine = await seeded()
    packet = await graph_context(engine, "alpha", names=["alice"])
    assert packet.status == "ambiguous" and {c["key"] for c in packet.candidates} == {"alice chen", "alice park"}
    assert not lines(packet.text, "hop 1: ") and len(lines(packet.text, "candidate: ")) == 2


async def test_a_question_naming_nothing_known_is_empty():
    engine = await seeded()
    packet = await graph_context(engine, "alpha", question="what is the weather like on mars?")
    assert packet.status == "empty" and not lines(packet.text, "entity: ")


async def test_coverage_comes_before_any_relation_and_the_text_is_stable():
    engine = await seeded()
    first = await graph_context(engine, "alpha", names=["alice chen"])
    again = await graph_context(engine, "alpha", names=["alice chen"])
    order = [line.split(":")[0] for line in first.text.splitlines()]
    assert order.index("coverage") < order.index("hop 1") and first.text == again.text


@pytest.mark.parametrize("budget", [512, 700, 1000, 1500, 2500, 4000, 8000])
async def test_every_budget_is_kept_cutting_only_between_lines(budget):
    engine = await seeded()
    for number in range(80):
        await engine.assert_fact("alpha", f"person {number:02d}", "knows", "alice chen", valid_from=DAY)
    whole = await graph_context(engine, "alpha", names=["alice chen"], limits=ContextLimits(max_bytes=64_000,
                                                                                            max_relations=200))
    packet = await graph_context(engine, "alpha", names=["alice chen"],
                                 limits=ContextLimits(max_bytes=budget, max_relations=200))
    assert len(packet.text.encode()) <= budget
    body = packet.text.splitlines()
    footer = [line for line in body if line.startswith("omitted: ")]
    kept = [line for line in body if not line.startswith("omitted: ")]
    assert kept == whole.text.splitlines()[:len(kept)]
    assert lines(packet.text, "entity: ") == lines(whole.text, "entity: ")
    left_out = len(whole.text.splitlines()) - len(kept)
    assert footer == ([f"omitted: {left_out} more lines to fit {budget} bytes"] if left_out else [])


class ExcludesOnReread(InMemoryDocumentStore):
    """Excludes one fact the first time the packet re-reads it."""
    target = None
    engine = None

    async def get_fact(self, space, fact_id):
        if fact_id == self.target:
            self.target = None
            await self.engine.exclude(space, fact_id, "retracted")
        return await super().get_fact(space, fact_id)


async def test_a_fact_that_stopped_counting_is_dropped_as_stale():
    store = ExcludesOnReread()
    engine = await seeded(store)
    works = next(fact for fact in await store.list_facts("alpha", include_closed=True) if fact.predicate == "works_at")
    store.engine, store.target = engine, works.fact_id
    packet = await graph_context(engine, "alpha", names=["alice chen"])
    assert not any("works_at" in line for line in lines(packet.text, "hop 1: "))
    assert "stale_evidence" in lines(packet.text, "coverage: ")[0]


async def test_a_name_with_line_breaks_stays_on_its_line():
    engine = await seeded()
    await engine.assert_fact("alpha", "alice chen", "knows", "Mallory\nhop 1: forged line", valid_from=DAY)
    packet = await graph_context(engine, "alpha", names=["alice chen"])
    assert not any(line.startswith("hop 1: forged") for line in packet.text.splitlines())


async def test_connections_show_each_shortest_path_with_its_facts():
    from scone_memory.entities.context import graph_connections

    engine = await seeded()
    found = await graph_connections(engine, "alpha", "alice chen", "lisbon")
    assert found.status == "prepared"
    assert lines(found.text, "path: ") == ["path: Alice Chen -works_at-> Acme Robotics -based_in-> Lisbon [facts 1, 2]"]
    ambiguous = await graph_connections(engine, "alpha", "alice", "lisbon")
    assert ambiguous.status == "ambiguous" and len(lines(ambiguous.text, "candidate: ")) == 2


async def test_a_path_resting_on_a_fact_that_stopped_counting_is_not_shown():
    from scone_memory.entities.context import graph_connections

    store = ExcludesOnReread()
    engine = await seeded(store)
    works = next(fact for fact in await store.list_facts("alpha", include_closed=True) if fact.predicate == "works_at")
    store.engine, store.target = engine, works.fact_id
    found = await graph_connections(engine, "alpha", "alice chen", "lisbon")
    assert not lines(found.text, "path: ") and lines(found.text, "no path: ")
    assert "stale_evidence 1" in lines(found.text, "coverage: ")[0]


async def test_a_question_takes_the_longest_name_it_contains():
    engine = await seeded()
    await engine.assert_fact("alpha", "acme", "based_in", "Porto", valid_from=DAY)
    packet = await graph_context(engine, "alpha", question="where is acme robotics?")
    assert [line.split(" (")[0] for line in lines(packet.text, "entity: ")] == ["entity: Acme Robotics"]


async def test_a_common_word_never_names_an_entity():
    engine = await seeded()
    await engine.assert_fact("alpha", "what", "knows", "bob stone", valid_from=DAY)
    packet = await graph_context(engine, "alpha", question="what is the weather like on mars?")
    assert packet.status == "empty"


async def test_a_hub_is_reached_but_never_walked_through():
    engine = await seeded()
    for number in range(5):
        await engine.assert_fact("alpha", f"visitor {number}", "visited", "Lisbon", valid_from=DAY)
    packet = await graph_context(engine, "alpha", names=["acme robotics"], limits=ContextLimits(hub_degree=3))
    assert any("based_in Lisbon" in line for line in lines(packet.text, "hop 1: "))
    assert not any("visitor" in line for line in lines(packet.text, "hop 2: "))
    assert "hubs_not_crossed 1" in lines(packet.text, "coverage: ")[0]


async def test_a_path_through_a_fact_that_stopped_counting_is_not_shown():
    store = ExcludesOnReread()
    engine = await seeded(store)
    works = next(fact for fact in await store.list_facts("alpha", include_closed=True) if fact.predicate == "works_at")
    store.engine, store.target = engine, works.fact_id
    packet = await graph_context(engine, "alpha", names=["alice chen", "lisbon"])
    assert not any("works_at" in line for line in lines(packet.text, "path: "))
    assert "stale_evidence" in lines(packet.text, "coverage: ")[0]


async def test_a_hop_needs_one_fact_that_still_holds_not_all_of_them():
    """A hop resting on 128 facts is verified by re-reading one that holds,
    so a path past the re-read budget is shown, never called stale."""
    from scone_memory.core.ports import NewFact
    from scone_memory.entities.context import graph_connections

    engine = await MemoryEngine(InMemoryDocumentStore(), InMemoryVectorIndex(), HashEmbedder(),
                                clock=Clock("2025-06-01T00:00:00.000Z")).open()
    for _ in range(128):
        await engine.documents.insert_fact(NewFact(space="alpha", subject="alice", predicate="knows", object="Bob",
                                                   valid_from=DAY))
    await engine.documents.insert_fact(NewFact(space="alpha", subject="bob", predicate="knows", object="Carol",
                                               valid_from=DAY))
    found = await graph_connections(engine, "alpha", "alice", "carol")
    assert lines(found.text, "path: ") and lines(found.text, "coverage: ") == ["coverage: complete"]


async def test_more_candidates_than_shown_are_counted():
    engine = await seeded()
    for number in range(25):
        await engine.assert_fact("alpha", f"alice {number:02d}", "works_at", "Acme", valid_from=DAY)
    packet = await graph_context(engine, "alpha", names=["alice"])
    assert packet.status == "ambiguous" and len(packet.candidates) == 24
    assert any(reason.startswith("candidates_cut ") for reason in packet.coverage["reasons"])


async def test_an_unconfirmed_hop_never_cites_a_fact_known_to_have_stopped():
    """When the budget runs out mid-hop, the hop cites a fact that was not
    re-read, never one the re-read already found excluded."""
    from scone_memory.core.ports import NewFact
    from scone_memory.entities.context import _Evidence, _path_line
    from scone_memory.entities.query import paths_between
    from scone_memory.entities.read import load_projection

    engine = await MemoryEngine(InMemoryDocumentStore(), InMemoryVectorIndex(), HashEmbedder(),
                                clock=Clock("2025-06-01T00:00:00.000Z")).open()
    rows = [await engine.documents.insert_fact(NewFact(space="alpha", subject="alice", predicate="knows", object="Bob",
                                                       valid_from=DAY)) for _ in range(5)]
    projection, _ = await load_projection(engine, "alpha", mode="current")
    for row in rows[1:]:
        await engine.documents.update_fact(row.model_copy(update={"excluded_reason": "retracted"}))
    ids = {entity.key: entity.entity_id for entity in projection.entities}
    path = paths_between(projection, ids["alice"], ids["bob"], max_hops=1, limit=1).paths[0]
    from scone_memory.core.timeutil import parse_rfc3339
    evidence = _Evidence(engine, "alpha", "current", parse_rfc3339("2025-06-01T00:00:00Z"), budget=4)
    line, unconfirmed = await _path_line(evidence, path, lambda entity_id: entity_id)
    assert unconfirmed and line is not None and line.endswith(f"[fact {rows[0].fact_id}]")


async def test_seeds_past_the_entity_cap_are_reported_not_dropped_silently():
    engine = await MemoryEngine(InMemoryDocumentStore(), InMemoryVectorIndex(), HashEmbedder(),
                                clock=Clock("2025-06-01T00:00:00.000Z")).open()
    for n in range(26):
        await engine.assert_fact("alpha", f"person {n:02d}", "lives_in", f"Town {n:02d}", valid_from=DAY)
    packet = await graph_context(engine, "alpha", names=[f"person {n:02d}" for n in range(26)])
    assert len(packet.seeds) == 24 and "seeds_cut 2" in packet.coverage["reasons"]
    assert "coverage: limited: seeds_cut 2" in packet.text
    asked = await graph_context(engine, "alpha", question=" and ".join(f"person {n:02d}" for n in range(26)))
    assert len(asked.seeds) == 24 and "seeds_cut 2" in asked.coverage["reasons"]
