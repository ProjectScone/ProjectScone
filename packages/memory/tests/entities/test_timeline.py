"""One entity's claims in valid time: what held when, and what replaced it.

A timeline lists every fact an entity takes part in, as subject, as object
or through its values, in lanes by role and predicate, ordered by when each
held rather than when it was written. Supersession and stored links between
its facts are relations. A marker says which facts held at a chosen moment.
Every item is re-read, and the page reads again if the space moved.
"""
from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from scone_memory import HashEmbedder, InMemoryDocumentStore, InMemoryVectorIndex, MemoryEngine
from scone_memory.api import create_app
from scone_memory.core.ports import NewFactLink
from scone_memory.entities.timeline import timeline_view
from scone_memory.testing import Clock


def auth() -> dict:
    return {"Authorization": "Bearer key-a"}


@pytest.fixture
async def moves():
    engine = await MemoryEngine(InMemoryDocumentStore(), InMemoryVectorIndex(), HashEmbedder(),
                                clock=Clock("2025-06-01T00:00:00.000Z")).open()
    porto = await engine.assert_fact("alpha", "alice chen", "lives_in", "Porto", valid_from="2020-01-01T00:00:00Z")
    lisbon = await engine.assert_fact("alpha", "alice chen", "lives_in", "Lisbon", valid_from="2023-01-01T00:00:00Z")
    braga = await engine.assert_fact("alpha", "alice chen", "lives_in", "Braga", valid_from="2021-01-01T00:00:00Z")
    await engine.assert_fact("alpha", "alice chen", "lives_in", "Lisbon", valid_from="2023-01-01T00:00:00Z")
    joined = await engine.assert_fact("alpha", "alice chen", "joined_on", "May 2021", valid_from="2021-05-01T00:00:00Z")
    known = await engine.assert_fact("alpha", "bob stone", "knows", "alice chen", valid_from="2022-01-01T00:00:00Z")
    hidden = await engine.assert_fact("alpha", "alice chen", "met", "Mallory", valid_from="2022-06-01T00:00:00Z")
    await engine.exclude("alpha", hidden.fact_id, "private")
    yield engine, {"porto": porto, "lisbon": lisbon, "braga": braga, "joined": joined, "known": known,
                   "hidden": hidden}
    await engine.close()


async def test_a_chain_is_ordered_by_when_each_held_not_when_it_was_written(moves):
    engine, facts = moves
    view = await timeline_view(engine, "alpha", "alice chen")
    lane = next(lane for lane in view["lanes"] if lane["id"] == "subject:lives_in")
    assert lane["items"] == [facts["porto"].fact_id, facts["braga"].fact_id, facts["lisbon"].fact_id]
    superseded = {(r["from_fact"], r["to_fact"]) for r in view["relations"] if r["kind"] == "superseded_by"}
    assert superseded == {(facts["porto"].fact_id, facts["braga"].fact_id),
                          (facts["braga"].fact_id, facts["lisbon"].fact_id)}


async def test_a_restatement_is_one_item_and_every_role_has_its_lane(moves):
    engine, facts = moves
    view = await timeline_view(engine, "alpha", "alice chen")
    ids = [item["fact_id"] for item in view["items"]]
    assert len(ids) == len(set(ids)) == 6
    lanes = {lane["id"]: lane["items"] for lane in view["lanes"]}
    assert lanes["value:joined_on"] == [facts["joined"].fact_id]
    assert lanes["object:knows"] == [facts["known"].fact_id]
    hidden = next(item for item in view["items"] if item["fact_id"] == facts["hidden"].fact_id)
    assert hidden["excluded"] is True and hidden["holds_at_as_of"] is False


async def test_the_marker_says_what_held_at_the_moment_asked_for(moves):
    engine, facts = moves
    then = await timeline_view(engine, "alpha", "alice chen", as_of="2021-06-01T00:00:00Z")
    held = {item["fact_id"] for item in then["items"] if item["holds_at_as_of"]}
    assert facts["braga"].fact_id in held and facts["porto"].fact_id not in held
    assert facts["lisbon"].fact_id not in held and then["as_of"] == "2021-06-01T00:00:00.000Z"


async def test_stored_links_between_its_facts_are_relations(moves):
    engine, facts = moves
    await engine.documents.insert_fact_link(NewFactLink(space="alpha", from_fact=facts["known"].fact_id,
                                                        to_fact=facts["lisbon"].fact_id, kind="contradicts",
                                                        created_at="2025-01-01T00:00:00Z"))
    view = await timeline_view(engine, "alpha", "alice chen")
    assert {(r["kind"], r["from_fact"], r["to_fact"]) for r in view["relations"] if r["kind"] == "contradicts"} == {
        ("contradicts", facts["known"].fact_id, facts["lisbon"].fact_id)}


async def test_a_long_history_keeps_the_newest_and_says_so(moves):
    engine, _ = moves
    for year in range(2000, 2012):
        await engine.assert_fact("alpha", "alice chen", f"visited_{year}", "Faro", valid_from=f"{year}-01-01T00:00:00Z")
    view = await timeline_view(engine, "alpha", "alice chen", limit=5)
    assert len(view["items"]) == 5 and view["coverage"]["truncated"] is True
    assert view["coverage"]["items_total"] == 18 and "item_limit" in view["coverage"]["reasons"]
    assert min(item["valid_from"] for item in view["items"]) >= "2021"


async def test_the_timeline_route_resolves_names_and_refuses_ambiguity(moves):
    engine, facts = moves
    await engine.assert_fact("alpha", "alice park", "knows", "bob stone", valid_from="2024-01-01T00:00:00Z")
    with TestClient(create_app(engine, {"key-a": "alpha"})) as client:
        found = client.get("/v1/graph/timeline", params={"entity": "Alice Chen"}, headers=auth())
        ambiguous = client.get("/v1/graph/timeline", params={"entity": "alice"}, headers=auth())
        missing = client.get("/v1/graph/timeline", params={"entity": "nobody"}, headers=auth())
        features = client.get("/v1/capabilities", headers=auth()).json()["features"]
    assert found.status_code == 200 and found.json()["entity"]["key"] == "alice chen"
    assert ambiguous.status_code == 409 and len(ambiguous.json()["candidates"]) == 2
    assert missing.status_code == 404 and features["graph.timeline"] is True


class WritesOnReread(InMemoryDocumentStore):
    """Writes a new fact about the entity the first time an item is re-read."""
    engine = None
    armed = False

    async def get_fact(self, space, fact_id):
        if self.armed:
            self.armed = False
            await self.engine.assert_fact(space, "alice chen", "visited", "Faro", valid_from="2024-06-01T00:00:00Z")
        return await super().get_fact(space, fact_id)


async def test_a_timeline_reads_again_when_the_space_moves_while_it_reads():
    """A fact written while the timeline re-reads its items would be missing
    from a page that stopped at its first read; the fence reads again."""
    store = WritesOnReread()
    engine = await MemoryEngine(store, InMemoryVectorIndex(), HashEmbedder(),
                                clock=Clock("2025-06-01T00:00:00.000Z")).open()
    await engine.assert_fact("alpha", "alice chen", "lives_in", "Lisbon", valid_from="2023-01-01T00:00:00Z")
    store.engine, store.armed = engine, True
    view = await timeline_view(engine, "alpha", "alice chen")
    assert {item["predicate"] for item in view["items"]} == {"lives_in", "visited"} and view["consistent"] is True
