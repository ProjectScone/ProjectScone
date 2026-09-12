"""An optional entity lane in recall: passages naming what a question is
about, or the things one relation away from it.

"Which city is Alice Chen's employer in?" shares no word with "Acme
Robotics is headquartered in Lisbon", so neither the text nor the vector
lane may find it. The ledger knows Alice Chen works at Acme Robotics; the
entity lane looks for passages naming Alice Chen or her neighbours and
fuses them in. It is off by default, adds exactly one lexical search, and
keeps every filter the other lanes keep.
"""
from __future__ import annotations

import pytest

from scone_memory import HashEmbedder, InMemoryDocumentStore, InMemoryVectorIndex, MemoryEngine
from scone_memory.testing import Clock

QUESTION = "Which city is Alice Chen's employer in?"
DAY = "2024-01-01T00:00:00Z"


class Counting(InMemoryDocumentStore):
    searches = 0

    async def search_text(self, space, query, limit, scope=None):
        self.searches += 1
        return await super().search_text(space, query, limit, scope)


@pytest.fixture
async def bridged():
    store = Counting()
    engine = await MemoryEngine(store, InMemoryVectorIndex(), HashEmbedder(), clock=Clock()).open()
    joined = await engine.remember("alpha", "Alice Chen joined Acme Robotics last spring.", tags=["public"])
    based = await engine.remember("alpha", "Acme Robotics is headquartered in Lisbon beside the river.", tags=["public"])
    for number in range(12):
        await engine.remember("alpha", f"City council meeting {number} discussed the employer survey in the city.",
                              tags=["public"])
    await engine.remember("alpha", "Acme Robotics keeps its private roadmap in a locked drawer.", tags=["private"])
    await engine.assert_fact("alpha", "alice chen", "works_at", "Acme Robotics", valid_from=DAY)
    await engine.assert_fact("alpha", "acme robotics", "based_in", "Lisbon", valid_from=DAY)
    yield engine, store, joined, based
    await engine.close()


def episodes(result) -> list[int]:
    return [item.episode_id for item in result.items]


async def test_the_lane_brings_in_a_passage_one_relation_away(bridged):
    engine, _, _, based = bridged
    plain = await engine.recall("alpha", QUESTION, limit=5, tags=["public"])
    boosted = await engine.recall("alpha", QUESTION, limit=5, tags=["public"], graph_boost=True)
    assert based.episode_id not in episodes(plain) and based.episode_id in episodes(boosted)
    item = next(item for item in boosted.items if item.episode_id == based.episode_id)
    assert "entity" in dict(item.lanes)
    assert {(entity.key, entity.role) for entity in boosted.entities} >= {("alice chen", "seed"),
                                                                          ("acme robotics", "neighbour")}


async def test_off_by_default_and_unchanged_when_off(bridged):
    engine, _, _, _ = bridged
    default = await engine.recall("alpha", QUESTION, limit=5, tags=["public"])
    off = await engine.recall("alpha", QUESTION, limit=5, tags=["public"], graph_boost=False)
    assert default.model_dump(exclude={"event_id"}) == off.model_dump(exclude={"event_id"})
    assert off.entities == [] and not any("entity" in dict(item.lanes) for item in off.items)


async def test_the_lane_costs_exactly_one_more_lexical_search(bridged):
    engine, store, _, _ = bridged
    before = store.searches
    await engine.recall("alpha", QUESTION, limit=5, tags=["public"])
    plain = store.searches - before
    before = store.searches
    await engine.recall("alpha", QUESTION, limit=5, tags=["public"], graph_boost=True)
    assert store.searches - before == plain + 1


async def test_the_lane_keeps_the_recall_filters(bridged):
    engine, _, _, _ = bridged
    boosted = await engine.recall("alpha", QUESTION, limit=10, tags=["public"], graph_boost=True)
    assert all("roadmap" not in item.text for item in boosted.items)


async def test_no_projection_in_time_leaves_the_other_lanes_and_says_so(bridged, monkeypatch):
    from scone_memory.entities import service

    engine, _, joined, _ = bridged

    async def building(*args, **kwargs):
        raise service.ProjectionBuilding("alpha")

    monkeypatch.setattr(engine.entities, "projection", building)
    result = await engine.recall("alpha", QUESTION, limit=5, tags=["public"], graph_boost=True)
    assert "entity: projection_building" in result.degraded and result.items and result.entities == []


async def test_the_recall_route_takes_the_boost_and_names_its_entities(bridged):
    from fastapi.testclient import TestClient

    from scone_memory.api import create_app

    engine, _, _, based = bridged
    with TestClient(create_app(engine, {"key-a": "alpha"})) as client:
        auth = {"Authorization": "Bearer key-a"}
        plain = client.get("/v1/recall", params={"q": QUESTION, "tags": "public"}, headers=auth).json()
        boosted = client.get("/v1/recall", params={"q": QUESTION, "tags": "public", "graph_boost": "true"},
                             headers=auth).json()
        features = client.get("/v1/capabilities", headers=auth).json()["features"]
    assert "entities" not in plain and {entity["key"] for entity in boosted["entities"]} >= {"alice chen"}
    assert based.episode_id in [item["episode_id"] for item in boosted["items"]]
    assert features["recall.graph_boost"] is True


async def test_a_passage_sharing_only_a_word_with_a_name_is_not_in_the_lane(bridged):
    engine, _, _, _ = bridged
    fair = await engine.remember("alpha", "The robotics fair drew record crowds downtown.", tags=["public"])
    boosted = await engine.recall("alpha", QUESTION, limit=20, tags=["public"], graph_boost=True)
    assert not any(item.episode_id == fair.episode_id and "entity" in dict(item.lanes) for item in boosted.items)


async def test_the_lane_ranks_second_hop_passages_before_first_hop_ones(bridged):
    """A passage naming the question's own entity is found by the other
    lanes anyway; the lane's first place goes to one naming only a neighbour."""
    from scone_memory.core.ports import TextFilter
    from scone_memory.entities.read import load_projection
    from scone_memory.retrieval.entity_lane import entity_lane

    engine, _, joined, based = bridged
    projection, _ = await load_projection(engine, "alpha", mode="current")
    lane, _ = await entity_lane(engine.documents, projection, "alpha", QUESTION, 20, TextFilter(tags=("public",)))
    owner = {chunk.chunk_id: chunk.episode_id
             for chunk in await engine.documents.get_chunks("alpha", [chunk_id for chunk_id, _ in lane])}
    assert [owner[chunk_id] for chunk_id, _ in lane][:2] == [based.episode_id, joined.episode_id]


async def test_the_bridge_benchmark_shows_the_lane_reaching_the_second_hop():
    from scone_memory.testing.entity_lane_benchmark import run_bridge_benchmark

    report = await run_bridge_benchmark(count=6)
    assert report.second_hop_recall_on > report.second_hop_recall_off
    assert report.first_hop_recall_on >= report.first_hop_recall_off


async def test_an_invalid_moment_is_refused_the_same_with_the_lane_on(bridged):
    from fastapi.testclient import TestClient

    from scone_memory.api import create_app

    engine, _, _, _ = bridged
    with TestClient(create_app(engine, {"key-a": "alpha"})) as client:
        auth = {"Authorization": "Bearer key-a"}
        for boost in ("false", "true"):
            response = client.get("/v1/recall", params={"q": QUESTION, "as_of": "yesterday", "graph_boost": boost},
                                  headers=auth)
            assert response.status_code == 422, boost


async def test_a_store_failure_building_the_graph_degrades_only_the_lane(bridged, monkeypatch):
    engine, store, _, _ = bridged

    async def broken(*args, **kwargs):
        raise ConnectionError("store unreachable")

    monkeypatch.setattr(store, "page_facts", broken)  # the graph's ledger read; recall's own reads still work
    result = await engine.recall("alpha", QUESTION, limit=5, tags=["public"], graph_boost=True)
    assert result.items and any(note.startswith("entity: ConnectionError") for note in result.degraded)


async def test_a_capped_graph_read_is_named_in_degraded(bridged, monkeypatch):
    from scone_memory.entities import read

    engine, _, _, _ = bridged
    monkeypatch.setattr(read, "MAX_FACTS", 1)
    result = await engine.recall("alpha", QUESTION, limit=5, tags=["public"], graph_boost=True)
    assert "entity: graph_read_capped fact_limit" in result.degraded


async def test_symbols_and_accents_that_make_a_name_are_kept():
    engine = await MemoryEngine(InMemoryDocumentStore(), InMemoryVectorIndex(), HashEmbedder(), clock=Clock()).open()
    await engine.assert_fact("alpha", "c++ labs", "based_in", "Lisbon", valid_from=DAY)
    await engine.assert_fact("alpha", "c# labs", "based_in", "Porto", valid_from=DAY)
    await engine.assert_fact("alpha", "jose", "knows", "Ana", valid_from=DAY)
    accented = await engine.remember("alpha", "Jose\u0301 visited the old market alone.")  # decomposed e-acute
    result = await engine.recall("alpha", "Where is C++ Labs based?", limit=5, graph_boost=True)
    assert {entity.key for entity in result.entities} == {"c++ labs", "lisbon"}
    other = await engine.recall("alpha", "Who does Jose know?", limit=10, graph_boost=True)
    assert not any(item.episode_id == accented.episode_id and "entity" in dict(item.lanes) for item in other.items)



def test_a_name_is_matched_as_whole_words_of_the_name_tokenizer():
    """Passages are read with the same words names and questions are: a
    combining mark belongs to its letter, and symbols inside a word are
    part of it, so neither "Jose" nor "C" nor "R" is found where it is not."""
    from scone_memory.retrieval.entity_lane import text_phrases

    assert "jose" not in text_phrases("Jose\u0301 mentions Chen")
    assert "jose" not in text_phrases("Jose\u0338 mentions Chen")
    assert "c" not in text_phrases("We write C++ daily") and "r" not in text_phrases("The R&D team met")
    assert "chen" in text_phrases("Jose\u0338 mentions Chen.") and "alice chen" in text_phrases("(Alice Chen) said")
    assert "c++" in text_phrases("We write C++ daily")



async def test_a_name_longer_than_six_words_is_still_matched():
    engine = await MemoryEngine(InMemoryDocumentStore(), InMemoryVectorIndex(), HashEmbedder(), clock=Clock()).open()
    long_name = "Royal Society for the Protection of Birds"
    await engine.assert_fact("alpha", "alice chen", "works_at", long_name, valid_from=DAY)
    await engine.assert_fact("alpha", long_name.casefold(), "based_in", "Sandy", valid_from=DAY)  # a subject: a thing
    passage = await engine.remember("alpha", f"The {long_name} opened a new wetland reserve.")
    result = await engine.recall("alpha", "Where does Alice Chen work?", limit=5, graph_boost=True)
    assert {entity.key for entity in result.entities} >= {"alice chen", long_name.casefold()}
    assert any(item.episode_id == passage.episode_id and "entity" in dict(item.lanes) for item in result.items)
    asked = await engine.recall("alpha", f"What does the {long_name} do?", limit=5, graph_boost=True)
    assert long_name.casefold() in {entity.key for entity in asked.entities if entity.role == "seed"}
