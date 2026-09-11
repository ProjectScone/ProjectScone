"""The knowledge map over HTTP: entities and how they relate, with the
facts behind every relation, what each view left out, and nothing a
bounded sample could be mistaken for."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from scone_memory import HashEmbedder, InMemoryDocumentStore, InMemoryVectorIndex, MemoryEngine
from scone_memory.api import create_app


def auth(key: str = "key-a") -> dict:
    return {"Authorization": f"Bearer {key}"}


@pytest.fixture
async def seeded():
    engine = await MemoryEngine(InMemoryDocumentStore(), InMemoryVectorIndex(), HashEmbedder()).open()
    works = await engine.assert_fact("alpha", "alice chen", "works_at", "Acme Robotics", valid_from="2024-01-01T00:00:00Z")
    await engine.assert_fact("alpha", "acme robotics", "based_in", "Lisbon", valid_from="2024-01-01T00:00:00Z")
    await engine.assert_fact("alpha", "alice chen", "joined_on", "May 2021", valid_from="2024-01-01T00:00:00Z")
    old = await engine.assert_fact("alpha", "alice chen", "lived_in", "Porto", valid_from="2020-01-01T00:00:00Z")
    await engine.close_fact("alpha", old.fact_id, "moved")
    await engine.assert_fact("alpha", "alice chen", "knows", "Bob", proposed=True)
    hidden = await engine.assert_fact("alpha", "alice chen", "met", "Mallory", valid_from="2024-01-01T00:00:00Z")
    await engine.exclude("alpha", hidden.fact_id, "private")
    await engine.assert_fact("beta", "zed", "works_at", "Globex", valid_from="2024-01-01T00:00:00Z")
    app = create_app(engine, {"key-a": "alpha", "key-b": "beta"})
    with TestClient(app) as client:
        yield client, works


def labels(view: dict) -> dict[str, str]:
    return {entity["id"]: entity["label"] for entity in view["entities"]}


def edges(view: dict) -> set[tuple[str, str, str]]:
    names = labels(view)
    return {(names[r["subject_id"]], r["predicate"], names[r["object_id"]]) for r in view["relations"]}


def test_the_knowledge_map_shows_things_and_the_facts_behind_each_relation(seeded):
    client, works = seeded
    view = client.get("/v1/graph/knowledge", headers=auth()).json()
    assert edges(view) == {("alice chen", "works_at", "Acme Robotics"), ("Acme Robotics", "based_in", "Lisbon")}
    relation = next(r for r in view["relations"] if r["predicate"] == "works_at")
    assert relation["fact_ids"] == [works.fact_id]
    assert relation["support"]["active"] == 1 and relation["support"]["unsourced"] == 1
    assert {(a["predicate"], a["value"], a["literal_kind"]) for a in view["attributes"]} == {("joined_on", "May 2021", "date")}
    assert view["projection"]["version"] == "scone.entities/1" and len(view["projection"]["digest"]) == 64
    assert view["filters"] == {"status": "current", "as_of": view["filters"]["as_of"]}
    assert view["coverage"]["truncated"] is False and view["coverage"]["reasons"] == []


@pytest.mark.parametrize(("status", "shown", "hidden"), [
    ("current", {"works_at"}, {"lived_in", "knows", "met"}),
    ("history", {"works_at", "lived_in"}, {"knows", "met"}),
    ("proposed", {"knows"}, {"works_at", "lived_in", "met"}),
    ("all", {"works_at", "lived_in", "knows", "met"}, set()),
])
def test_each_status_mode_shows_what_it_says(seeded, status, shown, hidden):
    client, _ = seeded
    view = client.get("/v1/graph/knowledge", params={"status": status}, headers=auth()).json()
    predicates = {r["predicate"] for r in view["relations"]}
    assert shown <= predicates and not hidden & predicates
    assert view["filters"]["status"] == status


def test_as_of_shows_what_held_then(seeded):
    client, _ = seeded
    view = client.get("/v1/graph/knowledge", params={"as_of": "2021-06-01T00:00:00Z"}, headers=auth()).json()
    assert {r["predicate"] for r in view["relations"]} == {"lived_in"}


def test_a_budget_is_reported_instead_of_passing_for_the_whole_graph(seeded):
    client, _ = seeded
    view = client.get("/v1/graph/knowledge", params={"limit": 2}, headers=auth()).json()
    coverage = view["coverage"]
    assert len(view["entities"]) == 2 and coverage["entities_shown"] == 2 and coverage["entities_total"] == 3
    assert coverage["truncated"] is True and "entity_limit" in coverage["reasons"]
    shown = {entity["id"] for entity in view["entities"]}
    assert all(r["subject_id"] in shown and r["object_id"] in shown for r in view["relations"])


def test_entities_list_their_names_and_filter_by_text(seeded):
    client, _ = seeded
    listed = client.get("/v1/entities", headers=auth()).json()
    assert {entity["key"] for entity in listed["entities"]} == {"alice chen", "acme robotics", "lisbon"}
    found = client.get("/v1/entities", params={"q": "ACME"}, headers=auth()).json()
    assert [entity["label"] for entity in found["entities"]] == ["Acme Robotics"]
    assert found["entities"][0]["names"][0] == {"text": "Acme Robotics", "count": 1}


def test_the_map_is_per_space_and_needs_a_key(seeded):
    client, _ = seeded
    assert client.get("/v1/graph/knowledge").status_code == 401
    assert client.get("/v1/entities").status_code == 401
    beta = client.get("/v1/graph/knowledge", headers=auth("key-b")).json()
    assert edges(beta) == {("zed", "works_at", "Globex")}


def test_bad_parameters_are_refused(seeded):
    client, _ = seeded
    assert client.get("/v1/graph/knowledge", params={"status": "maybe"}, headers=auth()).status_code == 422
    assert client.get("/v1/graph/knowledge", params={"limit": 0}, headers=auth()).status_code == 422
    assert client.get("/v1/graph/knowledge", params={"as_of": "yesterday"}, headers=auth()).status_code == 422


def test_capabilities_advertise_the_entity_routes(seeded):
    client, _ = seeded
    features = client.get("/v1/capabilities", headers=auth()).json()["features"]
    assert features["entities.read"] is True and features["graph.knowledge"] is True


async def test_a_view_is_classified_only_from_the_facts_it_counts():
    engine = await MemoryEngine(InMemoryDocumentStore(), InMemoryVectorIndex(), HashEmbedder()).open()
    await engine.assert_fact("alpha", "alice", "status", "tired", valid_from="2020-01-01T00:00:00Z")
    hidden = await engine.assert_fact("alpha", "alice", "works_at", "Acme", valid_from="2020-01-01T00:00:00Z")
    await engine.exclude("alpha", hidden.fact_id, "private")
    await engine.assert_fact("alpha", "tired", "knows", "Bob", valid_from="2030-01-01T00:00:00Z")
    with TestClient(create_app(engine, {"key-a": "alpha"})) as client:
        view = client.get("/v1/graph/knowledge", params={"as_of": "2021-01-01T00:00:00Z"}, headers=auth()).json()
        listed = client.get("/v1/entities", params={"as_of": "2021-01-01T00:00:00Z"}, headers=auth()).json()
    assert view["relations"] == []
    assert [(a["predicate"], a["value"]) for a in view["attributes"]] == [("status", "tired")]
    alice = next(entity for entity in view["entities"] if entity["key"] == "alice")
    assert alice["kind"] is None and hidden.fact_id not in alice["kind_basis"]
    assert [entity["key"] for entity in listed["entities"]] == ["alice"]


@pytest.fixture
async def teams():
    engine = await MemoryEngine(InMemoryDocumentStore(), InMemoryVectorIndex(), HashEmbedder()).open()
    for team in (("Ana", "Ben", "Cho", "Dev"), ("Eli", "Fay", "Gus", "Hal")):
        for left in team:
            for right in team:
                if left < right:
                    await engine.assert_fact("alpha", left, "knows", right, valid_from="2024-01-01T00:00:00Z")
    bridge = await engine.assert_fact("alpha", "Dev", "mentors", "Eli", valid_from="2024-01-01T00:00:00Z")
    with TestClient(create_app(engine, {"key-a": "alpha", "key-b": "beta"})) as client:
        yield client, bridge


def test_the_report_names_communities_central_entities_and_surprises(teams):
    client, bridge = teams
    report = client.get("/v1/graph/report", headers=auth()).json()
    assert report["analysis"]["version"] == "scone.analysis/1" and report["analysis"]["modularity"] > 0.3
    assert len(report["communities"]) == 2 and all(c["label"] for c in report["communities"])
    assert report["surprising_connections"][0]["fact_ids"] == [bridge.fact_id]
    assert {e["key"] for e in report["central_entities"][:2]} <= {"dev", "eli", "ana", "ben", "cho", "fay", "gus", "hal"}
    assert any(q["kind"] == "connection" and q["fact_ids"] == [bridge.fact_id] for q in report["suggestions"])
    assert report["projection"]["digest"] and report["coverage"]["truncated"] is False


def test_the_report_reads_as_markdown(teams):
    client, _ = teams
    response = client.get("/v1/graph/report", params={"format": "markdown"}, headers=auth())
    assert response.status_code == 200 and response.headers["content-type"].startswith("text/markdown")
    text = response.text
    assert text.startswith("# Knowledge report") and "## Communities" in text and "## Surprising connections" in text
    assert "Dev" in text and "Eli" in text


def test_groupings_are_computed_and_kept_apart_from_recorded_relations(teams):
    client, _ = teams
    plain = client.get("/v1/graph/knowledge", headers=auth()).json()
    assert "groupings" not in plain
    view = client.get("/v1/graph/knowledge", params={"groupings": "true", "limit": 6}, headers=auth()).json()
    groupings = view["groupings"]
    assert groupings["basis"] == "computed" and groupings["method"] == "scone.analysis/1"
    shown = {entity["id"] for entity in view["entities"]}
    assert set(groupings["membership"]) == shown
    assert {item["entity_id"] for item in groupings["importance"]} == shown
    assert all(set(community["members"]) <= shown for community in groupings["communities"])


def test_the_report_is_per_space_and_advertised(teams):
    client, _ = teams
    assert client.get("/v1/graph/report").status_code == 401
    beta = client.get("/v1/graph/report", headers=auth("key-b")).json()
    assert beta["communities"] == [] and beta["surprising_connections"] == []
    assert client.get("/v1/graph/report", params={"format": "pdf"}, headers=auth()).status_code == 422
    assert client.get("/v1/capabilities", headers=auth()).json()["features"]["graph.report"] is True
