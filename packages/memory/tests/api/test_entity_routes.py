"""The knowledge map over HTTP: entities and how they relate, with the
facts behind every relation, what each view left out, and nothing a
bounded sample could be mistaken for."""

from __future__ import annotations

import asyncio
import time

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
    assert view["coverage"]["read_mode"] == "paged"


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
    assert groupings["coverage"] == {"entities_total": 8, "entities_analysed": 8, "isolated_entities": 0,
                                     "truncated": False, "reasons": [], "betweenness": "exact",
                                     "betweenness_estimated": False, "levels": groupings["coverage"]["levels"],
                                     "resolution": 1.0}


async def test_estimated_betweenness_is_disclosed_wherever_it_is_shown():
    """Past the exact limit, betweenness is estimated. The groupings and both
    report forms say so, apart from the view's paging and the analysis caps."""
    engine = await MemoryEngine(InMemoryDocumentStore(), InMemoryVectorIndex(), HashEmbedder()).open()
    for number in range(500):
        await engine.assert_fact("alpha", f"person{number:03d}", "knows", "hub", valid_from="2024-01-01T00:00:00Z")
    with TestClient(create_app(engine, {"key-a": "alpha"})) as client:
        view = client.get("/v1/graph/knowledge", params={"groupings": "true", "limit": 1000}, headers=auth()).json()
        report = client.get("/v1/graph/report", headers=auth()).json()
        markdown = client.get("/v1/graph/report", params={"format": "markdown"}, headers=auth()).text
    analysed = view["groupings"]["coverage"]
    assert analysed["betweenness"] == "sampled:64" and analysed["betweenness_estimated"] is True
    assert analysed["entities_analysed"] == 501 and analysed["truncated"] is False
    assert view["coverage"]["truncated"] is False and view["coverage"]["reasons"] == []
    assert max(item["betweenness"] for item in view["groupings"]["importance"]) <= 1.0
    assert report["analysis"]["coverage"] == analysed
    assert "Betweenness (estimated)" in markdown and "estimated from 64 sampled sources" in markdown


def test_the_report_is_per_space_and_advertised(teams):
    client, _ = teams
    assert client.get("/v1/graph/report").status_code == 401
    beta = client.get("/v1/graph/report", headers=auth("key-b")).json()
    assert beta["communities"] == [] and beta["surprising_connections"] == []
    assert client.get("/v1/graph/report", params={"format": "pdf"}, headers=auth()).status_code == 422
    features = client.get("/v1/capabilities", headers=auth()).json()["features"]
    assert features["graph.report"] is True and features["graph.path"] is True


@pytest.fixture
async def quoted():
    engine = await MemoryEngine(InMemoryDocumentStore(), InMemoryVectorIndex(), HashEmbedder()).open()
    note = await engine.remember("alpha", "Dr. Alice Chen joined Acme Robotics. Acme Robotics is based in Lisbon.")
    works = await engine.assert_fact("alpha", "alice chen", "works_at", "Acme Robotics", source_episode_id=note.episode_id,
                                     quote="Dr. Alice Chen joined Acme Robotics", valid_from="2024-01-01T00:00:00Z")
    based = await engine.assert_fact("alpha", "acme robotics", "based_in", "Lisbon", valid_from="2024-01-01T00:00:00Z")
    await engine.assert_fact("alpha", "bob stone", "lives_in", "Lisbon", valid_from="2024-01-01T00:00:00Z")
    await engine.assert_fact("alpha", "alice park", "joined_on", "May 2021", valid_from="2024-01-01T00:00:00Z")
    with TestClient(create_app(engine, {"key-a": "alpha", "key-b": "beta"})) as client:
        yield client, works, based


def test_a_name_resolves_or_reports_ambiguity(quoted):
    client, _, _ = quoted
    resolved = client.get("/v1/entities/resolve", params={"name": "ACME Robotics"}, headers=auth()).json()
    assert resolved["status"] == "resolved" and resolved["tier"] == "key"
    assert [c["label"] for c in resolved["candidates"]] == ["Acme Robotics"]
    ambiguous = client.get("/v1/entities/resolve", params={"name": "alice"}, headers=auth()).json()
    assert ambiguous["status"] == "ambiguous" and len(ambiguous["candidates"]) == 2


def test_an_entity_page_groups_relations_and_rechecks_quotes(quoted):
    client, works, based = quoted
    acme = client.get("/v1/entities/resolve", params={"name": "acme robotics"}, headers=auth()).json()["candidates"][0]["id"]
    page = client.get(f"/v1/entities/{acme}", headers=auth()).json()
    assert page["entity"]["label"] == "Acme Robotics"
    assert [(group["predicate"], [r["subject"]["label"] for r in group["relations"]]) for group in page["incoming"]] == \
        [("works_at", ["Alice Chen"])]
    assert [(group["predicate"], [r["object"]["label"] for r in group["relations"]]) for group in page["outgoing"]] == \
        [("based_in", ["Lisbon"])]
    facts = {fact["fact_id"]: fact for fact in page["facts"]}
    assert facts[works.fact_id]["grounding"] == "quote_verified"
    assert facts[based.fact_id]["grounding"] == "stated"
    assert client.get("/v1/entities/ent:000000000000000000000000", headers=auth()).status_code == 404


def test_a_path_joins_two_names_and_refuses_an_ambiguous_one(quoted):
    client, works, based = quoted
    route = client.get("/v1/graph/path", params={"from": "Alice Chen", "to": "Bob Stone"}, headers=auth()).json()
    assert route["status"] == "found"
    hops = route["paths"][0]["hops"]
    assert [(hop["predicate"], hop["direction"]) for hop in hops] == [("works_at", "forward"), ("based_in", "forward"),
                                                                      ("lives_in", "reverse")]
    assert hops[0]["fact_ids"] == [works.fact_id] and hops[0]["subject"]["label"] == "Alice Chen"
    ambiguous = client.get("/v1/graph/path", params={"from": "alice", "to": "Bob Stone"}, headers=auth())
    assert ambiguous.status_code == 409 and len(ambiguous.json()["candidates"]) == 2
    assert client.get("/v1/graph/path", params={"from": "Globex", "to": "Bob Stone"}, headers=auth()).status_code == 404
    assert client.get("/v1/graph/path", params={"from": "Alice Chen", "to": "Bob Stone"}).status_code == 401


@pytest.mark.parametrize("format", ["json", "graphml", "cypher", "csv", "jsonld", "obsidian"])
def test_the_graph_downloads_in_each_format_named_by_its_projection(seeded, format):
    client, _ = seeded
    digest = client.get("/v1/graph/knowledge", headers=auth()).json()["projection"]["digest"]
    response = client.get("/v1/graph/export", params={"format": format}, headers=auth())
    assert response.status_code == 200 and response.content
    assert response.headers["x-scone-projection-digest"] == digest
    assert response.headers["content-disposition"].startswith("attachment; filename=")
    assert response.headers["x-scone-truncated"] == "false"


def test_an_export_holds_what_its_view_counts_and_says_what_it_read(seeded):
    client, works = seeded
    current = client.get("/v1/graph/export", headers=auth()).json()
    history = client.get("/v1/graph/export", params={"status": "history"}, headers=auth()).json()
    assert {link["predicate"] for link in current["links"]} == {"works_at", "based_in"}
    assert "lived_in" in {link["predicate"] for link in history["links"]}
    assert all("met" != link["predicate"] for link in history["links"])
    about = current["graph"]["about"]
    assert about["status"] == "current" and about["as_of"]
    assert about["coverage"]["truncated"] is False and about["coverage"]["facts_counted"] >= 3
    assert client.get("/v1/graph/export", params={"format": "pdf"}, headers=auth()).status_code == 422
    assert client.get("/v1/graph/export").status_code == 401
    assert client.get("/v1/capabilities", headers=auth()).json()["features"]["graph.export"] is True


async def test_an_export_of_a_capped_read_is_marked_partial(monkeypatch):
    from scone_memory.entities import read

    monkeypatch.setattr(read, "MAX_FACTS", 2)
    engine = await MemoryEngine(InMemoryDocumentStore(), InMemoryVectorIndex(), HashEmbedder()).open()
    for left, right in (("Ana", "Ben"), ("Ben", "Cho"), ("Cho", "Dev")):
        await engine.assert_fact("alpha", left, "knows", right, valid_from="2024-01-01T00:00:00Z")
    with TestClient(create_app(engine, {"key-a": "alpha"})) as client:
        response = client.get("/v1/graph/export", headers=auth())
    assert response.headers["x-scone-truncated"] == "true"
    assert response.json()["graph"]["about"]["coverage"]["reasons"] == ["fact_limit"]


class WritesDuringReread(InMemoryDocumentStore):
    """Runs a write the moment the entity page re-reads a fact."""
    during = None
    every_time = False

    async def get_fact(self, space, fact_id):
        if self.during is not None:
            during = self.during
            if not self.every_time:
                self.during = None
            await during()
        return await super().get_fact(space, fact_id)


async def test_an_entity_page_rebuilds_when_the_ledger_moves_while_it_reads():
    """An exclusion that lands mid-page would leave relations counting a
    fact the page's own re-read shows excluded. The page reads again, and
    says so when the space will not hold still."""
    from scone_memory.entities.ids import key_id

    store = WritesDuringReread()
    engine = await MemoryEngine(store, InMemoryVectorIndex(), HashEmbedder()).open()
    note = await engine.remember("alpha", "Alice works at Acme.")
    await engine.assert_fact("alpha", "alice", "based_in", "Lisbon", valid_from="2024-01-01T00:00:00Z")
    works = await engine.assert_fact("alpha", "alice", "works_at", "Acme", source_episode_id=note.episode_id,
                                     quote="Alice works at Acme.", valid_from="2024-01-01T00:00:00Z")
    store.during = lambda: engine.exclude("alpha", works.fact_id, "no longer evidence")
    with TestClient(create_app(engine, {"key-a": "alpha"})) as client:
        page = client.get(f"/v1/entities/{key_id('alpha', 'alice')}", headers=auth()).json()
        assert [group["predicate"] for group in page["outgoing"]] == ["based_in"]
        assert [fact["excluded"] for fact in page["facts"]] == [False] and page["consistent"] is True
        store.every_time = True
        store.during = lambda: store.bump_revision("alpha")
        moving = client.get(f"/v1/entities/{key_id('alpha', 'alice')}", headers=auth()).json()
    assert moving["consistent"] is False and "ledger_changed_during_read" in moving["coverage"]["reasons"]


@pytest.fixture
async def capped(monkeypatch):
    """A chain a-b-c-d read with room for only the two newest facts."""
    from scone_memory.entities import read

    engine = await MemoryEngine(InMemoryDocumentStore(), InMemoryVectorIndex(), HashEmbedder()).open()
    for left, right in (("a", "b"), ("b", "c"), ("c", "d")):
        await engine.assert_fact("alpha", left, "knows", right, valid_from="2024-01-01T00:00:00Z")
    monkeypatch.setattr(read, "MAX_FACTS", 2)
    with TestClient(create_app(engine, {"key-a": "alpha"})) as client:
        yield client


def test_a_capped_read_never_claims_a_name_is_absent(capped):
    found = capped.get("/v1/entities/resolve", params={"name": "a"}, headers=auth()).json()
    assert found["status"] == "not_found_in_read" and found["complete"] is False
    assert found["coverage"]["reasons"] == ["fact_limit"] and found["coverage"]["truncated"] is True


def test_a_capped_read_never_claims_two_entities_are_disconnected(capped):
    unknown = capped.get("/v1/graph/path", params={"from": "a", "to": "d"}, headers=auth())
    assert unknown.status_code == 404 and unknown.json()["complete"] is False
    assert unknown.json()["coverage"]["reasons"] == ["fact_limit"]
    # b-c and c-d were read, so b to d is found; the result still says the read was capped.
    found = capped.get("/v1/graph/path", params={"from": "b", "to": "d"}, headers=auth()).json()
    assert found["status"] == "found" and found["complete"] is False


def test_a_capped_read_marks_an_entity_page_incomplete(capped):
    from scone_memory.entities.ids import key_id

    page = capped.get(f"/v1/entities/{key_id('alpha', 'b')}", headers=auth()).json()
    assert page["incoming"] == [] and page["complete"] is False
    assert page["coverage"]["truncated"] is True and "fact_limit" in page["coverage"]["reasons"]


async def test_many_candidates_are_counted_and_the_cut_is_said():
    engine = await MemoryEngine(InMemoryDocumentStore(), InMemoryVectorIndex(), HashEmbedder()).open()
    for letter in "ABCDEFGHIJKLMNOPQRSTU":
        await engine.assert_fact("alpha", f"alice {letter}", "knows", "bob", valid_from="2024-01-01T00:00:00Z")
    with TestClient(create_app(engine, {"key-a": "alpha"})) as client:
        first = client.get("/v1/entities/resolve", params={"name": "alice"}, headers=auth()).json()
        every = client.get("/v1/entities/resolve", params={"name": "alice", "limit": 50}, headers=auth()).json()
        path = client.get("/v1/graph/path", params={"from": "alice", "to": "bob"}, headers=auth())
    assert first["status"] == "ambiguous" and len(first["candidates"]) == 20
    assert first["candidates_total"] == 21 and first["truncated"] is True
    assert len(every["candidates"]) == 21 and every["truncated"] is False
    assert path.status_code == 409 and path.json()["candidates_total"] == 21 and path.json()["truncated"] is True


async def test_a_quote_is_verified_only_against_its_own_source():
    """A store that hands back another space's or another id's episode must
    not have the quote checked against it."""
    from scone_memory.entities.ids import key_id

    store = InMemoryDocumentStore()
    engine = await MemoryEngine(store, InMemoryVectorIndex(), HashEmbedder()).open()
    note = await engine.remember("alpha", "Alice works at Acme.")
    await engine.assert_fact("alpha", "alice", "works_at", "Acme", source_episode_id=note.episode_id,
                             quote="Alice works at Acme.", valid_from="2024-01-01T00:00:00Z")
    real = store.get_episode

    async def someone_elses(space, episode_id):
        episode = await real(space, episode_id)
        return episode.model_copy(update={"space": "beta", "episode_id": episode_id + 100})

    store.get_episode = someone_elses
    with TestClient(create_app(engine, {"key-a": "alpha"})) as client:
        page = client.get(f"/v1/entities/{key_id('alpha', 'alice')}", headers=auth()).json()
    assert [fact["grounding"] for fact in page["facts"]] == ["quote_source_mismatch"]


class WritesAfterListing(InMemoryDocumentStore):
    """Runs a write just after the whole-ledger read returns its rows,
    paged or listed."""
    after = None

    async def _then(self, rows):
        if self.after is not None:
            after, self.after = self.after, None
            await after()
        return rows

    async def list_facts(self, space, **options):
        return await self._then(await super().list_facts(space, **options))

    async def page_facts(self, space, before_id, limit):
        return await self._then(await super().page_facts(space, before_id, limit))


async def test_a_write_right_after_the_ledger_read_is_not_mistaken_for_the_read():
    """The projection records the revision from before its rows; read after
    them, it would vouch for rows that no longer hold."""
    from scone_memory.entities.ids import key_id

    store = WritesAfterListing()
    engine = await MemoryEngine(store, InMemoryVectorIndex(), HashEmbedder()).open()
    await engine.assert_fact("alpha", "alice", "based_in", "Lisbon", valid_from="2024-01-01T00:00:00Z")
    works = await engine.assert_fact("alpha", "alice", "works_at", "Acme", valid_from="2024-01-01T00:00:00Z")
    store.after = lambda: engine.exclude("alpha", works.fact_id, "no longer evidence")
    with TestClient(create_app(engine, {"key-a": "alpha"})) as client:
        page = client.get(f"/v1/entities/{key_id('alpha', 'alice')}", headers=auth()).json()
    assert [group["predicate"] for group in page["outgoing"]] == ["based_in"] and page["consistent"] is True


async def test_entities_joined_only_beyond_a_capped_read_are_not_called_disconnected(monkeypatch):
    from scone_memory.entities import read

    engine = await MemoryEngine(InMemoryDocumentStore(), InMemoryVectorIndex(), HashEmbedder()).open()
    for left, right in (("b", "c"), ("a", "b"), ("c", "d")):
        await engine.assert_fact("alpha", left, "knows", right, valid_from="2024-01-01T00:00:00Z")
    with TestClient(create_app(engine, {"key-a": "alpha"})) as client:
        whole = client.get("/v1/graph/path", params={"from": "a", "to": "d"}, headers=auth()).json()
        monkeypatch.setattr(read, "MAX_FACTS", 2)
        part = client.get("/v1/graph/path", params={"from": "a", "to": "d"}, headers=auth()).json()
    assert whole["status"] == "found" and whole["complete"] is True
    assert part["status"] == "not_connected_in_read" and part["complete"] is False and part["paths"] == []


async def test_a_slow_view_answers_come_back_shortly(monkeypatch):
    from scone_memory.entities import service

    class Slow(InMemoryDocumentStore):
        async def page_facts(self, space, before_id, limit):
            await asyncio.sleep(0.3)
            return await super().page_facts(space, before_id, limit)

    monkeypatch.setattr(service, "BUILD_TIMEOUT", 0.01)
    engine = await MemoryEngine(Slow(), InMemoryVectorIndex(), HashEmbedder()).open()
    await engine.assert_fact("alpha", "alice", "works_at", "Acme", valid_from="2024-01-01T00:00:00Z")
    with TestClient(create_app(engine, {"key-a": "alpha"})) as client:
        busy = client.get("/v1/graph/knowledge", headers=auth())
        assert busy.status_code == 503 and busy.headers["retry-after"] == "1"
        assert busy.json()["code"] == "projection_building"
        time.sleep(0.6)
        ready = client.get("/v1/graph/knowledge", headers=auth())
    assert ready.status_code == 200 and ready.json()["entities"]


def test_a_path_names_its_projection_filters_and_the_bounds_it_applied(quoted):
    client, _, _ = quoted
    found = client.get("/v1/graph/path", params={"from": "alice chen", "to": "lisbon", "max_hops": 3, "limit": 2,
                                                  "hub_degree": 50}, headers=auth()).json()
    view = client.get("/v1/graph/knowledge", params={"status": "history"}, headers=auth()).json()
    assert found["schema_version"] == 1 and found["space"] == "alpha"
    assert found["projection"]["digest"] and found["projection"]["version"] == "scone.entities/1"
    assert found["filters"]["status"] == "current" and found["filters"]["as_of"]
    assert found["policy"] == {"max_hops": 3, "limit": 2, "hub_degree": 50}
    assert view["projection"]["version"] == found["projection"]["version"]


def test_the_graph_context_packet_is_served_for_names_or_a_question(quoted):
    client, _, _ = quoted
    named = client.get("/v1/graph/context", params=[("names", "alice chen"), ("names", "lisbon")], headers=auth()).json()
    assert named["schema_version"] == 1 and named["space"] == "alpha" and named["status"] == "prepared"
    assert any(line.startswith("path: ") for line in named["text"].splitlines()) and len(named["seeds"]) == 2
    asked = client.get("/v1/graph/context", params={"q": "who works in lisbon?"}, headers=auth()).json()
    assert asked["status"] == "prepared" and asked["coverage"]["reasons"] == []
    assert client.get("/v1/graph/context", headers=auth()).status_code == 422
    assert client.get("/v1/graph/context", params={"q": "x", "max_bytes": 511}, headers=auth()).status_code == 422
    assert client.get("/v1/capabilities", headers=auth()).json()["features"]["graph.context"] is True


async def test_the_report_can_leave_hubs_out_of_its_central_entities():
    """A hub linked to everything tops every ranking; excluding the top
    percentile by degree lists it apart instead."""
    engine = await MemoryEngine(InMemoryDocumentStore(), InMemoryVectorIndex(), HashEmbedder()).open()
    for number in range(30):
        await engine.assert_fact("alpha", f"person {number:02d}", "works_at", "Acme Corp", valid_from="2024-01-01T00:00:00Z")
    for left, right in (("person 01", "person 02"), ("person 02", "person 03"), ("person 03", "person 01")):
        await engine.assert_fact("alpha", left, "knows", right, valid_from="2024-01-01T00:00:00Z")
    with TestClient(create_app(engine, {"key-a": "alpha"})) as client:
        plain = client.get("/v1/graph/report", headers=auth()).json()
        trimmed = client.get("/v1/graph/report", params={"exclude_hubs": 95, "resolution": 1.5}, headers=auth()).json()
        refused = client.get("/v1/graph/report", params={"resolution": 0}, headers=auth())
    assert plain["central_entities"][0]["key"] == "acme corp"
    assert "acme corp" not in {item["key"] for item in trimmed["central_entities"]}
    assert [hub["key"] for hub in trimmed["hubs_excluded"]] == ["acme corp"]
    assert trimmed["analysis"]["resolution"] == 1.5 and refused.status_code == 422



def test_groupings_are_found_at_the_resolution_asked_for(teams):
    client, _ = teams
    usual = client.get("/v1/graph/knowledge", params={"groupings": "true"}, headers=auth()).json()["groupings"]
    fine = client.get("/v1/graph/knowledge", params={"groupings": "true", "resolution": 10},
                      headers=auth()).json()["groupings"]
    assert usual["coverage"]["resolution"] == 1.0 and fine["coverage"]["resolution"] == 10.0
    assert len(fine["communities"]) > len(usual["communities"]) == 2
