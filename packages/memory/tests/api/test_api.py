"""The HTTP surface, checked with the sibling ``scone-client`` models when
that package is on disk so a drift between the two stacks fails here."""

from __future__ import annotations

from ..paths import REPO_ROOT

import asyncio
import importlib.util
import json
import pathlib
import sys

import pytest
from fastapi.testclient import TestClient

from scone_memory import HashEmbedder, InMemoryDocumentStore, InMemoryVectorIndex, MemoryEngine
from scone_memory.api import create_app

REPO = REPO_ROOT
#: The HTTP client lives at packages/scone-client after the repository
#: cleanup; the older location is checked second during the move.
CLIENT_MODEL_PATHS = (
    REPO / "packages" / "scone-client" / "scone" / "models.py",
    REPO / "clients" / "python" / "scone" / "models.py",
)


def load_client_models():
    found = next((path for path in CLIENT_MODEL_PATHS if path.exists()), None)
    if found is None:
        pytest.skip("scone-client not checked out beside this package")
    spec = importlib.util.spec_from_file_location("scone_client_models", found)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module  # dataclasses resolve their module by name
    spec.loader.exec_module(module)  # type: ignore[union-attr]
    return module


@pytest.fixture
async def client():
    engine = await MemoryEngine(InMemoryDocumentStore(), InMemoryVectorIndex(), HashEmbedder()).open()
    app = create_app(engine, {"key-a": "alpha", "key-b": "beta"})
    with TestClient(app) as c:
        yield c


def auth(key: str = "key-a") -> dict:
    return {"authorization": f"Bearer {key}"}


def test_no_key_or_wrong_key_is_401(client):
    assert client.get("/v1/status").status_code == 401
    r = client.get("/v1/status", headers=auth("nope"))
    assert r.status_code == 401
    assert r.json() == {"error": "unknown key"}


def test_capabilities_are_authenticated_explicit_and_read_only(client):
    from importlib.util import find_spec
    expected = json.loads((REPO / "tests/fixtures/http-capabilities.json").read_text())["python"]
    expected['features']['documents.pdf'] = find_spec('pypdf') is not None
    assert client.get("/v1/capabilities").status_code == 401
    assert client.get("/v1/capabilities", headers=auth("wrong")).status_code == 401
    before = client.get("/v1/status", headers=auth()).json()
    for key in ("key-a", "key-b"):
        response = client.get("/v1/capabilities", headers=auth(key))
        assert response.status_code == 200
        assert response.json() == expected
    assert client.get("/v1/status", headers=auth()).json() == before


def test_key_decides_the_space(client):
    client.post("/v1/episodes", json={"content": "alpha's secret garden"}, headers=auth("key-a"))
    seen_by_b = client.get("/v1/recall", params={"q": "secret garden"}, headers=auth("key-b")).json()
    assert seen_by_b["items"] == []
    assert client.get("/v1/status", headers=auth("key-b")).json()["space"] == "beta"


def test_unknown_field_is_refused(client):
    r = client.post("/v1/episodes", json={"content": "x", "colour": "red"}, headers=auth())
    assert r.status_code == 422
    assert "colour" in r.json()["error"]


def test_engine_errors_map_to_status_codes(client):
    assert client.post("/v1/episodes", json={"content": "   "}, headers=auth()).status_code == 422
    assert client.post("/v1/facts/42/close", json={"reason": "gone"}, headers=auth()).status_code == 404
    assert client.delete("/v1/episodes/42", headers=auth()).status_code == 404
    assert client.get("/v1/recall", params={"q": ""}, headers=auth()).status_code == 422


def test_round_trip_parses_with_the_shared_client(client):
    models = load_client_models()
    added = client.post(
        "/v1/episodes",
        json={"content": "Moved to Lisbon in March", "tags": ["life"], "created_at": "2024-03-02"},
        headers=auth(),
    )
    assert added.status_code == 200
    parsed_added = models.Added.from_json(added.json())
    assert parsed_added.episode_id == 1 and parsed_added.chunks == 1

    client.post("/v1/facts", json={"subject": "mark", "predicate": "lives_in", "object": "Lisbon"}, headers=auth())
    recall = client.get("/v1/recall", params={"q": "lisbon", "limit": "3"}, headers=auth())
    parsed = models.Recall.from_json(recall.json())
    assert [m.episode_id for m in parsed.items] == [1]
    assert parsed.items[0].created_at == "2024-03-02T00:00:00.000Z"
    assert [f.object for f in parsed.facts] == ["Lisbon"]
    assert parsed.context_reduction == 0.0

    facts = models.Fact  # every fact field the client reads is present
    for f in client.get("/v1/facts", headers=auth()).json()["facts"]:
        facts.from_json(f)
    closed = client.post("/v1/facts/1/close", json={"reason": "moved again"}, headers=auth()).json()
    assert closed == {"closed": 1, "reason": "moved again"}

    status = models.Status.from_json(client.get("/v1/status", headers=auth()).json())
    assert (status.space, status.episodes, status.chunks) == ("alpha", 1, 1)
    tags = [models.Tag.from_json(t) for t in client.get("/v1/tags", headers=auth()).json()["tags"]]
    assert [(t.name, t.count) for t in tags] == [("life", 1)]
    profile = models.Profile.from_json(client.get("/v1/profile", headers=auth()).json())
    assert profile.dynamic == ["Moved to Lisbon in March"]


def test_health_needs_no_key(client):
    assert client.get("/healthz").json() == {"ok": True}


def test_where_filter_over_http(client):
    for user, seat in (("alice", "window"), ("bob", "aisle")):
        client.post(
            "/v1/episodes",
            json={"content": f"{user} prefers the {seat} seat", "metadata": {"user_id": user}},
            headers=auth(),
        )
    r = client.get("/v1/recall", params={"q": "seat", "where": "user_id:bob"}, headers=auth()).json()
    assert [i["metadata"]["user_id"] for i in r["items"]] == ["bob"]
    assert client.get("/v1/recall", params={"q": "seat", "where": "user_id"}, headers=auth()).status_code == 422




def test_review_and_exclusion_over_http(client):
    h = auth()
    held = client.post("/v1/facts", json={"subject": "mark", "predicate": "lives_in", "object": "Austin", "valid_from": "2022-01-01"}, headers=h).json()
    proposed = client.post(
        "/v1/facts",
        json={"subject": "mark", "predicate": "lives_in", "object": "Lisbon", "valid_from": "2024-03-02", "origin": "extracted", "proposed": True, "confidence": 0.7},
        headers=h,
    ).json()
    assert (proposed["status"], proposed["origin"]) == ("proposed", "extracted")
    assert [f["object"] for f in client.get("/v1/facts", headers=h).json()["facts"]] == ["Austin"]
    assert [f["fact_id"] for f in client.get("/v1/facts", params={"status": "proposed"}, headers=h).json()["facts"]] == [proposed["fact_id"]]
    assert client.get("/v1/status", headers=h).json()["pending_review"] == 1
    approved = client.post(f"/v1/facts/{proposed['fact_id']}/approve", headers=h).json()
    assert approved["status"] == "active"
    assert client.get("/v1/facts", params={"all": "true"}, headers=h).json()["facts"][0]["closed_reason"] == f"superseded by fact {proposed['fact_id']}"
    excluded = client.post(f"/v1/facts/{proposed['fact_id']}/exclude", json={"reason": "private"}, headers=h).json()
    assert excluded["excluded_reason"] == "private"
    assert client.get("/v1/facts", headers=h).json()["facts"] == []
    assert [f["fact_id"] for f in client.get("/v1/facts", params={"excluded": "true"}, headers=h).json()["facts"]] == [proposed["fact_id"]]
    assert client.post(f"/v1/facts/{proposed['fact_id']}/include", headers=h).json()["excluded_reason"] is None
    assert client.post(f"/v1/facts/{held['fact_id']}/decline", json={"reason": "x"}, headers=h).status_code == 422
    assert client.post("/v1/facts/999/approve", headers=h).status_code == 404


def test_an_episode_can_be_read_back_verbatim(client):
    h = auth()
    added = client.post("/v1/episodes", json={"content": "  Café ☕ verbatim\n", "created_at": "2024-01-02", "metadata": {"user_id": "ana"}}, headers=h).json()
    body = client.get(f"/v1/episodes/{added['episode_id']}", headers=h).json()
    assert (body["content"], body["kind"], body["created_at"], body["metadata"]) == ("  Café ☕ verbatim\n", "note", "2024-01-02T00:00:00.000Z", {"user_id": "ana"})
    assert client.get("/v1/episodes/999", headers=h).status_code == 404
    assert client.get(f"/v1/episodes/{added['episode_id']}", headers=auth("key-b")).status_code == 404





















def test_history_is_opt_in_over_http(client):
    h = auth()
    client.post("/v1/facts", json={"subject": "mark", "predicate": "lives_in", "object": "Austin", "valid_from": "2019-08-01"}, headers=h)
    lisbon = client.post("/v1/facts", json={"subject": "mark", "predicate": "lives_in", "object": "Lisbon", "valid_from": "2024-03-02"}, headers=h).json()
    plain = client.get("/v1/recall", params={"q": "mark lives"}, headers=h).json()
    assert [f["object"] for f in plain["facts"]] == ["Lisbon"] and plain["history"] == []
    with_history = client.get("/v1/recall", params={"q": "mark lives", "history": "true"}, headers=h).json()
    [austin] = with_history["history"]
    assert (austin["object"], austin["status"], austin["superseded_by"]) == ("Austin", "closed", lisbon["fact_id"])
    assert austin["valid_until"] == lisbon["valid_from"]
    assert client.get("/v1/recall", params={"q": "mark lives", "history": "maybe"}, headers=h).status_code == 422


def test_serve_builds_the_engine_on_the_loop_it_serves_from(monkeypatch, capsys):
    """Async database clients bind to the loop they were opened on. The
    compose smoke test found every Mongo request failing with "Cannot use
    AsyncMongoClient in different event loop" because the engine was built
    with asyncio.run and then served from uvicorn's own loop."""
    import uvicorn

    from scone_memory.api import __main__ as serve
    from scone_memory.runtime.config import Settings

    loops: dict[str, object] = {}

    async def fake_build(settings):
        loops["built"] = asyncio.get_running_loop()
        return await MemoryEngine(InMemoryDocumentStore(), InMemoryVectorIndex(), HashEmbedder()).open()

    class FakeServer:
        def __init__(self, config):
            loops["app"] = config.app

        async def serve(self):
            loops["served"] = asyncio.get_running_loop()

    monkeypatch.setattr(serve, "build_engine", fake_build)
    monkeypatch.setattr(uvicorn, "Server", FakeServer)
    serve.main(Settings.from_env({"SCONE_API_KEY": "k"}))
    assert loops["built"] is loops["served"], "one loop for building and serving"
    assert loops["app"].state.engine.documents.name == "memory"
    assert "documents=memory" in capsys.readouterr().err


def test_recall_narrows_over_http():
    engine = asyncio.run(MemoryEngine(InMemoryDocumentStore(), InMemoryVectorIndex(), HashEmbedder()).open())

    async def seed():
        out = []
        for kind, text, source, day in [
            ("note", "deploy runbook: rotate the staging keys first", None, "2024-01-10"),
            ("file", "deploy runbook: rotate the staging keys, then restart", "/ops/runbooks/deploy.md", "2024-02-10"),
            ("conversation", "user: where is the deploy runbook for staging keys?", "session-42", "2024-04-10"),
        ]:
            out.append((await engine.remember("alpha", text, kind=kind, source=source, created_at=day)).episode_id)
        return out

    ids = asyncio.run(seed())
    with TestClient(create_app(engine, {"key-a": "alpha"})) as client:
        def got(**params):
            r = client.get("/v1/recall", params={"q": "deploy runbook staging keys", **params}, headers=auth())
            assert r.status_code == 200, r.text
            return sorted({i["episode_id"] for i in r.json()["items"]})

        assert got() == ids
        assert got(kind="file") == [ids[1]]
        assert got(source_prefix="session-") == [ids[2]]
        assert got(until="2024-01-31") == [ids[0]]
        assert got(since="2024-03-01") == [ids[2]]
        assert client.get("/v1/recall", params={"q": "deploy", "kind": "chat"}, headers=auth()).status_code == 422


def test_the_facts_list_carries_the_space_revision(client):
    """A review page freezes the list it renders, so it needs the revision
    the list was read at: the same counter /v1/status reports, read in the
    same request as the facts. A different revision later means the space
    moved and the frozen set no longer matches the screen."""
    h = auth()
    first = client.get("/v1/facts", headers=h).json()
    assert first["revision"] == client.get("/v1/status", headers=h).json()["revision"]

    client.post("/v1/facts", json={"subject": "mark", "predicate": "lives_in", "object": "Lisbon"}, headers=h)
    after = client.get("/v1/facts", headers=h).json()
    assert after["revision"] > first["revision"]
    assert after["revision"] == client.get("/v1/status", headers=h).json()["revision"]


def test_many_episodes_read_in_one_request(client):
    """The review page's source panel asks for one episode per row. One
    request answers them all: episodes in the order asked for, and ids that
    are confirmed gone listed separately, so a missing source stays
    distinguishable from a failed request."""
    h = auth()
    ids = [client.post("/v1/episodes", json={"content": f"note {n}"}, headers=h).json()["episode_id"] for n in (1, 2, 3)]
    other_space = client.post("/v1/episodes", json={"content": "beta's own"}, headers=auth("key-b")).json()["episode_id"]

    body = client.get("/v1/episodes", params={"ids": f"{ids[1]},{ids[2]},999,{ids[0]},{other_space}"}, headers=h).json()
    assert [e["episode_id"] for e in body["episodes"]] == [ids[1], ids[2], ids[0]]
    assert [e["content"] for e in body["episodes"]] == ["note 2", "note 3", "note 1"]
    # 999 never existed and other_space belongs to beta: from alpha both are
    # simply absent, the same answer the single-episode read gives with a 404.
    assert body["missing"] == [999, other_space]

    # Proposed facts share a source, so a page of rows asks for the same
    # episode many times; it is read once, at the position first asked for.
    repeated = client.get("/v1/episodes", params={"ids": f"{ids[1]},{ids[0]},{ids[1]}"}, headers=h).json()
    assert [e["episode_id"] for e in repeated["episodes"]] == [ids[1], ids[0]]


def test_a_batch_episode_read_is_bounded(client):
    """A page cannot ask for the whole space in one request: over the bound
    the answer is a refusal, never a silent truncation."""
    h = auth()
    too_many = ",".join(str(n) for n in range(1, 102))
    assert client.get("/v1/episodes", params={"ids": too_many}, headers=h).status_code == 422
    assert client.get("/v1/episodes", params={"ids": ""}, headers=h).status_code == 422


def test_forgetting_over_http_previews_then_reports_its_impact(client):
    h = auth()
    episode = client.post("/v1/episodes", json={"content": "Acme is headquartered in Lisbon, near the river."}, headers=h).json()
    fact = client.post("/v1/facts", json={"subject": "acme", "predicate": "based_in", "object": "lisbon",
                                          "source_episode_id": episode["episode_id"], "quote": "headquartered in Lisbon"}, headers=h).json()
    preview = client.get(f"/v1/episodes/{episode['episode_id']}/impact", headers=h)
    assert preview.status_code == 200
    assert preview.json()["facts_citing"] == [fact["fact_id"]] and preview.json()["chunks"] >= 1
    assert client.get(f"/v1/episodes/{episode['episode_id']}", headers=h).status_code == 200, "a preview removes nothing"
    gone = client.delete(f"/v1/episodes/{episode['episode_id']}", headers=h)
    assert gone.status_code == 200 and gone.json()["forgotten"] == episode["episode_id"] and gone.json()["forgotten_at"]
    assert {k: v for k, v in gone.json().items() if k not in ("forgotten", "forgotten_at")} == {k: v for k, v in preview.json().items() if k != "forgotten_at"}
    assert client.get(f"/v1/episodes/{episode['episode_id']}/impact", headers=h).status_code == 410, "forgotten, not unknown"
    assert client.get(f"/v1/facts/{fact['fact_id']}", headers=h).json()["fact"]["status"] == "active", "the claim stands"


def test_a_keyed_episode_reports_duplicate_or_updated_over_http(client):
    h = auth()
    first = client.post("/v1/episodes", json={"content": "The office is in Lisbon.", "dedup_key": "doc:office"}, headers=h).json()
    assert first["outcome"] == "accepted" and first["replaced"] is None
    dropped = client.post("/v1/episodes", json={"content": "The office moved to Porto.", "dedup_key": "doc:office"}, headers=h).json()
    assert dropped["outcome"] == "duplicate" and dropped["episode_id"] == first["episode_id"], "the client learns its update did not land"
    updated = client.post("/v1/episodes", json={"content": "The office moved to Porto.", "dedup_key": "doc:office", "replace": True}, headers=h).json()
    assert updated["outcome"] == "updated" and updated["episode_id"] != first["episode_id"]
    assert updated["replaced"]["episode_id"] == first["episode_id"] and updated["replaced"]["chunks"] >= 1
    assert client.get(f"/v1/episodes/{first['episode_id']}", headers=h).status_code == 410, "replaced is forgotten on purpose"
    plain = client.post("/v1/episodes", json={"content": "no key, no replace"}, headers=h).json()
    assert plain["outcome"] == "accepted"
    assert client.post("/v1/episodes", json={"content": "x", "replace": True}, headers=h).status_code == 422, "replace needs a key"


def test_a_batch_of_episodes_answers_per_item_and_lands_whole_or_not_at_all(client):
    """G14: one bounded request, one outcome per item in the order sent,
    and either every record lands or none does."""
    h = auth()
    body = {"records": [
        {"content": "first note of the batch", "tags": ["batch"]},
        {"content": "second note, keyed", "dedup_key": "doc:two"},
        {"content": "first note of the batch"},
    ]}
    r = client.post("/v1/episodes/batch", json=body, headers=h)
    assert r.status_code == 200, r.text
    items = r.json()["items"]
    assert [i["outcome"] for i in items] == ["accepted", "accepted", "duplicate"]
    assert items[2]["episode_id"] == items[0]["episode_id"]
    assert r.json()["counts"] == {"accepted": 2, "duplicate": 1, "updated": 0}
    again = client.post("/v1/episodes/batch", json={"records": [{"content": "changed text", "dedup_key": "doc:two"}]}, headers=h).json()
    assert again["items"][0]["outcome"] == "duplicate" and again["items"][0]["episode_id"] == items[1]["episode_id"]

    before = client.get("/v1/status", headers=h).json()["episodes"]
    broken = client.post("/v1/episodes/batch", json={"records": [{"content": "lands?"}, {"content": "   "}]}, headers=h)
    assert broken.status_code == 422
    assert client.get("/v1/status", headers=h).json()["episodes"] == before, "a bad record fails the whole batch"
    assert client.post("/v1/episodes/batch", json={"records": []}, headers=h).status_code == 422
    too_many = {"records": [{"content": f"note {i}"} for i in range(501)]}
    assert client.post("/v1/episodes/batch", json=too_many, headers=h).status_code == 422
    assert client.post("/v1/episodes/batch", json={"records": [{"content": "x", "bogus": 1}]}, headers=h).status_code == 422
    one_at_a_time = client.post("/v1/episodes/batch", json={"records": [{"content": "y", "dedup_key": "k", "replace": True}]}, headers=h)
    assert one_at_a_time.status_code == 422 and "one record at a time" in one_at_a_time.text
    assert client.post("/v1/episodes/batch", json=body).status_code == 401


def test_a_forgotten_episode_answers_gone_with_the_date(client):
    h = auth()
    episode = client.post("/v1/episodes", json={"content": "soon gone"}, headers=h).json()
    gone = client.delete(f"/v1/episodes/{episode['episode_id']}", headers=h).json()
    assert gone["forgotten_at"]
    read = client.get(f"/v1/episodes/{episode['episode_id']}", headers=h)
    assert read.status_code == 410 and read.json()["forgotten_at"] == gone["forgotten_at"] and "forgotten" in read.json()["error"]
    assert client.get(f"/v1/episodes/{episode['episode_id']}/impact", headers=h).status_code == 410
    assert client.delete(f"/v1/episodes/{episode['episode_id']}", headers=h).status_code == 410
    assert client.get("/v1/episodes/999999", headers=h).status_code == 404, "never existed is still 404"
    assert client.get(f"/v1/episodes/{episode['episode_id']}", headers=auth("key-b")).status_code == 404, "another space learns nothing"


def test_a_temporal_question_is_answered_by_computation(client):
    for when, text in (("2023-03-11T09:00:00Z", "I sold homemade baked goods at the farmers market."),
                       ("2023-04-01T18:30:00Z", "I ran the charity bake-off at the village hall.")):
        assert client.post("/v1/episodes", json={"content": text, "created_at": when}, headers=auth()).status_code == 200
    answer = client.get("/v1/answers/temporal", params={
        "q": "How many weeks passed between the time I sold homemade baked goods and the time I ran the "
             "charity bake-off?", "now": "2023-04-20T10:12:00Z"}, headers=auth()).json()
    assert answer["status"] == "computed" and answer["value"]["asked"] == 3
    assert answer["plan"]["kind"] == "between" and len(answer["anchors"]) == 2
    assert "answer: 3 weeks (21 days)" in answer["text"]
    left = client.get("/v1/answers/temporal", params={"q": "What did I bake?"}, headers=auth()).json()
    assert left["status"] == "not_temporal" and left["value"] == {}
    refused = client.get("/v1/answers/temporal", params={"q": "How long ago?", "limit": 0}, headers=auth())
    assert refused.status_code == 422


def test_a_temporal_answer_stays_inside_its_space(client):
    assert client.post("/v1/episodes", json={"content": "I met Emma for coffee near the river.",
                                             "created_at": "2023-04-11T12:00:00Z"},
                       headers=auth("key-a")).status_code == 200
    answer = client.get("/v1/answers/temporal", params={"q": "How many days ago did I meet Emma?",
                                                        "now": "2023-04-20T10:12:00Z"}, headers=auth("key-b")).json()
    assert answer["status"] == "ungrounded" and answer["space"] == "beta"


def test_a_multi_part_question_can_be_searched_a_part_at_a_time(client):
    """The receipt is the point, so the response has to carry it: which part
    placed each passage, and whether anything judged the evidence."""
    for text in ("We reverted the billing change after the invoices came out wrong.",
                 "At the Thursday meeting were Priya, Tomas and the auditor from Lisbon."):
        assert client.post("/v1/episodes", json={"content": text}, headers=auth()).status_code == 200
    asked = {"q": "What did I decide about billing, and who was at the meeting?", "limit": 2}
    parted = client.get("/v1/recall/parts", params=asked, headers=auth()).json()
    assert parted["decomposition"]["split"] is True
    assert [p["text"] for p in parted["decomposition"]["parts"]] == [
        "What did I decide about billing", "who was at the meeting?"]
    assert len(parted["items"]) == 2 and sorted(parted["placed_by"]) == [0, 1]
    assert parted["judged"] is False and "not evidence" in parted["why"]
    assert all(p["found"] >= p["contributed"] for p in parted["per_part"])
    whole = client.get("/v1/recall/parts", params={"q": "Where can I buy salt and pepper?"},
                       headers=auth()).json()
    assert whole["decomposition"]["split"] is False and "one thing" in whole["why"]
    assert client.get("/v1/recall/parts", params={"q": ""}, headers=auth()).status_code == 422
    assert client.get("/v1/recall/parts", params=asked).status_code == 401


def test_parked_records_can_be_retried_on_purpose(client):
    """Restarting the server retries everything; this retries what you
    name. Without a worker there is nothing parked in this process, and
    the endpoint says so rather than pretending it cleared something."""
    assert client.post("/v1/episodes", json={"content": "Alice Chen works at Acme Robotics."},
                       headers=auth()).status_code == 200
    refused = client.post("/v1/consolidate/retry", json={}, headers=auth())
    assert refused.status_code == 501 and "nothing is parked" in refused.json()["error"]
    assert client.post("/v1/consolidate/retry", json={}).status_code == 401
    assert client.post("/v1/consolidate/retry", json={"episodes": [1], "extra": 1},
                       headers=auth()).status_code == 422


def test_a_parted_search_applies_the_same_narrowing_as_an_ordinary_one(client):
    """A filter a caller passed and the server dropped is worse than one it
    refused: the page shows results that look narrowed and are not. The
    record echoes what was applied so a reader can tell."""
    for text, kind in (("We reverted the billing change after the invoices came out wrong.", "note"),
                       ("At the Thursday meeting were Priya, Tomas and the auditor.", "file")):
        assert client.post("/v1/episodes", json={"content": text, "kind": kind},
                           headers=auth()).status_code == 200
    asked = {"q": "What did I decide about billing, and who was at the meeting?",
             "limit": 3, "kind": "note"}
    narrowed = client.get("/v1/recall/parts", params=asked, headers=auth()).json()
    assert narrowed["applied"]["kind"] == "note", narrowed["applied"]
    mine = client.get("/v1/status", headers=auth()).json()["space"]
    assert narrowed["space"] == mine, "the echo names the space the key authenticated for"
    assert all("meeting were Priya" not in item["text"] for item in narrowed["items"]), narrowed["items"]
    unfiltered = {name: value for name, value in asked.items() if name != "kind"}
    wide = client.get("/v1/recall/parts", params=unfiltered, headers=auth()).json()
    assert "kind" not in wide["applied"], wide["applied"]
    assert len(wide["items"]) > len(narrowed["items"]), (wide["items"], narrowed["items"])


def test_a_parted_search_refuses_a_filter_it_cannot_honour_rather_than_dropping_it(client):
    """`history` is the closed chain behind matched facts, which is a
    per-query notion with no defined meaning merged across parts. Inventing
    one silently would be worse than not offering it."""
    assert client.post("/v1/episodes", json={"content": "Billing was reverted."},
                       headers=auth()).status_code == 200
    asked = {"q": "What did I decide about billing, and who was at the meeting?", "history": True}
    refused = client.get("/v1/recall/parts", params=asked, headers=auth())
    assert refused.status_code == 422, refused.json()
    assert "history" in refused.json()["error"] and "part" in refused.json()["error"]


def test_a_retry_will_not_take_a_boolean_or_a_string_for_an_episode_id(client):
    """Pydantic coerces true, "1" and 1.0 to the integer 1, so a caller
    could clear episode 1 without ever naming it. An id is an id."""
    for wrong in (True, "1", 1.0, None):
        refused = client.post("/v1/consolidate/retry", json={"episodes": [wrong]}, headers=auth())
        assert refused.status_code == 422, (wrong, refused.status_code, refused.json())
    assert client.post("/v1/consolidate/retry", json={"episodes": [1]},
                       headers=auth()).status_code in (200, 501)


def test_a_recall_can_withhold_what_a_caller_must_not_receive(client):
    """The way-out half of redaction. Capture scrubs the agent feed; this
    memory arrived through the episodes endpoint, which does not scrub, so
    retrieval is the only place left to catch it."""
    assert client.post("/v1/episodes", json={
        "content": "Write to ana.alves@meridian-health.example about AKIAIOSFODNN7EXAMPLE."},
        headers=auth()).status_code == 200
    asked = {"q": "write about", "limit": 3}
    plain = client.get("/v1/recall", params=asked, headers=auth()).json()
    assert "ana.alves@meridian-health.example" in plain["items"][0]["text"]
    assert "withheld" not in plain, "withholding is opt-in"

    held = client.get("/v1/recall", params={**asked, "withhold": "email,secret"},
                      headers=auth()).json()
    assert "ana.alves@meridian-health.example" not in held["items"][0]["text"]
    assert "AKIAIOSFODNN7EXAMPLE" not in held["items"][0]["text"]
    assert held["withheld"]["count"] == 2, held["withheld"]
    assert held["withheld"]["by_kind"] == {"email": 1, "secret": 1}, held["withheld"]
    # The sentence that stops the report being read as a safety claim.
    assert "not a finding" in held["withheld"]["why"], held["withheld"]["why"]
    # The byte count has to describe what was actually handed back.
    assert held["returned_bytes"] != plain["returned_bytes"]
    assert client.get("/v1/recall", params={**asked, "withhold": "astrology"},
                      headers=auth()).status_code == 422


def test_withholding_covers_the_whole_answer_and_not_only_the_passage(client):
    """The report said one match and nothing unscanned while the same
    address came back in the item's source, tags and metadata and as the
    object of a fact. A report of coverage has to be true of everything
    handed back, or it is a safety claim about surfaces nobody scanned."""
    address = "ana.alves@meridian-health.example"
    assert client.post("/v1/episodes", json={
        "content": f"Write to {address} about the rota.", "source": address,
        "tags": [address], "metadata": {"contact": address}},
        headers=auth()).status_code == 200
    assert client.post("/v1/facts", json={
        "subject": "Ana", "predicate": "email", "object": address},
        headers=auth()).status_code == 200

    asked = {"q": "Ana write rota email", "limit": 3}
    held = client.get("/v1/recall", params={**asked, "withhold": "email"}, headers=auth())
    assert held.status_code == 200, held.text
    body = held.json()
    assert body["facts"], "this test needs a fact in the answer to be about anything"
    leaked = [section for section, value in body.items() if address in json.dumps(value)]
    assert leaked == [], f"{address} still in {leaked}"
    assert "facts" in body["withheld"]["surfaces"], body["withheld"]

    # A section the policy cannot reach is refused, not served under a
    # report that reads as clean.
    refused = client.get("/v1/recall", params={**asked, "withhold": "email",
                                               "evidence_graph": "true"}, headers=auth())
    assert refused.status_code == 422, refused.text
    assert "evidence_graph" in refused.text, refused.text


async def test_an_unknown_withhold_kind_is_refused_before_any_recall_runs():
    """Naming a kind that does not exist is a mistake in the request, and
    answering it by searching first and raising afterwards spends the
    whole search -- and logs a recall event -- for an answer nobody
    receives. Counted through the engine rather than inferred, because
    "it was refused" is true either way and only one of the two is right.
    """
    engine = await MemoryEngine(InMemoryDocumentStore(), InMemoryVectorIndex(),
                                HashEmbedder()).open()
    searches = 0
    real = engine.recall

    async def counted(*args, **kwargs):
        nonlocal searches
        searches += 1
        return await real(*args, **kwargs)

    engine.recall = counted  # type: ignore[method-assign]
    try:
        with TestClient(create_app(engine, {"key-a": "alpha"})) as spy:
            assert spy.post("/v1/episodes", json={"content": "Anything at all."},
                            headers=auth()).status_code == 200
            refused = spy.get("/v1/recall", params={"q": "anything", "withhold": "astrology"},
                              headers=auth())
            assert refused.status_code == 422, refused.text
            assert searches == 0, "the search ran for an answer that was never returned"
            assert spy.get("/v1/recall", params={"q": "anything", "withhold": "email"},
                           headers=auth()).status_code == 200
            assert searches == 1, "a valid policy must not stop the search"
    finally:
        await engine.close()
