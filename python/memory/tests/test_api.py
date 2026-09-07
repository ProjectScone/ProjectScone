"""The HTTP surface, checked with the sibling ``scone-client`` models when
that package is on disk so a drift between the two stacks fails here."""

from __future__ import annotations

import asyncio
import importlib.util
import json
import pathlib
import sys

import pytest
from fastapi.testclient import TestClient

from scone_memory import HashEmbedder, InMemoryDocumentStore, InMemoryVectorIndex, MemoryEngine
from scone_memory.api import create_app

REPO = pathlib.Path(__file__).resolve().parents[3]
#: The HTTP client lives at python/scone-client after the repository
#: cleanup; the older location is checked second during the move.
CLIENT_MODEL_PATHS = (
    REPO / "python" / "scone-client" / "scone" / "models.py",
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
    expected = json.loads((REPO / "tests/fixtures/http-capabilities.json").read_text())["python"]
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


def test_console_is_served_with_the_key_baked_in():
    """Whichever page generation is packaged, one configured key reaches the
    page and several keys do not; the exact carrier is tested separately."""
    engine = asyncio.run(MemoryEngine(InMemoryDocumentStore(), InMemoryVectorIndex(), HashEmbedder()).open())
    with TestClient(create_app(engine, {"solo": "default"}, console_key="solo")) as c:
        page = c.get("/")
        assert page.status_code == 200 and page.headers["content-type"].startswith("text/html")
        assert "solo" in page.text and "__SCONE_TOKEN__" not in page.text
    with TestClient(create_app(engine, {"a": "x", "b": "y"})) as c:
        text = c.get("/").text
        assert "x" not in text.split("<body>")[-1][:0] and 'data-token="x"' not in text and 'const KEY="x"' not in text
    with TestClient(create_app(engine, {"a": "x"}, console=False)) as c:
        assert c.get("/").status_code == 404


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


def test_playground_is_served_with_the_same_key_handling():
    import scone_memory.api.app as app_module

    if not app_module.PLAYGROUND.exists():
        pytest.skip("playground asset not packaged in this checkout; the Webapp build emits it")
    engine = asyncio.run(MemoryEngine(InMemoryDocumentStore(), InMemoryVectorIndex(), HashEmbedder()).open())
    with TestClient(create_app(engine, {"solo": "default"}, console_key="solo")) as c:
        page = c.get("/playground")
        assert page.status_code == 200 and page.headers["content-type"].startswith("text/html")
        assert "__SCONE_TOKEN__" not in page.text and "solo" in page.text
    with TestClient(create_app(engine, {"a": "x", "b": "y"})) as c:
        assert "__SCONE_TOKEN__" in c.get("/playground").text  # several keys: the page asks
        assert c.head("/playground").status_code == 200, "the console probes with HEAD"


def test_reload_pages_serves_edits_without_a_restart(tmp_path, monkeypatch):
    import scone_memory.api.app as app_module

    fake = tmp_path / "playground.html"
    fake.write_text("<html>v1 __SCONE_TOKEN__</html>", encoding="utf-8")
    monkeypatch.setattr(app_module, "PLAYGROUND", fake)
    engine = asyncio.run(MemoryEngine(InMemoryDocumentStore(), InMemoryVectorIndex(), HashEmbedder()).open())
    with TestClient(create_app(engine, {"solo": "default"}, console_key="solo", reload_pages=True)) as c:
        first = c.get("/playground")
        assert first.text == "<html>v1 solo</html>"
        assert first.headers["cache-control"] == "no-store" and first.headers["etag"].startswith('"')
        assert c.head("/playground").headers["etag"] == first.headers["etag"]
        import os, time
        fake.write_text("<html>v2 __SCONE_TOKEN__</html>", encoding="utf-8")
        os.utime(fake, ns=(time.time_ns(), time.time_ns() + 5_000_000))  # a distinct mtime even on a coarse clock
        second = c.get("/playground")
        assert second.text == "<html>v2 solo</html>", "development mode re-reads the file"
        assert second.headers["etag"] != first.headers["etag"], "HEAD pollers see a new revision"
        assert "solo" not in second.headers["etag"], "the revision carries no key"
    with TestClient(create_app(engine, {"solo": "default"}, console_key="solo")) as c:
        fake.write_text("<html>v3 __SCONE_TOKEN__</html>", encoding="utf-8")
        assert c.get("/playground").text == "<html>v2 solo</html>", "normal mode reads once at startup"


def test_memory_is_the_canonical_console_address_and_root_still_works():
    engine = asyncio.run(MemoryEngine(InMemoryDocumentStore(), InMemoryVectorIndex(), HashEmbedder()).open())
    with TestClient(create_app(engine, {"solo": "default"}, console_key="solo")) as c:
        a, b = c.get("/memory"), c.get("/")
        assert a.status_code == b.status_code == 200 and a.text == b.text
        assert c.head("/memory").status_code == 200
        assert "__SCONE_MARK__" not in a.text, "the mark placeholder is always substituted"


def test_memory_only_server_serves_workspace_deep_links_without_advertising_conversations():
    """A refresh on /conversations must not 404 on a memory-only host: the
    packaged workspace owns that address and shows its own readiness state.
    Catches the missing deep link, HTML leaking into a /v1 miss, and the
    capability list claiming a service that is not mounted."""
    engine = asyncio.run(MemoryEngine(InMemoryDocumentStore(), InMemoryVectorIndex(), HashEmbedder()).open())
    with TestClient(create_app(engine, {"solo": "default"}, console_key="solo")) as c:
        canonical = c.get("/memory")
        for path in ("/learn", "/learn/how-it-works", "/learn/graph-memory"):
            page = c.get(path)
            assert page.status_code == 200 and page.headers["content-type"].startswith("text/html"), path
            assert "solo" not in page.text and 'id="root"' in page.text, "a public page carries no configured key"
            assert c.head(path).status_code == 200
        for path in ("/conversations", "/conversations/session-one"):
            page = c.get(path)
            assert page.status_code == 200 and page.headers["content-type"].startswith("text/html"), path
            assert page.text == canonical.text, path
            head = c.head(path)
            assert head.status_code == 200 and head.content == b""
            assert head.headers["content-length"] == page.headers["content-length"]
        assert c.get("/conversations/session-one/not-a-route").status_code == 404
        assert c.post("/conversations").status_code in {404, 405}
        miss = c.get("/v1/conversations/capabilities", headers={"Authorization": "Bearer solo"})
        assert miss.status_code == 404 and miss.headers["content-type"].startswith("application/json")
        features = c.get("/v1/capabilities", headers={"Authorization": "Bearer solo"}).json()["features"]
        assert not [name for name in features if name.startswith("conversations")], features
    with TestClient(create_app(engine, {"solo": "default"}, console=False)) as c:
        for path in ("/conversations", "/conversations/session-one", "/learn", "/learn/how-it-works", "/learn/graph-memory"):
            assert c.get(path).status_code == 404, path
        assert c.get("/learn/anything-else").status_code == 404, "no catch-all"


def test_console_key_reaches_either_page_generation(tmp_path, monkeypatch):
    import scone_memory.api.app as app_module

    engine = asyncio.run(MemoryEngine(InMemoryDocumentStore(), InMemoryVectorIndex(), HashEmbedder()).open())
    react = tmp_path / "console.html"
    react.write_text('<html><script type="module">const KEY="__SCONE_TOKEN__";</script></html>', encoding="utf-8")
    monkeypatch.setattr(app_module, "CONSOLE", react)
    with TestClient(create_app(engine, {"solo": "default"}, console_key="solo")) as c:
        text = c.get("/memory").text
        assert 'const KEY="solo"' in text and "__SCONE_TOKEN__" not in text and "data-token" not in text
    with TestClient(create_app(engine, {"a": "x", "b": "y"})) as c:
        assert "__SCONE_TOKEN__" in c.get("/memory").text, "several keys: the page asks"
    legacy = tmp_path / "legacy.html"
    legacy.write_text("<html><script>const TOKEN = document.currentScript.dataset.token;</script></html>", encoding="utf-8")
    monkeypatch.setattr(app_module, "CONSOLE", legacy)
    with TestClient(create_app(engine, {"solo": "default"}, console_key="solo")) as c:
        assert '<script data-token="solo">' in c.get("/memory").text


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
    assert gone.status_code == 200 and gone.json()["forgotten"] == episode["episode_id"]
    assert {k: v for k, v in gone.json().items() if k != "forgotten"} == preview.json()
    assert client.get(f"/v1/episodes/{episode['episode_id']}/impact", headers=h).status_code == 404
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
    assert client.get(f"/v1/episodes/{first['episode_id']}", headers=h).status_code == 404
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
