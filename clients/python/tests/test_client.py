"""Unit tests: the client against a scripted stub of the Scone API.

The payload shapes here are copied from crates/scone/src/serve.rs, so a
server that changes shape should break these before it breaks a user.
"""

from __future__ import annotations

import pytest
from stub_server import StubScone

from scone import Fact, Memory, Profile, Scone, SconeError, Status, Tag


@pytest.fixture
def stub():
    with StubScone() as server:
        yield server


@pytest.fixture
def client(stub):
    with Scone(stub.base_url, "sk-test") as c:
        yield c


# ---------------------------------------------------------------------
# auth
# ---------------------------------------------------------------------


def test_every_request_carries_the_bearer_key(stub, client):
    stub.set_default(200, {"tags": []})
    client.tags()
    client.status()
    assert len(stub.requests) == 2
    for request in stub.requests:
        assert request.headers["authorization"] == "Bearer sk-test"


def test_api_key_is_read_from_the_environment(stub, monkeypatch):
    monkeypatch.setenv("SCONE_API_KEY", "sk-from-env")
    monkeypatch.setenv("SCONE_URL", stub.base_url)
    stub.set_default(200, {"tags": []})
    with Scone() as client:
        assert client.base_url == stub.base_url
        client.tags()
    assert stub.requests[0].headers["authorization"] == "Bearer sk-from-env"


def test_a_missing_key_fails_before_any_request_is_made(stub, monkeypatch):
    monkeypatch.delenv("SCONE_API_KEY", raising=False)
    with pytest.raises(SconeError) as excinfo:
        Scone(stub.base_url)
    assert "SCONE_API_KEY" in str(excinfo.value)
    assert excinfo.value.status is None
    assert stub.requests == []


def test_explicit_arguments_beat_the_environment(stub, monkeypatch):
    monkeypatch.setenv("SCONE_API_KEY", "sk-from-env")
    monkeypatch.setenv("SCONE_URL", "http://not-used.invalid")
    stub.set_default(200, {"tags": []})
    with Scone(stub.base_url, "sk-explicit") as client:
        client.tags()
    assert stub.requests[0].headers["authorization"] == "Bearer sk-explicit"


# ---------------------------------------------------------------------
# errors
# ---------------------------------------------------------------------


def test_unauthorized_carries_the_servers_message_and_status(stub, client):
    stub.set_default(401, {"error": "unknown key"})
    with pytest.raises(SconeError) as excinfo:
        client.status()
    assert excinfo.value.status == 401
    assert excinfo.value.message == "unknown key"
    assert "unknown key" in str(excinfo.value)


def test_unprocessable_content_carries_the_bound_message(stub, client):
    stub.set_default(422, {"error": "content must be 1..=100000 bytes"})
    with pytest.raises(SconeError) as excinfo:
        client.add("anything")
    assert excinfo.value.status == 422
    assert excinfo.value.message == "content must be 1..=100000 bytes"


def test_engine_failure_surfaces_as_a_500_sconeerror(stub, client):
    # serve.rs maps every engine error, NotFound included, to 500.
    stub.set_default(500, {"error": "not found: active fact 99 in space default"})
    with pytest.raises(SconeError) as excinfo:
        client.close_fact(99, "superseded")
    assert excinfo.value.status == 500
    assert "active fact 99" in excinfo.value.message


def test_a_non_json_rejection_body_still_becomes_a_sconeerror(stub, client):
    # axum's own extractor rejections answer plain text, not {"error": ...}.
    stub.set_default(400, "Failed to deserialize query string: missing field `q`")
    with pytest.raises(SconeError) as excinfo:
        client.recall("anything")
    assert excinfo.value.status == 400
    assert "missing field `q`" in excinfo.value.message


def test_an_empty_error_body_still_reports_the_status(stub, client):
    stub.set_default(404, "")
    with pytest.raises(SconeError) as excinfo:
        client.status()
    assert excinfo.value.status == 404
    assert "404" in excinfo.value.message


def test_a_connection_failure_is_a_sconeerror_with_no_status():
    # Port 1 on loopback refuses immediately; no server is ever involved.
    with Scone("http://127.0.0.1:1", "sk-test") as client:
        with pytest.raises(SconeError) as excinfo:
            client.status()
    assert excinfo.value.status is None


# ---------------------------------------------------------------------
# requests the client builds
# ---------------------------------------------------------------------


def test_add_sends_content_and_tags_under_the_servers_names(stub, client):
    stub.route("POST", "/v1/episodes", 201,
               {"episode_id": 7, "chunks": 2, "deduplicated": False})
    added = client.add("deploys happen on Thursdays", tags=["ops", "release"])
    request = stub.requests[0]
    assert request.method == "POST"
    assert request.path == "/v1/episodes"
    assert request.json == {
        "content": "deploys happen on Thursdays",
        "tags": ["ops", "release"],
    }
    assert added.episode_id == 7
    assert added.chunks == 2
    assert added.deduplicated is False


def test_add_omits_the_tags_key_when_there_are_none(stub, client):
    stub.route("POST", "/v1/episodes", 201,
               {"episode_id": 1, "chunks": 1, "deduplicated": False})
    client.add("a bare note")
    assert stub.requests[0].json == {"content": "a bare note"}


def test_a_deduplicated_store_reports_no_chunks(stub, client):
    stub.route("POST", "/v1/episodes", 200, {"episode_id": 7, "deduplicated": True})
    added = client.add("the same note twice")
    assert added.deduplicated is True
    assert added.episode_id == 7
    assert added.chunks is None


def test_add_refuses_a_source_the_server_would_drop(stub, client):
    with pytest.raises(SconeError) as excinfo:
        client.add("a note", source="notebook.md")
    assert "source" in str(excinfo.value)
    assert stub.requests == [], "the doomed request must not be sent"


def test_add_enforces_the_servers_bounds_locally(stub, client):
    with pytest.raises(SconeError):
        client.add("")
    with pytest.raises(SconeError):
        client.add("x" * 100_001)
    with pytest.raises(SconeError):
        client.add("fine", tags=[f"t{i}" for i in range(11)])
    assert stub.requests == []


def test_recall_maps_its_arguments_onto_the_query_string(stub, client):
    stub.route("GET", "/v1/recall", 200, {"facts": [], "items": []})
    client.recall("checklist", limit=3, as_of="2026-01-01T00:00:00Z", tags=["ops", "wiki"])
    query = stub.requests[0].query
    assert query["q"] == ["checklist"]
    assert query["limit"] == ["3"]
    assert query["as_of"] == ["2026-01-01T00:00:00Z"]
    assert query["tags"] == ["ops,wiki"], "tags are one comma-separated value"


def test_recall_sends_only_q_when_nothing_else_is_given(stub, client):
    stub.route("GET", "/v1/recall", 200, {"facts": [], "items": []})
    client.recall("checklist")
    assert stub.requests[0].query == {"q": ["checklist"]}


def test_recall_refuses_a_blank_query_the_server_would_500_on(stub, client):
    for blank in ("", "   ", "\n\t"):
        with pytest.raises(SconeError):
            client.recall(blank)
    with pytest.raises(SconeError):
        client.recall("x" * 1001)
    assert stub.requests == []


def test_facts_asks_for_closed_ones_only_when_told_to(stub, client):
    stub.set_default(200, {"facts": []})
    client.facts()
    assert stub.requests[0].query == {}
    client.facts(include_closed=True)
    assert stub.requests[1].query == {"all": ["true"]}


def test_close_fact_puts_the_id_in_the_path_and_the_reason_in_the_body(stub, client):
    stub.set_default(200, {"closed": 12, "reason": "superseded"})
    assert client.close_fact(12, "superseded") == 12
    request = stub.requests[0]
    assert request.method == "POST"
    assert request.path == "/v1/facts/12/close"
    assert request.json == {"reason": "superseded"}


def test_close_fact_enforces_the_reason_bounds_locally(stub, client):
    with pytest.raises(SconeError):
        client.close_fact(1, "")
    with pytest.raises(SconeError):
        client.close_fact(1, "x" * 501)
    assert stub.requests == []


# ---------------------------------------------------------------------
# parsing
# ---------------------------------------------------------------------


RECALL_BODY = {
    "facts": [
        {
            "fact_id": 3,
            "subject": "alice",
            "predicate": "works_at",
            "object": "acme",
            "confidence": 0.87,
            "valid_from": "2026-01-01T00:00:00Z",
            "valid_until": None,
            "status": "active",
        }
    ],
    "items": [
        {
            "episode_id": 5,
            "text": "deploy checklist lives in the wiki",
            "score": 0.42,
            "source": None,
            "created_at": "2026-08-30T12:00:00.000Z",
        }
    ],
    "degraded": ["vectors"],
    "returned_bytes": 34,
    "space_bytes": 3400,
    "context_reduction": 0.99,
}


def test_recall_parses_into_facts_and_memories(stub, client):
    stub.route("GET", "/v1/recall", 200, RECALL_BODY)
    pack = client.recall("checklist")

    assert isinstance(pack.items[0], Memory)
    assert pack.items[0].episode_id == 5
    assert pack.items[0].text == "deploy checklist lives in the wiki"
    assert pack.items[0].score == pytest.approx(0.42)
    assert pack.items[0].source is None, "HTTP-ingested notes never carry a source"
    assert pack.items[0].day == "2026-08-30"

    assert isinstance(pack.facts[0], Fact)
    assert pack.facts[0].fact_id == 3
    assert pack.facts[0].subject == "alice"
    assert pack.facts[0].valid_until is None
    assert pack.facts[0].status == "active"

    assert pack.degraded == ["vectors"]
    assert pack.returned_bytes == 34
    assert pack.space_bytes == 3400
    assert pack.context_reduction == pytest.approx(0.99)
    assert len(pack) == 1
    assert [m.text for m in pack] == ["deploy checklist lives in the wiki"]


def test_a_recall_with_no_hits_parses_to_empty_lists(stub, client):
    stub.route("GET", "/v1/recall", 200, {
        "facts": [], "items": [], "degraded": [],
        "returned_bytes": 0, "space_bytes": 0, "context_reduction": 0.0,
    })
    pack = client.recall("nothing here")
    assert pack.facts == []
    assert pack.items == []
    assert len(pack) == 0


def test_profile_facts_parse_without_validity_or_status(stub, client):
    # /v1/profile emits a narrower fact than /v1/recall: no valid_from,
    # no valid_until, no status. Parsing must not require them.
    stub.route("GET", "/v1/profile", 200, {
        "static_facts": [{
            "fact_id": 1,
            "subject": "alice",
            "predicate": "prefers",
            "object": "rust",
            "confidence": 0.9,
        }],
        "dynamic": ["alice ships rust code"],
    })
    profile = client.profile()
    assert isinstance(profile, Profile)
    fact = profile.static_facts[0]
    assert fact.fact_id == 1
    assert fact.predicate == "prefers"
    assert fact.valid_from is None
    assert fact.valid_until is None
    assert fact.status is None
    assert profile.dynamic == ["alice ships rust code"]


def test_an_empty_profile_parses(stub, client):
    stub.route("GET", "/v1/profile", 200, {"static_facts": [], "dynamic": []})
    profile = client.profile()
    assert profile.static_facts == []
    assert profile.dynamic == []


def test_status_parses_every_field(stub, client):
    stub.route("GET", "/v1/status", 200, {
        "space": "default",
        "episodes": 12,
        "chunks": 30,
        "revision": 4,
        "semantic_lane": "paused",
        "pending_distill": 2,
    })
    status = client.status()
    assert isinstance(status, Status)
    assert status.space == "default"
    assert status.episodes == 12
    assert status.chunks == 30
    assert status.revision == 4
    assert status.semantic_lane == "paused"
    assert status.semantic_lane_active is False
    assert status.pending_distill == 2


def test_status_tolerates_a_body_carrying_only_the_space(stub, client):
    stub.route("GET", "/v1/status", 200, {"space": "default"})
    status = client.status()
    assert status.space == "default"
    assert status.episodes == 0
    assert status.semantic_lane is None
    assert status.semantic_lane_active is False


def test_tags_parse_with_their_counts(stub, client):
    stub.route("GET", "/v1/tags", 200,
               {"tags": [{"name": "ops", "count": 3}, {"name": "wiki", "count": 1}]})
    tags = client.tags()
    assert all(isinstance(t, Tag) for t in tags)
    assert [(t.name, t.count) for t in tags] == [("ops", 3), ("wiki", 1)]


def test_a_tag_with_no_count_parses_as_zero(stub, client):
    stub.route("GET", "/v1/tags", 200, {"tags": [{"name": "orphan"}]})
    assert client.tags() == [Tag(name="orphan", count=0)]


def test_a_missing_top_level_array_parses_as_empty(stub, client):
    stub.set_default(200, {})
    assert client.facts() == []
    assert client.tags() == []
    assert client.profile().dynamic == []
    assert client.recall("x").items == []
