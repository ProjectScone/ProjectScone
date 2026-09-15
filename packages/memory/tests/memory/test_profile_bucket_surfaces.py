"""Profile buckets on the route and the command line: off unless asked for.

``GET /v1/profile`` and ``scone profile`` answer as they always have. Asked
for buckets, they add them beside the profile, with each bucket's bounds,
whether they cut, and the rule that placed each claim.
"""
from __future__ import annotations

import asyncio
import io
import json

from fastapi.testclient import TestClient

from scone_memory import HashEmbedder, InMemoryDocumentStore, InMemoryVectorIndex, MemoryEngine
from scone_memory.api import create_app
from scone_memory.runtime.cli import main
from scone_memory.testing import Clock

AUTH = {"Authorization": "Bearer k"}


def seeded() -> MemoryEngine:
    async def build() -> MemoryEngine:
        engine = await MemoryEngine(InMemoryDocumentStore(), InMemoryVectorIndex(), HashEmbedder(),
                                    clock=Clock("2025-06-01T00:00:00.000Z")).open()
        await engine.assert_fact("default", "mark", "name", "Mark", valid_from="2020-01-01T00:00:00Z")
        await engine.assert_fact("default", "mark", "role", "surveyor", valid_from="2021-01-01T00:00:00Z")
        await engine.assert_fact("default", "mark", "works_on", "the crane survey", valid_from="2025-05-20T00:00:00Z")
        return engine
    return asyncio.run(build())


def test_the_route_answers_as_before_unless_buckets_are_asked_for():
    with TestClient(create_app(seeded(), {"k": "default"})) as client:
        plain = client.get("/v1/profile", headers=AUTH).json()
        assert "buckets" not in plain
        body = client.get("/v1/profile?buckets=both&static_limit=1", headers=AUTH).json()
        assert {key: body[key] for key in plain if key != "coverage"} == {
            key: plain[key] for key in plain if key != "coverage"}
        buckets = body["buckets"]
        assert [fact["object"] for fact in buckets["static"]] == ["surveyor"]
        assert [fact["object"] for fact in buckets["dynamic"]] == ["the crane survey"]
        coverage = buckets["coverage"]
        assert coverage["static"]["cut"] == "count" and coverage["static"]["omitted"] == 1
        assert coverage["dynamic"]["cut"] is None
        dynamic_id = str(buckets["dynamic"][0]["fact_id"])
        assert coverage["placed"][dynamic_id]["rule"] == "tenure"
        only = client.get("/v1/profile?buckets=dynamic&dynamic_max_bytes=100", headers=AUTH).json()["buckets"]
        assert only["static"] == [] and "static" not in only["coverage"]
        assert only["coverage"]["dynamic"]["max_bytes"] == 100


def test_the_route_refuses_bad_bucket_requests():
    with TestClient(create_app(seeded(), {"k": "default"})) as client:
        assert client.get("/v1/profile?buckets=all", headers=AUTH).status_code == 422
        assert client.get("/v1/profile?buckets=both&static_limit=0", headers=AUTH).status_code == 422
        refused = client.get("/v1/profile?static_limit=3", headers=AUTH)
        assert refused.status_code == 422 and "buckets" in refused.json()["error"]


def test_the_cli_prints_buckets_only_when_asked(tmp_path):
    env = {"SCONE_SQLITE_PATH": str(tmp_path / "memory.db"), "SCONE_EMBEDDER": "hash"}
    for args in (["assert", "mark", "name", "Mark", "--valid-from", "2020-01-01"],
                 ["assert", "mark", "works_on", "the crane survey"]):
        assert main(args, env=env, out=io.StringIO()) == 0
    out = io.StringIO()
    assert main(["--json", "profile"], env=env, out=out) == 0
    assert "buckets" not in json.loads(out.getvalue())
    out = io.StringIO()
    assert main(["--json", "profile", "--buckets", "both", "--dynamic-limit", "1"], env=env, out=out) == 0
    buckets = json.loads(out.getvalue())["buckets"]
    assert [fact["object"] for fact in buckets["static"]] == ["Mark"]
    assert [fact["object"] for fact in buckets["dynamic"]] == ["the crane survey"]
    assert buckets["coverage"]["dynamic"]["limit"] == 1
    out = io.StringIO()
    assert main(["profile", "--buckets", "static", "--static-max-bytes", "300"], env=env, out=out) == 0
    text = out.getvalue()
    assert "static (1 shown" in text and "rule tenure" in text and "dynamic" not in text
