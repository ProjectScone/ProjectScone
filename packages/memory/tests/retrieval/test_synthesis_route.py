"""The synthesize route: asked for by name, never chosen by the rule, and honest without a model."""

from __future__ import annotations

import asyncio
import json

import pytest

from scone_memory import HashEmbedder, InMemoryDocumentStore, InMemoryVectorIndex, MemoryEngine
from scone_memory.core.errors import InvalidInput
from scone_memory.providers.llm import FakeChat
from scone_memory.retrieval.router import ROUTES, answer_question

SPACE = "s"
TEXTS = ("Priya moved the launch to March because the audit ran late.",
         "Tomas said the audit found two billing errors in February.",
         "The launch party is booked for the twelfth of March.")
QUESTION = "What happened with the launch?"


async def memory() -> MemoryEngine:
    engine = await MemoryEngine(InMemoryDocumentStore(), InMemoryVectorIndex(), HashEmbedder()).open()
    for text in TEXTS:
        await engine.remember(SPACE, text)
    return engine


def citing_reply(prompt: str) -> str:
    import re
    rows = [{"sentence": f"Note on {m.group(1)}.", "passage": m.group(1), "quote": " ".join(m.group(2).split()[:3])}
            for m in re.finditer(r"^\[(chunk:\d+)\] (.+)$", prompt, flags=re.M)]
    return json.dumps({"notes": rows})


class CitingChat:
    def __init__(self) -> None:
        self.calls = 0

    async def complete(self, system: str, user: str) -> str:
        self.calls += 1
        return citing_reply(user)


async def test_the_rule_never_chooses_synthesis_and_a_named_route_needs_a_model():
    assert "synthesize" not in ROUTES
    engine = await memory()
    chosen = await answer_question(engine, SPACE, QUESTION, synthesis=CitingChat())
    assert chosen.route == "recall", "a model on hand does not change the rule"
    with pytest.raises(InvalidInput, match="needs a model"):
        await answer_question(engine, SPACE, QUESTION, route="synthesize")


async def test_the_synthesize_route_cites_every_sentence_and_records_what_it_read():
    engine = await memory()
    model = CitingChat()
    answered = await answer_question(engine, SPACE, QUESTION, route="synthesize", synthesis=model, limit=2)
    assert answered.route == "synthesize" and "asked for" in answered.why
    assert model.calls == 1
    assert answered.detail["status"] == "synthesized"
    assert answered.detail["passages"]["given"] == 2, "limit bounds the passages read; three are there"
    assert len(answered.detail["sentences"]) == 2 and answered.detail["verified_accuracy"] is False
    for line in answered.text.splitlines():
        assert line.endswith("]") and "[chunk:" in line
    assert answered.shown == {}
    record = answered.record(SPACE)
    assert record["route"] == "synthesize" and record["detail"]["notes"]["kept"] == 2


async def test_a_synthesis_with_nothing_to_say_says_so():
    engine = await memory()
    answered = await answer_question(engine, SPACE, QUESTION, route="synthesize", synthesis=FakeChat(['{"notes": []}']))
    assert answered.detail["status"] == "no_evidence" and answered.text.startswith("nothing to say")


def test_over_http_the_route_needs_the_server_to_have_a_model():
    from fastapi.testclient import TestClient

    from scone_memory.api import create_app

    engine = asyncio.run(memory())
    auth = {"Authorization": "Bearer key-a"}
    with TestClient(create_app(engine, {"key-a": SPACE})) as client:
        refused = client.get("/v1/answer", params={"q": QUESTION, "route": "synthesize"}, headers=auth)
        assert refused.status_code == 422 and "needs a model" in refused.json()["error"]
    with TestClient(create_app(engine, {"key-a": SPACE}, synthesis_factory=CitingChat)) as client:
        said = client.get("/v1/answer", params={"q": QUESTION, "route": "synthesize", "limit": 2}, headers=auth).json()
        assert said["route"] == "synthesize" and said["detail"]["status"] == "synthesized"
        assert len(said["detail"]["sentences"]) == 2
        still = client.get("/v1/answer", params={"q": QUESTION}, headers=auth).json()
        assert still["route"] == "recall", "the rule is unchanged by a model being available"


async def test_the_command_line_refuses_the_route_without_a_configured_model(monkeypatch):
    import io

    from scone_memory.runtime.cli import build_parser, run

    for name in ("SCONE_CHAT_URL", "SCONE_CHAT_MODEL"):
        monkeypatch.delenv(name, raising=False)
    engine = await memory()
    out, err = io.StringIO(), io.StringIO()
    monkeypatch.setattr("sys.stderr", err)
    code = await run(build_parser().parse_args(["--space", SPACE, "answer", QUESTION, "--route", "synthesize"]),
                     engine, io.StringIO(""), out)
    assert code == 2 and "SCONE_CHAT_URL" in err.getvalue()


async def test_the_synthesize_route_takes_a_mode_and_records_it():
    engine = await memory()
    model = CitingChat()
    answered = await answer_question(engine, SPACE, QUESTION, route="synthesize", synthesis=model, limit=3,
                                     synthesis_mode="accumulate")
    assert model.calls == 3, "accumulate asks once per passage"
    assert answered.detail["mode"] == "accumulate" and answered.detail["model_calls"] == 3
    assert answered.detail["passages"]["cited"] == 3
    default = await answer_question(engine, SPACE, QUESTION, route="synthesize", synthesis=CitingChat(), limit=3)
    assert default.detail["mode"] == "evidence"


async def test_a_mode_is_refused_when_unknown_or_without_the_synthesize_route():
    engine = await memory()
    with pytest.raises(InvalidInput, match="mode"):
        await answer_question(engine, SPACE, QUESTION, route="synthesize", synthesis=CitingChat(), synthesis_mode="tree")
    model = CitingChat()
    with pytest.raises(InvalidInput, match="mode"):
        await answer_question(engine, SPACE, QUESTION, route="synthesize", synthesis=model, synthesis_mode="")
    assert model.calls == 0, "an empty mode is refused like any unknown one, not read as the default"
    with pytest.raises(InvalidInput, match="synthesize route"):
        await answer_question(engine, SPACE, QUESTION, synthesis_mode="refine")


def test_over_http_the_mode_is_a_query_parameter():
    from fastapi.testclient import TestClient

    from scone_memory.api import create_app

    engine = asyncio.run(memory())
    auth = {"Authorization": "Bearer key-a"}
    with TestClient(create_app(engine, {"key-a": SPACE}, synthesis_factory=CitingChat)) as client:
        said = client.get("/v1/answer", params={"q": QUESTION, "route": "synthesize", "limit": 2,
                                                "synthesis_mode": "refine"}, headers=auth).json()
        assert said["detail"]["mode"] == "refine" and said["detail"]["status"] == "synthesized"
        refused = client.get("/v1/answer", params={"q": QUESTION, "route": "synthesize", "synthesis_mode": "tree"},
                             headers=auth)
        assert refused.status_code == 422 and "mode" in refused.json()["error"]
        empty = client.get("/v1/answer", params={"q": QUESTION, "route": "synthesize", "synthesis_mode": ""},
                           headers=auth)
        assert empty.status_code == 422 and "mode" in empty.json()["error"]


async def test_the_command_line_passes_the_mode(monkeypatch):
    import io

    from scone_memory.runtime import config as runtime_config
    from scone_memory.runtime.cli import build_parser, run

    model = CitingChat()
    monkeypatch.setattr(runtime_config, "build_chat", lambda settings: model)
    engine = await memory()
    out = io.StringIO()
    args = build_parser().parse_args(["--space", SPACE, "--json", "answer", QUESTION, "--route", "synthesize",
                                      "--limit", "2", "--synthesis-mode", "accumulate"])
    code = await run(args, engine, io.StringIO(""), out)
    assert code == 0 and model.calls == 2
    assert json.loads(out.getvalue())["detail"]["mode"] == "accumulate"
