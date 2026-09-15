"""SCONE_FOLLOWUP_QUERIES turns follow-up queries on for served text conversations, and only coherently."""

from __future__ import annotations

import pytest

from scone_memory import HashEmbedder, InMemoryDocumentStore, InMemoryVectorIndex, MemoryEngine
from scone_memory.core.errors import InvalidInput
from scone_memory.providers.llm import OpenAICompatibleChat
from scone_memory.runtime.config import Settings
from scone_memory.runtime.conversation_followup import build_followup


def environment(tmp_path, **changes: str) -> dict[str, str]:
    return {"SCONE_API_KEY": "fixture-key", "SCONE_CONVERSATIONS_JOURNAL": str(tmp_path / "sessions.db"),
            "SCONE_FOLLOWUP_QUERIES": "carry", **changes}


REWRITE = {"SCONE_FOLLOWUP_QUERIES": "rewrite", "SCONE_FOLLOWUP_URL": "http://127.0.0.1:11434/v1",
           "SCONE_FOLLOWUP_MODEL": "condenser", "SCONE_FOLLOWUP_TIMEOUT": "4"}


def test_off_by_default_adds_no_conversation_options():
    settings = Settings.from_env({})
    assert settings.followup_queries == "off" and build_followup(settings) == {}


def test_carry_reads_the_environment_and_becomes_a_conversation_option(tmp_path):
    settings = Settings.from_env(environment(tmp_path))
    assert settings.followup_queries == "carry"
    assert build_followup(settings) == {"followup_queries": "carry"}


def test_rewrite_builds_its_own_self_hosted_model_with_its_own_deadline(tmp_path):
    settings = Settings.from_env(environment(tmp_path, **REWRITE, SCONE_FOLLOWUP_API_KEY="secret"))
    options = build_followup(settings)
    model = options["followup_model"]
    assert options["followup_queries"] == "rewrite" and options["followup_timeout"] == 4.0
    assert isinstance(model, OpenAICompatibleChat) and model.model == "condenser" and model.api_key == "secret"
    assert model.trust_env is False and model.timeout == 4.0
    assert "secret" not in repr(settings)


@pytest.mark.parametrize("changes", [
    {"SCONE_FOLLOWUP_QUERIES": "sometimes"},
    {**REWRITE, "SCONE_FOLLOWUP_QUERIES": "sometimes"},
    {"SCONE_CONVERSATIONS_JOURNAL": ""},
    {"SCONE_ADAPTIVE_RETRIEVAL": "1", "SCONE_ADAPTIVE_URL": "http://127.0.0.1:11434/v1", "SCONE_ADAPTIVE_MODEL": "a"},
    {"SCONE_FOLLOWUP_URL": "http://127.0.0.1:11434/v1"},
    {"SCONE_FOLLOWUP_MODEL": "condenser"},
    {"SCONE_FOLLOWUP_API_KEY": "secret"},
    {"SCONE_FOLLOWUP_TIMEOUT": "4"},
    {**REWRITE, "SCONE_FOLLOWUP_URL": ""},
    {**REWRITE, "SCONE_FOLLOWUP_MODEL": ""},
    {**REWRITE, "SCONE_FOLLOWUP_URL": "https://api.openai.com/v1"},
    {**REWRITE, "SCONE_FOLLOWUP_API_KEY": "bad\nkey"},
    {**REWRITE, "SCONE_FOLLOWUP_API_KEY": "   "},
    {**REWRITE, "SCONE_FOLLOWUP_MODEL": "bad\tmodel"},
    {**REWRITE, "SCONE_FOLLOWUP_TIMEOUT": "61"},
    {**REWRITE, "SCONE_FOLLOWUP_TIMEOUT": "nan"},
])
def test_incoherent_followup_configuration_refuses_startup(tmp_path, changes):
    with pytest.raises(InvalidInput):
        Settings.from_env(environment(tmp_path, **changes))


@pytest.mark.parametrize("timeout", [True, "4", -1.0, 0.0])
def test_a_directly_built_rewrite_timeout_must_be_a_positive_number(tmp_path, timeout):
    with pytest.raises(InvalidInput, match="SCONE_FOLLOWUP_TIMEOUT"):
        Settings(conversations_journal=str(tmp_path / "j.db"), followup_queries="rewrite",
                 followup_url="http://127.0.0.1:11434/v1", followup_model="condenser", followup_timeout=timeout)


@pytest.mark.parametrize("mode", ["sometimes", "Carry"])
def test_an_unknown_mode_is_named_as_such(tmp_path, mode):
    with pytest.raises(InvalidInput, match="SCONE_FOLLOWUP_QUERIES must be off, carry, or rewrite"):
        Settings.from_env(environment(tmp_path, SCONE_FOLLOWUP_QUERIES=mode))


def test_tool_mode_chooses_its_own_searches(tmp_path):
    tools = environment(tmp_path, SCONE_CONVERSATIONS_TOOL_MODE="native",
                        SCONE_MODEL_CONNECTIONS=str(tmp_path / "models.json"))
    assert Settings.from_env({**tools, "SCONE_FOLLOWUP_QUERIES": "off"}).conversations_tool_mode == "native"
    with pytest.raises(InvalidInput, match="tool mode"):
        Settings.from_env(tools)


def test_rewrite_without_an_endpoint_names_what_it_needs(tmp_path):
    with pytest.raises(InvalidInput, match="SCONE_FOLLOWUP_URL and SCONE_FOLLOWUP_MODEL"):
        Settings.from_env(environment(tmp_path, **{**REWRITE, "SCONE_FOLLOWUP_URL": ""}))


def test_a_setting_that_does_nothing_without_the_mode_names_the_mode():
    with pytest.raises(InvalidInput, match="SCONE_FOLLOWUP_QUERIES=rewrite"):
        Settings.from_env({"SCONE_FOLLOWUP_MODEL": "condenser"})


async def test_the_served_conversation_path_hands_the_option_to_the_runtime(tmp_path):
    import httpx

    from scone_memory.api.conversations import create_conversation_app

    received: list[dict[str, object]] = []

    class Runtime:
        async def reply(self, text):
            return {}

        async def close(self):
            pass

    def factory(space, sid, **options):
        received.append(options)
        return Runtime()

    engine = await MemoryEngine(InMemoryDocumentStore(), InMemoryVectorIndex(), HashEmbedder()).open()
    app = create_conversation_app(engine, {"fixture-key": "default"}, tmp_path / "sessions.db", factory,
                                  followup={"followup_queries": "carry"})
    try:
        async with app.router.lifespan_context(app):
            async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://scone.test",
                                         headers={"Authorization": "Bearer fixture-key"}) as client:
                created = await client.post("/v1/conversations", json={"request_id": "session", "capture": True})
    finally:
        await engine.close()
    assert created.status_code == 200, created.text
    assert received == [{"followup_queries": "carry"}]


async def test_a_served_conversation_carries_the_earlier_turn_into_the_follow_up(tmp_path, monkeypatch):
    import asyncio
    import sys
    from types import ModuleType

    import httpx

    from scone_memory.api.__main__ import build_app
    from scone_memory.realtime.events import ReplyCompleted, TextDelta

    class Model:
        async def respond(self, messages):
            yield TextDelta("Noted.")
            yield ReplyCompleted()

        async def aclose(self):
            pass

    module = ModuleType("followup_served_provider")
    module.create = Model  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, module.__name__, module)
    env = environment(tmp_path, SCONE_CONVERSATIONS_MODEL_FACTORY="followup_served_provider:create")
    engine = await MemoryEngine(InMemoryDocumentStore(), InMemoryVectorIndex(), HashEmbedder()).open()
    await engine.remember("default", "Alice Chen has worked at Acme Robotics since May 2021.", kind="file")
    app = build_app(Settings.from_env(env), engine)
    receipts = []
    try:
        async with app.router.lifespan_context(app):
            async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://scone.test",
                                         headers={"Authorization": "Bearer fixture-key"}) as client:
                session = (await client.post("/v1/conversations", json={"request_id": "session", "capture": True})).json()
                base = "/v1/conversations/" + session["session_id"]
                revision = session["revision"]
                for number, text in enumerate(["Where does Alice Chen work?", "since when?"]):
                    sent = await client.post(base + "/turns", json={"request_id": f"turn{number}", "text": text,
                                                                     "expected_revision": revision})
                    assert sent.status_code == 202, sent.text
                    async with asyncio.timeout(10):
                        while (receipt := (await client.get(base + f"/turns/turn{number}")).json())["status"] == "pending":
                            await asyncio.sleep(0.01)
                    assert receipt["status"] == "completed", receipt
                    receipts.append(receipt)
                    revision = (await client.get(base)).json()["revision"]
    finally:
        await engine.close()
    first, second = (receipt["result"]["memory_context"]["followup"] for receipt in receipts)
    assert "first turn" in first["reason"] and first["applied"] is False
    assert second["carried"] == ["Alice Chen"] and second["query"] == "Alice Chen since when?"
