"""The standard server wires configured answer review into native sessions."""
from __future__ import annotations

import asyncio
import json
import sys
from types import ModuleType

import pytest
from fastapi.testclient import TestClient
import httpx

from scone_memory import HashEmbedder, InMemoryDocumentStore, InMemoryVectorIndex, MemoryEngine
from scone_memory.api.__main__ import build_app
from scone_memory.runtime.config import Settings
from scone_memory.core.errors import InvalidInput

AUTH = {"Authorization": "Bearer fixture-key"}


def environment(tmp_path):
    return {"SCONE_API_KEY": "fixture-key", "SCONE_CONVERSATIONS_JOURNAL": str(tmp_path / "sessions.db"),
            "SCONE_ANSWER_REVIEW_POLICY": "require_supported", "SCONE_ANSWER_REVIEW_URL": "http://127.0.0.1:11434/v1",
            "SCONE_ANSWER_REVIEW_MODEL": "review-model", "SCONE_ANSWER_REVIEW_TIMEOUT": "12"}


def test_review_settings_are_explicit_and_do_not_borrow_extraction_credentials(tmp_path):
    settings = Settings.from_env(environment(tmp_path) | {"SCONE_CHAT_API_KEY": "extraction-key"})
    assert settings.answer_review_policy == "require_supported"
    assert settings.answer_review_model == "review-model"
    assert settings.answer_review_timeout == 12
    assert settings.answer_review_api_key is None
    assert Settings.from_env({}).answer_review_policy == "off"


@pytest.mark.parametrize("changes", [
    {"SCONE_ANSWER_REVIEW_POLICY": "magic"}, {"SCONE_ANSWER_REVIEW_POLICY": "off"},
    {"SCONE_ANSWER_REVIEW_URL": ""}, {"SCONE_ANSWER_REVIEW_MODEL": ""},
    {"SCONE_ANSWER_REVIEW_TIMEOUT": "nan"}, {"SCONE_ANSWER_REVIEW_TIMEOUT": "0"},
    {"SCONE_ANSWER_REVIEW_TIMEOUT": "181"}, {"SCONE_ANSWER_REVIEW_URL": "https://api.openai.com/v1"},
    {"SCONE_CONVERSATIONS_JOURNAL": ""},
    {"SCONE_ANSWER_REVIEW_QUOTE_MODE": "auto"}, {"SCONE_ANSWER_REVIEW_QUOTE_MODE": ""},
])
def test_incomplete_or_invalid_review_configuration_refuses_startup(tmp_path, changes):
    with pytest.raises(InvalidInput):
        Settings.from_env(environment(tmp_path) | changes)


def test_standard_serve_passes_review_policy_to_custom_model_sessions(tmp_path, monkeypatch):
    from scone_memory.realtime import text
    from scone_memory.providers.answer_reviewer import SelfHostedAnswerReviewer

    received = []
    original = text.TextConversation

    def native(*args, **kwargs):
        received.append(kwargs)
        return original(*args, **kwargs)

    monkeypatch.setattr(text, "TextConversation", native)
    module = ModuleType("review_fixture_provider")
    module.create = lambda: None
    monkeypatch.setitem(sys.modules, module.__name__, module)
    settings = Settings.from_env(environment(tmp_path) | {
        "SCONE_CONVERSATIONS_MODEL_FACTORY": "review_fixture_provider:create"})
    engine = asyncio.run(MemoryEngine(InMemoryDocumentStore(), InMemoryVectorIndex(), HashEmbedder()).open())
    try:
        with TestClient(build_app(settings, engine)) as client:
            capabilities = client.get("/v1/conversations/capabilities", headers=AUTH).json()
            assert capabilities["answer_review"] == {"configured": True, "policy": "require_supported"}
            response = client.post("/v1/conversations", headers=AUTH,
                                   json={"request_id": "review-session", "capture": True})
            assert response.status_code == 200
            assert isinstance(received[0]["answer_reviewer"], SelfHostedAnswerReviewer)
            assert received[0]["review_policy"] == "require_supported"
            assert received[0]["review_limits"].timeout_s == 12
    finally:
        asyncio.run(engine.close())


@pytest.mark.parametrize("mode,policy,outcome", [
    (mode, "require_supported", outcome)
    for mode in ("custom", "persona", "saved") for outcome in ("supported", "uncertain")
] + [
    ("custom", "report", "uncertain"),
    ("custom", "report", "provider_failure"),
    ("custom", "require_supported", "provider_failure"),
    ("custom", "report", "deleted_source"),
    ("custom", "require_supported", "deleted_source"),
])
async def test_reviewed_http_turn_buffers_draft_and_obeys_publication_policy(tmp_path, monkeypatch, mode, policy, outcome):
    from scone_memory.providers import answer_reviewer
    from scone_memory.realtime.answer_review import AnswerIssue, AnswerReviewDecision
    from scone_memory.realtime.events import ReplyCompleted, TextDelta
    from scone_memory.realtime.providers import ProviderRegistry
    from scone_memory.runtime.model_connections import ModelConnection, ModelConnectionStore
    from scone_memory.runtime import model_runtime

    entered, release = asyncio.Event(), asyncio.Event()

    class Model:
        async def respond(self, messages):
            yield TextDelta("Mira works on Billing.")
            yield ReplyCompleted()

        async def aclose(self):
            pass

    class Reviewer:
        async def review(self, question, answer, evidence, evidence_ids):
            assert "Mira works on Platform." in evidence
            entered.set()
            await release.wait()
            if outcome == "provider_failure":
                raise RuntimeError("PRIVATE provider credential and source details")
            if outcome == "deleted_source":
                await engine.forget("default", source.episode_id)
                return AnswerReviewDecision(status="supported")
            if outcome == "uncertain":
                return AnswerReviewDecision(status="uncertain")
            if "Billing" in answer:
                return AnswerReviewDecision(status="needs_revision", issues=(AnswerIssue(
                    code="unsupported_claim", answer_quote="Billing", evidence_ids=evidence_ids[:1]),),
                    revised_answer="Mira works on Platform.")
            return AnswerReviewDecision(status="supported")

    monkeypatch.setattr(answer_reviewer, "SelfHostedAnswerReviewer", lambda *args, **kwargs: Reviewer())
    module = ModuleType("review_turn_provider")
    module.create = Model
    monkeypatch.setitem(sys.modules, module.__name__, module)
    env = environment(tmp_path)
    env["SCONE_ANSWER_REVIEW_POLICY"] = policy
    body = {"request_id": "session", "capture": True}
    if mode == "custom":
        env["SCONE_CONVERSATIONS_MODEL_FACTORY"] = "review_turn_provider:create"
    elif mode == "persona":
        path = tmp_path / "personas.json"
        path.write_text(json.dumps([{"schema_version": 1, "id": "helper", "name": "Helper",
            "instructions": "Answer briefly.", "reply": {"provider": "stub", "model": "reply"},
            "transcription": {"provider": "stub", "model": "ears"},
            "speech": {"provider": "stub", "model": "mouth", "voice": "alto"}}]))
        module.registry = lambda: ProviderRegistry(reply={("stub", "reply"): Model},
            transcription={("stub", "ears"): lambda: None}, speech={("stub", "mouth", "alto"): lambda: None})
        env.update(SCONE_CONVERSATIONS_PERSONAS=str(path), SCONE_CONVERSATIONS_REGISTRY="review_turn_provider:registry")
        body["persona"] = "helper"
    else:
        path = tmp_path / "models.json"
        store = ModelConnectionStore(path, {})
        store.replace("chat", ModelConnection(base_url="http://127.0.0.1:11434/v1", model="reply"), expected_revision=0)
        env["SCONE_MODEL_CONNECTIONS"] = str(path)
        monkeypatch.setattr(model_runtime, "OpenAICompatibleTextModel", lambda *args, **kwargs: Model())
    engine = await MemoryEngine(InMemoryDocumentStore(), InMemoryVectorIndex(), HashEmbedder()).open()
    try:
        source = await engine.remember("default", "Mira works on Platform.")
        app = build_app(Settings.from_env(env), engine)
        async with app.router.lifespan_context(app):
            async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://scone.test", headers=AUTH) as client:
                created = await client.post("/v1/conversations", json=body)
                assert created.status_code == 200, created.text
                session = created.json()
                base = "/v1/conversations/" + session["session_id"]
                sent = await client.post(base + "/turns", json={"request_id": "turn", "text": "Which team does Mira work on?",
                    "expected_revision": session["revision"]})
                assert sent.status_code == 202
                await asyncio.wait_for(entered.wait(), 2)
                assert (await client.get(base + "/turns/turn")).json()["status"] == "pending"
                episodes = await engine.episodes("default", {"session_id": session["session_id"]})
                assert not any(item.metadata.get("role") == "assistant" for item in episodes)
                release.set()
                for _ in range(100):
                    receipt = (await client.get(base + "/turns/turn")).json()
                    if receipt["status"] != "pending":
                        break
                    await asyncio.sleep(0.01)
                stream = await client.get(base + "/turns/turn/stream")
                assert "PRIVATE" not in stream.text + json.dumps(receipt)
                episodes = await engine.episodes("default", {"session_id": session["session_id"]})
                answers = [item.content for item in episodes if item.metadata.get("role") == "assistant"]
                if outcome == "supported":
                    assert "Billing" not in stream.text
                    assert receipt["status"] == "completed", receipt
                    assert receipt["result"]["text"] == "Mira works on Platform."
                    assert receipt["result"]["answer_review"]["revised"] is True
                    assert answers == ["Mira works on Platform."]
                elif policy == "report" and outcome != "deleted_source":
                    assert receipt["status"] == "completed", receipt
                    assert receipt["result"]["text"] == "Mira works on Billing."
                    assert receipt["result"]["answer_review"]["status"] == (
                        "unavailable" if outcome == "provider_failure" else "uncertain")
                    assert receipt["result"]["answer_review"]["source_status"] == "retained"
                    assert "event: terminal" in stream.text
                    assert answers == ["Mira works on Billing."]
                else:
                    assert "Billing" not in stream.text
                    assert receipt["status"] == "failed", receipt
                    if outcome == "deleted_source":
                        assert receipt["answer_review"]["source_status"] == "stale"
                    else:
                        assert receipt["answer_review"]["status"] == (
                            "unavailable" if outcome == "provider_failure" else "uncertain")
                    assert "review" in receipt["error"].lower()
                    assert answers == []
    finally:
        release.set()
        await engine.close()
    assert "PRIVATE" not in (tmp_path / "sessions.db").read_bytes().decode("utf-8", errors="ignore")


@pytest.mark.parametrize('quote_mode', ['text', 'spans'])
def test_review_connection_receives_only_its_explicit_credentials(tmp_path, monkeypatch, quote_mode):
    from scone_memory.providers import answer_reviewer
    from scone_memory.runtime.conversation_review import build_conversation_review

    received = []

    class Reviewer:
        def __init__(self, *args, **kwargs):
            received.append((args, kwargs))

        async def review(self, *args):
            raise AssertionError("configuration must not invoke inference")

    monkeypatch.setattr(answer_reviewer, "SelfHostedAnswerReviewer", Reviewer)
    configured = build_conversation_review(Settings.from_env(environment(tmp_path) | {
        "SCONE_CHAT_API_KEY": "extraction-key", "SCONE_ANSWER_REVIEW_API_KEY": "review-key",
        "SCONE_ANSWER_REVIEW_QUOTE_MODE": quote_mode}))
    assert configured is not None
    assert received == [(("http://127.0.0.1:11434/v1", "review-model"),
                        {"api_key": "review-key", "timeout": 12.0, "quote_mode": quote_mode})]


def test_span_protocol_requires_an_enabled_reviewer():
    assert Settings.from_env({}).answer_review_quote_mode == 'text'
    with pytest.raises(InvalidInput, match='require SCONE_ANSWER_REVIEW_POLICY'):
        Settings.from_env({'SCONE_ANSWER_REVIEW_QUOTE_MODE':'spans'})


@pytest.mark.parametrize("review_enabled", [True, False])
def test_review_capability_does_not_imply_text_model_availability(tmp_path, review_enabled):
    env = environment(tmp_path) if review_enabled else {
        "SCONE_API_KEY": "fixture-key", "SCONE_CONVERSATIONS_JOURNAL": str(tmp_path / "sessions.db")}
    engine = asyncio.run(MemoryEngine(InMemoryDocumentStore(), InMemoryVectorIndex(), HashEmbedder()).open())
    try:
        with TestClient(build_app(Settings.from_env(env), engine)) as client:
            capabilities = client.get("/v1/conversations/capabilities", headers=AUTH).json()
            assert capabilities["text_configured"] is False
            assert capabilities["answer_review"] == {
                "configured": review_enabled, "policy": "require_supported" if review_enabled else "off"}
    finally:
        asyncio.run(engine.close())


async def test_malformed_review_failure_is_sanitized_and_settles_turn(tmp_path):
    from scone_memory.api.conversations import create_conversation_app
    from scone_memory.realtime.text import _AnswerReviewFailure

    class BrokenRuntime:
        async def reply(self, text):
            error = _AnswerReviewFailure("PRIVATE provider detail", {})
            error.answer_review = {"status": "PRIVATE invalid provider result"}
            raise error

        async def close(self):
            pass

    engine = await MemoryEngine(InMemoryDocumentStore(), InMemoryVectorIndex(), HashEmbedder()).open()
    journal = tmp_path / "sessions.db"
    app = create_conversation_app(engine, {"fixture-key": "default"}, journal, lambda *args: BrokenRuntime())
    try:
        async with app.router.lifespan_context(app):
            async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://scone.test", headers=AUTH) as client:
                session = (await client.post("/v1/conversations", json={"request_id": "session", "capture": True})).json()
                base = "/v1/conversations/" + session["session_id"]
                sent = await client.post(base + "/turns", json={"request_id": "turn", "text": "Hello",
                    "expected_revision": session["revision"]})
                assert sent.status_code == 202
                for _ in range(100):
                    receipt = (await client.get(base + "/turns/turn")).json()
                    if receipt["status"] != "pending":
                        break
                    await asyncio.sleep(0.01)
                assert receipt["status"] == "failed", receipt
                assert receipt["answer_review"]["errors"] == ["invalid_review"]
                assert receipt["answer_review"]["source_status"] == "unavailable"
                assert "PRIVATE" not in json.dumps(receipt)
    finally:
        await engine.close()
    assert "PRIVATE" not in journal.read_bytes().decode("utf-8", errors="ignore")
