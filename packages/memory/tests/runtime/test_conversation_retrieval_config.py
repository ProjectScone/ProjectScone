"""The served conversation path exposes bounded, scoped adaptive retrieval."""
import asyncio
import json
import sys
from types import ModuleType

import httpx
import pytest

from scone_memory import HashEmbedder, InMemoryDocumentStore, InMemoryVectorIndex, MemoryEngine
from scone_memory.api.__main__ import build_app
from scone_memory.core.errors import InvalidInput
from scone_memory.runtime.config import Settings

AUTH = {"Authorization": "Bearer fixture-key"}


def environment(tmp_path):
    return {"SCONE_API_KEY": "fixture-key", "SCONE_CONVERSATIONS_JOURNAL": str(tmp_path / "sessions.db"),
        "SCONE_ADAPTIVE_RETRIEVAL": "1", "SCONE_ADAPTIVE_URL": "http://127.0.0.1:11434/v1",
        "SCONE_ADAPTIVE_MODEL": "assessor", "SCONE_ADAPTIVE_TIMEOUT": "5", "SCONE_ADAPTIVE_GRAPH_HOPS": "3"}


@pytest.mark.parametrize("changes", [
    {"SCONE_ADAPTIVE_RETRIEVAL": "maybe"}, {"SCONE_ADAPTIVE_RETRIEVAL": "0"},
    {"SCONE_ADAPTIVE_URL": ""}, {"SCONE_ADAPTIVE_MODEL": ""},
    {"SCONE_ADAPTIVE_URL": "https://api.openai.com/v1"}, {"SCONE_ADAPTIVE_API_KEY": "bad\nkey"},
    {"SCONE_ADAPTIVE_TIMEOUT": "nan"}, {"SCONE_ADAPTIVE_TIMEOUT": "0"},
    {"SCONE_ADAPTIVE_MAX_ROUNDS": "5"}, {"SCONE_ADAPTIVE_MAX_QUERIES": "13"},
    {"SCONE_ADAPTIVE_CANDIDATE_LIMIT": "101"}, {"SCONE_ADAPTIVE_MAX_EVIDENCE_BYTES": "128001"},
    {"SCONE_ADAPTIVE_GRAPH_HOPS": "7"}, {"SCONE_CONVERSATIONS_JOURNAL": ""},
])
def test_invalid_adaptive_configuration_refuses_startup(tmp_path, changes):
    with pytest.raises(InvalidInput):
        Settings.from_env(environment(tmp_path) | changes)


@pytest.mark.parametrize('changes', [
    {'SCONE_ADAPTIVE_SEARCH_HISTORY': 'maybe'},
    {'SCONE_ADAPTIVE_SEARCH_HISTORY': '1', 'SCONE_ADAPTIVE_RETRIEVAL': '0'},
])
def test_invalid_search_history_configuration_refuses_startup(tmp_path, changes):
    with pytest.raises(InvalidInput):
        Settings.from_env(environment(tmp_path) | changes)


def test_search_history_alone_requires_adaptive_retrieval():
    with pytest.raises(InvalidInput, match='SCONE_ADAPTIVE_RETRIEVAL'):
        Settings.from_env({'SCONE_ADAPTIVE_SEARCH_HISTORY': '1'})


async def test_configured_history_reaches_real_assessor_request(tmp_path, monkeypatch, engine):
    from scone_memory.providers import evidence_assessor
    from scone_memory.runtime.conversation_retrieval import build_adaptive_retrieval
    from scone_memory.retrieval.recall_scope import RecallScope

    requests = []

    def respond(request):
        body = json.loads(request.content)
        requests.append(body)
        data = json.loads(body['messages'][1]['content'])
        content = json.dumps({'status': 'sufficient', 'selected_ids': [data['candidates'][0]['id']],
                              'followup_queries': []})
        return httpx.Response(200, json={'choices': [{'message': {'content': content}, 'finish_reason': 'stop'}]})

    original = evidence_assessor.SelfHostedEvidenceAssessor
    monkeypatch.setattr(evidence_assessor, 'SelfHostedEvidenceAssessor',
        lambda *args, **kwargs: original(*args, **kwargs, transport=httpx.MockTransport(respond)))
    settings = Settings.from_env(environment(tmp_path) | {
        'SCONE_ADAPTIVE_SEARCH_HISTORY': '1', 'SCONE_ADAPTIVE_GRAPH_HOPS': '0'})
    await engine.remember('alpha', 'Cedar retains Aster records in a private journal.')
    strategy = build_adaptive_retrieval(settings, engine)
    result = await strategy.retrieve('alpha', 'Cedar Aster journal', scope=RecallScope.validated())
    assert result.status == 'sufficient' and result.recall.items
    data = json.loads(requests[0]['messages'][1]['content'])
    assert data['search_history']['searches'][0]['query'] == 'Cedar Aster journal'
    assert data['search_history']['rounds_remaining'] == 2
    assert len(requests) == 1


def test_assessor_configuration_uses_only_its_own_credentials(tmp_path, monkeypatch):
    from dataclasses import replace
    from scone_memory.providers import evidence_assessor
    from scone_memory.runtime.conversation_retrieval import build_adaptive_retrieval

    received = []

    class Assessor:
        def __init__(self, *args, **kwargs):
            received.append((args, kwargs))

        async def assess(self, *args):
            pytest.fail("configuration must not invoke inference")

    monkeypatch.setattr(evidence_assessor, "SelfHostedEvidenceAssessor", Assessor)
    settings = Settings.from_env(environment(tmp_path) | {"SCONE_CHAT_API_KEY": "extraction-secret"})
    engine = MemoryEngine(InMemoryDocumentStore(), InMemoryVectorIndex(), HashEmbedder())
    strategy = build_adaptive_retrieval(settings, engine)
    assert strategy.memory is engine
    assert strategy.graph_limits.max_hops == 3
    assert strategy.limits.timeout_s == 5
    assert strategy.evidence_policy == "original_and_selected"
    assert strategy.failure_policy == strategy.empty_selection_policy == "retain_verified"
    assert received[0][0] == ("http://127.0.0.1:11434/v1", "assessor")
    assert received[0][1] == {"api_key": None, "timeout": 5.0, "group_relations": True, "max_evidence_bytes": 16000}
    assert build_adaptive_retrieval(Settings.from_env({}), engine) is None
    explicit = replace(settings, adaptive_api_key="assessor-secret")
    build_adaptive_retrieval(explicit, engine)
    assert received[1][1]["api_key"] == "assessor-secret"
    assert "assessor-secret" not in repr(explicit)


@pytest.mark.parametrize("mode", ["custom", "saved", "persona"])
@pytest.mark.parametrize("outcome", ["followup", "provider_failure", "deleted_bridge"])
async def test_served_adaptive_graph_reaches_native_reply_with_scope_and_receipts(tmp_path, monkeypatch, mode, outcome, engine):
    from scone_memory.providers import evidence_assessor
    from scone_memory.realtime.events import ReplyCompleted, TextDelta
    from scone_memory.realtime.providers import ProviderRegistry
    from scone_memory.retrieval.adaptive import EvidenceDecision
    from scone_memory.retrieval.multihop import BoundedIncidentLinks, BoundedSubjectFacts
    from scone_memory.runtime import model_runtime
    from scone_memory.runtime.model_connections import ModelConnection, ModelConnectionStore

    calls, requests = [], []
    graph_supported = isinstance(engine.documents, BoundedIncidentLinks) and isinstance(engine.documents, BoundedSubjectFacts)

    class Assessor:
        async def assess(self, question, candidates):
            calls.append((question, candidates))
            assert all("PRIVATE" not in candidate.text for candidate in candidates)
            ids = tuple(candidate.id for candidate in candidates if candidate.id.startswith("fact:"))
            if outcome == "provider_failure":
                raise RuntimeError("PRIVATE assessor credential")
            if outcome == "deleted_bridge":
                await engine.forget("default", episodes[1].episode_id)
            return EvidenceDecision(status="insufficient" if outcome == "followup" and len(calls) == 1 else "sufficient",
                selected_ids=ids, selected_groups=(ids,) if len(ids) > 1 else (),
                followup_queries=("beacon",) if outcome == "followup" and len(calls) == 1 else ())

    class Model:
        async def respond(self, messages):
            requests.append(messages)
            yield TextDelta("Recorded evidence inspected.")
            yield ReplyCompleted()

        async def aclose(self):
            pass

    monkeypatch.setattr(evidence_assessor, "SelfHostedEvidenceAssessor", lambda *args, **kwargs: Assessor())
    module = ModuleType("adaptive_http_provider")
    module.create = Model
    monkeypatch.setitem(sys.modules, module.__name__, module)
    env = environment(tmp_path)
    reviewed = graph_supported and mode == "custom" and outcome == "followup"
    if reviewed:
        from scone_memory.providers import answer_reviewer
        from scone_memory.realtime.answer_review import AnswerReviewDecision

        class Reviewer:
            async def review(self, question, answer, evidence, evidence_ids):
                assert "denver" in evidence and "PRIVATE" not in evidence
                return AnswerReviewDecision(status="supported")

        monkeypatch.setattr(answer_reviewer, "SelfHostedAnswerReviewer", lambda *args, **kwargs: Reviewer())
        env.update(SCONE_ANSWER_REVIEW_POLICY="require_supported", SCONE_ANSWER_REVIEW_URL="http://127.0.0.1:11434/v1",
                   SCONE_ANSWER_REVIEW_MODEL="reviewer")
    body = {"request_id": "session", "capture": True,
            "recall_scope": {"where": {"team": "science"}, "kind": "file", "source_prefix": "manuals/"}}
    if mode == "custom":
        env["SCONE_CONVERSATIONS_MODEL_FACTORY"] = "adaptive_http_provider:create"
    elif mode == "saved":
        path = tmp_path / "models.json"
        store = ModelConnectionStore(path, {})
        store.replace("chat", ModelConnection(base_url="http://127.0.0.1:11434/v1", model="reply"), expected_revision=0)
        env["SCONE_MODEL_CONNECTIONS"] = str(path)
        monkeypatch.setattr(model_runtime, "OpenAICompatibleTextModel", lambda *args, **kwargs: Model())
    else:
        path = tmp_path / "personas.json"
        path.write_text(json.dumps([{"schema_version": 1, "id": "helper", "name": "Helper",
            "instructions": "Use retained sources.", "reply": {"provider": "stub", "model": "reply"},
            "transcription": {"provider": "stub", "model": "ears"},
            "speech": {"provider": "stub", "model": "mouth", "voice": "alto"}}]))
        module.registry = lambda: ProviderRegistry(reply={("stub", "reply"): Model},
            transcription={("stub", "ears"): lambda: None}, speech={("stub", "mouth", "alto"): lambda: None})
        env.update(SCONE_CONVERSATIONS_PERSONAS=str(path), SCONE_CONVERSATIONS_REGISTRY="adaptive_http_provider:registry")
        body["persona"] = "helper"
    facts, episodes = [], []
    for left, right in [("aster", "beacon"), ("beacon", "cedar"), ("cedar", "denver")]:
        quote = f"{left} depends on {right}."
        episode = await engine.remember("default", quote, kind="file", source=f"manuals/{left}", metadata={"team": "science"})
        facts.append(await engine.assert_fact("default", left, "depends on", right, source_episode_id=episode.episode_id, quote=quote))
        episodes.append(episode)
    for space, metadata, source in [("foreign", {"team": "science"}, "manuals/private"),
            ("default", {"team": "legal"}, "manuals/private"), ("default", {"team": "science"}, "private/secret")]:
        await engine.remember(space, "aster beacon PRIVATE distractor", kind="file", source=source, metadata=metadata)
    assert [fact.fact_id for fact in (await engine.recall("default", "aster", where={"team": "science"})).facts] == [facts[0].fact_id]
    app = build_app(Settings.from_env(env), engine)
    async with app.router.lifespan_context(app):
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://scone.test", headers=AUTH) as client:
            capability = (await client.get("/v1/conversations/capabilities")).json()["adaptive_retrieval"]
            assert capability["configured"] is True and capability["graph_max_hops"] == 3
            assert capability["limits"]["timeout_s"] == 5
            created = await client.post("/v1/conversations", json=body)
            assert created.status_code == 200, created.text
            session = created.json()
            base = "/v1/conversations/" + session["session_id"]
            sent = await client.post(base + "/turns", json={"request_id": "turn", "text": "aster", "expected_revision": session["revision"]})
            assert sent.status_code == 202
            # Observe the full turn, including capture after the five-second
            # retrieval budget, without imposing a shorter backend deadline.
            async with asyncio.timeout(15):
                while True:
                    response = await client.get(base + "/turns/turn")
                    assert response.status_code == 200, response.text
                    receipt = response.json()
                    if receipt["status"] != "pending":
                        break
                    await asyncio.sleep(.01)
            assert receipt["status"] == "completed", receipt
            context = receipt["result"]["memory_context"]
            if reviewed:
                assert receipt["result"]["answer_review"]["status"] == "supported"
            expansion = context["adaptive_graph_expansions"][0]
            expected = {f"fact:{fact.fact_id}" for fact in facts} if graph_supported else {f"fact:{facts[0].fact_id}"}
            assert {candidate.id for candidate in calls[0][1] if candidate.id.startswith("fact:")} == expected
            assert "PRIVATE" not in json.dumps(requests) + json.dumps(receipt)
            if not graph_supported:
                assert expansion["complete"] is False
                assert any(reason.startswith("unsupported_") for reason in expansion["reasons"])
                assert expansion["added_count"] == 0
                return
            assert expansion["added_count"] == 2
            if outcome == "deleted_bridge":
                assert context["adaptive_status"] == "uncertain"
                assert "atomic_group_omitted" in context["adaptive_reasons"]
                assert context.get("claim_count", 0) == 0
                packets = [json.loads(message["content"].split("\n", 1)[1]) for message in requests[0]
                           if message["content"].startswith("Scone retrieved source material:")]
                assert all(not packet.get("claims") and not packet.get("paths") for packet in packets)
            else:
                packet = next(json.loads(message["content"].split("\n", 1)[1]) for message in requests[0]
                              if message["content"].startswith("Scone retrieved source material:"))
                assert any(path["fact_ids"] == [fact.fact_id for fact in facts] for path in packet["paths"])
                if outcome == "followup":
                    assert context["adaptive_round_count"] == context["adaptive_queries_used"] == 2
                    assert context["adaptive_evidence_basis"] == "original_and_selected"
                else:
                    assert context["adaptive_fallback_status"] == "retained"
                    assert context["adaptive_status"] == "uncertain"


@pytest.mark.parametrize('enabled,history', [(True, True), (True, False), (False, False)])
async def test_adaptive_capability_is_separate_from_reply_model_availability(tmp_path, enabled, history):
    env = environment(tmp_path) if enabled else {"SCONE_API_KEY": "fixture-key",
        "SCONE_CONVERSATIONS_JOURNAL": str(tmp_path / "sessions.db")}
    if history:
        env['SCONE_ADAPTIVE_SEARCH_HISTORY'] = '1'
    engine = await MemoryEngine(InMemoryDocumentStore(), InMemoryVectorIndex(), HashEmbedder()).open()
    try:
        app = build_app(Settings.from_env(env), engine)
        async with app.router.lifespan_context(app):
            async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://scone.test", headers=AUTH) as client:
                capability = (await client.get("/v1/conversations/capabilities")).json()
                assert capability["text_configured"] is False
                assert capability["adaptive_retrieval"]["configured"] is enabled
                assert capability['adaptive_retrieval']['search_history'] is history
                if not enabled:
                    assert capability["adaptive_retrieval"] == {
                        "configured": False, "limits": None, "graph_max_hops": 0, 'search_history': False}
    finally:
        await engine.close()


@pytest.mark.parametrize("outcome", ["graph_off", "assessment_timeout", "verification_timeout", "turn_timeout", "cancel"])
async def test_served_retrieval_respects_graph_optout_timeout_and_cancellation(tmp_path, monkeypatch, outcome):
    from scone_memory.providers import evidence_assessor
    from scone_memory.realtime.events import ReplyCompleted, TextDelta
    from scone_memory.retrieval.adaptive import EvidenceDecision

    entered, stopped = asyncio.Event(), asyncio.Event()
    observed = []
    clock_value = [0.0]
    if outcome == "verification_timeout":
        from types import SimpleNamespace
        from scone_memory.retrieval import adaptive
        monkeypatch.setattr(adaptive, "time", SimpleNamespace(monotonic=lambda: clock_value[0]))

    class Assessor:
        async def assess(self, question, candidates):
            entered.set()
            if outcome == "graph_off":
                return EvidenceDecision(status="sufficient", selected_ids=tuple(candidate.id for candidate in candidates))
            try:
                if outcome == "verification_timeout":
                    clock_value[0] = 1.01
                    raise TimeoutError()
                if outcome == "assessment_timeout":
                    # Provider timeout with time left to verify the fallback.
                    # Waiting for the outer timer can exhaust that reserve on CI.
                    raise TimeoutError()
                await asyncio.Event().wait()
            finally:
                stopped.set()

    class Model:
        async def respond(self, messages):
            observed.append(messages)
            yield TextDelta("Recorded source inspected.")
            yield ReplyCompleted()

        async def aclose(self):
            pass

    monkeypatch.setattr(evidence_assessor, "SelfHostedEvidenceAssessor", lambda *args, **kwargs: Assessor())
    module = ModuleType("adaptive_wait_provider")
    module.create = Model
    monkeypatch.setitem(sys.modules, module.__name__, module)
    env = environment(tmp_path) | {"SCONE_ADAPTIVE_GRAPH_HOPS": "0", "SCONE_ADAPTIVE_TIMEOUT": "1",
        "SCONE_CONVERSATIONS_MODEL_FACTORY": "adaptive_wait_provider:create"}
    if outcome == "turn_timeout":
        env["SCONE_CHAT_TIMEOUT"] = ".15"
    engine = await MemoryEngine(InMemoryDocumentStore(), InMemoryVectorIndex(), HashEmbedder()).open()
    try:
        await engine.remember("default", "aster depends on beacon.")
        app = build_app(Settings.from_env(env), engine)
        async with app.router.lifespan_context(app):
            async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://scone.test", headers=AUTH) as client:
                session = (await client.post("/v1/conversations", json={"request_id": "session", "capture": True})).json()
                base = "/v1/conversations/" + session["session_id"]
                sent = await client.post(base + "/turns", json={"request_id": "turn", "text": "aster", "expected_revision": session["revision"]})
                assert sent.status_code == 202
                await asyncio.wait_for(entered.wait(), 2)
                if outcome == "cancel":
                    assert (await client.post(base + "/turns/turn/cancel")).json()["status"] == "cancelled"
                for _ in range(200):
                    receipt = (await client.get(base + "/turns/turn")).json()
                    if receipt["status"] != "pending":
                        break
                    await asyncio.sleep(.01)
                episodes = await engine.episodes("default", {"session_id": session["session_id"]})
                if outcome in ("turn_timeout", "cancel"):
                    assert receipt["status"] == ("failed" if outcome == "turn_timeout" else "cancelled")
                    assert observed == []
                    assert not any(episode.metadata.get("role") == "assistant" for episode in episodes)
                    assert stopped.is_set()
                else:
                    assert receipt["status"] == "completed", receipt
                    context = receipt["result"]["memory_context"]
                    assert "adaptive_graph_expansions" not in context
                    if outcome == "verification_timeout":
                        assert "aster depends on beacon." not in json.dumps(observed)
                        assert context["status"] == "empty"
                        assert context["references"] == []
                        assert context["adaptive_fallback_status"] == "verification_timeout"
                        assert context["adaptive_evidence_basis"] == "none"
                    else:
                        assert "aster depends on beacon." in json.dumps(observed)
                    if outcome in ("assessment_timeout", "verification_timeout"):
                        assert stopped.is_set()
                        if outcome == "assessment_timeout":
                            assert context["adaptive_fallback_status"] == "retained"
                        assert "timeout" in context["adaptive_errors"]
                        assert context["adaptive_status"] == "uncertain"
    finally:
        await engine.close()


def test_embedded_service_rejects_adaptive_retriever_bound_to_another_engine(tmp_path):
    from scone_memory.api.conversations import create_conversation_app
    from scone_memory.retrieval.adaptive import AdaptiveRetriever

    first = MemoryEngine(InMemoryDocumentStore(), InMemoryVectorIndex(), HashEmbedder())
    second = MemoryEngine(InMemoryDocumentStore(), InMemoryVectorIndex(), HashEmbedder())
    with pytest.raises(ValueError, match="same|this memory engine"):
        create_conversation_app(first, {"fixture-key": "default"}, tmp_path / "journal.db", None,
                                adaptive_retriever=AdaptiveRetriever(second, object()))
    assert not (tmp_path / "journal.db").exists()
