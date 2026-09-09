import asyncio
import json

import httpx
import pytest

from scone_memory.providers.evidence_assessor import SelfHostedEvidenceAssessor, EvidenceAssessmentError
from scone_memory.retrieval.adaptive import EvidenceCandidate


def candidates():
    return (EvidenceCandidate(id="chunk:1", episode_id=1, text="Iris forwards records to Fern."),
            EvidenceCandidate(id="chunk:2", episode_id=2, text="Fern uses a private journal."))


def transport_for(content, seen, finish="stop"):
    def handle(request):
        seen.append(json.loads(request.content))
        return httpx.Response(200, json={"choices": [{"message": {"content": content}, "finish_reason": finish}]})
    return httpx.MockTransport(handle)


async def test_assessor_selects_existing_evidence_without_requesting_reasoning():
    seen = []
    content = json.dumps({"status": "sufficient", "selected_ids": ["chunk:1", "chunk:2"], "followup_queries": []})
    assessor = SelfHostedEvidenceAssessor("http://127.0.0.1:11434/v1", "fixture", transport=transport_for(content, seen))
    decision = await assessor.assess("Where do Iris records go?", candidates())
    assert decision.status == "sufficient"
    assert decision.selected_ids == ("chunk:1", "chunk:2")
    body = seen[0]
    assert body["max_tokens"] <= 1024 and body["reasoning_effort"] == "none"
    assert "think" not in body
    schema = body["response_format"]["json_schema"]["schema"]
    assert schema["properties"]["selected_ids"]["items"]["enum"] == ["chunk:1", "chunk:2"]
    assert "reasoning" not in schema["properties"]
    assert json.loads(body["messages"][1]["content"])["candidates"][0]["text"] == candidates()[0].text


@pytest.mark.parametrize("content", [
    '{"status":"sufficient","selected_ids":["chunk:999"],"followup_queries":[]}',
    '{"status":"sufficient","selected_ids":["chunk:1","chunk:1"],"followup_queries":[]}',
    '{"status":"sufficient","selected_ids":[],"followup_queries":[]}',
    '{"status":"sufficient","selected_ids":["chunk:1"],"followup_queries":[],"space":"other"}',
    '{"status":"insufficient","status":"sufficient","selected_ids":["chunk:1"],"followup_queries":[]}',
    '{"status":"insufficient","selected_ids":[],"followup_queries":["a","b","c","d"]}',
    'private-model-output-not-json',
])
async def test_invalid_decisions_are_sanitized_without_repairs(content):
    seen = []
    assessor = SelfHostedEvidenceAssessor("http://localhost:11434/v1", "fixture", transport=transport_for(content, seen))
    with pytest.raises(EvidenceAssessmentError) as error:
        await assessor.assess("Iris storage", candidates())
    assert str(error.value) == "evidence assessment failed"
    assert error.value.reason == "invalid_assessment"
    assert len(seen) == 1


async def test_truncated_valid_json_is_not_an_assessment():
    content = '{"status":"sufficient","selected_ids":["chunk:1"],"followup_queries":[]}'
    assessor = SelfHostedEvidenceAssessor("http://localhost:11434/v1", "fixture", transport=transport_for(content, [], "length"))
    with pytest.raises(EvidenceAssessmentError):
        await assessor.assess("Iris storage", candidates())


async def test_empty_evidence_needs_no_model_call():
    seen = []
    assessor = SelfHostedEvidenceAssessor("http://localhost:11434/v1", "fixture", transport=transport_for("{}", seen))
    decision = await assessor.assess("Iris storage", ())
    assert decision.status == "insufficient" and decision.selected_ids == ()
    assert seen == []


@pytest.mark.parametrize("endpoint", ["https://api.openai.com/v1", "http://127.0.0.1:11434/v1?key=secret"])
def test_assessor_requires_operator_managed_endpoint(endpoint):
    with pytest.raises(ValueError):
        SelfHostedEvidenceAssessor(endpoint, "fixture")


@pytest.mark.parametrize("timeout", [True, 0, float("inf"), 181])
def test_assessor_timeout_is_bounded(timeout):
    with pytest.raises(ValueError):
        SelfHostedEvidenceAssessor("http://localhost:11434/v1", "fixture", timeout=timeout)


async def test_serialized_unicode_budget_is_checked_before_request():
    seen = []
    assessor = SelfHostedEvidenceAssessor("http://localhost:11434/v1", "fixture", transport=transport_for("{}", seen))
    evidence = (EvidenceCandidate(id="chunk:1", episode_id=1, text="界" * 50000),)
    with pytest.raises(ValueError, match="128000 UTF-8 bytes"):
        await assessor.assess("What does this source say?", evidence)
    assert seen == []


async def test_http_assessment_cancellation_propagates():
    entered = asyncio.Event()
    async def handle(request):
        entered.set()
        await asyncio.Event().wait()
        return httpx.Response(500)
    assessor = SelfHostedEvidenceAssessor("http://localhost:11434/v1", "fixture", transport=httpx.MockTransport(handle))
    task = asyncio.create_task(assessor.assess("Iris storage", candidates()))
    await asyncio.wait_for(entered.wait(), timeout=1)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task


@pytest.mark.parametrize("failure,reason", [
    (httpx.ReadTimeout("private request"), "assessment_timeout"),
    (httpx.ConnectError("private endpoint"), "assessment_provider_failed"),
])
async def test_transport_failure_has_safe_distinct_reason(failure, reason):
    def handle(request):
        raise failure
    assessor = SelfHostedEvidenceAssessor("http://localhost:11434/v1", "fixture", transport=httpx.MockTransport(handle))
    with pytest.raises(EvidenceAssessmentError) as error:
        await assessor.assess("Iris storage", candidates())
    assert error.value.reason == reason
    assert str(error.value) == "evidence assessment failed"
    assert error.value.__cause__ is None


def connected_facts():
    return (
        EvidenceCandidate(id="fact:1", episode_id=1, text="Iris forwards to Fern.", subject="iris", predicate="forwards_to", object="fern"),
        EvidenceCandidate(id="fact:2", episode_id=2, text="Fern persists through Willow.", subject="fern", predicate="persists_through", object="willow"),
        EvidenceCandidate(id="fact:3", episode_id=3, text="Willow uses a private journal.", subject="willow", predicate="uses", object="private journal"),
        EvidenceCandidate(id="fact:4", episode_id=4, text="Copper uses an unrelated archive.", subject="copper", predicate="uses", object="unrelated archive"),
    )


async def test_group_selection_returns_all_connected_fact_ids_and_atomic_contract():
    seen = []
    def handle(request):
        body = json.loads(request.content)
        seen.append(body)
        data = json.loads(body["messages"][1]["content"])
        group = next(record for record in data["candidates"] if record["id"].startswith("group:"))
        assert {record["id"] for record in group["members"]} == {"fact:1", "fact:2", "fact:3"}
        content = json.dumps({"status": "sufficient", "selected_ids": [group["id"]], "followup_queries": []})
        return httpx.Response(200, json={"choices": [{"message": {"content": content}, "finish_reason": "stop"}]})
    assessor = SelfHostedEvidenceAssessor("http://localhost:11434/v1", "fixture", group_relations=True,
                                         transport=httpx.MockTransport(handle))
    decision = await assessor.assess("Where do Iris records end up?", connected_facts())
    assert set(decision.selected_ids) == {"fact:1", "fact:2", "fact:3"}
    assert decision.selected_groups == (("fact:1", "fact:2", "fact:3"),)
    schema = seen[0]["response_format"]["json_schema"]["schema"]
    assert "fact:1" not in schema["properties"]["selected_ids"]["items"]["enum"]
    assert "selected_groups" not in schema["properties"]


@pytest.mark.parametrize("extra", [False, True])
async def test_model_cannot_select_partial_group_or_inject_atomic_contract(extra):
    content = {"status": "sufficient", "selected_ids": ["fact:1"], "followup_queries": []}
    if extra:
        content["selected_groups"] = [["fact:1", "fact:2"]]
    assessor = SelfHostedEvidenceAssessor("http://localhost:11434/v1", "fixture", group_relations=True,
                                         transport=transport_for(json.dumps(content), []))
    with pytest.raises(EvidenceAssessmentError) as error:
        await assessor.assess("Iris storage?", connected_facts())
    assert error.value.reason == "invalid_assessment"


async def test_grouped_byte_budget_fails_before_model_call():
    seen = []
    assessor = SelfHostedEvidenceAssessor("http://localhost:11434/v1", "fixture", group_relations=True,
        max_evidence_bytes=100, transport=transport_for("{}", seen))
    with pytest.raises(ValueError):
        await assessor.assess("Iris storage?", connected_facts())
    assert seen == []


@pytest.mark.parametrize("options", [{"group_relations": 1}, {"max_evidence_bytes": True},
    {"max_evidence_bytes": 1}, {"max_evidence_bytes": 128001}])
def test_grouping_configuration_validates_before_network(options):
    with pytest.raises(ValueError):
        SelfHostedEvidenceAssessor("http://localhost:11434/v1", "fixture", **options)


async def test_grouped_provider_flows_through_native_retrieval_and_context():
    from scone_memory import HashEmbedder, InMemoryDocumentStore, InMemoryVectorIndex, MemoryEngine
    from scone_memory.realtime.context import MemoryContext, _PREFIX
    from scone_memory.retrieval.adaptive import AdaptiveLimits, AdaptiveRetriever

    memory = await MemoryEngine(InMemoryDocumentStore(), InMemoryVectorIndex(), HashEmbedder()).open()
    try:
        fact_ids = set()
        for candidate in connected_facts():
            source = await memory.remember("alpha", candidate.text)
            fact = await memory.assert_fact("alpha", candidate.subject, candidate.predicate, candidate.object,
                                           source_episode_id=source.episode_id, quote=candidate.text)
            if candidate.id != "fact:4":
                fact_ids.add(fact.fact_id)

        def handle(request):
            data = json.loads(json.loads(request.content)["messages"][1]["content"])
            group = next(record for record in data["candidates"] if record["id"].startswith("group:"))
            content = json.dumps({"status": "sufficient", "selected_ids": [group["id"]], "followup_queries": []})
            return httpx.Response(200, json={"choices": [{"message": {"content": content}, "finish_reason": "stop"}]})

        assessor = SelfHostedEvidenceAssessor("http://localhost:11434/v1", "fixture", group_relations=True,
                                             transport=httpx.MockTransport(handle))
        adaptive = AdaptiveRetriever(memory, assessor, limits=AdaptiveLimits(timeout_s=1.0))
        request, receipt = await MemoryContext(memory, "alpha", "current", adaptive_retriever=adaptive).prepare([
            {"role": "user", "content": "Where do Iris records go through Fern and Willow?"}])
        block = next(message["content"] for message in request if message["content"].startswith(_PREFIX.rstrip("\n")))
        payload = json.loads(block[block.index("{"):])
        assert {claim["fact_id"] for claim in payload["claims"]} == fact_ids
        assert payload["coverage"]["adaptive"]["selection_complete"] is True
        assert receipt["adaptive_atomic_group_omitted_count"] == 0
        assert payload["paths"] and receipt["path_count"] >= 1
        assert "unrelated archive" not in block
    finally:
        await memory.close()
