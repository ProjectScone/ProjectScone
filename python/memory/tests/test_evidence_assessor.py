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
    assert body["max_tokens"] <= 1024 and body["think"] is False
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
