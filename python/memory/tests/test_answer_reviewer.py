"""Self-hosted answer review: bounded structured output and no implicit services."""
import json

import httpx
import pytest

from scone_memory.providers.answer_reviewer import SelfHostedAnswerReviewer
from scone_memory.realtime.answer_review import AnswerReviewError


def response(payload):
    return httpx.Response(200, json={"choices": [{"message": {"content": json.dumps(payload)}, "finish_reason": "stop"}]})


def decision(status="supported", **extra):
    return {"status": status, "issues": [], "revised_answer": None, **extra}


async def test_private_provider_sends_only_given_evidence_and_strict_schema():
    requests = []

    async def serve(request):
        requests.append(json.loads(request.content))
        return response(decision())

    provider = SelfHostedAnswerReviewer("http://127.0.0.1:11434/v1", "installed-model", transport=httpx.MockTransport(serve))
    reviewed = await provider.review("Where does it finish?", "It finishes at the vault.",
                                     '{"claims":["source text"]}', ("fact:1",))
    assert reviewed.status == "supported"
    assert len(requests) == 1
    request = requests[0]
    assert request["model"] == "installed-model"
    assert request["temperature"] == 0
    assert request["max_tokens"] == 2048
    schema = request["response_format"]["json_schema"]["schema"]
    branches = {branch["properties"]["status"]["const"]: branch for branch in schema["anyOf"]}
    assert set(branches) == {"supported", "needs_revision", "uncertain"}
    for branch in branches.values():
        assert branch["additionalProperties"] is False
        assert set(branch["required"]) == {"status", "issues", "revised_answer"}
        properties = branch["properties"]
        assert "maxLength" not in properties["revised_answer"]
        assert "maxLength" not in properties["issues"]["items"]["properties"]["answer_quote"]
    assert branches["supported"]["properties"]["issues"]["maxItems"] == 0
    assert branches["supported"]["properties"]["revised_answer"] == {"type": "null"}
    assert branches["needs_revision"]["properties"]["issues"]["minItems"] == 1
    assert branches["needs_revision"]["properties"]["revised_answer"]["type"] == ["string", "null"]
    assert branches["uncertain"]["properties"]["revised_answer"] == {"type": "null"}
    payload = json.loads(request["messages"][-1]["content"])
    assert payload == {"question": "Where does it finish?", "answer": "It finishes at the vault.",
                       "evidence": '{"claims":["source text"]}', "evidence_ids": ["fact:1"]}
    assert "source text" not in request["messages"][0]["content"]


@pytest.mark.parametrize("oversized", ["quote", "revision"])
async def test_host_enforces_string_bounds_without_expanding_wire_grammar(oversized):
    draft = "x" * 2001 if oversized == "quote" else "answer"
    raw = decision("needs_revision", issues=[{"code": "contradiction", "answer_quote": draft,
        "evidence_ids": ["fact:1"]}], revised_answer="é" * 400 if oversized == "revision" else None)
    provider = SelfHostedAnswerReviewer("http://127.0.0.1:11434/v1", "installed-model",
        max_answer_bytes=512 if oversized == "revision" else 64000,
        transport=httpx.MockTransport(lambda request: response(raw)))
    with pytest.raises(AnswerReviewError) as error:
        await provider.review("question", draft, "evidence", ("fact:1",))
    assert error.value.reason == "invalid_review"


async def test_proposed_revision_is_returned_for_independent_second_review():
    raw = decision("needs_revision", issues=[{"code": "broken_path", "answer_quote": "Birch",
        "evidence_ids": ["fact:1", "fact:2"]}], revised_answer="The records reach the archive.")
    provider = SelfHostedAnswerReviewer("http://127.0.0.1:11434/v1", "installed-model",
                                        transport=httpx.MockTransport(lambda request: response(raw)))
    reviewed = await provider.review("Which destination?", "Birch", "route evidence", ("fact:1", "fact:2"))
    assert reviewed.revised_answer == "The records reach the archive."
    assert reviewed.issues[0].evidence_ids == ("fact:1", "fact:2")


@pytest.mark.parametrize("raw", [
    decision(extra="unrequested"),
    decision("unsupported-status"),
    decision("needs_revision"),
    decision(revised_answer="unasked rewrite"),
    decision(issues=[{"code": "contradiction", "answer_quote": "answer", "evidence_ids": ["fact:1"]}]),
    decision("uncertain", revised_answer="unasked rewrite"),
    decision("needs_revision", issues=[{"code": "contradiction", "answer_quote": "answer", "evidence_ids": ["fact:99"]}]),
    decision("needs_revision", issues=[{"code": "unknown", "answer_quote": "answer", "evidence_ids": []}]),
    decision("needs_revision", issues=[{"code": "contradiction", "answer_quote": "not in draft", "evidence_ids": ["fact:1"]}]),
])
async def test_invalid_review_output_is_sanitized_without_repair(raw):
    calls = []
    def serve(request):
        calls.append(request)
        return response(raw)
    provider = SelfHostedAnswerReviewer("http://127.0.0.1:11434/v1", "installed-model", transport=httpx.MockTransport(serve))
    with pytest.raises(AnswerReviewError) as error:
        await provider.review("question", "answer", "evidence", ("fact:1",))
    assert error.value.reason == "invalid_review"
    assert len(calls) == 1


@pytest.mark.parametrize("content", ['{"status":"supported","status":"uncertain","issues":[],"revised_answer":null}', "not JSON", "x" * 200000])
async def test_malformed_duplicate_or_oversized_response_is_rejected(content):
    provider = SelfHostedAnswerReviewer("http://127.0.0.1:11434/v1", "installed-model",
        transport=httpx.MockTransport(lambda request: httpx.Response(200, json={"choices": [{"message": {"content": content}, "finish_reason": "stop"}]})))
    with pytest.raises(AnswerReviewError) as error:
        await provider.review("question", "answer", "evidence", ("fact:1",))
    assert error.value.reason == "invalid_review"


@pytest.mark.parametrize("failure,reason", [(httpx.ReadTimeout("private details"), "review_timeout"),
                                            (httpx.ConnectError("private details"), "review_provider_failed")])
async def test_provider_failures_are_content_free(failure, reason):
    def serve(request):
        raise failure
    provider = SelfHostedAnswerReviewer("http://127.0.0.1:11434/v1", "installed-model", transport=httpx.MockTransport(serve))
    with pytest.raises(AnswerReviewError) as error:
        await provider.review("question", "answer", "evidence", ("fact:1",))
    assert error.value.reason == reason
    assert "private" not in str(error.value)


@pytest.mark.parametrize("endpoint", ["https://api.openai.com/v1", "https://8.8.8.8/v1", "http://user:secret@localhost/v1"])
def test_external_or_credentialed_endpoint_is_rejected(endpoint):
    with pytest.raises(ValueError):
        SelfHostedAnswerReviewer(endpoint, "installed-model")


@pytest.mark.parametrize("ids", [["fact:1"], ("fact:1", "fact:1"), ("fact:0",), ("invalid",)])
async def test_invalid_evidence_ids_fail_before_network(ids):
    provider = SelfHostedAnswerReviewer("http://localhost:11434/v1", "installed-model",
        transport=httpx.MockTransport(lambda request: pytest.fail("unexpected network")))
    with pytest.raises(ValueError):
        await provider.review("question", "answer", "evidence", ids)
