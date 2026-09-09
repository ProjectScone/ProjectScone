"""Small structured selections never introduce model-authored answer prose."""
import json

import httpx
import pytest


def reply(content):
    return httpx.Response(200, json={"choices": [{"message": {"content": content}, "finish_reason": "stop"}]})


async def test_selector_receives_exact_cards_and_returns_only_ids():
    from scone_memory.providers.evidence_selector import SelfHostedEvidenceSelector
    from scone_memory.realtime.evidence_answer import EvidenceCard
    requests = []
    def serve(request):
        requests.append(json.loads(request.content))
        return reply('{"card_ids":["card:1"]}')
    card = EvidenceCard(id="card:1", kind="claim", text="Spruce routes to Birch.", evidence_ids=("fact:1",))
    provider = SelfHostedEvidenceSelector("http://127.0.0.1:11434/v1", "installed", transport=httpx.MockTransport(serve))
    selected = await provider.select("Where does Spruce route?", (card,))
    assert selected.card_ids == ("card:1",) and len(requests) == 1
    payload = requests[0]
    assert payload["max_tokens"] == 256 and payload["temperature"] == 0
    data = json.loads(payload["messages"][-1]["content"])
    assert data == {"question": "Where does Spruce route?", "cards": [card.model_dump(mode="json")]}
    assert "Spruce" not in payload["messages"][0]["content"]
    schema = payload["response_format"]["json_schema"]["schema"]
    assert schema["properties"]["card_ids"]["items"]["enum"] == ["card:1"]


async def test_selector_serializes_claim_identity_and_revalidates_nested_witnesses():
    from scone_memory.providers.evidence_selector import SelfHostedEvidenceSelector
    from scone_memory.realtime.evidence_answer import EvidenceCard, EvidenceClaim
    from pydantic import ValidationError
    requests = []

    def serve(request):
        requests.append(json.loads(request.content))
        return reply('{"card_ids":["card:1"]}')

    claim = EvidenceClaim(fact_id=1, subject="atlas", predicate="routes to", object="birch", origin="inferred", source_episode_id=7)
    card = EvidenceCard(id="card:1", kind="claim", text="A broad source quote.", evidence_ids=("fact:1",), claims=(claim,))
    provider = SelfHostedEvidenceSelector("http://localhost:11434/v1", "installed", transport=httpx.MockTransport(serve))
    await provider.select("Where does atlas route?", (card,))
    payload = json.loads(requests[0]["messages"][-1]["content"])
    assert payload["cards"][0]["claims"] == [claim.model_dump(mode="json")]
    prompt = requests[0]["messages"][0]["content"]
    assert "not a stated causal relationship" in prompt and "not proof" in prompt
    object.__setattr__(card.claims[0], "fact_id", 99)
    with pytest.raises(ValidationError):
        await provider.select("Where does atlas route?", (card,))
    assert len(requests) == 1


@pytest.mark.parametrize("content", [
    '{"card_ids":["card:99"]}', '{"card_ids":["card:1","card:1"]}',
    '{"card_ids":[],"answer":"invented prose"}', '{"card_ids":[],"card_ids":["card:1"]}',
    '{"card_ids":"card:1"}', 'not json', 'x' * 5000,
])
async def test_invalid_selections_are_sanitized_without_retry(content):
    from scone_memory.providers.evidence_selector import SelfHostedEvidenceSelector
    from scone_memory.realtime.evidence_answer import EvidenceAnswerError, EvidenceCard
    calls = []
    def serve(request):
        calls.append(request)
        return reply(content)
    provider = SelfHostedEvidenceSelector("http://localhost:11434/v1", "installed", transport=httpx.MockTransport(serve))
    with pytest.raises(EvidenceAnswerError) as error:
        await provider.select("question", (EvidenceCard(id="card:1", kind="claim", text="source", evidence_ids=("fact:1",)),))
    assert error.value.reason == "invalid_selection" and len(calls) == 1


async def test_empty_card_input_returns_empty_without_contacting_provider():
    from scone_memory.providers.evidence_selector import SelfHostedEvidenceSelector
    provider = SelfHostedEvidenceSelector("http://localhost:11434/v1", "installed",
        transport=httpx.MockTransport(lambda request: pytest.fail("unexpected network")))
    assert (await provider.select("question", ())).card_ids == ()


@pytest.mark.parametrize("error_type,reason", [(httpx.ReadTimeout, "selection_timeout"), (httpx.ConnectError, "selection_provider_failed")])
async def test_private_provider_error_is_redacted(error_type, reason):
    from scone_memory.providers.evidence_selector import SelfHostedEvidenceSelector
    from scone_memory.realtime.evidence_answer import EvidenceAnswerError, EvidenceCard
    def serve(request):
        raise error_type("private credentials")
    provider = SelfHostedEvidenceSelector("http://localhost:11434/v1", "installed", transport=httpx.MockTransport(serve))
    with pytest.raises(EvidenceAnswerError) as error:
        await provider.select("question", (EvidenceCard(id="card:1", kind="claim", text="source", evidence_ids=("fact:1",)),))
    assert error.value.reason == reason and "private" not in str(error.value)


@pytest.mark.parametrize("endpoint", ["https://api.openai.com/v1", "https://8.8.8.8/v1", "http://user:secret@localhost/v1"])
def test_selector_requires_operator_managed_endpoint(endpoint):
    from scone_memory.providers.evidence_selector import SelfHostedEvidenceSelector
    with pytest.raises(ValueError):
        SelfHostedEvidenceSelector(endpoint, "installed")
