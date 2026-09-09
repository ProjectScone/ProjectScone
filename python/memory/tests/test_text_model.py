"""An OpenAI-compatible chat endpoint as a native TextModel: the reply
streams as public deltas and ends with an explicit completion, or fails."""

from __future__ import annotations

import json

import httpx
import pytest

from scone_memory import HashEmbedder, InMemoryDocumentStore, InMemoryVectorIndex, MemoryEngine
from scone_memory.providers.llm import ChatError, OpenAICompatibleTextModel
from scone_memory.realtime.events import ReplyCompleted, TextDelta
from scone_memory.realtime.text import TextConversation

MESSAGES = [{"role": "system", "content": "Answer briefly."}, {"role": "user", "content": "Where is Juniper?"}]


def sse(*chunks, done=True):
    lines = [": keep-alive", ""]
    for chunk in chunks:
        lines += ["data: " + json.dumps(chunk), ""]
    if done:
        lines += ["data: [DONE]", ""]
    return "\n".join(lines).encode()


def delta(text, finish=None):
    return {"choices": [{"delta": {"content": text}, "finish_reason": finish}]}


async def collect(model):
    events = []
    async for event in model.respond(MESSAGES):
        events.append(event)
    return events


async def test_a_streamed_reply_arrives_as_deltas_and_ends_with_completion():
    seen = []

    def handle(request):
        seen.append(request)
        return httpx.Response(200, content=sse(delta("Juniper "), delta(""), delta("points north."), delta("", "stop")),
                              headers={"content-type": "text/event-stream"})

    model = OpenAICompatibleTextModel("http://llm.local/v1/", "qwen3", api_key="k", think=False,
                                      temperature=0.3, transport=httpx.MockTransport(handle))
    try:
        events = await collect(model)
    finally:
        await model.aclose()
    assert events == [TextDelta("Juniper "), TextDelta("points north."), ReplyCompleted()], "empty deltas are not public text"
    [request] = seen
    assert str(request.url) == "http://llm.local/v1/chat/completions"
    assert request.headers["authorization"] == "Bearer k"
    body = json.loads(request.content)
    assert body["messages"] == MESSAGES and body["stream"] is True and body["temperature"] == 0.3
    assert body["reasoning_effort"] == "none" and "think" not in body


async def test_completion_is_the_finish_reason_when_no_done_marker_comes():
    def handle(request):
        return httpx.Response(200, content=sse(delta("Yes."), delta("", "stop"), done=False),
                              headers={"content-type": "text/event-stream"})

    model = OpenAICompatibleTextModel("http://llm.local/v1", "gpt", transport=httpx.MockTransport(handle))
    assert await collect(model) == [TextDelta("Yes."), ReplyCompleted()]


async def test_a_server_that_ignores_streaming_still_delivers_one_reply():
    def handle(request):
        return httpx.Response(200, json={"choices": [{"message": {"role": "assistant", "content": "Whole reply."}}]})

    model = OpenAICompatibleTextModel("http://llm.local/v1", "gpt", transport=httpx.MockTransport(handle))
    assert await collect(model) == [TextDelta("Whole reply."), ReplyCompleted()]


@pytest.mark.parametrize("response", [
    pytest.param(httpx.Response(500, text="boom"), id="status-500"),
    pytest.param(httpx.Response(200, content=b"data: not json\n\n", headers={"content-type": "text/event-stream"}), id="bad-data"),
    pytest.param(httpx.Response(200, content=sse(delta("half"), done=False), headers={"content-type": "text/event-stream"}), id="no-completion"),
])
async def test_a_reply_that_does_not_complete_is_a_chat_error_not_a_silent_success(response):
    model = OpenAICompatibleTextModel("http://llm.local/v1", "gpt", transport=httpx.MockTransport(lambda request: response))
    with pytest.raises(ChatError) as error:
        await collect(model)
    assert "boom" not in str(error.value), "a server's body is not echoed into the error"


async def test_an_empty_completion_is_reported_as_such_for_the_runtime_to_refuse():
    response = httpx.Response(200, content=sse(done=True), headers={"content-type": "text/event-stream"})
    model = OpenAICompatibleTextModel("http://llm.local/v1", "gpt", transport=httpx.MockTransport(lambda request: response))
    assert await collect(model) == [ReplyCompleted()], "the adapter reports; TextConversation refuses an empty reply"


@pytest.mark.parametrize("streamed", [True, False])
async def test_token_limit_finish_is_not_a_completed_reply(streamed):
    response = (httpx.Response(200, content=sse(delta("Partial answer", "length")),
                              headers={"content-type": "text/event-stream"}) if streamed else
                httpx.Response(200, json={"choices": [{"message": {"content": "Partial answer"},
                                                       "finish_reason": "length"}]}))
    model = OpenAICompatibleTextModel("http://llm.local/v1", "gpt", transport=httpx.MockTransport(lambda request: response))
    try:
        with pytest.raises(ChatError, match="finish"):
            await collect(model)
    finally:
        await model.aclose()


async def test_optional_output_token_budget_is_forwarded():
    bodies = []
    def handle(request):
        bodies.append(json.loads(request.content))
        return httpx.Response(200, content=sse(delta("Done", "stop")), headers={"content-type": "text/event-stream"})
    model = OpenAICompatibleTextModel("http://llm.local/v1", "gpt", max_output_tokens=512,
                                    transport=httpx.MockTransport(handle))
    try:
        await collect(model)
    finally:
        await model.aclose()
    assert bodies[0]["max_tokens"] == 512


@pytest.mark.parametrize("budget", [True, 0, -1, 1.5, 32769])
def test_invalid_output_token_budget_is_rejected_before_opening_transport(budget):
    with pytest.raises(ValueError, match="max_output_tokens"):
        OpenAICompatibleTextModel("http://llm.local/v1", "gpt", max_output_tokens=budget)


async def test_the_model_drives_a_native_text_conversation():
    engine = await MemoryEngine(InMemoryDocumentStore(), InMemoryVectorIndex(), HashEmbedder()).open()
    bodies = []

    def handle(request):
        bodies.append(json.loads(request.content))
        return httpx.Response(200, content=sse(delta("Polaris."), delta("", "stop")), headers={"content-type": "text/event-stream"})

    factory = lambda: OpenAICompatibleTextModel("http://llm.local/v1", "gpt", transport=httpx.MockTransport(handle))  # noqa: E731
    conversation = TextConversation(engine, "default", "session-1", factory, system_prompt="Answer briefly.")
    try:
        reply = await conversation.reply("Where does Juniper point?")
    finally:
        await conversation.close()
    assert reply["text"] == "Polaris."
    assert bodies[0]["messages"][0] == {"role": "system", "content": "Answer briefly."}
    assert bodies[0]["messages"][-1] == {"role": "user", "content": "Where does Juniper point?"}
