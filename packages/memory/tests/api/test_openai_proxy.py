"""The OpenAI-compatible route: an application sends chat completions to
Scone, the key's memory is put in front of the conversation as one
delimited block, the server's own model answers, and the turn is kept.
FakeChat keeps every call, so each test can read what reached the model."""

from __future__ import annotations

import asyncio
import hashlib
import json

import httpx
import pytest
from fastapi.testclient import TestClient

from scone_memory import HashEmbedder, InMemoryDocumentStore, InMemoryVectorIndex, MemoryEngine
from scone_memory.api import create_app, openai_proxy
from scone_memory.providers.llm import ChatError, FakeChat

ROUTE = "/v1/openai/chat/completions"
KEYS = {"key-a": "alpha", "key-b": "beta", "reader": "alpha"}
ROLES = {"reader": "read"}
QUESTION = "When does the vault door code rotate?"
PASSAGE = "The vault door code rotates on the 14th of each month, set by the facilities desk."


def auth(key: str = "key-a") -> dict:
    return {"authorization": f"Bearer {key}"}


class Host:
    """The app and the one model it is configured with; ``model`` may be
    replaced by a test before the request, as the server rebuilds it per request."""

    def __init__(self, engine, client, chat):
        self.engine, self.client, self.chat = engine, client, chat

    def remember(self, content: str, key: str = "key-a", **fields) -> int:
        response = self.client.post("/v1/episodes", json={"content": content, **fields}, headers=auth(key))
        assert response.status_code == 200, response.text
        return response.json()["episode_id"]

    def ask(self, messages, key: str = "key-a", headers=None, **fields):
        body = {"model": "gpt-4o", "messages": messages, **fields}
        return self.client.post(ROUTE, json=body, headers={**auth(key), **(headers or {})})

    def episodes(self, key: str = "key-a") -> list[dict]:
        return self.client.get("/v1/sources", params={"limit": 100}, headers=auth(key)).json()["items"]


@pytest.fixture
async def host():
    engine = await MemoryEngine(InMemoryDocumentStore(), InMemoryVectorIndex(), HashEmbedder()).open()
    holder = {"chat": FakeChat(["It rotates on the 14th."] * 4)}
    app = create_app(engine, KEYS, roles=ROLES, synthesis_factory=lambda: holder["chat"])
    with TestClient(app) as client:
        made = Host(engine, client, holder["chat"])
        made.holder = holder
        yield made


def user(text: str) -> dict:
    return {"role": "user", "content": text}


def test_recalled_passage_reaches_the_model_inside_one_delimited_system_block(host):
    episode = host.remember(PASSAGE, source="ops-handbook")
    response = host.ask([{"role": "system", "content": "Be brief."}, user(QUESTION)])
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["object"] == "chat.completion"
    assert body["choices"] == [{"index": 0, "message": {"role": "assistant", "content": "It rotates on the 14th."},
                                "finish_reason": "stop"}]
    [(system, asked)] = host.chat.calls
    assert asked == QUESTION
    assert system.startswith("Be brief.\n\n")
    scone = body["scone"]
    boundary = scone["injected"]["boundary"]
    opening, closing = f'<scone-memory boundary="{boundary}">', f'</scone-memory boundary="{boundary}">'
    assert system.count(opening) == 1 and system.endswith(closing)
    block = system[system.index(opening):]
    assert PASSAGE in block
    assert scone["injected"]["role"] == "system"
    assert scone["injected"]["bytes"] == len(block.encode())
    assert scone["injected"]["sha256"] == hashlib.sha256(block.encode()).hexdigest()
    assert scone["recall"]["status"] == "prepared"
    [item] = scone["recall"]["items"]
    assert item["episode_id"] == episode and item["source"] == "ops-handbook"
    payload = json.loads(block[block.index("{"):block.rindex("}") + 1])
    assert [source["episode_id"] for source in payload["sources"]] == [episode]
    assert response.headers["x-scone-recall-status"] == "prepared"
    assert response.headers["x-scone-injected-bytes"] == str(scone["injected"]["bytes"])
    assert response.headers["x-scone-recalled-episodes"] == str(episode)


def test_nothing_recalled_still_forwards_and_says_so(host):
    response = host.ask([user(QUESTION)])
    assert response.status_code == 200, response.text
    [(system, asked)] = host.chat.calls
    assert system == "" and asked == QUESTION
    scone = response.json()["scone"]
    assert scone["recall"]["status"] == "empty" and scone["recall"]["items"] == []
    assert scone["injected"] is None
    assert response.headers["x-scone-recall-status"] == "empty"
    assert response.headers["x-scone-injected-bytes"] == "0"
    assert response.headers["x-scone-recalled-episodes"] == ""


def test_another_space_is_never_recalled(host):
    host.remember(PASSAGE, key="key-b")
    response = host.ask([user(QUESTION)])
    assert response.json()["scone"]["recall"]["items"] == []
    assert PASSAGE not in host.chat.calls[0][0]


def test_turn_and_reply_are_kept_as_conversation_episodes(host):
    response = host.ask([user(QUESTION)])
    scone = response.json()["scone"]
    assert scone["capture"]["status"] == "captured"
    assert response.headers["x-scone-capture-status"] == "captured"
    ids = scone["capture"]["episode_ids"]
    read = host.client.get("/v1/episodes", params={"ids": ",".join(map(str, ids))}, headers=auth()).json()
    stored = {episode["metadata"]["role"]: episode for episode in read["episodes"]}
    assert stored["user"]["content"] == QUESTION
    assert stored["assistant"]["content"] == "It rotates on the 14th."
    assert scone["session_id"] == "openai:" + scone["session"] == "openai:" + response.headers["x-scone-session"]
    for role, episode in stored.items():
        assert episode["kind"] == "conversation"
        assert episode["source"] == scone["session_id"]
        assert episode["metadata"]["session_id"] == scone["session_id"]
        assert episode["metadata"]["turn_id"] == scone["turn_id"]
        assert episode["metadata"]["integration"] == "scone-openai-proxy"
    assert stored["assistant"]["metadata"]["provider_completion"] == "unverified"
    assert stored["user"]["metadata"]["capture_status"] == "submitted"
    assert stored["assistant"]["metadata"]["capture_status"] == "aggregated"
    assert {episode["metadata"]["representation"] for episode in stored.values()} == {"aggregated_text"}


def test_the_same_words_twice_in_a_session_are_two_turns(host):
    session = {"x-scone-session": "support-chat-7"}
    first = host.ask([user(QUESTION)], headers=session).json()["scone"]["capture"]["episode_ids"]
    second = host.ask([user(QUESTION)], headers=session).json()["scone"]["capture"]["episode_ids"]
    assert len(set(first + second)) == 4


def test_a_named_session_does_not_recall_its_own_turns_but_another_session_does(host):
    session = {"x-scone-session": "support-chat-7"}
    host.ask([user("My locker number is 4417 on the third floor.")], headers=session)
    again = host.ask([user("My locker number is 4417 on the third floor."), {"role": "assistant", "content": "Noted."},
                      user("Which floor is my locker on?")], headers=session)
    assert again.json()["scone"]["session"] == "support-chat-7" == again.headers["x-scone-session"]
    assert again.json()["scone"]["session_id"] == "openai:support-chat-7"
    assert "4417" not in host.chat.calls[1][0]
    assert again.json()["scone"]["recall"]["session_turns"]["admitted_turn_ids"] == []
    elsewhere = host.ask([user("Which floor is my locker on?")], headers={"x-scone-session": "support-chat-8"})
    assert "4417" in host.chat.calls[2][0]
    assert elsewhere.json()["scone"]["recall"]["status"] == "prepared"


def test_a_named_session_recalls_its_own_turns_the_request_does_not_resend(host):
    session = {"x-scone-session": "user-42"}
    first = host.ask([user("My locker number is 4417 on the third floor.")], headers=session).json()["scone"]
    trimmed = host.ask([user("Which floor is my locker on?")], headers=session).json()["scone"]
    assert "4417" in host.chat.calls[1][0]
    assert trimmed["recall"]["status"] == "prepared"
    assert trimmed["recall"]["session_turns"]["status"] == "probed"
    assert trimmed["recall"]["session_turns"]["admitted_turn_ids"] == [first["turn_id"]]
    assert first["capture"]["episode_ids"][0] in [item["episode_id"] for item in trimmed["recall"]["items"]]


def test_a_session_passage_without_a_turn_is_not_admitted_as_a_turn(host):
    host.remember("My locker is on the third floor, wing B.", kind="conversation",
                  metadata={"session_id": "openai:user-42"})
    turns = host.ask([user("Which floor is my locker on?")],
                     headers={"x-scone-session": "user-42"}).json()["scone"]["recall"]["session_turns"]
    assert turns["probed"] == 1 and turns["admitted_turn_ids"] == []


def test_the_session_probe_limit_cuts_and_says_it_cut(host, monkeypatch):
    session = {"x-scone-session": "user-42"}
    host.ask([user("My locker number is 4417 on the third floor.")], headers=session)
    turns = host.ask([user("Which floor is my locker on?")], headers=session).json()["scone"]["recall"]["session_turns"]
    assert turns["probe_limit"] == openai_proxy.SESSION_PROBE_LIMIT and turns["limit_reached"] is False
    monkeypatch.setattr(openai_proxy, "SESSION_PROBE_LIMIT", 1)
    turns = host.ask([user("Which floor is my locker on?")], headers=session).json()["scone"]["recall"]["session_turns"]
    assert turns["probe_limit"] == 1 and turns["probed"] == 1 and turns["limit_reached"] is True


def test_a_failed_session_probe_admits_nothing_and_says_so(host, monkeypatch):
    session = {"x-scone-session": "user-42"}
    host.ask([user("My locker number is 4417 on the third floor.")], headers=session)
    recall = host.engine.recall

    async def refuse_scoped(space, query, **kwargs):
        if kwargs.get("where"):
            raise RuntimeError("scoped recall is down")
        return await recall(space, query, **kwargs)
    monkeypatch.setattr(host.engine, "recall", refuse_scoped)
    response = host.ask([user("Which floor is my locker on?")], headers=session)
    assert response.status_code == 200
    turns = response.json()["scone"]["recall"]["session_turns"]
    assert turns["status"] == "failed" and turns["error_type"] == "RuntimeError" and turns["admitted_turn_ids"] == []
    assert "4417" not in host.chat.calls[1][0]


def test_requests_without_a_session_are_each_their_own_session(host):
    first = host.ask([user("My locker number is 4417 on the third floor.")])
    second = host.ask([user("Which floor is my locker on?")])
    assert first.json()["scone"]["session_id"] != second.json()["scone"]["session_id"]
    assert first.json()["scone"]["session_id"] == "openai:" + first.headers["x-scone-session"]
    assert "4417" in host.chat.calls[1][0]
    assert second.json()["scone"]["recall"]["session_turns"]["status"] == "skipped"


def test_a_session_name_shares_no_identity_with_a_document_source_or_a_native_conversation(host):
    episode = host.remember(PASSAGE, source="ops-handbook")
    response = host.ask([user(QUESTION)], headers={"x-scone-session": "ops-handbook"})
    scone = response.json()["scone"]
    assert PASSAGE in host.chat.calls[0][0]
    assert [item["episode_id"] for item in scone["recall"]["items"]] == [episode]
    assert [each["episode_id"] for each in host.episodes() if each["source"] == "ops-handbook"] == [episode]
    native_like = "0123456789abcdef0123456789abcdef"  # the shape of a native conversation id
    kept = host.ask([user(QUESTION)], headers={"x-scone-session": native_like}).json()["scone"]
    read = host.client.get("/v1/episodes", params={"ids": ",".join(map(str, kept["capture"]["episode_ids"]))},
                           headers=auth()).json()["episodes"]
    assert {(each["source"], each["metadata"]["session_id"]) for each in read} == {(f"openai:{native_like}",) * 2}


def test_the_longest_session_name_is_accepted_and_one_longer_is_refused(host):
    longest = "s" * openai_proxy.MAX_SESSION_CHARS
    assert len("openai:" + longest) == 128
    assert host.ask([user(QUESTION)], headers={"x-scone-session": longest}).status_code == 200
    assert "x-scone-session" in refused(host.ask([user(QUESTION)], headers={"x-scone-session": longest + "s"}))


def test_earlier_turns_are_flattened_into_the_user_prompt_and_system_messages_join(host):
    host.ask([{"role": "system", "content": "Be brief."}, user("Hello there, I need facilities help."),
              {"role": "assistant", "content": "Sure, what do you need?"},
              {"role": "developer", "content": "Answer in English."},
              {"role": "user", "content": [{"type": "text", "text": "When does the vault"}, {"type": "text", "text": " door code rotate?"}]}])
    [(system, asked)] = host.chat.calls
    assert system == "Be brief.\n\nAnswer in English."
    earlier, latest = asked.split("\n\nLatest user message:\n")
    assert latest == QUESTION
    assert json.loads(earlier.split("\n", 1)[1]) == [
        {"role": "user", "content": "Hello there, I need facilities help."},
        {"role": "assistant", "content": "Sure, what do you need?"}]


def test_request_model_and_sampling_fields_are_named_as_not_forwarded(host):
    host.holder["chat"].model = "local-7b"
    response = host.ask([user(QUESTION)], temperature=0.2, max_tokens=64, top_p=0.9, user="end-user-1", stream=False)
    body = response.json()
    assert response.status_code == 200, response.text
    assert body["model"] == "local-7b"
    assert body["scone"]["requested_model"] == "gpt-4o"
    assert body["scone"]["not_forwarded"] == ["max_tokens", "model", "temperature", "top_p", "user"]
    assert body["scone"]["provider_completion"] == "unverified"
    assert "usage" not in body


def test_a_model_without_a_name_is_reported_as_unknown(host):
    assert host.ask([user(QUESTION)]).json()["model"] == "unknown"


def test_recall_limit_cuts_and_says_it_cut(host, monkeypatch):
    monkeypatch.setattr(openai_proxy, "RECALL_LIMIT", 1)
    host.remember(PASSAGE)
    host.remember("The vault door code rotation is logged by the facilities desk in the door ledger.")
    recall = host.ask([user(QUESTION)]).json()["scone"]["recall"]
    assert recall["limit"] == 1
    assert len(recall["items"]) == 1 and recall["omitted_count"] >= 1


def test_context_byte_budget_cuts_and_says_it_cut(host, monkeypatch):
    monkeypatch.setattr(openai_proxy, "MAX_CONTEXT_BYTES", 512)
    host.remember("vault door code " + "rotation schedule details " * 40)
    recall = host.ask([user(QUESTION)]).json()["scone"]["recall"]
    assert recall["max_context_bytes"] == 512
    assert recall["items"] == [] and recall["omitted_count"] >= 1


def test_the_answer_is_returned_when_capture_fails(host, monkeypatch):
    async def refuse(*_args, **_kwargs):
        raise RuntimeError("store is down")
    monkeypatch.setattr(host.engine, "remember_many", refuse)
    response = host.ask([user(QUESTION)])
    assert response.status_code == 200
    assert response.json()["choices"][0]["message"]["content"] == "It rotates on the 14th."
    assert response.json()["scone"]["capture"] == {
        "status": "failed", "episode_ids": [], "error_type": "RuntimeError",
        "user": {"status": "failed", "episode_id": None, "error_type": "RuntimeError"},
        "assistant": {"status": "not_attempted", "episode_id": None, "error_type": None}}
    assert response.headers["x-scone-capture-status"] == "failed"


def test_a_blank_reply_keeps_the_user_turn_and_says_the_reply_was_blank(host):
    host.holder["chat"].replies[:] = [" \n"]
    response = host.ask([user("Remember my badge is B-77 please.")])
    assert response.status_code == 200
    capture = response.json()["scone"]["capture"]
    assert capture["status"] == "partial" and capture["error_type"] is None
    assert capture["assistant"] == {"status": "blank", "episode_id": None, "error_type": None}
    assert capture["user"]["status"] == "captured" and capture["episode_ids"] == [capture["user"]["episode_id"]]
    assert [each["preview"] for each in host.episodes()] == ["Remember my badge is B-77 please."]


def test_a_reply_the_store_refuses_still_keeps_the_user_turn(host, monkeypatch):
    remember_many = host.engine.remember_many

    async def refuse_replies(space, records, **kwargs):
        if records[0].metadata["role"] == "assistant":
            raise RuntimeError("store refused the reply")
        return await remember_many(space, records, **kwargs)
    monkeypatch.setattr(host.engine, "remember_many", refuse_replies)
    capture = host.ask([user(QUESTION)]).json()["scone"]["capture"]
    assert capture["status"] == "partial" and capture["error_type"] == "RuntimeError"
    assert capture["user"]["status"] == "captured" and capture["episode_ids"] == [capture["user"]["episode_id"]]
    assert capture["assistant"] == {"status": "failed", "episode_id": None, "error_type": "RuntimeError"}
    assert [each["preview"] for each in host.episodes()] == [QUESTION]


def test_a_failing_model_is_502_says_nothing_of_the_provider_and_nothing_is_kept(host):
    host.holder["chat"].replies[:] = [ChatError('chat server returned 401: {"error": "Incorrect API key sk-proj-ab**yz"}')]
    response = host.ask([user(QUESTION)])
    assert response.status_code == 502
    assert response.json() == {"error": "the configured chat model did not return a reply"}
    assert host.episodes() == []


def refused(response, status: int = 422) -> str:
    assert response.status_code == status, response.text
    return response.json()["error"]


def test_stream_is_refused_in_this_version(host):
    assert "stream" in refused(host.ask([user(QUESTION)], stream=True))
    assert host.ask([user(QUESTION)], stream=False).status_code == 200
    assert len(host.chat.calls) == 1


@pytest.mark.parametrize("messages", [
    pytest.param([{"role": "system", "content": "Be brief."}], id="system-only"),
    pytest.param([user(QUESTION), {"role": "assistant", "content": "It rotates"}], id="ends-with-assistant"),
    pytest.param([user("   ")], id="blank-user"),
])
def test_a_request_without_a_latest_user_message_is_refused(host, messages):
    assert "user message" in refused(host.ask(messages))
    assert host.chat.calls == []


def test_no_messages_is_refused(host):
    assert "messages" in refused(host.ask([]))


def test_an_oversize_body_is_refused_whether_declared_or_streamed(host, monkeypatch):
    monkeypatch.setattr(openai_proxy, "MAX_BODY_BYTES", 300)
    body = json.dumps({"model": "gpt-4o", "messages": [user("vault " * 80)]}).encode()
    assert "a chat completion request takes at most 300 bytes" in refused(host.client.post(ROUTE, content=body, headers=auth()))
    chunks = iter([body[:200], body[200:]])
    assert "a chat completion request takes at most 300 bytes" in refused(host.client.post(ROUTE, content=chunks, headers=auth()))
    assert host.chat.calls == [] and host.episodes() == []


def test_an_oversize_message_is_refused(host, monkeypatch):
    monkeypatch.setattr(openai_proxy, "MAX_MESSAGE_BYTES", 20)
    assert "at most 20 bytes" in refused(host.ask([{"role": "system", "content": "x" * 21}, user("vault door?")]))
    assert host.ask([user("x" * 20)]).status_code == 200


def test_too_many_messages_are_refused(host, monkeypatch):
    monkeypatch.setattr(openai_proxy, "MAX_MESSAGES", 2)
    assert "at most 2 messages" in refused(host.ask([user("a vault"), {"role": "assistant", "content": "b"}, user(QUESTION)]))
    assert host.ask([{"role": "assistant", "content": "b"}, user(QUESTION)]).status_code == 200


@pytest.mark.parametrize("factory", [pytest.param(None, id="no-factory"), pytest.param(lambda: None, id="no-model")])
async def test_an_unconfigured_model_is_refused_and_nothing_is_kept(factory):
    engine = await MemoryEngine(InMemoryDocumentStore(), InMemoryVectorIndex(), HashEmbedder()).open()
    with TestClient(create_app(engine, KEYS, synthesis_factory=factory)) as client:
        response = client.post(ROUTE, json={"model": "gpt-4o", "messages": [user(QUESTION)]}, headers=auth())
        assert "none is configured" in refused(response)
        assert client.get("/v1/sources", headers=auth()).json()["items"] == []


@pytest.mark.parametrize("field", ["base_url", "api_key", "tools"])
def test_a_caller_cannot_name_an_endpoint_a_credential_or_tools(host, field):
    assert field in refused(host.ask([user(QUESTION)], **{field: "https://example.invalid"}))
    assert host.chat.calls == []


@pytest.mark.parametrize("message,needle", [
    pytest.param({"role": "tool", "content": "42"}, "role", id="tool-role"),
    pytest.param({"role": "user", "content": "vault", "name": "bob"}, "name", id="named-message"),
    pytest.param({"role": "user", "content": [{"type": "image_url", "image_url": {"url": "x"}}]}, "text", id="image-part"),
    pytest.param({"role": "user", "content": None}, "content", id="null-content"),
    pytest.param({"role": "user", "content": [{"type": "text", "text": 5}]}, "text", id="non-string-text-part"),
    pytest.param({"role": "user", "content": [{"type": "image_url", "text": "a caption", "image_url": {"url": "x"}}]},
                 "text", id="captioned-image-part"),
])
def test_content_this_version_cannot_carry_is_refused(host, message, needle):
    assert needle in refused(host.ask([message, user(QUESTION)]))
    assert host.chat.calls == []


def test_malformed_json_and_a_bad_session_header_are_refused(host):
    assert "JSON" in refused(host.client.post(ROUTE, content=b"{not json", headers=auth()))
    assert "x-scone-session" in refused(host.ask([user(QUESTION)], headers={"x-scone-session": "bad session!"}))
    assert "x-scone-session" in refused(host.ask([user(QUESTION)], headers={"x-scone-session": "s" * 122}))
    assert host.chat.calls == []


def test_a_read_only_key_cannot_use_the_route_because_it_writes(host):
    assert refused(host.ask([user(QUESTION)], key="reader"), 403)
    assert host.client.post(ROUTE, json={"model": "m", "messages": [user(QUESTION)]}).status_code == 401


def test_a_long_question_is_recalled_through_a_bounded_formulation(host):
    host.remember(PASSAGE)
    question = QUESTION + " " + "Please include every detail you have about the facilities schedule. " * 25
    recall = host.ask([user(question)]).json()["scone"]["recall"]
    assert recall["query_formulation"]["source_chars"] == len(question)
    assert recall["query_formulation"]["query_chars"] <= 1000


def test_recalled_claims_are_named_by_fact_id(host):
    episode = host.remember("Aurora uses LedgerDB for its records.", source="docs/aurora")
    fact = host.client.post("/v1/facts", headers=auth(), json={
        "subject": "Aurora", "predicate": "uses", "object": "LedgerDB",
        "source_episode_id": episode, "quote": "Aurora uses LedgerDB"}).json()
    recall = host.ask([user("What does Aurora use?")]).json()["scone"]["recall"]
    assert recall["claim_ids"] == [fact["fact_id"]]


def test_a_key_withdrawn_while_the_model_answers_keeps_nothing(host):
    class Withdrawing:
        async def complete(self, system: str, user: str) -> str:
            host.client.app.state.keys.pop("key-a")
            return "It rotates on the 14th."
    host.holder["chat"] = Withdrawing()
    assert host.ask([user(QUESTION)]).status_code == 401
    assert host.episodes(key="key-b") == []
    host.client.app.state.keys["key-a"] = "alpha"
    assert host.episodes() == []


async def test_capture_does_not_take_the_ingest_lane_so_a_full_lane_loses_no_turn():
    """The reply has gone out before the turn is kept, so a lane that
    refused would leave the client nothing to retry; native conversation
    turns do not take the lane either."""
    engine = await MemoryEngine(InMemoryDocumentStore(), InMemoryVectorIndex(), HashEmbedder()).open()
    release, holding = asyncio.Event(), asyncio.Event()
    remember = engine.remember

    async def slow_remember(*args, **kwargs):
        holding.set()
        await release.wait()
        return await remember(*args, **kwargs)

    engine.remember = slow_remember  # type: ignore[method-assign]
    app = create_app(engine, KEYS, ingest_concurrency=1, synthesis_factory=lambda: FakeChat(["It rotates on the 14th."]))
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://scone") as client:
        write = asyncio.create_task(client.post("/v1/episodes", json={"content": PASSAGE}, headers=auth()))
        await holding.wait()
        response = await client.post(ROUTE, json={"model": "gpt-4o", "messages": [user(QUESTION)]}, headers=auth())
        release.set()
        assert (await write).status_code == 200
    assert response.status_code == 200
    capture = response.json()["scone"]["capture"]
    assert capture["status"] == "captured" and len(capture["episode_ids"]) == 2
