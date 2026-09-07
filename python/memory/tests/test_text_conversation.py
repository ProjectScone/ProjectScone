"""Native Scone scheduling, scripted provider streams and real memory."""

import asyncio
import copy

import pytest

from scone_memory import HashEmbedder, InMemoryDocumentStore, InMemoryVectorIndex, MemoryEngine
from scone_memory.realtime.events import TextDelta, ReplyCompleted
from scone_memory.realtime.text import TextConversation

@pytest.fixture
async def memory():
    return await MemoryEngine(InMemoryDocumentStore(), InMemoryVectorIndex(), HashEmbedder()).open()


class ScriptedModel:
    def __init__(self, text="Scripted reply.", *, mode="normal"):
        self.text, self.mode = text, mode
        self.requests = []
        self.entered, self.release = asyncio.Event(), asyncio.Event()
        self.closes = 0

    async def respond(self, messages):
        self.requests.append(copy.deepcopy(messages))
        if self.mode == "mutate":
            messages[0]["content"] = "mutated system"
            messages[-1]["content"] = "mutated user"
        self.entered.set()
        if self.mode == "wait":
            await self.release.wait()
        if self.mode == "error":
            raise RuntimeError("private provider diagnostics")
        if self.mode in {"tools", "interrupted"}:
            yield object()
        yield TextDelta(self.text)
        if self.mode != "partial":
            yield ReplyCompleted()
            if self.mode == "late_error":
                raise RuntimeError("private finalization error")

    async def aclose(self):
        self.closes += 1
        if self.mode == "slow_cancel":
            raise RuntimeError("provider close failed")


async def test_cancelled_turn_with_failed_processor_cleanup_closes_conversation(memory):
    class BrokenCleanup(ScriptedModel):
        async def aclose(self):
            await super().aclose()
            raise RuntimeError("processor cleanup failed")

    model = BrokenCleanup(mode="wait")
    conversation = TextConversation(memory, "alpha", "cleanup-failure", lambda: model)
    pending = asyncio.create_task(conversation.reply("first"))
    await asyncio.wait_for(model.entered.wait(), 5)
    pending.cancel()
    with pytest.raises(RuntimeError, match="cleanup"):
        await pending
    assert conversation.closed is True
    with pytest.raises(RuntimeError, match="closed"):
        await conversation.reply("must not continue")


async def test_multi_turn_context_and_memory_preserve_only_public_messages(memory):
    await memory.remember("alpha", "Polaris calibrates Juniper.", metadata={"collection": "manuals"})
    models = []

    def factory():
        model = ScriptedModel("First reply." if not models else "Second reply.")
        models.append(model)
        return model

    conversation = TextConversation(memory, "alpha", "text-session", factory,
                                           where={"collection": "manuals"})
    first = await conversation.reply("How is Juniper calibrated?")
    second = await conversation.reply("And its name?")
    assert first["text"] == "First reply."
    assert second["text"] == "Second reply."
    assert first["memory_context"]["status"] == "prepared"
    messages = models[1].requests[0]
    assert [(m["role"], m["content"]) for m in messages if "Scone retrieved" not in m["content"]] == [
        ("system", "You are a helpful assistant."),
        ("user", "How is Juniper calibrated?"), ("assistant", "First reply."),
        ("user", "And its name?"),
    ]
    episodes = await memory.episodes("alpha", {"session_id": "text-session"})
    assert [e.content for e in episodes] == ["How is Juniper calibrated?", "First reply.", "And its name?", "Second reply."]
    assert [e.metadata["role"] for e in episodes] == ["user", "assistant", "user", "assistant"]
    assert episodes[1].metadata["capture_status"] == "aggregated"
    assert episodes[1].metadata["completion_evidence"] == "adapter_end_and_stream_closed"
    assert first["provider_completion"] == "unverified"
    assert [e.episode_id for e in episodes] == [first["user_episode_id"], first["assistant_episode_id"], second["user_episode_id"], second["assistant_episode_id"]]
    assert not (await memory.episodes("beta", {"session_id": "text-session"}))
    await conversation.close()


async def test_each_turn_uses_the_original_full_scope_despite_caller_metadata_mutation(memory):
    allowed = await memory.remember("alpha", "Juniper calibration approved manual", kind="file", source="manuals/one",
                                    created_at="2026-09-02T10:00:00Z", metadata={"team": "science"})
    await memory.remember("alpha", "Juniper calibration other department", kind="file", source="manuals/two",
                          created_at="2026-09-02T10:00:00Z", metadata={"team": "legal"})
    await memory.remember("alpha", "Juniper calibration wrong source", kind="file", source="private/one",
                          created_at="2026-09-02T10:00:00Z", metadata={"team": "science"})
    await memory.remember("alpha", "Juniper calibration old manual", kind="file", source="manuals/old",
                          created_at="2026-08-02T10:00:00Z", metadata={"team": "science"})
    models = []
    def factory():
        model = ScriptedModel("Juniper calibration public reply")
        models.append(model)
        return model
    where = {"team": "science"}
    conversation = TextConversation(memory, "alpha", "fixed-scope", factory, where=where,
                                           kind="file", source_prefix="manuals/",
                                           since="2026-09-01T00:00:00Z", until="2026-09-06T23:59:59Z")
    where["team"] = "legal"
    try:
        for question in ["Juniper calibration?", "Juniper calibration follow-up?"]:
            result = await conversation.reply(question)
            assert {r["episode_id"] for r in result["memory_context"]["references"]} == {allowed.episode_id}
        for model in models:
            source = next(m["content"] for m in model.requests[0] if m["content"].startswith("Scone retrieved"))
            assert "approved manual" in source
            assert all(text not in source for text in ["other department", "wrong source", "old manual", "public reply"])
        assert len(await memory.episodes("alpha", {"session_id": "fixed-scope"})) == 4
    finally:
        await conversation.close()


@pytest.mark.parametrize("scope", [{"kind": "unknown"}, {"source_prefix": False}, {"since": ""}, {"until": "tomorrow"},
                                  {"where": []}, {"where": {2: "invalid"}},
                                  {"since": "2026-09-06T00:00:00Z", "until": "2026-09-01T00:00:00Z"}])
async def test_invalid_text_scope_fails_before_factory_or_capture(memory, scope):
    def must_not_start():
        raise AssertionError("invalid scope must not start a model")
    with pytest.raises(ValueError):
        TextConversation(memory, "alpha", "invalid-scope", must_not_start, **scope)
    assert await memory.episodes("alpha", {"session_id": "invalid-scope"}) == []


@pytest.mark.parametrize("mode,text", [("partial", "Unfinished"), ("error", ""), ("normal", "")])
async def test_failed_or_incomplete_reply_never_becomes_completed_memory(memory, mode, text):
    conversation = TextConversation(memory, "alpha", "failed", lambda: ScriptedModel(text, mode=mode), turn_timeout=1)
    with pytest.raises(RuntimeError) as error:
        await conversation.reply("question")
    assert "private provider diagnostics" not in str(error.value)
    assert [e.metadata["role"] for e in await memory.episodes("alpha", {"session_id": "failed"})] == ["user"]
    with pytest.raises(RuntimeError, match="closed"):
        await conversation.reply("do not retry")


async def test_close_cancels_pending_turn_and_rejects_simultaneous_send(memory):
    model = ScriptedModel(mode="wait")
    conversation = TextConversation(memory, "alpha", "cancelled", lambda: model)
    pending = asyncio.create_task(conversation.reply("first"))
    await asyncio.wait_for(model.entered.wait(), 3)
    with pytest.raises(RuntimeError, match="active"):
        await conversation.reply("second")
    await asyncio.wait_for(conversation.close(), 3)
    with pytest.raises(asyncio.CancelledError):
        await pending
    model.release.set()
    assert [e.content for e in await memory.episodes("alpha", {"session_id": "cancelled"})] == ["first"]


async def test_timeout_closes_instance_without_assistant_capture(memory):
    model = ScriptedModel(mode="wait")
    conversation = TextConversation(memory, "alpha", "timeout", lambda: model, turn_timeout=0.3)
    with pytest.raises(TimeoutError):
        await conversation.reply("question")
    assert len(model.requests) == 1
    with pytest.raises(RuntimeError, match="closed"):
        await conversation.reply("retry")
    assert [e.metadata["role"] for e in await memory.episodes("alpha", {"session_id": "timeout"})] == ["user"]


@pytest.mark.parametrize("message", ["", "  ", None, "é" * 16001], ids=["empty", "blank", "null", "oversized"])
async def test_bad_input_does_not_start_or_capture_a_turn(memory, message):
    models = []
    def factory():
        models.append(ScriptedModel())
        return models[-1]
    conversation = TextConversation(memory, "alpha", "invalid", factory)
    with pytest.raises(ValueError):
        await conversation.reply(message)
    assert models == []
    assert not await memory.episodes("alpha", {"session_id": "invalid"})
    await conversation.close()


@pytest.mark.parametrize("options,text", [({"max_reply_bytes": 512}, "é" * 257), ({"max_history_bytes": 512}, "reply" * 110)], ids=["reply", "history"])
async def test_output_and_history_budgets_reject_without_completed_capture(memory, options, text):
    conversation = TextConversation(memory, "alpha", "bounded", lambda: ScriptedModel(text), **options)
    with pytest.raises(RuntimeError, match="byte limit"):
        await conversation.reply("question")
    assert [e.metadata["role"] for e in await memory.episodes("alpha", {"session_id": "bounded"})] == ["user"]


async def test_assistant_store_failure_is_not_reported_as_success_or_retried(memory, monkeypatch):
    remember = memory.remember_many
    async def fail_assistant(space, records):
        records = list(records)
        if records[0].metadata["role"] == "assistant":
            raise OSError("injected store failure")
        return await remember(space, records)
    monkeypatch.setattr(memory, "remember_many", fail_assistant)
    model = ScriptedModel()
    conversation = TextConversation(memory, "alpha", "store-failure", lambda: model)
    with pytest.raises(OSError, match="injected"):
        await conversation.reply("question")
    with pytest.raises(RuntimeError, match="closed"):
        await conversation.reply("retry")
    assert len(model.requests) == 1
    assert [e.metadata["role"] for e in await memory.episodes("alpha", {"session_id": "store-failure"})] == ["user"]


async def test_provider_error_during_finalization_does_not_claim_success(memory):
    conversation = TextConversation(memory, "alpha", "late-error", lambda: ScriptedModel(mode="late_error"))
    with pytest.raises(RuntimeError, match="provider"):
        await conversation.reply("question")
    assert [e.metadata["role"] for e in await memory.episodes("alpha", {"session_id": "late-error"})] == ["user"]


@pytest.mark.parametrize("mode", ["tools", "slow_cancel", "interrupted"])
async def test_unsupported_tool_flow_or_cleanup_timeout_does_not_succeed(memory, mode):
    conversation = TextConversation(memory, "alpha", "unsupported", lambda: ScriptedModel(mode=mode), turn_timeout=3)
    with pytest.raises(RuntimeError):
        await conversation.reply("question")
    assert [e.metadata["role"] for e in await memory.episodes("alpha", {"session_id": "unsupported"})] == ["user"]


async def test_model_context_mutation_cannot_rewrite_saved_conversation_history(memory):
    models = []
    def factory():
        model = ScriptedModel(mode="mutate" if not models else "normal")
        models.append(model)
        return model
    conversation = TextConversation(memory, "alpha", "history-copy", factory, where={"collection": "manuals"})
    await conversation.reply("original user")
    await conversation.reply("second user")
    assert models[1].requests[0][:2] == [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "original user"},
    ]
    await conversation.close()


class ChunkedModel:
    def __init__(self, chunks=("Hello ", "🌍"), *, release_end=None):
        self.chunks, self.release_end = chunks, release_end

    async def respond(self, messages):
        for chunk in self.chunks:
            yield TextDelta(chunk)
        if self.release_end:
            await self.release_end.wait()
        yield ReplyCompleted()

    async def aclose(self):
        pass


async def test_public_chunks_arrive_before_response_end_and_only_final_text_is_saved(memory):
    received, release_end, first_chunk = [], asyncio.Event(), asyncio.Event()

    async def observe(text):
        received.append(text)
        first_chunk.set()

    conversation = TextConversation(memory, "alpha", "stream", lambda: ChunkedModel(release_end=release_end))
    pending = asyncio.create_task(conversation.reply("greet me", on_text=observe))
    try:
        await asyncio.wait_for(first_chunk.wait(), 3)
        assert not pending.done()
        assert received[0] == "Hello "
        assert [e.content for e in await memory.episodes("alpha", {"session_id": "stream"})] == ["greet me"]
        release_end.set()
        result = await asyncio.wait_for(pending, 3)
        assert received == ["Hello ", "🌍"]
        assert result["text"] == "Hello 🌍"
        assert result["provider_completion"] == "unverified"
        episodes = await memory.episodes("alpha", {"session_id": "stream"})
        assert [e.content for e in episodes] == ["greet me", "Hello 🌍"]
        assert episodes[1].metadata["representation"] == "aggregated_text"
    finally:
        release_end.set()
        await conversation.close()
        await asyncio.gather(pending, return_exceptions=True)


async def test_public_observer_is_serial_and_completion_waits_for_it(memory):
    received, entered, release = [], asyncio.Event(), asyncio.Event()

    async def observe(text):
        received.append(text)
        if text == "Hello ":
            entered.set()
            await release.wait()

    conversation = TextConversation(memory, "alpha", "slow-observer", ChunkedModel)
    pending = asyncio.create_task(conversation.reply("greet me", on_text=observe))
    try:
        await asyncio.wait_for(entered.wait(), 3)
        assert received == ["Hello "]
        assert not pending.done()
        release.set()
        assert (await asyncio.wait_for(pending, 3))["text"] == "Hello 🌍"
        assert received == ["Hello ", "🌍"]
    finally:
        release.set()
        await conversation.close()
        await asyncio.gather(pending, return_exceptions=True)


@pytest.mark.parametrize("failure", [ValueError("private observer error"), asyncio.CancelledError()])
async def test_observer_failure_closes_turn_without_completed_capture(memory, failure):
    received = []

    async def observe(text):
        received.append(text)
        raise failure

    conversation = TextConversation(memory, "alpha", "observer-error", ChunkedModel)
    with pytest.raises(RuntimeError, match="observer") as error:
        await asyncio.wait_for(conversation.reply("question", on_text=observe), 3)
    assert "private observer error" not in str(error.value)
    assert received == ["Hello "]
    assert conversation.closed
    assert [e.metadata["role"] for e in await memory.episodes("alpha", {"session_id": "observer-error"})] == ["user"]


async def test_chunk_budget_is_checked_before_observer_delivery(memory):
    received = []

    async def observe(text):
        received.append(text)

    conversation = TextConversation(memory, "alpha", "stream-limit", lambda: ChunkedModel(("a" * 510, "🌍")), max_reply_bytes=512)
    with pytest.raises(RuntimeError, match="byte limit"):
        await conversation.reply("question", on_text=observe)
    assert received == ["a" * 510]
    assert [e.metadata["role"] for e in await memory.episodes("alpha", {"session_id": "stream-limit"})] == ["user"]


@pytest.mark.parametrize("observer", [False, "not a callback"])
async def test_invalid_observer_rejected_before_capture(memory, observer):
    conversation = TextConversation(memory, "alpha", "invalid-observer", ChunkedModel)
    with pytest.raises(ValueError, match="on_text"):
        await conversation.reply("question", on_text=observer)
    assert not conversation.closed
    assert await memory.episodes("alpha", {"session_id": "invalid-observer"}) == []
    await conversation.close()


@pytest.mark.parametrize("cancel", [False, True], ids=["deadline", "caller-cancel"])
async def test_pending_observer_is_cancelled_without_capturing_partial_reply(memory, cancel):
    entered, exited = asyncio.Event(), asyncio.Event()

    async def observe(text):
        entered.set()
        try:
            await asyncio.Event().wait()
        finally:
            exited.set()

    conversation = TextConversation(memory, "alpha", "observer-cancel", ChunkedModel, turn_timeout=0.5 if not cancel else 5)
    pending = asyncio.create_task(conversation.reply("question", on_text=observe))
    try:
        await asyncio.wait_for(entered.wait(), 3)
        if cancel:
            pending.cancel()
        with pytest.raises(asyncio.CancelledError if cancel else TimeoutError):
            await asyncio.wait_for(pending, 3)
        assert exited.is_set()
        assert conversation.closed  # cancelled observer effects are uncertain
        assert [e.metadata["role"] for e in await memory.episodes("alpha", {"session_id": "observer-cancel"})] == ["user"]
    finally:
        await conversation.close()
        await asyncio.gather(pending, return_exceptions=True)


async def test_observer_cannot_close_its_own_pipeline_and_deadlock(memory):
    conversation = TextConversation(memory, "alpha", "observer-reentrant", ChunkedModel)
    received_errors = []

    async def observe(text):
        try:
            await conversation.close()
        except RuntimeError as exc:
            received_errors.append(str(exc))

    try:
        result = await asyncio.wait_for(conversation.reply("question", on_text=observe), 3)
        assert len(received_errors) == 2
        assert result["text"] == "Hello 🌍"
        assert not conversation.closed
    finally:
        await conversation.close()


async def test_observer_does_not_leak_into_following_turn(memory):
    received = []

    async def observe(text):
        received.append(text)

    conversation = TextConversation(memory, "alpha", "observer-one-turn", lambda: ChunkedModel(("", "first", "")))
    try:
        await conversation.reply("one", on_text=observe)
        await conversation.reply("two")
        assert received == ["first"]
        assert [e.content for e in await memory.episodes("alpha", {"session_id": "observer-one-turn"})] == ["one", "first", "two", "first"]
    finally:
        await conversation.close()


async def test_non_awaitable_observer_fails_without_completed_capture(memory):
    conversation = TextConversation(memory, "alpha", "sync-observer", ChunkedModel)
    with pytest.raises(RuntimeError, match="observer"):
        await conversation.reply("question", on_text=lambda text: None)
    assert conversation.closed
    assert [e.metadata["role"] for e in await memory.episodes("alpha", {"session_id": "sync-observer"})] == ["user"]


async def test_cancel_swallowed_by_recall_never_starts_a_provider(memory, monkeypatch):
    result = await memory.recall("alpha", "empty")
    entered = asyncio.Event()
    async def stubborn_recall(*args, **kwargs):
        entered.set()
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            return result
    monkeypatch.setattr(memory, "recall", stubborn_recall)
    models = []
    def factory():
        models.append(ScriptedModel())
        return models[-1]
    conversation = TextConversation(memory, "alpha", "cancel-recall", factory)
    pending = asyncio.create_task(conversation.reply("Question"))
    await entered.wait()
    pending.cancel()
    with pytest.raises(asyncio.CancelledError):
        await pending
    assert models == []
    assert [e.content for e in await memory.episodes("alpha", {"session_id": "cancel-recall"})] == ["Question"]
