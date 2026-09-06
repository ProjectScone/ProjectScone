"""Real scheduling with scripted model frames; no network model or live data."""

import asyncio
import copy

import pytest

pytest.importorskip("pipecat")
from pipecat.frames.frames import (
    CancelFrame, EndFrame, ErrorFrame, LLMContextFrame, LLMFullResponseEndFrame,
    LLMFullResponseStartFrame, LLMTextFrame, LLMThoughtTextFrame, FunctionCallsStartedFrame, InterruptionFrame,
)
from pipecat.processors.frame_processor import FrameProcessor

from scone_memory import HashEmbedder, InMemoryDocumentStore, InMemoryVectorIndex, MemoryEngine
from scone_memory.integrations.pipecat_text import PipecatTextConversation


@pytest.fixture
async def memory():
    return await MemoryEngine(InMemoryDocumentStore(), InMemoryVectorIndex(), HashEmbedder()).open()


class ScriptedModel(FrameProcessor):
    def __init__(self, text="Scripted reply.", *, mode="normal"):
        super().__init__()
        self.text, self.mode = text, mode
        self.requests = []
        self.entered, self.release = asyncio.Event(), asyncio.Event()

    async def process_frame(self, frame, direction):
        await super().process_frame(frame, direction)
        if not isinstance(frame, LLMContextFrame):
            if isinstance(frame, CancelFrame) and self.mode == "slow_cancel":
                await self.release.wait()
            await self.push_frame(frame, direction)
            return
        self.requests.append(copy.deepcopy(frame.context.get_messages()))
        if self.mode == "mutate":
            frame.context.get_messages()[0]["content"] = "mutated system"
            frame.context.get_messages()[-1]["content"] = "mutated user"
        self.entered.set()
        if self.mode == "wait":
            await self.release.wait()
        if self.mode == "error":
            await self.push_error_frame(ErrorFrame("private provider diagnostics"))
            return
        if self.mode == "tools":
            await self.push_frame(FunctionCallsStartedFrame([]))
        if self.mode == "interrupted":
            await self.push_frame(InterruptionFrame())
        await self.push_frame(LLMFullResponseStartFrame())
        await self.push_frame(LLMThoughtTextFrame("not public"))
        await self.push_frame(LLMTextFrame(self.text))
        if self.mode == "partial":
            await self.push_frame(EndFrame())
        else:
            await self.push_frame(LLMFullResponseEndFrame())
            if self.mode == "late_error":
                await self.push_error_frame(ErrorFrame("private finalization error"))


async def test_cancelled_turn_with_failed_processor_cleanup_closes_conversation(memory):
    class BrokenCleanup(ScriptedModel):
        async def cleanup(self):
            await super().cleanup()
            raise RuntimeError("processor cleanup failed")

    model = BrokenCleanup(mode="wait")
    conversation = PipecatTextConversation(memory, "alpha", "cleanup-failure", lambda: model)
    pending = asyncio.create_task(conversation.reply("first"))
    await asyncio.wait_for(model.entered.wait(), 5)
    pending.cancel()
    with pytest.raises(asyncio.CancelledError):
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

    conversation = PipecatTextConversation(memory, "alpha", "text-session", factory,
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
    assert episodes[1].metadata["completion_evidence"] == "response_end_frame"
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
    conversation = PipecatTextConversation(memory, "alpha", "fixed-scope", factory, where=where,
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
        PipecatTextConversation(memory, "alpha", "invalid-scope", must_not_start, **scope)
    assert await memory.episodes("alpha", {"session_id": "invalid-scope"}) == []


@pytest.mark.parametrize("mode,text", [("partial", "Unfinished"), ("error", ""), ("normal", "")])
async def test_failed_or_incomplete_reply_never_becomes_completed_memory(memory, mode, text):
    conversation = PipecatTextConversation(memory, "alpha", "failed", lambda: ScriptedModel(text, mode=mode), turn_timeout=1)
    with pytest.raises(RuntimeError) as error:
        await conversation.reply("question")
    assert "private provider diagnostics" not in str(error.value)
    assert [e.metadata["role"] for e in await memory.episodes("alpha", {"session_id": "failed"})] == ["user"]
    with pytest.raises(RuntimeError, match="closed"):
        await conversation.reply("do not retry")


async def test_close_cancels_pending_turn_and_rejects_simultaneous_send(memory):
    model = ScriptedModel(mode="wait")
    conversation = PipecatTextConversation(memory, "alpha", "cancelled", lambda: model)
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
    conversation = PipecatTextConversation(memory, "alpha", "timeout", lambda: model, turn_timeout=0.3)
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
    conversation = PipecatTextConversation(memory, "alpha", "invalid", factory)
    with pytest.raises(ValueError):
        await conversation.reply(message)
    assert models == []
    assert not await memory.episodes("alpha", {"session_id": "invalid"})
    await conversation.close()


@pytest.mark.parametrize("options,text", [({"max_reply_bytes": 512}, "é" * 257), ({"max_history_bytes": 512}, "reply" * 110)], ids=["reply", "history"])
async def test_output_and_history_budgets_reject_without_completed_capture(memory, options, text):
    conversation = PipecatTextConversation(memory, "alpha", "bounded", lambda: ScriptedModel(text), **options)
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
    conversation = PipecatTextConversation(memory, "alpha", "store-failure", lambda: model)
    with pytest.raises(OSError, match="injected"):
        await conversation.reply("question")
    with pytest.raises(RuntimeError, match="closed"):
        await conversation.reply("retry")
    assert len(model.requests) == 1
    assert [e.metadata["role"] for e in await memory.episodes("alpha", {"session_id": "store-failure"})] == ["user"]


async def test_pipeline_error_during_finalization_does_not_claim_success(memory):
    conversation = PipecatTextConversation(memory, "alpha", "late-error", lambda: ScriptedModel(mode="late_error"))
    with pytest.raises(RuntimeError, match="pipeline"):
        await conversation.reply("question")
    assert [e.metadata["role"] for e in await memory.episodes("alpha", {"session_id": "late-error"})] == ["user"]


async def test_timeout_during_pipeline_shutdown_is_not_swallowed(memory):
    conversation = PipecatTextConversation(memory, "alpha", "slow-end", lambda: ScriptedModel(mode="slow_cancel"), turn_timeout=0.3)
    with pytest.raises(TimeoutError):
        await conversation.reply("question")
    assert [e.metadata["role"] for e in await memory.episodes("alpha", {"session_id": "slow-end"})] == ["user"]


@pytest.mark.parametrize("mode", ["tools", "slow_cancel", "interrupted"])
async def test_unsupported_tool_flow_or_cleanup_timeout_does_not_succeed(memory, mode):
    conversation = PipecatTextConversation(memory, "alpha", "unsupported", lambda: ScriptedModel(mode=mode), turn_timeout=3)
    with pytest.raises(RuntimeError):
        await conversation.reply("question")
    assert [e.metadata["role"] for e in await memory.episodes("alpha", {"session_id": "unsupported"})] == ["user"]


async def test_model_context_mutation_cannot_rewrite_saved_conversation_history(memory):
    models = []
    def factory():
        model = ScriptedModel(mode="mutate" if not models else "normal")
        models.append(model)
        return model
    conversation = PipecatTextConversation(memory, "alpha", "history-copy", factory, where={"collection": "manuals"})
    await conversation.reply("original user")
    await conversation.reply("second user")
    assert models[1].requests[0][:2] == [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "original user"},
    ]
    await conversation.close()
