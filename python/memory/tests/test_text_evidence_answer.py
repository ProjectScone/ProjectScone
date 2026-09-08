"""Optional extractive answers bypass generation and preserve public turn safety."""

import asyncio

import pytest

from scone_memory import HashEmbedder, InMemoryDocumentStore, InMemoryVectorIndex, MemoryEngine
from scone_memory.realtime.events import ReplyCompleted, TextDelta
from scone_memory.realtime.text import TextConversation


@pytest.fixture
async def memory():
    engine = await MemoryEngine(InMemoryDocumentStore(), InMemoryVectorIndex(), HashEmbedder()).open()
    yield engine
    await engine.close()


class Selector:
    def __init__(self):
        self.calls = []

    async def select(self, question, cards):
        from scone_memory.realtime.evidence_answer import EvidenceSelection
        self.calls.append((question, cards))
        return EvidenceSelection(card_ids=(cards[0].id,))


class Model:
    def __init__(self):
        self.closes = 0

    async def respond(self, messages):
        yield TextDelta("Normal ")
        yield TextDelta("reply.")
        yield ReplyCompleted()

    async def aclose(self):
        self.closes += 1


def unused_factory():
    raise AssertionError("extractive mode must not create a generation provider")


async def test_extractive_answer_uses_checked_cards_and_only_final_callback_history_capture(memory):
    await memory.remember("alpha", "Juniper calibration uses Polaris.", metadata={"team": "science"})
    await memory.remember("alpha", "Juniper calibration PRIVATE record.", metadata={"team": "legal"})
    selector = Selector()
    conversation = TextConversation(memory, "alpha", "extractive", unused_factory,
        evidence_selector=selector, where={"team": "science"})
    observed = []

    async def observe(text):
        observed.append(text)

    result = await conversation.reply("Juniper calibration?", on_text=observe)
    assert observed == [result["text"]] and "Juniper calibration uses Polaris." in result["text"]
    assert "PRIVATE" not in str(selector.calls) and len(selector.calls) == 1
    assert result["evidence_answer"]["verified_accuracy"] is False and "answer_review" not in result
    assert conversation._history[-1] == {"role": "assistant", "content": result["text"]}
    episodes = await memory.episodes("alpha", {"session_id": "extractive"})
    assert [episode.content for episode in episodes] == ["Juniper calibration?", result["text"]]
    assert episodes[-1].metadata["completion_evidence"] == "source_checked_extractive_answer"
    await conversation.close()


@pytest.mark.parametrize("question", ["Hello!", "Juniper calibration?"])
async def test_selector_skips_without_prepared_scoped_memory_and_preserves_normal_stream(memory, question):
    await memory.remember("alpha", "Juniper calibration PRIVATE record.", metadata={"team": "legal"})
    selector, model = Selector(), Model()
    conversation = TextConversation(memory, "alpha", "skip-extractive", lambda: model,
        evidence_selector=selector, where={"team": "science"})
    observed = []

    async def observe(text):
        observed.append(text)

    result = await conversation.reply(question, on_text=observe)
    assert observed == ["Normal ", "reply."] and selector.calls == [] and model.closes == 1
    assert result["evidence_answer"] == {"status": "skipped", "reason": "no_memory_evidence", "verified_accuracy": False}
    await conversation.close()


async def test_source_deleted_during_selection_prevents_emission_and_capture(memory):
    source = await memory.remember("alpha", "Juniper calibration uses Polaris.")

    class DeleteSource(Selector):
        async def select(self, question, cards):
            selected = await super().select(question, cards)
            await memory.forget("alpha", source.episode_id)
            return selected

    conversation = TextConversation(memory, "alpha", "stale-extractive", unused_factory, evidence_selector=DeleteSource())
    observed = []

    async def observe(text):
        observed.append(text)

    with pytest.raises(RuntimeError, match="evidence answer") as error:
        await conversation.reply("Juniper calibration?", on_text=observe)
    assert error.value.evidence_answer["errors"] == ["stale_evidence"] and observed == []
    assert [episode.metadata["role"] for episode in await memory.episodes("alpha", {"session_id": "stale-extractive"})] == ["user"]


async def test_selection_failure_is_sanitized_without_generator_fallback(memory):
    await memory.remember("alpha", "Juniper calibration uses Polaris.")

    class BrokenSelector:
        async def select(self, *args):
            raise RuntimeError("PRIVATE provider details")

    conversation = TextConversation(memory, "alpha", "failed-extractive", unused_factory, evidence_selector=BrokenSelector())
    with pytest.raises(RuntimeError, match="evidence answer") as error:
        await conversation.reply("Juniper calibration?")
    assert error.value.evidence_answer["errors"] == ["selection_provider_failed"]
    assert "PRIVATE" not in str(error.value) and conversation.closed
    assert [episode.metadata["role"] for episode in await memory.episodes("alpha", {"session_id": "failed-extractive"})] == ["user"]


async def test_cancel_during_selection_never_emits_or_captures_answer(memory):
    await memory.remember("alpha", "Juniper calibration uses Polaris.")
    entered = asyncio.Event()

    class WaitingSelector:
        async def select(self, *args):
            entered.set()
            await asyncio.Event().wait()

    observed = []

    async def observe(text):
        observed.append(text)

    conversation = TextConversation(memory, "alpha", "cancel-extractive", unused_factory, evidence_selector=WaitingSelector())
    pending = asyncio.create_task(conversation.reply("Juniper calibration?", on_text=observe))
    await asyncio.wait_for(entered.wait(), 2)
    pending.cancel()
    with pytest.raises(asyncio.CancelledError):
        await pending
    assert observed == []
    assert [episode.metadata["role"] for episode in await memory.episodes("alpha", {"session_id": "cancel-extractive"})] == ["user"]
    await conversation.close()


async def test_extractive_final_observer_failure_does_not_capture_answer(memory):
    await memory.remember("alpha", "Juniper calibration uses Polaris.")
    observed = []

    async def broken(text):
        observed.append(text)
        raise RuntimeError("PRIVATE observer failure")

    conversation = TextConversation(memory, "alpha", "observer-extractive", unused_factory, evidence_selector=Selector())
    with pytest.raises(RuntimeError, match="public text observer failed"):
        await conversation.reply("Juniper calibration?", on_text=broken)
    assert len(observed) == 1 and conversation.closed
    assert [episode.metadata["role"] for episode in await memory.episodes("alpha", {"session_id": "observer-extractive"})] == ["user"]


async def test_proof_preparation_and_selection_share_deadline(memory, monkeypatch):
    from scone_memory.realtime.review_evidence import prepare_review_evidence

    await memory.remember("alpha", "Juniper calibration uses Polaris.")

    async def slow_proof(*args, **kwargs):
        await asyncio.sleep(0.6)
        return await prepare_review_evidence(*args, **kwargs)

    class SlowSelector(Selector):
        async def select(self, question, cards):
            await asyncio.sleep(0.6)
            return await super().select(question, cards)

    monkeypatch.setattr("scone_memory.realtime.text.prepare_review_evidence", slow_proof)
    conversation = TextConversation(memory, "alpha", "deadline-extractive", unused_factory,
        evidence_selector=SlowSelector(), evidence_answer_timeout=1.0)
    with pytest.raises(RuntimeError, match="evidence answer") as error:
        await conversation.reply("Juniper calibration?")
    assert error.value.evidence_answer["errors"] == ["selection_timeout"]
    assert [episode.metadata["role"] for episode in await memory.episodes("alpha", {"session_id": "deadline-extractive"})] == ["user"]


async def test_invalid_extractive_settings_fail_before_capture(memory):
    for value in [True, False, 0, 0.5, 181, float("inf"), float("nan"), "20", None]:
        with pytest.raises(ValueError):
            TextConversation(memory, "alpha", "invalid-extractive", Model,
                evidence_selector=Selector(), evidence_answer_timeout=value)
    for options in [{"evidence_selector": object()}, {"evidence_answer_timeout": 1.0},
                    {"evidence_selector": Selector(), "answer_reviewer": type("Reviewer", (), {"review": lambda self: None})()}]:
        with pytest.raises(ValueError):
            TextConversation(memory, "alpha", "invalid-extractive", Model, **options)
    assert await memory.episodes("alpha", {"session_id": "invalid-extractive"}) == []


async def test_default_generation_retains_stream_and_has_no_extractive_receipt(memory):
    model = Model()
    observed = []

    async def observe(text):
        observed.append(text)

    conversation = TextConversation(memory, "alpha", "default-extractive", lambda: model)
    result = await conversation.reply("Hello!", on_text=observe)
    assert observed == ["Normal ", "reply."] and model.closes == 1
    assert "evidence_answer" not in result and "answer_review" not in result
    await conversation.close()


@pytest.mark.parametrize("external_cancel", [False, True])
async def test_proof_cannot_swallow_deadline_or_cancellation_and_start_selection(memory, monkeypatch, external_cancel):
    from scone_memory.realtime.review_evidence import prepare_review_evidence

    await memory.remember("alpha", "Juniper calibration uses Polaris.")
    entered = asyncio.Event()

    async def swallow(*args, **kwargs):
        entered.set()
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            return await prepare_review_evidence(*args, **kwargs)

    monkeypatch.setattr("scone_memory.realtime.text.prepare_review_evidence", swallow)
    selector = Selector()
    conversation = TextConversation(memory, "alpha", "proof-extractive", unused_factory,
        evidence_selector=selector, evidence_answer_timeout=1.0)
    pending = asyncio.create_task(conversation.reply("Juniper calibration?"))
    await asyncio.wait_for(entered.wait(), 2)
    if external_cancel:
        pending.cancel()
        with pytest.raises(asyncio.CancelledError):
            await pending
    else:
        with pytest.raises(RuntimeError, match="evidence answer") as error:
            await pending
        assert error.value.evidence_answer["errors"] == ["source_validation_timeout"]
    assert selector.calls == []
    assert [episode.metadata["role"] for episode in await memory.episodes("alpha", {"session_id": "proof-extractive"})] == ["user"]
    await conversation.close()


async def test_extractive_history_budget_is_checked_before_callback_and_capture(memory):
    await memory.remember("alpha", "Juniper calibration uses Polaris. " * 25)
    observed = []

    async def observe(text):
        observed.append(text)

    conversation = TextConversation(memory, "alpha", "budget-extractive", unused_factory,
        evidence_selector=Selector(), system_prompt="Answer.", max_history_bytes=512)
    with pytest.raises(RuntimeError, match="conversation history byte limit"):
        await conversation.reply("Juniper calibration?", on_text=observe)
    assert observed == []
    assert [episode.metadata["role"] for episode in await memory.episodes("alpha", {"session_id": "budget-extractive"})] == ["user"]
