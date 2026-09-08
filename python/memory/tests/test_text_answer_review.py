"""Native optional answer review buffers drafts and captures only final replies."""

import asyncio
import copy

import pytest

from scone_memory import HashEmbedder, InMemoryDocumentStore, InMemoryVectorIndex, MemoryEngine
from scone_memory.realtime.events import ReplyCompleted, TextDelta
from scone_memory.realtime.text import TextConversation


@pytest.fixture
async def memory():
    engine = await MemoryEngine(InMemoryDocumentStore(), InMemoryVectorIndex(), HashEmbedder()).open()
    yield engine
    await engine.close()


class Model:
    def __init__(self, *parts, cleanup_failure=False):
        self.parts = parts or ("Juniper uses Vega.",)
        self.requests = []
        self.closes = 0
        self.cleanup_failure = cleanup_failure

    async def respond(self, messages):
        self.requests.append(copy.deepcopy(messages))
        for part in self.parts:
            yield TextDelta(part)
        yield ReplyCompleted()

    async def aclose(self):
        self.closes += 1
        if self.cleanup_failure:
            raise RuntimeError("PRIVATE cleanup diagnostics")


class Reviewer:
    def __init__(self, decide):
        self.decide = decide
        self.calls = []

    async def review(self, question, answer, evidence, evidence_ids):
        self.calls.append((question, answer, evidence, evidence_ids))
        return self.decide(question, answer, evidence, evidence_ids)


def supported(question, answer, evidence, evidence_ids):
    from scone_memory.realtime.answer_review import AnswerReviewDecision
    return AnswerReviewDecision(status="supported")


def correct_vega(question, answer, evidence, evidence_ids):
    from scone_memory.realtime.answer_review import AnswerIssue, AnswerReviewDecision
    if "Vega" in answer:
        return AnswerReviewDecision(status="needs_revision", revised_answer="Juniper uses Polaris.",
            issues=(AnswerIssue(code="contradiction", answer_quote="Vega", evidence_ids=(evidence_ids[0],)),))
    return AnswerReviewDecision(status="supported")


async def test_supported_revision_is_only_callback_history_and_assistant_capture(memory):
    await memory.remember("alpha", "Juniper calibration uses Polaris.")
    models = [Model("Juniper uses ", "Vega."), Model("Juniper remains calibrated.")]
    reviewer = Reviewer(correct_vega)
    conversation = TextConversation(memory, "alpha", "reviewed", lambda: models.pop(0), answer_reviewer=reviewer)
    observed = []

    async def observe(text):
        observed.append(text)

    first = await conversation.reply("Juniper calibration?", on_text=observe)
    second_model = models[0]
    await conversation.reply("Juniper calibration again?")
    assert first["text"] == "Juniper uses Polaris." and observed == ["Juniper uses Polaris."]
    assert first["answer_review"]["status"] == "supported" and first["answer_review"]["revised"] is True
    assert first["answer_review"]["verified_accuracy"] is False and first["answer_review"]["source_status"] == "retained"
    assert [call[1] for call in reviewer.calls[:2]] == ["Juniper uses Vega.", "Juniper uses Polaris."]
    assert all("Scone retrieved source material" in call[2] or '"sources"' in call[2] for call in reviewer.calls)
    assert all(call[3] for call in reviewer.calls)
    history = [message["content"] for message in second_model.requests[0] if message["role"] == "assistant"]
    assert history == ["Juniper uses Polaris."]
    episodes = await memory.episodes("alpha", {"session_id": "reviewed"})
    assert [episode.content for episode in episodes if episode.metadata["role"] == "assistant"] == [
        "Juniper uses Polaris.", "Juniper remains calibrated."]
    await conversation.close()


@pytest.mark.parametrize("policy", ["report", "require_supported"])
async def test_reviewer_failure_obeys_policy_without_exposing_draft_early(memory, policy):
    await memory.remember("alpha", "Juniper calibration uses Polaris.")

    def fail(*args):
        raise RuntimeError("PRIVATE reviewer details")

    model = Model("Juniper uses Polaris.")
    observed = []

    async def observe(text):
        observed.append(text)

    conversation = TextConversation(memory, "alpha", "failed-review", lambda: model,
                                    answer_reviewer=Reviewer(fail), review_policy=policy)
    if policy == "report":
        result = await conversation.reply("Juniper calibration?", on_text=observe)
        assert result["text"] == "Juniper uses Polaris." and observed == [result["text"]]
        assert result["answer_review"]["status"] == "unavailable"
        assert result["answer_review"]["source_status"] == "retained"
        assert "PRIVATE" not in str(result["answer_review"])
    else:
        with pytest.raises(RuntimeError, match="answer review") as error:
            await conversation.reply("Juniper calibration?", on_text=observe)
        assert observed == [] and "PRIVATE" not in str(error.value)
        assert [episode.metadata["role"] for episode in await memory.episodes("alpha", {"session_id": "failed-review"})] == ["user"]
    assert model.closes == 1
    await conversation.close()


@pytest.mark.parametrize("policy", ["report", "require_supported"])
async def test_source_deleted_during_review_suppresses_all_output(memory, policy):
    episode = await memory.remember("alpha", "Juniper calibration uses Polaris.")

    class DeleteSource:
        async def review(self, question, answer, evidence, evidence_ids):
            await memory.forget("alpha", episode.episode_id)
            return supported(question, answer, evidence, evidence_ids)

    model = Model("Juniper uses Polaris.")
    observed = []

    async def observe(text):
        observed.append(text)

    conversation = TextConversation(memory, "alpha", "stale-review", lambda: model,
                                    answer_reviewer=DeleteSource(), review_policy=policy)
    with pytest.raises(RuntimeError, match="answer review") as error:
        await conversation.reply("Juniper calibration?", on_text=observe)
    assert error.value.answer_review["source_status"] == "stale"
    assert observed == [] and model.closes == 1
    assert [episode.metadata["role"] for episode in await memory.episodes("alpha", {"session_id": "stale-review"})] == ["user"]


async def test_cancel_during_review_never_emits_or_captures_draft(memory):
    await memory.remember("alpha", "Juniper calibration uses Polaris.")
    entered = asyncio.Event()

    class WaitingReviewer:
        async def review(self, *args):
            entered.set()
            await asyncio.Event().wait()
            raise AssertionError("unreachable")

    model = Model("Juniper uses ", "Vega.")
    observed = []

    async def observe(text):
        observed.append(text)

    conversation = TextConversation(memory, "alpha", "cancel-review", lambda: model, answer_reviewer=WaitingReviewer())
    pending = asyncio.create_task(conversation.reply("Juniper calibration?", on_text=observe))
    await asyncio.wait_for(entered.wait(), 2)
    assert observed == [] and model.closes == 1
    pending.cancel()
    with pytest.raises(asyncio.CancelledError):
        await pending
    assert observed == []
    assert [episode.metadata["role"] for episode in await memory.episodes("alpha", {"session_id": "cancel-review"})] == ["user"]
    await conversation.close()


async def test_final_observer_failure_keeps_cleanup_and_capture_safety(memory):
    await memory.remember("alpha", "Juniper calibration uses Polaris.")
    model = Model()
    observed = []

    async def broken_observer(text):
        observed.append(text)
        raise RuntimeError("PRIVATE observer effect")

    conversation = TextConversation(memory, "alpha", "observer-failure", lambda: model, answer_reviewer=Reviewer(correct_vega))
    with pytest.raises(RuntimeError, match="public text observer failed"):
        await conversation.reply("Juniper calibration?", on_text=broken_observer)
    assert observed == ["Juniper uses Polaris."] and model.closes == 1 and conversation.closed
    assert [episode.metadata["role"] for episode in await memory.episodes("alpha", {"session_id": "observer-failure"})] == ["user"]


async def test_no_reviewer_preserves_public_stream_chunks_and_receipt_shape(memory):
    model = Model("First ", "second.")
    observed = []

    async def observe(text):
        observed.append(text)

    conversation = TextConversation(memory, "alpha", "legacy-stream", lambda: model)
    result = await conversation.reply("Hello!", on_text=observe)
    assert observed == ["First ", "second."] and result["text"] == "First second."
    assert "answer_review" not in result and model.closes == 1
    await conversation.close()


@pytest.mark.parametrize("question", ["Hello!", "Unmatched question?"])
async def test_no_prepared_memory_skips_review_but_emits_final_once(memory, question):
    reviewer = Reviewer(supported)
    model = Model("Conversation ", "reply.")
    observed = []

    async def observe(text):
        observed.append(text)

    conversation = TextConversation(memory, "alpha", "skip-review", lambda: model, answer_reviewer=reviewer)
    result = await conversation.reply(question, on_text=observe)
    assert result["answer_review"] == {"status": "skipped", "reason": "no_memory_evidence", "verified_accuracy": False}
    assert observed == ["Conversation reply."] and reviewer.calls == []
    await conversation.close()


async def test_invalid_review_configuration_fails_before_capture(memory):
    reviewer = Reviewer(supported)
    for options in [{"review_policy": "unsupported"}, {"review_policy": True}, {"answer_reviewer": object()},
                    {"review_limits": object()}, {"review_policy": "require_supported"}]:
        with pytest.raises(ValueError):
            TextConversation(memory, "alpha", "invalid-review", Model, **options)
    with pytest.raises(ValueError):
        TextConversation(memory, "alpha", "invalid-review", Model, answer_reviewer=reviewer, review_policy=False)
    assert await memory.episodes("alpha", {"session_id": "invalid-review"}) == []


async def test_unconfirmed_correction_never_reaches_callback_or_capture(memory):
    from scone_memory.realtime.answer_review import AnswerReviewDecision

    await memory.remember("alpha", "Juniper calibration uses Polaris.")

    def reject_revision(question, answer, evidence, evidence_ids):
        if "Vega" in answer:
            return correct_vega(question, answer, evidence, evidence_ids)
        return AnswerReviewDecision(status="uncertain")

    observed = []

    async def observe(text):
        observed.append(text)

    conversation = TextConversation(memory, "alpha", "unconfirmed", Model, answer_reviewer=Reviewer(reject_revision))
    result = await conversation.reply("Juniper calibration?", on_text=observe)
    assert result["text"] == "Juniper uses Vega." and observed == ["Juniper uses Vega."]
    assert result["answer_review"]["revised"] is False and result["answer_review"]["status"] == "uncertain"
    assert [episode.content for episode in await memory.episodes("alpha", {"session_id": "unconfirmed"})] == [
        "Juniper calibration?", "Juniper uses Vega."]
    await conversation.close()


async def test_source_preparation_timeout_suppresses_output_and_preserves_safe_receipt(memory, monkeypatch):
    from scone_memory.realtime.answer_review import AnswerReviewLimits

    await memory.remember("alpha", "Juniper calibration uses Polaris.")

    async def stalled(*args, **kwargs):
        await asyncio.Event().wait()

    monkeypatch.setattr("scone_memory.realtime.text.prepare_review_evidence", stalled)
    reviewer = Reviewer(supported)
    model = Model()
    observed = []

    async def observe(text):
        observed.append(text)

    conversation = TextConversation(memory, "alpha", "proof-timeout", lambda: model, answer_reviewer=reviewer,
        review_limits=AnswerReviewLimits(timeout_s=1.0))
    with pytest.raises(RuntimeError, match="answer review evidence unavailable") as error:
        await conversation.reply("Juniper calibration?", on_text=observe)
    assert error.value.answer_review["source_status"] == "unavailable"
    assert error.value.answer_review["errors"] == ["source_validation_timeout"]
    assert observed == [] and reviewer.calls == [] and model.closes == 1
    assert [episode.metadata["role"] for episode in await memory.episodes("alpha", {"session_id": "proof-timeout"})] == ["user"]


async def test_source_preparation_and_review_share_one_total_deadline(memory, monkeypatch):
    from scone_memory.realtime.answer_review import AnswerReviewLimits
    from scone_memory.realtime.review_evidence import prepare_review_evidence

    await memory.remember("alpha", "Juniper calibration uses Polaris.")

    async def slow_proof(*args, **kwargs):
        await asyncio.sleep(0.6)
        return await prepare_review_evidence(*args, **kwargs)

    class SlowReview:
        async def review(self, question, answer, evidence, evidence_ids):
            await asyncio.sleep(0.6)
            return supported(question, answer, evidence, evidence_ids)

    monkeypatch.setattr("scone_memory.realtime.text.prepare_review_evidence", slow_proof)
    observed = []

    async def observe(text):
        observed.append(text)

    conversation = TextConversation(memory, "alpha", "shared-budget", Model, answer_reviewer=SlowReview(),
        review_limits=AnswerReviewLimits(timeout_s=1.0))
    result = await conversation.reply("Juniper calibration?", on_text=observe)
    assert result["answer_review"]["status"] == "unavailable"
    assert result["answer_review"]["source_status"] == "retained"
    assert "review_timeout" in result["answer_review"]["errors"]
    assert observed == ["Juniper uses Vega."]
    await conversation.close()


async def test_provider_cleanup_failure_prevents_review_and_final_callback(memory):
    await memory.remember("alpha", "Juniper calibration uses Polaris.")
    model = Model(cleanup_failure=True)
    reviewer = Reviewer(supported)
    observed = []

    async def observe(text):
        observed.append(text)

    conversation = TextConversation(memory, "alpha", "cleanup-before-review", lambda: model, answer_reviewer=reviewer)
    with pytest.raises(RuntimeError, match="model provider cleanup failed"):
        await conversation.reply("Juniper calibration?", on_text=observe)
    assert reviewer.calls == [] and observed == [] and model.closes == 1
    assert [episode.metadata["role"] for episode in await memory.episodes("alpha", {"session_id": "cleanup-before-review"})] == ["user"]


async def test_revised_reply_budget_is_checked_before_final_callback(memory):
    from scone_memory.realtime.answer_review import AnswerIssue, AnswerReviewDecision

    await memory.remember("alpha", "Juniper calibration uses Polaris.")

    def oversized_revision(question, answer, evidence, evidence_ids):
        if "Vega" in answer:
            return AnswerReviewDecision(status="needs_revision", revised_answer="Polaris " * 100,
                issues=(AnswerIssue(code="contradiction", answer_quote="Vega", evidence_ids=(evidence_ids[0],)),))
        return AnswerReviewDecision(status="supported")

    observed = []

    async def observe(text):
        observed.append(text)

    conversation = TextConversation(memory, "alpha", "revision-budget", Model,
        answer_reviewer=Reviewer(oversized_revision), max_reply_bytes=512)
    with pytest.raises(RuntimeError, match="model reply exceeded"):
        await conversation.reply("Juniper calibration?", on_text=observe)
    assert observed == []
    assert [episode.metadata["role"] for episode in await memory.episodes("alpha", {"session_id": "revision-budget"})] == ["user"]


async def test_source_preparation_cannot_swallow_shared_review_deadline(memory, monkeypatch):
    from scone_memory.realtime.answer_review import AnswerReviewLimits
    from scone_memory.realtime.review_evidence import prepare_review_evidence

    await memory.remember("alpha", "Juniper calibration uses Polaris.")

    async def swallow_timeout(*args, **kwargs):
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            return await prepare_review_evidence(*args, **kwargs)

    monkeypatch.setattr("scone_memory.realtime.text.prepare_review_evidence", swallow_timeout)
    reviewer = Reviewer(supported)
    observed = []

    async def observe(text):
        observed.append(text)

    conversation = TextConversation(memory, "alpha", "swallowed-deadline", Model, answer_reviewer=reviewer,
        review_limits=AnswerReviewLimits(timeout_s=1.0))
    with pytest.raises(RuntimeError, match="answer review") as error:
        await conversation.reply("Juniper calibration?", on_text=observe)
    assert error.value.answer_review["source_status"] == "unavailable"
    assert observed == [] and reviewer.calls == []


async def test_source_preparation_cannot_swallow_external_cancellation_and_start_review(memory, monkeypatch):
    from scone_memory.realtime.review_evidence import prepare_review_evidence

    await memory.remember("alpha", "Juniper calibration uses Polaris.")
    entered = asyncio.Event()

    async def swallow_cancel(*args, **kwargs):
        entered.set()
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            return await prepare_review_evidence(*args, **kwargs)

    monkeypatch.setattr("scone_memory.realtime.text.prepare_review_evidence", swallow_cancel)
    reviewer = Reviewer(supported)
    conversation = TextConversation(memory, "alpha", "swallowed-cancel", Model, answer_reviewer=reviewer)
    pending = asyncio.create_task(conversation.reply("Juniper calibration?"))
    await asyncio.wait_for(entered.wait(), 2)
    pending.cancel()
    with pytest.raises(asyncio.CancelledError):
        await pending
    assert reviewer.calls == []
    await conversation.close()
