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


@pytest.mark.parametrize('question', ['Hello!', 'Juniper launch date?'])
async def test_required_evidence_never_falls_back_to_generation_on_empty_memory(memory, question):
    conversation = TextConversation(memory, 'alpha', 'required-empty', evidence_selector=Selector(),
        evidence_answer_policy='required')
    observed = []

    async def observe(text):
        observed.append(text)

    result = await conversation.reply(question, on_text=observe)
    assert observed == [result['text']]
    assert result['evidence_answer']['status'] == 'no_selection'
    assert result['evidence_answer']['source_status'] == 'none'
    assert result['evidence_answer']['evidence_ids'] == []
    episodes = await memory.episodes('alpha', {'session_id':'required-empty'})
    assert episodes[-1].metadata['completion_evidence'] == 'evidence_abstention'
    await conversation.close()


@pytest.mark.parametrize('policy', ['always', None, True])
def test_evidence_answer_policy_rejects_invalid_values(memory, policy):
    with pytest.raises(ValueError, match='evidence_answer_policy'):
        TextConversation(memory, 'alpha', 'policy', unused_factory, evidence_selector=Selector(),
                         evidence_answer_policy=policy)


def test_required_evidence_policy_requires_selector(memory):
    with pytest.raises(ValueError, match='evidence_selector'):
        TextConversation(memory, 'alpha', 'policy', unused_factory, evidence_answer_policy='required')


async def test_required_evidence_reports_retrieval_failure_without_generation(memory, monkeypatch):
    conversation = TextConversation(memory, 'alpha', 'required-failed', unused_factory,
        evidence_selector=Selector(), evidence_answer_policy='required')

    async def failed(messages):
        return messages, {'status':'failed'}

    monkeypatch.setattr(conversation._context, 'prepare', failed)
    with pytest.raises(RuntimeError, match='evidence answer'):
        await conversation.reply('Juniper launch date?')
    assert [row.metadata['role'] for row in await memory.episodes('alpha', {'session_id':'required-failed'})] == ['user']


@pytest.mark.parametrize('predicate', ['depends on', 'painted by'])
async def test_required_structured_answer_runs_through_scoped_retrieval_and_capture(engine, predicate):
    from scone_memory.core.ports import NewFact
    from scone_memory.realtime.structured_selector import StructuredEvidenceSelector
    from scone_memory.retrieval.structured_evidence import EvidenceRequirement
    quote = f'Juniper {predicate} Polaris.'
    for team, obj in [('blue', 'Polaris'), ('red', 'PRIVATE_OTHER_TEAM')]:
        text = f'Juniper {predicate} {obj}.'
        episode = await engine.remember('alpha', text, metadata={'team':team})
        await engine.documents.insert_fact(NewFact(space='alpha', subject='Juniper', predicate=predicate,
            object=obj, source_episode_id=episode.episode_id, quote=text, valid_from='2025-01-01T00:00:00Z'))
    question = 'What does Juniper depend on?'
    selector = StructuredEvidenceSelector(question, (
        EvidenceRequirement(kind='fact', subject='Juniper', predicate='depends on'),))
    conversation = TextConversation(engine, 'alpha', 'required-scoped', unused_factory,
        evidence_selector=selector, evidence_answer_policy='required', where={'team':'blue'})
    result = await conversation.reply(question)
    assert 'PRIVATE_OTHER_TEAM' not in result['text']
    assert (result['evidence_answer']['status'] == 'selected') is (predicate == 'depends on')
    if predicate == 'depends on':
        assert quote in result['text'] and result['evidence_answer']['atomic_selection'] is True
    else:
        assert quote not in result['text']
    episodes = await engine.episodes('alpha', {'session_id':'required-scoped'})
    assert episodes[-1].content == result['text']
    await conversation.close()


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


@pytest.mark.parametrize("demote_restated", [False, True])
async def test_extractive_history_budget_is_checked_before_callback_and_capture(memory: MemoryEngine, demote_restated: bool) -> None:
    memory.demote_restated = demote_restated
    history_limit = 512
    source_text = "Juniper calibration uses Polaris. " * 18
    added = await memory.remember("alpha", source_text)
    # One oversized chunk makes this a history-limit test regardless of rank.
    # The previous 25 repetitions also made a short tail that could rank first.
    chunks = await memory.documents.chunks_of("alpha", added.episode_id)
    assert len(chunks) == 1 and len(chunks[0].text.encode("utf-8")) > history_limit
    observed: list[str] = []

    async def observe(text: str) -> None:
        observed.append(text)

    selector = Selector()
    conversation = TextConversation(memory, "alpha", "budget-extractive", unused_factory,
        evidence_selector=selector, system_prompt="Answer.", max_history_bytes=history_limit)
    with pytest.raises(RuntimeError, match="conversation history byte limit"):
        await conversation.reply("Juniper calibration?", on_text=observe)
    assert len(selector.calls) == 1
    assert len(selector.calls[0][1][0].text.encode("utf-8")) > history_limit
    assert observed == [] and conversation.closed
    assert [episode.metadata["role"] for episode in await memory.episodes("alpha", {"session_id": "budget-extractive"})] == ["user"]
