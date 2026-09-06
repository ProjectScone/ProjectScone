"""Real Pipecat aggregation → Scone persistence/recall; no provider or microphone.

Run in the optional Pipecat environment. The frame text is a test fixture, not
captured user activity. These tests never open the application's live database.
"""

import asyncio
from dataclasses import dataclass, field
from datetime import datetime, timezone

import pytest

pytest.importorskip("pipecat.processors.aggregators.llm_response_universal")

from pipecat.frames.frames import (
    DataFrame, EndFrame, InterimTranscriptionFrame, InterruptionFrame,
    LLMFullResponseEndFrame, LLMFullResponseStartFrame, LLMTextFrame,
    LLMThoughtEndFrame, LLMThoughtStartFrame, LLMThoughtTextFrame,
    TranscriptionFrame,
)
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.worker import PipelineWorker
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.processors.aggregators.llm_response_universal import (
    AssistantTurnStoppedMessage, LLMContextAggregatorPair, UserTurnMessageAddedMessage,
)
from pipecat.processors.frame_processor import FrameProcessor
from pipecat.workers.runner import WorkerRunner

from scone_memory import HashEmbedder, InMemoryDocumentStore, InMemoryVectorIndex, MemoryEngine


@pytest.fixture
async def memory():
    engine = await MemoryEngine(
        InMemoryDocumentStore(), InMemoryVectorIndex(), HashEmbedder()
    ).open()
    yield engine


@dataclass
class Barrier(DataFrame):
    reached: asyncio.Event = field(default_factory=asyncio.Event)


class Acknowledge(FrameProcessor):
    async def process_frame(self, frame, direction):
        await super().process_frame(frame, direction)
        if isinstance(frame, Barrier):
            frame.reached.set()
        await self.push_frame(frame, direction)


async def run_frames(pair, frames):
    """Use the production pipeline lifecycle, with explicit interruption barriers."""
    worker = PipelineWorker(
        Pipeline([pair.user(), pair.assistant(), Acknowledge()]),
        enable_rtvi=False, cancel_on_idle_timeout=False,
    )
    started = asyncio.Event()

    @worker.event_handler("on_pipeline_started")
    async def on_started(worker, frame):
        started.set()

    async def produce():
        await asyncio.wait_for(started.wait(), 3)
        for frame in frames:
            await worker.queue_frame(frame)
            if isinstance(frame, Barrier):
                await asyncio.wait_for(frame.reached.wait(), 3)
        await worker.queue_frame(EndFrame())

    runner = WorkerRunner()
    await runner.add_workers(worker)
    await asyncio.wait_for(asyncio.gather(runner.run(), produce()), 10)


async def test_pipeline_transcripts_are_scoped_retrievable_episodes(memory):
    # Catches wrong event binding, fragment capture, role loss and wrong space.
    from scone_memory.integrations.pipecat import SconePipecatMemory

    context = LLMContext()
    pair = LLMContextAggregatorPair(context, realtime_service_mode=True)
    capture = SconePipecatMemory(memory, "voice-test", "conversation-1")
    capture.attach(pair)
    await run_frames(pair, [
        InterimTranscriptionFrame("not finalized", "speaker-1", "2026-09-06T10:00:00Z"),
        TranscriptionFrame("The telescope is named Juniper.", "speaker-1", "2026-09-06T10:00:01Z"),
        LLMFullResponseStartFrame(), LLMTextFrame("Juniper "),
        LLMTextFrame("is the telescope."), LLMFullResponseEndFrame(),
    ])
    capture.raise_if_failed()
    episodes = await memory.episodes("voice-test", {"session_id": "conversation-1"})
    assert [(e.metadata["role"], e.content) for e in episodes] == [
        ("user", "The telescope is named Juniper."),
        ("assistant", "Juniper is the telescope."),
    ]
    assert [e.metadata["capture_status"] for e in episodes] == ["context_message", "aggregated"]
    assert [e.metadata["seq"] for e in episodes] == ["0", "1"]
    assert all(e.kind == "conversation" and e.source == "conversation-1" for e in episodes)
    assert all(e.metadata["integration"] == "pipecat" for e in episodes)
    recall = await memory.recall("voice-test", "Juniper", where={"session_id": "conversation-1"})
    assert {e.episode_id for e in episodes} <= {item.episode_id for item in recall.items}
    assert not (await memory.recall("other-space", "Juniper")).items
    assert capture.stored_count == 2


async def test_interrupted_partial_is_not_labeled_complete(memory):
    from scone_memory.integrations.pipecat import SconePipecatMemory

    pair = LLMContextAggregatorPair(LLMContext(), realtime_service_mode=True)
    capture = SconePipecatMemory(memory, "voice-test", "interruptions")
    capture.attach(pair)
    await run_frames(pair, [
        LLMFullResponseStartFrame(), LLMTextFrame("A partial answer"), Barrier(),
        InterruptionFrame(), Barrier(),
        LLMFullResponseStartFrame(), LLMTextFrame("A revised answer"),
        LLMFullResponseEndFrame(),
    ])
    capture.raise_if_failed()
    episodes = await memory.episodes("voice-test", {"session_id": "interruptions"})
    assert [(e.content, e.metadata["capture_status"]) for e in episodes] == [
        ("A partial answer", "interrupted"), ("A revised answer", "aggregated"),
    ]


async def test_thought_frames_and_empty_turns_do_not_become_memory(memory):
    from scone_memory.integrations.pipecat import SconePipecatMemory

    pair = LLMContextAggregatorPair(LLMContext(), realtime_service_mode=True)
    capture = SconePipecatMemory(memory, "voice-test", "public-only")
    capture.attach(pair)
    await run_frames(pair, [
        LLMFullResponseStartFrame(), LLMThoughtStartFrame(),
        LLMThoughtTextFrame("synthetic private-channel test fixture"), LLMThoughtEndFrame(),
        LLMFullResponseEndFrame(),
        LLMFullResponseStartFrame(), LLMTextFrame("Public answer"), LLMFullResponseEndFrame(),
    ])
    capture.raise_if_failed()
    episodes = await memory.episodes("voice-test", {"session_id": "public-only"})
    assert [e.content for e in episodes] == ["Public answer"]
    assert capture.empty_count == 1


async def test_session_end_flush_is_aggregated_not_proof_of_response_completion(memory):
    from scone_memory.integrations.pipecat import SconePipecatMemory

    pair = LLMContextAggregatorPair(LLMContext(), realtime_service_mode=True)
    capture = SconePipecatMemory(memory, "voice-test", "ended")
    capture.attach(pair)
    await run_frames(pair, [LLMFullResponseStartFrame(), LLMTextFrame("Still speaking")])
    capture.raise_if_failed()
    episodes = await memory.episodes("voice-test", {"session_id": "ended"})
    assert [(e.content, e.metadata["capture_status"]) for e in episodes] == [("Still speaking", "aggregated")]


async def test_repeated_text_and_new_capture_instances_keep_distinct_records(memory):
    from scone_memory.integrations.pipecat import SconePipecatMemory

    for _ in range(2):
        capture = SconePipecatMemory(memory, "voice-test", "same-session")
        for _ in range(2):
            await capture.on_user_message(None, UserTurnMessageAddedMessage("yes", "2026-09-06T10:00:00Z", "speaker"))
        capture.raise_if_failed()
    episodes = await memory.episodes("voice-test", {"session_id": "same-session"})
    assert len(episodes) == 4
    assert all(e.content == "yes" and e.metadata["user_id"] == "speaker" for e in episodes)
    assert len({e.metadata["capture_id"] for e in episodes}) == 2
    assert all(datetime.fromisoformat(e.created_at) == datetime(2026, 9, 6, 10, tzinfo=timezone.utc) for e in episodes)


async def test_capture_errors_remain_observable_after_pipecat_swallows_handler_errors(memory):
    from scone_memory.integrations.pipecat import CaptureError, SconePipecatMemory

    # A real backend validation failure, not a mocked successful write.
    pair = LLMContextAggregatorPair(LLMContext(), realtime_service_mode=True)
    capture = SconePipecatMemory(memory, "voice-test", "bad-source")
    capture.attach(pair)
    await capture.on_user_message(None, UserTurnMessageAddedMessage("bad timestamp", "not-a-date"))
    await run_frames(pair, [LLMFullResponseStartFrame(), LLMTextFrame("later"), LLMFullResponseEndFrame()])
    with pytest.raises(CaptureError):
        capture.raise_if_failed()
    assert capture.stored_count == 0
    assert not await memory.episodes("voice-test", {"session_id": "bad-source"})


async def test_duplicate_attach_is_refused_and_detach_stops_future_capture(memory):
    from scone_memory.integrations.pipecat import SconePipecatMemory

    pair = LLMContextAggregatorPair(LLMContext(), realtime_service_mode=True)
    capture = SconePipecatMemory(memory, "voice-test", "detached")
    capture.attach(pair)
    with pytest.raises(ValueError):
        capture.attach(pair)
    capture.detach()
    await run_frames(pair, [LLMFullResponseStartFrame(), LLMTextFrame("not recorded"), LLMFullResponseEndFrame()])
    assert not await memory.episodes("voice-test", {"session_id": "detached"})


class WaitingEmbedder(HashEmbedder):
    """Control a slow dependency; storage, identity and recall remain real."""

    def __init__(self):
        super().__init__()
        self.started = asyncio.Event()
        self.release = asyncio.Event()

    async def embed(self, texts):
        self.started.set()
        await self.release.wait()
        return await super().embed(texts)


async def test_slow_write_times_out_and_latches_incomplete_capture(memory):
    from scone_memory.integrations.pipecat import CaptureError, SconePipecatMemory

    memory.embedder = WaitingEmbedder()
    capture = SconePipecatMemory(memory, "voice-test", "timeout", write_timeout=0.02)
    await asyncio.wait_for(capture.on_user_message(None, UserTurnMessageAddedMessage("pending", "")), 1)
    with pytest.raises(CaptureError):
        capture.raise_if_failed()
    assert isinstance(capture.error, TimeoutError)
    assert not await memory.episodes("voice-test", {"session_id": "timeout"})


async def test_backlog_is_bounded_and_later_events_are_counted_as_unrecorded(memory):
    from scone_memory.integrations.pipecat import CaptureError, SconePipecatMemory

    embedder = WaitingEmbedder()
    memory.embedder = embedder
    capture = SconePipecatMemory(memory, "voice-test", "backlog", max_pending=1)
    active = asyncio.create_task(capture.on_user_message(None, UserTurnMessageAddedMessage("first", "")))
    await asyncio.wait_for(embedder.started.wait(), 1)
    await capture.on_user_message(None, UserTurnMessageAddedMessage("overflow", ""))
    await capture.on_user_message(None, UserTurnMessageAddedMessage("after failure", ""))
    embedder.release.set()
    await active
    with pytest.raises(CaptureError):
        capture.raise_if_failed()
    episodes = await memory.episodes("voice-test", {"session_id": "backlog"})
    assert [e.content for e in episodes] == ["first"]
    assert capture.stored_count == 1
    assert capture.unrecorded_count == 2


async def test_cancellation_while_waiting_for_write_lock_is_not_silent(memory):
    from scone_memory.integrations.pipecat import CaptureError, SconePipecatMemory

    embedder = WaitingEmbedder()
    memory.embedder = embedder
    capture = SconePipecatMemory(memory, "voice-test", "cancelled")
    active = asyncio.create_task(capture.on_user_message(None, UserTurnMessageAddedMessage("first", "")))
    await asyncio.wait_for(embedder.started.wait(), 1)
    queued = asyncio.create_task(capture.on_user_message(None, UserTurnMessageAddedMessage("second", "")))
    await asyncio.sleep(0)  # allow the handler to begin waiting on the held lock
    queued.cancel()
    with pytest.raises(asyncio.CancelledError):
        await queued
    embedder.release.set()
    await active
    with pytest.raises(CaptureError):
        capture.raise_if_failed()
    assert capture.unrecorded_count == 1


@pytest.mark.parametrize("options", [
    {"write_timeout": 0}, {"write_timeout": float("inf")},
    {"write_timeout": float("nan")}, {"max_pending": 0}, {"max_pending": True},
])
async def test_invalid_limits_are_rejected_before_subscribing(memory, options):
    from scone_memory.integrations.pipecat import SconePipecatMemory

    with pytest.raises(ValueError):
        SconePipecatMemory(memory, "voice-test", "limits", **options)


async def test_sqlite_transcript_and_provenance_survive_reopen(tmp_path):
    from scone_memory.backends import SqliteDocumentStore, SqliteVectorIndex
    from scone_memory.integrations.pipecat import SconePipecatMemory

    path = tmp_path / "pipecat.db"
    documents, vectors = SqliteDocumentStore(path), SqliteVectorIndex(path)
    memory = await MemoryEngine(documents, vectors, HashEmbedder()).open()
    capture = SconePipecatMemory(memory, "voice-test", "durable-session")
    pair = LLMContextAggregatorPair(LLMContext(), realtime_service_mode=True)
    capture.attach(pair)
    try:
        await run_frames(pair, [
            TranscriptionFrame("Juniper telescope", "speaker-1", "2026-09-06T10:00:00Z"),
            LLMFullResponseStartFrame(), LLMTextFrame("Juniper telescope recorded"),
            LLMFullResponseEndFrame(),
        ])
        capture.raise_if_failed()
    finally:
        capture.detach()
        await documents.close()
        await vectors.close()

    documents, vectors = SqliteDocumentStore(path), SqliteVectorIndex(path)
    reopened = await MemoryEngine(documents, vectors, HashEmbedder()).open()
    try:
        recall = await reopened.recall("voice-test", "Juniper", where={"session_id": "durable-session"})
        assert {item.text for item in recall.items} == {"Juniper telescope", "Juniper telescope recorded"}
        assert {item.metadata["capture_id"] for item in recall.items} == {capture.capture_id}
        assert {item.metadata["capture_status"] for item in recall.items} == {"context_message", "aggregated"}
        assert not (await reopened.recall("other-space", "Juniper")).items
    finally:
        await documents.close()
        await vectors.close()


async def test_failed_write_latches_before_next_queued_write_acquires_lock(memory):
    from scone_memory.integrations.pipecat import CaptureError, SconePipecatMemory

    class FailingFirstEmbedder(WaitingEmbedder):
        async def embed(self, texts):
            if texts == ["first fails"]:
                self.started.set()
                await self.release.wait()
                raise RuntimeError("first embedding failed")
            return await HashEmbedder.embed(self, texts)

    embedder = FailingFirstEmbedder()
    memory.embedder = embedder
    capture = SconePipecatMemory(memory, "voice-test", "failure-order")
    first = asyncio.create_task(capture.on_user_message(None, UserTurnMessageAddedMessage("first fails", "")))
    await asyncio.wait_for(embedder.started.wait(), 1)
    second = asyncio.create_task(capture.on_user_message(None, UserTurnMessageAddedMessage("must not pass gap", "")))
    # Python 3.11 wait_for schedules a child task, unlike 3.12. Let both the
    # callback and that child reach the held lock before failing the first write.
    await asyncio.sleep(0)
    await asyncio.sleep(0)
    embedder.release.set()
    await asyncio.gather(first, second)
    with pytest.raises(CaptureError):
        capture.raise_if_failed()
    assert not await memory.episodes("voice-test", {"session_id": "failure-order"})
    assert capture.stored_count == 0
    assert capture.unrecorded_count == 2
