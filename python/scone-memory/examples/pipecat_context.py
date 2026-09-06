"""Scone source → Pipecat request → captured scripted reply, without a provider.

Run after installing .[pipecat] and provisioning NLTK as described in README:
    python examples/pipecat_context.py

All records are synthetic and stored only in this process. No microphone,
network model, application database or live session is used. The responder
inspects its request and emits a fixed fixture; it does not perform inference.
"""

import asyncio
import json

from loguru import logger
from pipecat.frames.frames import (
    EndFrame, LLMContextFrame, LLMFullResponseEndFrame,
    LLMFullResponseStartFrame, LLMTextFrame,
)
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.worker import PipelineWorker
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.processors.aggregators.llm_response_universal import LLMContextAggregatorPair
from pipecat.processors.frame_processor import FrameProcessor
from pipecat.workers.runner import WorkerRunner

from scone_memory import HashEmbedder, InMemoryDocumentStore, InMemoryVectorIndex, MemoryEngine
from scone_memory.events import InMemoryEventLog
from scone_memory.integrations.pipecat import SconePipecatMemory
from scone_memory.integrations.pipecat_context import SconeMemoryContextProcessor


class ScriptedResponder(FrameProcessor):
    """Explicit stand-in for a compatible text LLM; never claims model use."""

    def __init__(self):
        super().__init__()
        self.receipt = None
        self.source_supplied = False

    async def process_frame(self, frame, direction):
        await super().process_frame(frame, direction)
        if not isinstance(frame, LLMContextFrame):
            await self.push_frame(frame, direction)
            return
        self.receipt = frame.metadata.get("scone_memory")
        messages = frame.context.get_messages()
        self.source_supplied = len(messages) >= 2 and "Polaris" in messages[-2]["content"]
        await self.push_frame(LLMFullResponseStartFrame())
        await self.push_frame(LLMTextFrame("Scripted reply: the request received a Juniper source."))
        await self.push_frame(LLMFullResponseEndFrame())


async def main():
    logger.remove()
    memory = await MemoryEngine(
        InMemoryDocumentStore(), InMemoryVectorIndex(), HashEmbedder(),
        events=InMemoryEventLog(),
    ).open()
    await memory.remember(
        "context-demo", "Juniper telescope uses Polaris for calibration.",
        source="fixture-manual.txt", metadata={"collection": "manuals"},
    )
    context = LLMContext([{"role": "user", "content": "Juniper telescope calibration"}])
    pair = LLMContextAggregatorPair(context, realtime_service_mode=True)
    recall = SconeMemoryContextProcessor(
        memory, "context-demo", "scripted-session", where={"collection": "manuals"},
    )
    capture = SconePipecatMemory(memory, "context-demo", "scripted-session")
    capture.attach(pair)
    responder = ScriptedResponder()
    worker = PipelineWorker(
        Pipeline([pair.user(), recall, responder, pair.assistant()]),
        enable_rtvi=False, cancel_on_idle_timeout=False,
    )
    started = asyncio.Event()

    @worker.event_handler("on_pipeline_started")
    async def on_started(worker, frame):
        started.set()

    async def submit_request():
        await asyncio.wait_for(started.wait(), 5)
        await worker.queue_frames([LLMContextFrame(context), EndFrame()])

    runner = WorkerRunner()
    await runner.add_workers(worker)
    try:
        await asyncio.wait_for(asyncio.gather(runner.run(), submit_request()), 15)
        capture.raise_if_failed()
    finally:
        capture.detach()
    result = await memory.recall(
        "context-demo", "Juniper", where={"session_id": "scripted-session"},
    )
    print(json.dumps({
        "mode": "scripted responder; no model inference or live media",
        "request_receipt": responder.receipt,
        "source_supplied_to_responder": responder.source_supplied,
        "source_block_in_shared_history": any("Polaris" in m.get("content", "") for m in context.get_messages()),
        "stored_transcripts": capture.stored_count,
        "captured_recall": [{"episode_id": item.episode_id, "text": item.text} for item in result.items],
    }, indent=2))


if __name__ == "__main__":
    asyncio.run(main())
