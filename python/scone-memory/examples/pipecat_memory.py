"""Scripted Pipecat → Scone capture/recall demo, not a live voice conversation.

From python/scone-memory, after installing .[pipecat]:
    python examples/pipecat_memory.py

Uses an isolated in-memory store, synthetic public text, no credentials/devices,
no provider calls and no application database. Bring your own transport, STT,
LLM and TTS to use the same adapter with a real conversation pipeline.
"""

import asyncio
import json

from loguru import logger
from pipecat.frames.frames import (
    EndFrame, LLMFullResponseEndFrame, LLMFullResponseStartFrame,
    LLMTextFrame, TranscriptionFrame,
)
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.worker import PipelineWorker
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.processors.aggregators.llm_response_universal import (
    LLMContextAggregatorPair, LLMUserAggregatorParams,
)
from pipecat.turns.user_turn_strategies import ExternalUserTurnStrategies
from pipecat.workers.runner import WorkerRunner

from scone_memory import HashEmbedder, InMemoryDocumentStore, InMemoryVectorIndex, MemoryEngine
from scone_memory.integrations.pipecat import SconePipecatMemory


async def main():
    logger.remove()  # this standalone demo prints one labeled result
    memory = await MemoryEngine(
        InMemoryDocumentStore(), InMemoryVectorIndex(), HashEmbedder()
    ).open()
    pair = LLMContextAggregatorPair(
        LLMContext(), realtime_service_mode=True,
        user_params=LLMUserAggregatorParams(user_turn_strategies=ExternalUserTurnStrategies()),
    )
    capture = SconePipecatMemory(memory, "pipecat-demo", "scripted-demo")
    capture.attach(pair)
    worker = PipelineWorker(
        Pipeline([pair.user(), pair.assistant()]),
        enable_rtvi=False, cancel_on_idle_timeout=False,
    )
    started = asyncio.Event()

    @worker.event_handler("on_pipeline_started")
    async def on_started(worker, frame):
        started.set()

    async def scripted_input():
        await asyncio.wait_for(started.wait(), 5)
        await worker.queue_frames([
            TranscriptionFrame("Our test telescope is named Juniper.", "demo-speaker", "2026-09-06T10:00:00Z"),
            LLMFullResponseStartFrame(),
            LLMTextFrame("Juniper is "), LLMTextFrame("the test telescope."),
            LLMFullResponseEndFrame(), EndFrame(),
        ])

    runner = WorkerRunner()
    await runner.add_workers(worker)
    try:
        await asyncio.wait_for(asyncio.gather(runner.run(), scripted_input()), 15)
        capture.raise_if_failed()
    finally:
        capture.detach()

    result = await memory.recall("pipecat-demo", "Juniper", where={"session_id": "scripted-demo"})
    print(json.dumps({
        "mode": "scripted fixture; not live voice or generated model output",
        "stored_transcripts": capture.stored_count,
        "recall": [
            {"episode_id": item.episode_id, "text": item.text, "metadata": item.metadata}
            for item in result.items
        ],
    }, indent=2))


if __name__ == "__main__":
    asyncio.run(main())
