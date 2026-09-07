"""A conversation that remembers, as a stage.

Recall belongs in the pipeline, not bolted onto a provider: the stage
sees the same frames everything else does, so it can be placed, removed
or observed like any other part, and a turn that was cut off never
becomes a stored answer.
"""

from __future__ import annotations

import asyncio

import pytest

from scone_memory import HashEmbedder, InMemoryDocumentStore, InMemoryVectorIndex, MemoryEngine
from scone_memory.pipeline import Pipeline
from scone_memory.pipeline.memory import MemoryStage, Prompt, Remembered
from scone_memory.realtime.audio import Transcript
from scone_memory.realtime.events import ReplyCompleted, TextDelta


class Collect:
    def __init__(self):
        self.seen: list[object] = []

    async def handle(self, frame, emit):
        self.seen.append(frame)


@pytest.fixture
async def engine():
    memory = await MemoryEngine(InMemoryDocumentStore(), InMemoryVectorIndex(), HashEmbedder()).open()
    await memory.remember("alpha", "Mark takes his coffee black", tags=["habits"])
    return memory


async def run(pipeline, frames, settle=0.0):
    for frame in frames:
        await pipeline.push(frame)
    if settle:
        await asyncio.sleep(settle)
    await pipeline.drain()


async def test_a_finished_question_arrives_at_the_model_with_memory_in_front_of_it(engine):
    tail = Collect()
    stage = MemoryStage(engine, "alpha", session="call-1")
    pipeline = Pipeline([stage, tail])
    await pipeline.start()
    await run(pipeline, [Transcript(text="how do I take my coffee?", final=True)])

    [prompt] = [f for f in tail.seen if isinstance(f, Prompt)]
    assert prompt.messages[-1] == {"role": "user", "content": "how do I take my coffee?"}
    assert any("black" in m["content"] for m in prompt.messages if m["role"] == "system")
    assert prompt.receipt.injected is True and prompt.receipt.episode_ids
    await pipeline.stop()


async def test_a_partial_transcript_is_not_a_question(engine):
    tail = Collect()
    pipeline = Pipeline([MemoryStage(engine, "alpha", session="call-1"), tail])
    await pipeline.start()
    await run(pipeline, [Transcript(text="how do I take my", final=False)])
    assert not [f for f in tail.seen if isinstance(f, Prompt)], "recall waits for the whole question"
    await pipeline.stop()


async def test_both_sides_of_a_finished_turn_are_kept_and_found_next_time(engine):
    tail = Collect()
    pipeline = Pipeline([MemoryStage(engine, "alpha", session="call-7"), tail])
    await pipeline.start()
    await run(pipeline, [
        Transcript(text="remind me what I drive", final=True),
        TextDelta(text="You drive "), TextDelta(text="a green van."), ReplyCompleted(),
    ])

    [kept] = [f for f in tail.seen if isinstance(f, Remembered)]
    assert len(kept.episode_ids) == 2, "the question and the answer"
    found = await engine.recall("alpha", "what do I drive")
    assert any("green van" in item.text for item in found.items)
    episode = await engine.episode("alpha", kept.episode_ids[0])
    assert episode.source == "call-7" and episode.kind == "conversation"
    await pipeline.stop()


async def test_an_answer_that_was_cut_off_is_never_stored(engine):
    """The turn was interrupted, so the words the model was still saying
    belong to nobody: they are not what happened, and memory must not
    learn them."""
    tail = Collect()
    pipeline = Pipeline([MemoryStage(engine, "alpha", session="call-9"), tail])
    await pipeline.start()
    await pipeline.push(Transcript(text="tell me a long story", final=True))
    await pipeline.push(TextDelta(text="Once upon a time"))
    pipeline.interrupt()
    await pipeline.push(TextDelta(text=" there was a dragon"))
    await pipeline.push(ReplyCompleted())
    await pipeline.drain()

    assert not [f for f in tail.seen if isinstance(f, Remembered)], "a cut-off turn is not a memory"
    found = await engine.recall("alpha", "dragon")
    assert all("dragon" not in item.text for item in found.items)
    await pipeline.stop()


async def test_the_stage_does_not_hold_up_the_pipeline(engine):
    stage = MemoryStage(engine, "alpha", session="call-2")
    assert stage.buffered is True, "recall waits on a store, so it gets its own task"
    tail = Collect()
    pipeline = Pipeline([stage, tail])
    await pipeline.start()
    await pipeline.push(Transcript(text="anything", final=True))
    assert not tail.seen or not isinstance(tail.seen[-1], Prompt), "pushing returned before the recall finished"
    await pipeline.drain()
    await pipeline.stop()


async def test_words_from_a_cut_off_turn_do_not_leak_into_the_next_answer(engine):
    tail = Collect()
    pipeline = Pipeline([MemoryStage(engine, "alpha", session="call-10"), tail])
    await pipeline.start()
    await pipeline.push(Transcript(text="tell me a long story", final=True))
    await pipeline.push(TextDelta(text="Once upon a time there was a dragon"))
    await pipeline.drain()  # the model really did say it, and the stage heard it
    pipeline.interrupt()
    await pipeline.push(Transcript(text="what colour is the van?", final=True))
    await pipeline.push(TextDelta(text="It is green."))
    await pipeline.push(ReplyCompleted())
    await pipeline.drain()

    [kept] = [f for f in tail.seen if isinstance(f, Remembered)]
    answer = await engine.episode("alpha", kept.episode_ids[1])
    assert answer.content == "It is green.", "the abandoned sentence is not part of this answer"
    await pipeline.stop()


async def test_an_answer_with_no_words_is_not_a_memory(engine):
    tail = Collect()
    pipeline = Pipeline([MemoryStage(engine, "alpha", session="call-11"), tail])
    await pipeline.start()
    await run(pipeline, [Transcript(text="are you there?", final=True), ReplyCompleted()])
    assert not [f for f in tail.seen if isinstance(f, Remembered)], "silence is not an answer to keep"
    await pipeline.stop()
