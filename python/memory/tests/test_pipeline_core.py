"""Composable stages: frames flow through in order, a stage may answer
upstream, an interruption drops the work of the turn that was cut off,
a slow stage runs without holding up the rest, a raising stage does not
take the pipeline down, and one observer sees every delivery."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass

import pytest

from scone_memory.pipeline import DROPPED, HANDLED, Delivery, Failed, Pipeline, Stage, Started, Stopped


@dataclass(frozen=True)
class Word:
    text: str


@dataclass(frozen=True)
class Cancel:
    pass


class Collect:
    """Terminal stage: keeps what reached it, in arrival order."""

    def __init__(self):
        self.seen: list[object] = []

    async def handle(self, frame, emit):
        self.seen.append(frame)


class Relay(Collect):
    """Watches and passes everything on, the shape a logger or a tap takes."""

    async def handle(self, frame, emit):
        self.seen.append(frame)
        await emit(frame)


class Shout(Collect):
    async def handle(self, frame, emit):
        self.seen.append(frame)
        if isinstance(frame, Word):
            await emit(Word(frame.text.upper()))


class Split(Collect):
    """One frame in, several out: the fan-out an aggregator needs."""

    async def handle(self, frame, emit):
        self.seen.append(frame)
        if isinstance(frame, Word):
            for part in frame.text.split():
                await emit(Word(part))


async def test_frames_flow_through_the_stages_in_order():
    tail = Collect()
    pipeline = Pipeline([Shout(), tail])
    await pipeline.start()
    await pipeline.push(Word("hello"))
    await pipeline.drain()
    assert [f.text for f in tail.seen if isinstance(f, Word)] == ["HELLO"], "the tail sees what the head made of it"
    await pipeline.stop()


async def test_a_stage_may_answer_several_frames_and_reach_upstream():
    head = Relay()

    class Barge(Collect):
        async def handle(self, frame, emit):
            self.seen.append(frame)
            if isinstance(frame, Word) and frame.text == "stop":
                await emit.up(Cancel())

    barge = Barge()
    pipeline = Pipeline([head, Split(), barge])
    await pipeline.start()
    await pipeline.push(Word("one two stop"))
    await pipeline.drain()
    assert [f.text for f in barge.seen if isinstance(f, Word)] == ["one", "two", "stop"], "each part arrives"
    assert any(isinstance(f, Cancel) for f in head.seen), "upstream frames reach the stages before, not after"
    assert not any(isinstance(f, Cancel) for f in barge.seen), "and never the sender itself"
    await pipeline.stop()


async def test_an_interruption_drops_the_cut_off_turn_and_the_next_one_runs():
    tail = Collect()

    class Slow:
        buffered = True

        def __init__(self):
            self.started: list[str] = []

        async def handle(self, frame, emit):
            if not isinstance(frame, Word):
                return
            self.started.append(frame.text)
            await asyncio.sleep(0.05)
            await emit(Word(frame.text + "!"))

    slow = Slow()
    pipeline = Pipeline([slow, tail])
    await pipeline.start()
    await pipeline.push(Word("first"))
    await pipeline.push(Word("second"))
    await asyncio.sleep(0.01)
    pipeline.interrupt()
    await pipeline.push(Word("third"))
    await pipeline.drain()
    assert slow.started == ["first", "third"], "the queued frame of the cut-off turn is never started"
    assert [f.text for f in tail.seen if isinstance(f, Word)] == ["third!"], \
        "and the work in flight when it was cut does not reach the tail"
    await pipeline.stop()


async def test_a_buffered_stage_does_not_hold_up_the_stages_around_it():
    order: list[str] = []

    class Slow:
        buffered = True

        async def handle(self, frame, emit):
            if not isinstance(frame, Word):
                return
            await asyncio.sleep(0.03)
            order.append("slow")

    class Quick:
        async def handle(self, frame, emit):
            if isinstance(frame, Word):
                order.append("quick")

    pipeline = Pipeline([Slow(), Quick()])
    await pipeline.start()
    await pipeline.push(Word("a"))
    order.append("pushed")
    await pipeline.drain()
    assert order[0] == "pushed", "pushing into a buffered stage returns at once"
    assert order.count("slow") == 1
    await pipeline.stop()


async def test_a_raising_stage_reports_and_the_pipeline_keeps_running():
    tail = Collect()

    class Broken:
        async def handle(self, frame, emit):
            if isinstance(frame, Word) and frame.text == "bad":
                raise RuntimeError("no")
            await emit(frame)

    pipeline = Pipeline([Broken(), tail], name="voice")
    await pipeline.start()
    await pipeline.push(Word("bad"))
    await pipeline.push(Word("good"))
    await pipeline.drain()
    failures = [f for f in tail.seen if isinstance(f, Failed)]
    assert len(failures) == 1 and failures[0].stage == "Broken" and "no" in failures[0].error
    assert [f.text for f in tail.seen if isinstance(f, Word)] == ["good"], "the next frame still flows"
    await pipeline.stop()


async def test_the_lifecycle_and_the_observer_see_every_stage_and_delivery():
    seen: list[tuple[str, str, str]] = []

    class Watcher:
        async def observed(self, delivery):
            assert isinstance(delivery, Delivery)
            assert delivery.session == "run-7" and delivery.elapsed_ms >= 0.0
            assert delivery.outcome in (HANDLED, DROPPED)
            seen.append((delivery.stage, type(delivery.frame).__name__, delivery.direction))

    class Lifecycle(Collect):
        def __init__(self):
            super().__init__()
            self.calls: list[str] = []

        async def start(self):
            self.calls.append("start")

        async def stop(self):
            self.calls.append("stop")

    first, second = Lifecycle(), Lifecycle()
    pipeline = Pipeline([first, second], observer=Watcher(), session="run-7")
    await pipeline.start()
    await pipeline.push(Word("x"))
    await pipeline.drain()
    await pipeline.stop()
    assert first.calls == ["start", "stop"] and second.calls == ["start", "stop"], "each stage once"
    assert [s for s, f, d in seen if f == "Word"] == ["Lifecycle"], \
        "data goes where a stage sends it: this one sends nothing on, so the word stops"
    assert {d for _, _, d in seen} == {"down"}
    assert ("Lifecycle", "Started", "down") in seen and ("Lifecycle", "Stopped", "down") in seen, \
        "the lifecycle is frames too, so a stage can act on them"


def test_a_stage_is_anything_with_handle():
    assert isinstance(Collect(), Stage) and not isinstance(object(), Stage)


async def test_a_stage_that_raises_on_everything_reports_once_and_does_not_loop():
    """A failure is announced to the other stages, never back to the one
    that raised, and a stage that raises while being told about a failure
    is counted rather than announced again: a pipeline cannot spin on its
    own bad news."""
    tail = Collect()

    class Hopeless:
        async def handle(self, frame, emit):
            raise RuntimeError("always")

    pipeline = Pipeline([Hopeless(), tail])
    await pipeline.start()
    await pipeline.push(Word("x"))
    await pipeline.drain()
    failures = [f for f in tail.seen if isinstance(f, Failed)]
    assert failures and all(f.stage == "Hopeless" for f in failures)
    assert len(failures) == 2, "one for the start announcement, one for the word; not a storm"
    assert pipeline.failures == 2, "the stage's own bad news never comes back to it"
    await pipeline.stop()


async def test_the_observer_can_tell_finished_work_from_work_that_was_cut_off():
    """R19 wants stage timing and a correlation id; R07 wants cancelled
    work to be distinguishable from completed work. One delivery record
    carries both."""
    seen: list[Delivery] = []

    class Watcher:
        async def observed(self, delivery):
            seen.append(delivery)

    class Slow:
        buffered = True

        async def handle(self, frame, emit):
            if isinstance(frame, Word):
                await asyncio.sleep(0.02)

    pipeline = Pipeline([Slow()], observer=Watcher(), session="call-1")
    await pipeline.start()
    await pipeline.push(Word("kept"))
    await pipeline.push(Word("cut"))
    await asyncio.sleep(0.005)
    pipeline.interrupt()
    await pipeline.drain()
    await pipeline.stop()

    words = [d for d in seen if isinstance(d.frame, Word)]
    assert [(d.frame.text, d.outcome) for d in words] == [("kept", HANDLED), ("cut", DROPPED)]
    assert words[0].elapsed_ms >= 15, "the finished stage reports the time it took"
    assert words[1].elapsed_ms == 0.0, "work never entered is not timed"
    assert {d.session for d in seen} == {"call-1"} and pipeline.dropped == 1


async def test_two_stages_that_both_raise_on_bad_news_do_not_ping_pong():
    """Telling a stage about a failure can make it raise too. Counting
    that rather than announcing it is what keeps two broken stages from
    trading bad news forever."""

    class Hopeless:
        async def handle(self, frame, emit):
            raise RuntimeError("always")

    pipeline = Pipeline([Hopeless(), Hopeless()])
    await pipeline.start()
    assert pipeline.failures == 4, "each stage raises on the start announcement and again on the other's failure"
    await pipeline.push(Word("x"))
    await pipeline.drain()
    assert pipeline.failures == 6, "the word costs one raise each, and there it ends"
    await pipeline.stop()


async def test_a_full_buffered_stage_makes_the_sender_wait_rather_than_grow():
    """A stage that says how much backlog it will hold pushes back on
    whoever feeds it. Without that, a fast source outruns a slow stage
    and the queue is the only thing that grows."""
    release = asyncio.Event()
    taken: list[str] = []

    class Slow:
        buffered = True
        capacity = 1

        async def handle(self, frame, emit):
            if not isinstance(frame, Word):
                return
            taken.append(frame.text)
            await release.wait()

    pipeline = Pipeline([Slow()])
    await pipeline.start()
    await pipeline.push(Word("a"))
    for _ in range(100):
        if taken:
            break
        await asyncio.sleep(0)
    assert taken == ["a"], "the stage is busy with the first frame"
    await pipeline.push(Word("b"))
    third = asyncio.create_task(pipeline.push(Word("c")))
    for _ in range(100):
        await asyncio.sleep(0)
    assert not third.done(), "one frame is waiting, so the next sender waits too"
    release.set()
    await asyncio.wait_for(third, 1)
    await pipeline.drain()
    assert taken == ["a", "b", "c"], "everything arrives, in order, once there is room"
    await pipeline.stop()


async def test_a_backlog_size_that_is_not_a_count_of_frames_is_refused():
    """A capacity that cannot mean a number of frames is a mistake worth
    saying out loud, not a queue silently left unbounded."""

    class Odd:
        buffered = True
        capacity = -1

    with pytest.raises(ValueError, match="whole number of frames"):
        await Pipeline([Odd()]).start()


async def test_a_stage_the_run_cannot_do_without_ends_it_when_it_raises():
    """Most faults are worth reporting and carrying on from. A stage the
    run is built around is not: when the ear in a call stops working the
    call is over, and whoever started it has to hear why."""
    tail = Collect()

    class Ear:
        essential = True

        async def handle(self, frame, emit):
            if not isinstance(frame, Word):
                return
            if frame.text == "bad":
                raise RuntimeError("the recognizer went away")
            await emit(frame)

    pipeline = Pipeline([Ear(), tail])
    await pipeline.start()
    await pipeline.push(Word("good"))
    await pipeline.push(Word("bad"))
    with pytest.raises(RuntimeError, match="recognizer went away"):
        await asyncio.wait_for(pipeline.wait(), 1)
    assert any(isinstance(f, Failed) for f in tail.seen), "the stages after it still hear what happened"
    await pipeline.push(Word("later"))
    assert [f.text for f in tail.seen if isinstance(f, Word)] == ["good"], "nothing more flows once the run is over"
    await pipeline.stop()


async def test_a_run_that_ends_on_its_own_terms_has_nothing_to_raise():
    """Waiting on a run that finished cleanly returns rather than
    inventing a failure, so one place can wait for either ending."""
    pipeline = Pipeline([Collect()])
    await pipeline.start()
    await pipeline.stop()
    await asyncio.wait_for(pipeline.wait(), 1)
    assert pipeline.error is None
