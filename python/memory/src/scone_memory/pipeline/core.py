"""The pipeline itself: delivery, turns, buffering and lifecycle."""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass
from typing import Optional, Protocol, Sequence, runtime_checkable

#: Toward the stages after the sender, and toward the ones before it.
DOWN = "down"
UP = "up"

#: What became of one frame at one stage.
HANDLED = "handled"
DROPPED = "dropped"
FAILED = "failed"


@dataclass(frozen=True)
class Started:
    """The run has begun. Every stage sees it before any data."""


@dataclass(frozen=True)
class Stopped:
    """The run is over. Every stage sees it after the last data."""


@dataclass(frozen=True)
class Interrupted:
    """The turn named here was cut off; its work no longer matters."""

    turn: int


@dataclass(frozen=True)
class Failed:
    """A stage raised. The run continues; this says what happened."""

    stage: str
    error: str


#: Announcements about the run. They reach every stage and outlive a turn.
ANNOUNCEMENTS = (Started, Stopped, Interrupted, Failed)


@runtime_checkable
class Stage(Protocol):
    """Anything that reads a frame and sends what should continue.

    ``start`` and ``stop`` are optional. ``buffered = True`` asks for a
    queue and a task, for work that waits on something outside, and
    ``capacity`` says how much backlog that queue will hold.
    ``essential = True`` says the run cannot continue without it."""

    async def handle(self, frame: object, emit: "Emit") -> None: ...


@dataclass(frozen=True)
class Delivery:
    """One frame reaching one stage, and what became of it: the unit a
    trace, a latency figure or a dropped-work count is made of."""

    stage: str
    frame: object
    direction: str
    turn: int
    #: Correlates every delivery of one run, for a trace across services.
    session: str
    #: handled, dropped (its turn was cut off) or failed (the stage raised).
    outcome: str
    #: Time in the stage. Zero for a frame that never entered one.
    elapsed_ms: float


class Observer(Protocol):
    """Sees every delivery: for metrics, tracing or a log."""

    async def observed(self, delivery: Delivery) -> None: ...


class Emit:
    """A stage's way to send, bound to the turn it is working on, so
    anything it produces belongs to that turn and dies with it."""

    __slots__ = ("_pipeline", "_index", "turn")

    def __init__(self, pipeline: "Pipeline", index: int, turn: int) -> None:
        self._pipeline = pipeline
        self._index = index
        self.turn = turn

    async def __call__(self, frame: object) -> None:
        """Send to the next stage."""
        await self._pipeline._deliver(self._index + 1, frame, self.turn, DOWN)

    async def up(self, frame: object) -> None:
        """Send back to every stage before this one, nearest first: an
        answer from the far end, like a barge-in reaching the source."""
        for index in range(self._index - 1, -1, -1):
            await self._pipeline._deliver(index, frame, self.turn, UP)


class Pipeline:
    """An ordered list of stages that frames flow through."""

    def __init__(self, stages: Sequence[Stage], *, observer: Optional[Observer] = None,
                 name: str = "pipeline", session: str = "") -> None:
        if not stages:
            raise ValueError("a pipeline needs at least one stage")
        self.stages = list(stages)
        self.observer = observer
        self.name = name
        #: Correlation id carried by every observation of this run.
        self.session = session
        #: The turn frames belong to. Interrupting moves it on.
        self.turn = 1
        #: Frames dropped because their turn was cut off, for the record.
        self.dropped = 0
        #: Stages that raised, counted whether or not they were announced.
        self.failures = 0
        #: Set once the run is over, however it ended.
        self.ended = asyncio.Event()
        #: What ended it, when a stage the run needs raised.
        self.error: Optional[Exception] = None
        self._queues: dict[int, asyncio.Queue] = {}
        self._workers: list[asyncio.Task] = []
        self._inflight = 0
        self._idle = asyncio.Event()
        self._idle.set()
        self._running = False

    # -- lifecycle --------------------------------------------------------

    async def start(self) -> None:
        """Give the buffered stages their tasks, start every stage, and
        announce the run."""
        if self._running:
            return
        self._running = True
        for index, stage in enumerate(self.stages):
            if getattr(stage, "buffered", False):
                queue: asyncio.Queue = asyncio.Queue(self._capacity(stage))
                self._queues[index] = queue
                self._workers.append(asyncio.create_task(
                    self._work(index, stage, queue), name=f"{self.name}-stage-{index}"))
            begin = getattr(stage, "start", None)
            if callable(begin):
                await begin()
        await self._announce(Started())

    async def stop(self) -> None:
        """Announce the end, stop every stage, and let the tasks go."""
        if not self._running:
            return
        self._end(None)
        await self._announce(Stopped())
        for stage in reversed(self.stages):
            end = getattr(stage, "stop", None)
            if callable(end):
                await end()
        for queue in self._queues.values():
            await queue.put(None)
        for worker in self._workers:
            await worker
        self._workers.clear()
        self._queues.clear()
        self._running = False

    @staticmethod
    def _capacity(stage: Stage) -> int:
        """How much backlog a buffered stage will hold. The default, zero,
        holds anything; a positive number makes whoever is feeding the stage
        wait instead, so a fast source cannot outrun a slow one."""
        room = getattr(stage, "capacity", 0)
        if type(room) is not int or room < 0:
            raise ValueError("a stage's capacity must be a whole number of frames")
        return room

    # -- sending ----------------------------------------------------------

    async def push(self, frame: object) -> None:
        """Put a frame in at the head, as part of the current turn."""
        await self._deliver(0, frame, self.turn, DOWN)

    def interrupt(self) -> int:
        """Cut off the current turn and return the new one. Work already
        in flight finishes, but nothing it produces is delivered, and a
        backlog of the old turn is dropped rather than started."""
        self.turn += 1
        return self.turn

    async def drain(self) -> None:
        """Wait until no buffered stage has work left."""
        await self._idle.wait()

    async def wait(self) -> None:
        """Wait for the run to end, and raise whatever ended it. A run
        that finished on its own terms returns, so one place can wait for
        either ending."""
        await self.ended.wait()
        if self.error is not None:
            raise self.error

    def _end(self, error: Optional[Exception]) -> None:
        """The first ending is the one that counts; a fault while shutting
        down does not rewrite why the run stopped."""
        if self.ended.is_set():
            return
        self.error = error
        self.ended.set()

    # -- delivery ---------------------------------------------------------

    async def _announce(self, frame: object, skip: Optional[int] = None) -> None:
        for index in range(len(self.stages)):
            if index != skip:
                await self._deliver(index, frame, self.turn, DOWN)

    async def _deliver(self, index: int, frame: object, turn: int, direction: str) -> None:
        if not 0 <= index < len(self.stages):
            return
        # Once the run is over only announcements move: a stage still has
        # to be told the run ended, but no new work belongs to it.
        if (turn != self.turn or self.ended.is_set()) and not isinstance(frame, ANNOUNCEMENTS):
            await self._drop(index, frame, turn, direction)
            return
        stage = self.stages[index]
        queue = self._queues.get(index)
        # Upstream frames run inline even into a buffered stage: an
        # interruption that waits behind a backlog is not an interruption,
        # and with a bounded backlog it could not get through at all.
        if queue is not None and direction == DOWN:
            self._took()
            await queue.put((frame, turn))
            return
        await self._run(index, stage, frame, turn, direction)

    async def _drop(self, index: int, frame: object, turn: int, direction: str) -> None:
        """A frame of a turn that was cut off, counted and observed where
        it was found rather than carried further."""
        self.dropped += 1
        await self._saw(type(self.stages[index]).__name__, frame, direction, turn, DROPPED, 0.0)

    async def _saw(self, stage: str, frame: object, direction: str, turn: int,
                   outcome: str, elapsed_ms: float) -> None:
        if self.observer is not None:
            await self.observer.observed(Delivery(
                stage=stage, frame=frame, direction=direction, turn=turn,
                session=self.session, outcome=outcome, elapsed_ms=elapsed_ms))

    async def _run(self, index: int, stage: Stage, frame: object, turn: int, direction: str) -> None:
        name = type(stage).__name__
        began = time.perf_counter()
        try:
            await stage.handle(frame, Emit(self, index, turn))
        except asyncio.CancelledError:
            raise
        except Exception as error:  # noqa: BLE001 - a stage's fault is reported, not raised at the sender
            self.failures += 1
            await self._saw(name, frame, direction, turn, FAILED, (time.perf_counter() - began) * 1000)
            # Telling a stage about a failure must not be able to start
            # another round of it, and the stage that raised already knows.
            if not isinstance(frame, Failed):
                await self._announce(
                    Failed(stage=name, error=f"{type(error).__name__}: {error}"), skip=index)
            if getattr(stage, "essential", False):
                self._end(error)
        else:
            await self._saw(name, frame, direction, turn, HANDLED, (time.perf_counter() - began) * 1000)

    async def _work(self, index: int, stage: Stage, queue: asyncio.Queue) -> None:
        while True:
            item = await queue.get()
            if item is None:
                return
            frame, turn = item
            try:
                if turn == self.turn or isinstance(frame, ANNOUNCEMENTS):
                    await self._run(index, stage, frame, turn, DOWN)
                else:
                    await self._drop(index, frame, turn, DOWN)
            finally:
                self._done()

    # -- idleness ---------------------------------------------------------

    def _took(self) -> None:
        self._inflight += 1
        self._idle.clear()

    def _done(self) -> None:
        self._inflight -= 1
        if self._inflight <= 0:
            self._inflight = 0
            self._idle.set()
