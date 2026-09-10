"""A conversation that remembers, as a stage.

Recall belongs in the pipeline rather than bolted onto a provider: as a
stage it sees the same frames everything else does, so it can be placed,
removed, observed and interrupted like any other part. Put it before the
model and the model is handed what the person said together with what is
already known about it; leave it out and the same pipeline is a
conversation with no memory, which is a useful thing to be able to test.

What is stored is what actually happened. The words a model was still
saying when it was cut off belong to nobody, so a turn that never
completed is never learned.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping, Optional, Sequence

from ..integrations.chat import BUDGET, ContextReceipt, recall_context, remember_exchange
from ..memory.engine import MemoryEngine
from ..realtime.audio import Transcript
from ..realtime.events import ReplyCompleted, TextDelta


@dataclass(frozen=True)
class Prompt:
    """What to ask the model, and what was put in front of it."""

    messages: list[dict]
    receipt: ContextReceipt


@dataclass(frozen=True)
class Remembered:
    """A finished exchange that is now in memory."""

    episode_ids: tuple[int, ...]
    session: str


class MemoryStage:
    """Recalls when a question finishes, keeps the exchange when the
    answer does. Buffered: recall and writes wait on a store, and a
    conversation should not stop while they do."""

    #: Waits on a store, so it gets its own task and its own backlog.
    buffered = True

    def __init__(self, engine: MemoryEngine, space: str, *, session: str,
                 limit: int = 5, budget: int = BUDGET, tags: Sequence[str] = (),
                 metadata: Optional[Mapping[str, str]] = None, history: int = 20) -> None:
        self.engine = engine
        self.space = space
        self.session = session
        self.limit = limit
        self.budget = budget
        self.tags = tuple(tags)
        self.metadata = dict(metadata or {})
        #: Turns kept for context. Older ones fall off; memory holds them.
        self.history = history
        self._messages: list[dict] = []
        self._question: Optional[str] = None
        self._answer: list[str] = []

    async def handle(self, frame: object, emit) -> None:
        if isinstance(frame, Transcript):
            await self._asked(frame, emit)
        elif isinstance(frame, TextDelta):
            self._answer.append(frame.text)
        elif isinstance(frame, ReplyCompleted):
            await self._answered(emit)

    async def _asked(self, frame: Transcript, emit) -> None:
        """A finished question is the only thing worth searching for: a
        partial transcript is half a thought, and recalling on it wastes
        the search and misleads the model."""
        if not frame.final or frame.speaker != "user" or not frame.text.strip():
            return
        self._question = frame.text.strip()
        # A new question ends whatever the last answer was: anything left
        # from a turn that was cut off is not part of this one.
        self._answer = []
        self._messages = (self._messages + [{"role": "user", "content": self._question}])[-self.history:]
        prepared, receipt = await recall_context(
            self.engine, self.space, self._messages,
            limit=self.limit, budget=self.budget, tags=self.tags,
        )
        await emit(Prompt(messages=prepared, receipt=receipt))

    async def _answered(self, emit) -> None:
        answer = "".join(self._answer).strip()
        if self._question is None or not answer:
            return
        self._messages = (self._messages + [{"role": "assistant", "content": answer}])[-self.history:]
        written = await remember_exchange(
            self.engine, self.space,
            [{"role": "user", "content": self._question}, {"role": "assistant", "content": answer}],
            session=self.session, metadata=self.metadata,
        )
        self._question, self._answer = None, []
        await emit(Remembered(episode_ids=tuple(w.episode_id for w in written), session=self.session))
