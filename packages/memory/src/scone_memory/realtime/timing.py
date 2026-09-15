"""When a conversation's turn was heard, prepared, first answered, first spoken and finished.

A conversation is judged by how long a person waits, and the waits are
not one number: the time to prepare memory for the turn, the time to
the first token of the answer, the time to the first audio a listener
hears, and the whole turn. Each is measured from the moment the question
was heard (a transcript settled, or a text turn arrived), with the
process clock (``perf_counter``), and each is recorded only when its
moment came: a text turn has no first audio, a text turn nobody streams
has no first token, and a mark that did not happen is absent rather than
zero. A turn that failed records no event at all: the timing is noted
once the reply stands in memory. What is recorded is a
`conversation_turn` event, the same evidence log recall's timings go to,
and the metrics report reads it back.
"""

from __future__ import annotations

import time

#: The moments a turn can reach, in the order they come.
MARKS = ("context", "first_token", "first_audio", "done")
#: The clock, held here so a test can tick it without touching the process's.
_now = time.perf_counter


class TurnTiming:
    """Marks for one turn, measured from when it was heard."""

    __slots__ = ("heard", "marks")

    def __init__(self, heard: float | None = None) -> None:
        self.heard = _now() if heard is None else heard
        self.marks: dict[str, float] = {}

    def mark(self, name: str) -> None:
        """Record that a moment came, once: the first token is the first."""
        if name not in MARKS:
            raise ValueError(f"unknown mark {name!r}")
        if name not in self.marks:
            self.marks[name] = _now()

    def record(self) -> dict[str, float]:
        """Milliseconds from hearing to each moment that came; ``total`` is
        the turn's end when it ended."""
        out = {name: round((at - self.heard) * 1000, 3) for name, at in self.marks.items() if name != "done"}
        if "done" in self.marks:
            out["total"] = round((self.marks["done"] - self.heard) * 1000, 3)
        return out
