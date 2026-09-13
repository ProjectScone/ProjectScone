"""An agent's answer as it is written: events, in sequence, bound to a run.

The host publishes a running step's public text as SSE. Each ``text``
frame carries one delta and its sequence as the SSE ``id``; ``withdraw``
says text streamed before a tool turn was not the answer; ``gap`` says the
reader fell behind the host's bounded window; ``terminal`` says the run
has a receipt; ``end`` says the observation window closed. Sequences are
contiguous from the cursor, a gap being the only sanctioned jump, and an
``id`` must name its frame's sequence because it is what a reconnect will
send back. What arrives is provisional: ``terminal`` carries
``read_receipt``, and the receipt is what ``result()`` returns.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
from typing import Dict, Iterator, List, Optional, Tuple, Union

from ._wire import boolean, integer, invalid, record, text
from .errors import SconeError

MAX_SEQUENCE = 2**63 - 1


@dataclass(frozen=True)
class TextDelta:
    sequence: int
    text: str
    kind: str = "text"


@dataclass(frozen=True)
class Withdrawn:
    sequence: int
    kind: str = "withdraw"


@dataclass(frozen=True)
class Gap:
    after: int
    next_sequence: int
    kind: str = "gap"


@dataclass(frozen=True)
class Terminal:
    status: str
    read_receipt: bool
    kind: str = "terminal"


@dataclass(frozen=True)
class Ended:
    reason: str
    kind: str = "end"


AnswerEvent = Union[TextDelta, Withdrawn, Gap, Terminal, Ended]


def _exact(row: Dict[str, object], fields: Tuple[str, ...]) -> None:
    if set(row) != set(fields):
        raise invalid("answer frame fields")


class AnswerStream:
    """Answer events as the host publishes them, read frame by frame.

    Use as a context manager so the connection is released whether the
    stream ended, was refused, or the caller stopped early. ``cursor`` is
    the last sequence seen, ready for ``after=`` on a reconnect.
    """

    def __init__(self, lines: "StreamedLines", *, after: int) -> None:
        self._lines = lines
        self._closed = False
        self.cursor: int = after

    def __enter__(self) -> "AnswerStream":
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    def close(self) -> None:
        if not self._closed:
            self._closed = True
            self._lines.close()

    def __iter__(self) -> Iterator[AnswerEvent]:
        kind: Optional[str] = None
        identifier: Optional[str] = None
        data: List[bytes] = []
        for raw in self._lines:
            if raw == b"":
                if kind is not None or data:
                    event = self._frame(kind, identifier, b"\n".join(data))
                    kind, identifier, data = None, None, []
                    yield event
                    if isinstance(event, (Terminal, Ended)):
                        return
                continue
            if raw.startswith(b":"):
                continue
            field, _, value = raw.partition(b":")
            value = value[1:] if value.startswith(b" ") else value
            if field == b"event":
                kind = value.decode("utf-8", errors="strict")
            elif field == b"id":
                identifier = value.decode("utf-8", errors="strict")
            elif field == b"data":
                data.append(value)
        if kind is not None or data:
            raise invalid("answer stream ended inside a frame")

    def _sequenced(self, row: Dict[str, object], identifier: Optional[str]) -> int:
        sequence = integer(row.get("sequence"), 1, MAX_SEQUENCE)
        if sequence != self.cursor + 1:
            raise invalid("answer frame sequence")
        if identifier is None or identifier != str(sequence):
            raise invalid("answer frame id does not name its sequence")
        return sequence

    def _frame(self, kind: Optional[str], identifier: Optional[str], data: bytes) -> AnswerEvent:
        try:
            payload = json.loads(data.decode("utf-8"))
        except (ValueError, UnicodeError, RecursionError):
            raise invalid("answer frame") from None
        row = record(payload)
        if kind == "error":
            reason = row.get("reason")
            raise SconeError("answer stream refused: " + (reason if isinstance(reason, str) else "unknown"))
        if kind == "text":
            _exact(row, ("sequence", "text"))
            sequence = self._sequenced(row, identifier)
            value = row.get("text")
            if not isinstance(value, str) or not value:
                raise invalid("answer text")
            self.cursor = sequence
            return TextDelta(sequence, value)
        if kind == "withdraw":
            _exact(row, ("sequence",))
            sequence = self._sequenced(row, identifier)
            self.cursor = sequence
            return Withdrawn(sequence)
        if kind == "gap":
            _exact(row, ("after", "next_sequence"))
            if identifier is not None:
                raise invalid("answer frame id")
            after = integer(row.get("after"), 0, MAX_SEQUENCE)
            following = integer(row.get("next_sequence"), 1, MAX_SEQUENCE)
            if after != self.cursor or following <= after + 1:
                raise invalid("answer gap sequence")
            self.cursor = following - 1
            return Gap(after, following)
        if kind == "terminal":
            _exact(row, ("status", "read_receipt"))
            if identifier is not None or not boolean(row.get("read_receipt")):
                raise invalid("answer terminal without a read_receipt")
            return Terminal(text(row.get("status"), 64, "answer status"), True)
        if kind == "end":
            _exact(row, ("reason",))
            if identifier is not None:
                raise invalid("answer frame id")
            return Ended(text(row.get("reason"), 64, "answer end reason"))
        raise invalid("answer frame kind")


from ._wire import StreamedLines  # noqa: E402
