"""Conversations from the client: sessions, turns, the reply as it is written, and the record.

The service keeps a session's lifecycle in a journal (created, running,
stopping, ended, failed, interrupted) with a revision that every command
must name, and each turn as a receipt under a client-chosen request id
whose status is ``pending`` until it settles as ``completed``,
``failed``, ``interrupted`` or ``cancelled``. The reply text lives in the
memory, not the journal: a receipt's ``result_state`` says whether the
episode is still there. While a turn runs, the host publishes its text
provisionally as SSE; what arrives is not the answer until the terminal
frame says ``read_receipt`` and the receipt is read.

This client puts that on the wire without inventing anything: a command
names its ``request_id`` and ``expected_revision`` so a retry is the same
command and a stale one is a ``ConversationConflict`` carrying the
revision the server holds; reads are strict about the shapes they were
promised and lenient about fields they have not learned; the stream
decoder refuses a frame whose id does not name its sequence, a sequence
that skips without a gap, or a stream cut inside a frame; and nothing
here resumes work on its own -- ``cursor`` is what a caller sends back.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
import time
from typing import Dict, Iterator, List, Mapping, Optional, Tuple, Union

from ._wire import (ResourceClient, StreamedLines, address, boolean, bounded_body, identifier, integer, invalid,
                    items, record, text, timestamp)
from .errors import SconeError

MAX_SEQUENCE = 2**63 - 1
SESSION_STATES = ("created", "running", "stopping", "ended", "failed", "interrupted")
TURN_STATUSES = ("pending", "completed", "failed", "interrupted", "cancelled")
RESULT_STATES = ("available", "forgotten", "unavailable", "unreadable")
EVENT_ACTIONS = ("create", "start", "stop", "end", "fail", "interrupt")


class ConversationConflict(SconeError):
    """A 409: the command named a revision the server has moved past, or a stale persona.

    ``revision`` is the server's current revision when it said so, ready
    for the next ``expected_revision``; ``code`` and ``fingerprint`` carry
    a stale persona selection.
    """

    def __init__(self, message: str, *, revision: Optional[int] = None, code: Optional[str] = None,
                 fingerprint: Optional[str] = None, body: Optional[str] = None) -> None:
        super().__init__(message, 409, body=body)
        self.revision = revision
        self.code = code
        self.fingerprint = fingerprint

    @classmethod
    def from_error(cls, error: SconeError) -> "ConversationConflict":
        details: Dict[str, object] = {}
        if isinstance(error.body, str):
            try:
                payload = json.loads(error.body)
            except ValueError:
                payload = None
            if isinstance(payload, dict):
                details = payload
        revision = details.get("revision")
        code = details.get("code")
        fingerprint = details.get("fingerprint")
        return cls(str(error), revision=revision if type(revision) is int else None,
                   code=code if isinstance(code, str) else None,
                   fingerprint=fingerprint if isinstance(fingerprint, str) else None, body=error.body)


def _choice(value: object, allowed: Tuple[str, ...], label: str) -> str:
    if not isinstance(value, str) or value not in allowed:
        raise invalid(label)
    return value


def _optional_text(value: object, maximum: int, label: str) -> Optional[str]:
    return None if value is None else text(value, maximum, label)


@dataclass(frozen=True)
class PersonaRef:
    id: str
    name: str
    fingerprint: str
    current: Optional[bool] = None

    @classmethod
    def from_json(cls, value: object) -> "PersonaRef":
        row = record(value)
        current = row.get("current")
        return cls(identifier(row.get("id")), text(row.get("name"), 256, "persona name"),
                   text(row.get("fingerprint"), 64, "persona fingerprint"),
                   None if current is None else boolean(current))


@dataclass(frozen=True)
class Persona:
    id: str
    name: str
    fingerprint: str
    text_ready: bool
    voice_ready: bool

    @classmethod
    def from_json(cls, value: object) -> "Persona":
        row = record(value)
        return cls(identifier(row.get("id")), text(row.get("name"), 256, "persona name"),
                   text(row.get("fingerprint"), 64, "persona fingerprint"),
                   boolean(row.get("text_ready")), boolean(row.get("voice_ready")))


@dataclass(frozen=True)
class ConversationCapabilities:
    """What the conversation service will do; read leniently, as the service grows."""

    text_configured: bool
    streaming: bool
    turn_cancellation: bool
    session_deletion: bool
    transcript_pagination: bool
    personas: bool
    max_sessions: int
    max_turns: int

    @classmethod
    def from_json(cls, value: object) -> "ConversationCapabilities":
        row = record(value)
        if integer(row.get("schema_version"), 1, 1) != 1:
            raise invalid("conversation capability schema")
        stream = row.get("text_stream")
        if stream is not None and record(stream).get("transport") != "sse":
            raise invalid("conversation text stream transport")
        return cls(boolean(row.get("text_configured")), stream is not None,
                   boolean(row.get("turn_cancellation")), boolean(row.get("session_deletion")),
                   boolean(row.get("transcript_pagination")), bool(row.get("personas")),
                   integer(row.get("max_sessions"), 1), integer(row.get("max_turns"), 1))


@dataclass(frozen=True)
class Session:
    space: str
    session_id: str
    mode: str
    state: str
    revision: int
    created_at: str
    updated_at: str
    recall_scope: Mapping[str, object]
    persona: Optional[PersonaRef]
    active_request_id: Optional[str] = None
    latest_request_id: Optional[str] = None

    @property
    def closed(self) -> bool:
        return self.state in ("ended", "failed", "interrupted")

    @classmethod
    def from_json(cls, value: object, *, expected_space: str) -> "Session":
        row = record(value)
        if row.get("space") != expected_space:
            raise SconeError("conversation response does not match the expected space")
        persona = row.get("persona")
        active, latest = row.get("active_request_id"), row.get("latest_request_id")
        return cls(expected_space, identifier(row.get("session_id")), _choice(row.get("mode"), ("text", "voice"), "mode"),
                   _choice(row.get("state"), SESSION_STATES, "session state"), integer(row.get("revision"), 0),
                   timestamp(row.get("created_at")), timestamp(row.get("updated_at")),
                   dict(record(row.get("recall_scope", {}))), None if persona is None else PersonaRef.from_json(persona),
                   None if active is None else identifier(active), None if latest is None else identifier(latest))


@dataclass(frozen=True)
class TurnReceipt:
    request_id: str
    status: str
    result_state: Optional[str]
    result: Optional[Mapping[str, object]]
    error: Optional[str]
    answer_review: Optional[Mapping[str, object]] = None

    @property
    def settled(self) -> bool:
        return self.status != "pending"

    @property
    def text(self) -> Optional[str]:
        value = self.result.get("text") if self.result is not None else None
        return value if isinstance(value, str) else None

    @classmethod
    def from_json(cls, value: object, *, request_id: Optional[str] = None) -> "TurnReceipt":
        row = record(value)
        named = identifier(row.get("request_id"))
        if request_id is not None and named != request_id:
            raise invalid("turn receipt identity")
        status = _choice(row.get("status"), TURN_STATUSES, "turn status")
        state = row.get("result_state")
        result = row.get("result")
        review = row.get("answer_review")
        if status == "pending" and (state is not None or result is not None):
            raise invalid("pending turn with a result")
        return cls(named, status, None if state is None else _choice(state, RESULT_STATES, "result state"),
                   None if result is None else dict(record(result)), _optional_text(row.get("error"), 8192, "turn error"),
                   None if review is None else dict(record(review)))


@dataclass(frozen=True)
class SessionEvent:
    session_id: str
    request_id: str
    revision: int
    action: str
    previous_state: Optional[str]
    state: str
    recorded_at: str

    @classmethod
    def from_json(cls, value: object) -> "SessionEvent":
        row = record(value)
        previous = row.get("previous_state")
        return cls(identifier(row.get("session_id")), identifier(row.get("request_id")), integer(row.get("revision"), 1),
                   _choice(row.get("action"), EVENT_ACTIONS, "event action"),
                   None if previous is None else _choice(previous, SESSION_STATES, "event state"),
                   _choice(row.get("state"), SESSION_STATES, "event state"), timestamp(row.get("recorded_at")))


@dataclass(frozen=True)
class SessionPage:
    items: Tuple[Session, ...]
    next_after: Optional[str]
    has_more: bool


@dataclass(frozen=True)
class TurnPage:
    turns: Tuple[TurnReceipt, ...]
    next_after: Optional[str]
    has_more: bool


@dataclass(frozen=True)
class EventPage:
    events: Tuple[SessionEvent, ...]
    next_after: int
    has_more: bool


@dataclass(frozen=True)
class TranscriptEpisode:
    """One captured turn as the memory holds it; read leniently, like every memory shape."""

    episode_id: int
    kind: str
    content: str
    source: Optional[str]
    tags: Tuple[str, ...]
    metadata: Mapping[str, str]
    created_at: str

    @classmethod
    def from_json(cls, value: object) -> "TranscriptEpisode":
        row = record(value)
        tags = row.get("tags") or []
        metadata = row.get("metadata") or {}
        source = row.get("source")
        return cls(integer(row.get("episode_id"), 1), str(row.get("kind") or ""), str(row.get("content") or ""),
                   source if isinstance(source, str) else None,
                   tuple(str(tag) for tag in tags) if isinstance(tags, list) else (),
                   {str(k): str(v) for k, v in metadata.items()} if isinstance(metadata, dict) else {},
                   timestamp(row.get("created_at")))


@dataclass(frozen=True)
class TranscriptPage:
    episodes: Tuple[TranscriptEpisode, ...]
    next_before: Optional[str]
    has_more: bool


@dataclass(frozen=True)
class ReplyDelta:
    sequence: int
    text: str
    kind: str = "text"


@dataclass(frozen=True)
class ReplyGap:
    after: int
    next_sequence: int
    kind: str = "gap"


@dataclass(frozen=True)
class ReplyTerminal:
    request_id: str
    status: str
    read_receipt: bool
    kind: str = "terminal"


@dataclass(frozen=True)
class ReplyEnded:
    request_id: str
    reason: str
    kind: str = "end"


ReplyEvent = Union[ReplyDelta, ReplyGap, ReplyTerminal, ReplyEnded]


class ReplyStream:
    """A turn's reply as the host writes it, frame by frame, provisional until the receipt.

    Use as a context manager so the connection is released whether the
    stream ended, was refused, or the caller stopped early. ``cursor`` is
    the last sequence seen, ready for ``after=`` on a reconnect; nothing
    here reconnects on its own.
    """

    def __init__(self, lines: StreamedLines, *, request_id: str, after: int) -> None:
        self._lines = lines
        self._closed = False
        self.request_id = request_id
        self.cursor: int = after

    def __enter__(self) -> "ReplyStream":
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    def close(self) -> None:
        if not self._closed:
            self._closed = True
            self._lines.close()

    def __iter__(self) -> Iterator[ReplyEvent]:
        kind: Optional[str] = None
        event_id: Optional[str] = None
        data: List[bytes] = []
        for raw in self._lines:
            if raw == b"":
                if kind is not None or data:
                    event = self._frame(kind, event_id, b"\n".join(data))
                    kind, event_id, data = None, None, []
                    yield event
                    if isinstance(event, (ReplyTerminal, ReplyEnded)):
                        return
                continue
            if raw.startswith(b":"):
                continue
            field, _, value = raw.partition(b":")
            value = value[1:] if value.startswith(b" ") else value
            if field == b"event":
                kind = value.decode("utf-8", errors="strict")
            elif field == b"id":
                event_id = value.decode("utf-8", errors="strict")
            elif field == b"data":
                data.append(value)
        if kind is not None or data:
            raise invalid("reply stream ended inside a frame")

    def _frame(self, kind: Optional[str], event_id: Optional[str], data: bytes) -> ReplyEvent:
        try:
            payload = json.loads(data.decode("utf-8"))
        except (ValueError, UnicodeError, RecursionError):
            raise invalid("reply frame") from None
        row = record(payload)
        if kind == "text":
            sequence = integer(row.get("sequence"), 1, MAX_SEQUENCE)
            if sequence != self.cursor + 1:
                raise invalid("reply frame sequence")
            if event_id is None or event_id != str(sequence):
                raise invalid("reply frame id does not name its sequence")
            value = row.get("text")
            if not isinstance(value, str) or not value:
                raise invalid("reply text")
            if not boolean(row.get("provisional", True)):
                raise invalid("reply text claims to be final")
            self.cursor = sequence
            return ReplyDelta(sequence, value)
        if event_id is not None:
            raise invalid("reply frame id")
        if kind == "gap":
            after = integer(row.get("after"), 0, MAX_SEQUENCE)
            following = integer(row.get("next_sequence"), 1, MAX_SEQUENCE)
            if after != self.cursor or following <= after + 1:
                raise invalid("reply gap sequence")
            self.cursor = following - 1
            return ReplyGap(after, following)
        if kind == "terminal":
            if identifier(row.get("request_id")) != self.request_id or not boolean(row.get("read_receipt")):
                raise invalid("reply terminal without a read_receipt for this turn")
            return ReplyTerminal(self.request_id, _choice(row.get("status"), TURN_STATUSES[1:], "reply status"), True)
        if kind == "end":
            if identifier(row.get("request_id")) != self.request_id:
                raise invalid("reply end names another turn")
            return ReplyEnded(self.request_id, text(row.get("reason"), 64, "reply end reason"))
        if kind == "error":
            reason = row.get("reason")
            raise SconeError("reply stream refused: " + (reason if isinstance(reason, str) else "unknown"))
        raise invalid("reply frame kind")


class ConversationClient(ResourceClient):
    """Sessions and turns over HTTP; every command names its request and revision."""

    def capabilities(self) -> ConversationCapabilities:
        self._check("conversations")
        return ConversationCapabilities.from_json(self._client._request("GET", "/v1/conversations/capabilities"))

    def personas(self) -> Tuple[Persona, ...]:
        self._check("conversations")
        row = record(self._client._request("GET", "/v1/conversations/personas"))
        found = tuple(Persona.from_json(value) for value in items(row.get("personas"), 1000))
        if len({persona.id for persona in found}) != len(found):
            raise invalid("persona list")
        return found

    def sessions(self, *, limit: int = 100, after: Optional[str] = None) -> SessionPage:
        params = {"limit": str(integer(limit, 1, 200))}
        if after is not None:
            params["after"] = identifier(after)
        self._check("conversations")
        row = record(self._client._request("GET", "/v1/conversations", params=params))
        listed = tuple(Session.from_json(value, expected_space=self.expected_space) for value in items(row.get("items"), limit))
        if len({session.session_id for session in listed}) != len(listed):
            raise invalid("session page")
        following = row.get("next_after")
        return SessionPage(listed, None if following in (None, "") else identifier(following), boolean(row.get("has_more")))

    def create(self, *, request_id: str, mode: str = "text", recall_scope: Optional[Mapping[str, object]] = None,
               persona: Optional[str] = None, persona_fingerprint: Optional[str] = None) -> Session:
        """Open a session; the same ``request_id`` again is the same command, not a second session."""
        body: Dict[str, object] = {"request_id": identifier(request_id), "capture": True,
                                   "mode": _choice(mode, ("text", "voice"), "mode")}
        if recall_scope:
            body["recall_scope"] = dict(recall_scope)
        if persona is not None:
            body["persona"] = identifier(persona)
        if persona_fingerprint is not None:
            body["persona_fingerprint"] = text(persona_fingerprint, 64, "persona fingerprint")
        self._check("conversations", mutation=True)
        return Session.from_json(self._conflicts("POST", "/v1/conversations", json=bounded_body(body)),
                                 expected_space=self.expected_space)

    def session(self, session_id: str) -> Session:
        self._check("conversations")
        found = Session.from_json(self._client._request("GET", "/v1/conversations/" + address(session_id)),
                                  expected_space=self.expected_space)
        if found.session_id != session_id:
            raise invalid("session identity")
        return found

    def events(self, session_id: str, *, after: int = 0, limit: int = 100) -> EventPage:
        """The session's lifecycle as the journal recorded it; ``after`` is a revision."""
        params = {"after": str(integer(after, 0)), "limit": str(integer(limit, 1, 200))}
        self._check("conversations")
        row = record(self._client._request("GET", "/v1/conversations/" + address(session_id) + "/events", params=params))
        events = tuple(SessionEvent.from_json(value) for value in items(row.get("events"), limit))
        if any(event.session_id != session_id for event in events) or [e.revision for e in events] != sorted(
                {e.revision for e in events}) or any(event.revision <= after for event in events):
            raise invalid("event page")
        following = integer(row.get("next_after"), 0)
        if events and following != events[-1].revision:
            raise invalid("event page cursor")
        return EventPage(events, following, boolean(row.get("has_more")))

    def transcript(self, session_id: str, *, before: Optional[str] = None, limit: int = 200) -> TranscriptPage:
        """The captured turns, newest first; ``before`` is the opaque cursor a page hands back."""
        params = {"limit": str(integer(limit, 1, 200))}
        if before is not None:
            params["before"] = text(before, 1024, "transcript cursor")
        self._check("conversations")
        row = record(self._client._request("GET", "/v1/conversations/" + address(session_id) + "/transcript", params=params))
        episodes = tuple(TranscriptEpisode.from_json(value) for value in items(row.get("episodes"), limit))
        following = row.get("next_before")
        return TranscriptPage(episodes, None if following is None else text(following, 1024, "transcript cursor"),
                              boolean(row.get("has_more")))

    def submit(self, session_id: str, *, request_id: str, text_: str, expected_revision: int) -> TurnReceipt:
        """Ask a turn; the receipt is ``pending`` until it settles, and the same command again is the same turn."""
        body = bounded_body({"request_id": identifier(request_id), "expected_revision": integer(expected_revision, 1, 2**63 - 2),
                             "text": text(text_, 32000, "turn text")}, 40000)
        self._check("conversations", mutation=True)
        return TurnReceipt.from_json(self._conflicts("POST", "/v1/conversations/" + address(session_id) + "/turns", json=body),
                                     request_id=request_id)

    def turns(self, session_id: str, *, after: Optional[str] = None, limit: int = 100) -> TurnPage:
        params = {"limit": str(integer(limit, 1, 200))}
        if after is not None:
            params["after"] = identifier(after)
        self._check("conversations")
        row = record(self._client._request("GET", "/v1/conversations/" + address(session_id) + "/turns", params=params))
        listed = tuple(TurnReceipt.from_json(value) for value in items(row.get("turns"), limit))
        if len({turn.request_id for turn in listed}) != len(listed):
            raise invalid("turn page")
        following = row.get("next_after")
        return TurnPage(listed, None if following in (None, "") else identifier(following), boolean(row.get("has_more")))

    def turn(self, session_id: str, request_id: str) -> TurnReceipt:
        self._check("conversations")
        path = "/v1/conversations/" + address(session_id) + "/turns/" + address(request_id)
        return TurnReceipt.from_json(self._client._request("GET", path), request_id=request_id)

    def wait(self, session_id: str, request_id: str, *, timeout: float = 60.0, interval: float = 1.0) -> TurnReceipt:
        """Poll the receipt until it settles; a turn still pending at the deadline is an error, not a guess."""
        if not 0 < interval <= timeout:
            raise invalid("wait interval")
        deadline = time.monotonic() + timeout
        while True:
            receipt = self.turn(session_id, request_id)
            if receipt.settled:
                return receipt
            if time.monotonic() >= deadline:
                raise SconeError(f"turn {request_id} still pending after {timeout}s")
            time.sleep(interval)

    def stream_reply(self, session_id: str, request_id: str, *, after: int = 0) -> ReplyStream:
        """Read the reply as it is written; resume with ``cursor``; the receipt is the answer."""
        cursor = integer(after, 0, MAX_SEQUENCE)
        params: Dict[str, str] = {}
        headers: Dict[str, str] = {}
        if cursor:
            params["after"] = str(cursor)
            headers["Last-Event-ID"] = str(cursor)
        self._check("conversations")
        if not self.capabilities().streaming:
            raise SconeError("server does not stream conversation text")
        path = "/v1/conversations/" + address(session_id) + "/turns/" + address(request_id) + "/stream"
        return ReplyStream(self._client._stream(path, params=params, headers=headers), request_id=request_id, after=cursor)

    def cancel(self, session_id: str, request_id: str) -> TurnReceipt:
        """Cancel a pending turn; the returned receipt says ``cancelled`` or the call failed."""
        self._check("conversations", mutation=True)
        if not self.capabilities().turn_cancellation:
            raise SconeError("server does not cancel turns")
        path = "/v1/conversations/" + address(session_id) + "/turns/" + address(request_id) + "/cancel"
        receipt = TurnReceipt.from_json(self._conflicts("POST", path), request_id=request_id)
        if receipt.status != "cancelled":
            raise invalid("cancel acknowledgement")
        return receipt

    def stop(self, session_id: str, *, request_id: str, expected_revision: int) -> Session:
        body = bounded_body({"request_id": identifier(request_id), "expected_revision": integer(expected_revision, 1, 2**63 - 2)})
        self._check("conversations", mutation=True)
        found = Session.from_json(self._conflicts("POST", "/v1/conversations/" + address(session_id) + "/stop", json=body),
                                  expected_space=self.expected_space)
        if found.session_id != session_id:
            raise invalid("session identity")
        return found

    def delete(self, session_id: str) -> None:
        """Remove a closed session's record; a running one is a conflict, not a deletion."""
        self._check("conversations", mutation=True)
        if not self.capabilities().session_deletion:
            raise SconeError("server does not delete sessions")
        self._conflicts("DELETE", "/v1/conversations/" + address(session_id))

    def _conflicts(self, method: str, path: str, *, json: Optional[Dict[str, object]] = None) -> object:
        try:
            return self._client._request(method, path, json=json)
        except SconeError as error:
            if error.status == 409:
                raise ConversationConflict.from_error(error) from None
            raise
