"""Bounded agent receipts with evidence checked against current retention.

This is an event reader, not a task-completion inference or transcript store.
The window and receipt high-water mark are frozen; source retention stays live.
"""

from __future__ import annotations

import base64
import binascii
import hashlib
import hmac
from datetime import timedelta
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from ..core import forget_after
from ..core.errors import InvalidInput
from ..core.ports import DocumentStore, Event, EventLog
from ..core.timeutil import format_rfc3339, parse_rfc3339
from ..core.validation import check_space

MAX_SCAN = 200
MAX_EXCERPT_BYTES = 1024
MAX_CURSOR_BYTES = 4096
MAX_HOURS = 720
MAX_LIMIT = 50


class _Cursor(BaseModel):
    model_config = ConfigDict(strict=True, extra="forbid", frozen=True)

    version: Literal[1] = 1
    space: str
    hours: int = Field(ge=1, le=MAX_HOURS)
    project: str | None
    session_id: str | None
    since: str
    until: str
    high_water: int = Field(ge=0, le=2**63 - 1)
    after_id: int = Field(ge=0, le=2**63 - 1)


def _integer(value: int, name: str, maximum: int) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or not 1 <= value <= maximum:
        raise InvalidInput(f"{name} must be an integer from 1 through {maximum}")


def _scope(value: str | None, name: str, maximum: int) -> None:
    if value is not None and (not isinstance(value, str) or not value.strip() or len(value) > maximum):
        raise InvalidInput(f"{name} must be a nonempty string of at most {maximum} characters")


def _encode(cursor: _Cursor, key: bytes) -> str:
    payload = base64.urlsafe_b64encode(cursor.model_dump_json().encode()).decode("ascii")
    signature = hmac.new(key, payload.encode("ascii"), hashlib.sha256).hexdigest()
    return f"{payload}.{signature}"


def _decode(token: str, key: bytes) -> _Cursor:
    if not isinstance(token, str) or not 1 <= len(token) <= MAX_CURSOR_BYTES:
        raise InvalidInput("activity cursor must be a nonempty string of at most 4096 characters")
    try:
        payload, signature = token.split(".", 1)
        encoded = payload.encode("ascii")
        expected = hmac.new(key, encoded, hashlib.sha256).hexdigest()
        if not hmac.compare_digest(signature, expected):
            raise ValueError("signature mismatch")
        decoded = base64.b64decode(encoded, altchars=b"-_", validate=True)
        cursor = _Cursor.model_validate_json(decoded)
        if cursor.after_id > cursor.high_water:
            raise ValueError("cursor beyond snapshot")
        return cursor
    except (ValueError, UnicodeError, binascii.Error, ValidationError):
        raise InvalidInput("invalid activity cursor; restart the query after a server restart") from None


def _record(event: Event) -> dict[str, object]:
    """A deliberate metadata allowlist: payload text is never a fallback."""
    row: dict[str, object] = {"event_id": event.event_id, "recorded_at": event.ts}
    for name, maximum in (("agent", 120), ("session_id", 128), ("project", 120), ("event", 120),
                          ("tool_name", 120), ("tool_use_id", 128), ("model", 120), ("source_event_id", 128)):
        value = event.payload.get(name)
        if isinstance(value, str):
            row[name] = value[:maximum]
    if isinstance(event.payload.get("ok"), bool):
        row["ok"] = event.payload["ok"]
    return row


async def _evidence(documents: DocumentStore, space: str, event: Event, now: str) -> tuple[str, dict[str, object] | None]:
    episode_id = event.payload.get("episode_id")
    if isinstance(episode_id, bool) or not isinstance(episode_id, int) or episode_id <= 0:
        status = "unlinked_text_omitted" if "text" in event.payload else "metadata_only"
        return status, None
    episode = await documents.get_episode(space, episode_id)
    if episode is None or episode.space != space or episode.episode_id != episode_id:
        return "unavailable", None
    if forget_after.is_due(episode.metadata, parse_rfc3339(now)):
        return "expired", None
    chunks = await documents.chunks_of(space, episode_id)
    chunk = next((item for item in chunks if item.space == space and item.episode_id == episode_id
                  and item.text.strip()), None)
    if chunk is None:
        return "unavailable", None
    # Forget may run while chunks are read. Never fall back to the event copy.
    current = await documents.get_episode(space, episode_id)
    if current is None or current.space != space or current.episode_id != episode_id or current.content_hash != episode.content_hash:
        return "unavailable", None
    if forget_after.is_due(current.metadata, parse_rfc3339(now)):
        return "expired", None
    encoded = chunk.text.encode("utf-8")
    return "retained", {
        "episode_id": episode_id, "chunk_id": chunk.chunk_id, "source": current.source,
        "created_at": current.created_at, "ingested_at": current.ingested_at,
        "text": encoded[:MAX_EXCERPT_BYTES].decode("utf-8", errors="ignore"),
        "truncated": len(encoded) > MAX_EXCERPT_BYTES or len(chunks) > 1,
        "link_provenance": "connector_reported",
    }


async def activity(
    events: EventLog | None, documents: DocumentStore, space: str, *, now: str, cursor_key: bytes,
    hours: int = 2, limit: int = 20, project: str | None = None, session_id: str | None = None,
    cursor: str | None = None,
) -> dict[str, object]:
    """Read at most 200 agent receipts, with a bounded excerpt per result.

    Callers authorize space. Cursor signatures bind all filters to the server
    instance; continuation calls repeat their original hours/project/session.
    Retention and capture completeness cannot be inferred from a bounded log.
    """
    check_space(space)
    _integer(hours, "hours", MAX_HOURS)
    _integer(limit, "limit", MAX_LIMIT)
    _scope(project, "project", 120)
    _scope(session_id, "session_id", 128)
    moment = parse_rfc3339(now)
    if cursor is not None:
        state = _decode(cursor, cursor_key)
        if (state.space, state.hours, state.project, state.session_id) != (space, hours, project, session_id):
            raise InvalidInput("activity cursor does not match the requested space or filters")
    else:
        latest = await events.query(space, kind="agent", limit=1) if events is not None else []
        if latest and (latest[0].space != space or latest[0].kind != "agent" or latest[0].event_id <= 0):
            raise InvalidInput("activity event reader returned an invalid receipt")
        state = _Cursor(space=space, hours=hours, project=project, session_id=session_id,
                        since=format_rfc3339(moment - timedelta(hours=hours)), until=format_rfc3339(moment),
                        high_water=latest[0].event_id if latest else 0, after_id=0)
    output: list[dict[str, object]] = []
    scanned, after_id, has_more = 0, state.after_id, False
    if events is not None and after_id < state.high_water:
        batch = await events.query(space, kind="agent", since=state.since, after_id=after_id, limit=MAX_SCAN)
        floor, ceiling = parse_rfc3339(state.since), parse_rfc3339(state.until)
        for index, event in enumerate(batch):
            if event.space != space or event.kind != "agent" or event.event_id <= after_id:
                raise InvalidInput("activity event reader must return scoped ascending receipt IDs")
            if event.event_id > state.high_water:
                break
            after_id = event.event_id
            scanned += 1
            stamp = parse_rfc3339(event.ts)
            if not floor <= stamp <= ceiling:
                continue
            if project is not None and event.payload.get("project") != project:
                continue
            if session_id is not None and event.payload.get("session_id") != session_id:
                continue
            row = _record(event)
            status, evidence = await _evidence(documents, space, event, now)
            row.update(evidence_status=status, evidence=evidence)
            episode_id = event.payload.get("episode_id")
            if isinstance(episode_id, int) and not isinstance(episode_id, bool) and episode_id > 0:
                row["episode_id"] = episode_id
            output.append(row)
            if len(output) == limit:
                has_more = after_id < state.high_water and (index + 1 < len(batch) or len(batch) == MAX_SCAN)
                break
        else:
            has_more = after_id < state.high_water and len(batch) == MAX_SCAN
    next_cursor = _encode(state.model_copy(update={"after_id": after_id}), cursor_key) if has_more else None
    return {
        "schema_version": 1, "status": "available" if events is not None else "unavailable", "space": space,
        "window": {"since": state.since, "until": state.until, "hours": state.hours},
        "time_basis": "server_receipt_time", "order": "receipt_id_ascending",
        "events": output, "scanned": scanned, "next_cursor": next_cursor,
        "coverage": {
            "capture_complete": False, "scan_limited": scanned == MAX_SCAN and has_more,
            "has_more": has_more, "scope": "retained_recorded_agent_events",
            "limitations": ["Only explicitly recorded agent events are included; this is not proof of completed tasks.",
                            "Capture gaps and event retention losses are unknown, even when no more pages remain.",
                            "Source evidence is checked at read time; the event window is not a source snapshot."],
        },
    }
