"""The behavioural contract every EventLog sink must satisfy.

Import with ``from scone_memory.testing.events_contract import *`` and
provide a ``sink`` fixture (an opened, empty sink). The in-memory,
SQLite and MongoDB sinks pass these. Retention is sink-specific and is
tested beside each sink, not here.
"""

from __future__ import annotations

import pytest

from ..ports import DuplicateEvent, NewEvent


def ev(space, kind, ts="2025-01-01T00:00:00.000Z", **payload):
    return NewEvent(ts=ts, space=space, kind=kind, payload=payload)


async def test_events_come_back_newest_first_with_increasing_ids(sink):
    a = await sink.append(ev("alpha", "recall", "2025-01-01T00:00:00.000Z", query="one"))
    b = await sink.append(ev("alpha", "remember", "2025-01-02T00:00:00.000Z", fresh=1))
    c = await sink.append(ev("alpha", "recall", "2025-01-03T00:00:00.000Z", query="two"))
    assert a.event_id < b.event_id < c.event_id
    assert [e.event_id for e in await sink.query("alpha")] == [c.event_id, b.event_id, a.event_id]


async def test_kind_since_and_limit_compose(sink):
    a = await sink.append(ev("alpha", "recall", "2025-01-01T00:00:00.000Z", query="one"))
    b = await sink.append(ev("alpha", "remember", "2025-01-02T00:00:00.000Z", fresh=1))
    c = await sink.append(ev("alpha", "recall", "2025-01-03T00:00:00.000Z", query="two"))
    assert [e.event_id for e in await sink.query("alpha", kind="recall")] == [c.event_id, a.event_id]
    # since is inclusive on the event timestamp
    assert [e.event_id for e in await sink.query("alpha", since="2025-01-02T00:00:00.000Z")] == [c.event_id, b.event_id]
    assert [e.event_id for e in await sink.query("alpha", since="2025-01-02T00:00:00.000Z", kind="recall")] == [c.event_id]
    assert [e.event_id for e in await sink.query("alpha", limit=1)] == [c.event_id]
    assert [e.event_id for e in await sink.query("alpha", kind="recall", limit=1)] == [c.event_id]


async def test_spaces_are_isolated(sink):
    a = await sink.append(ev("alpha", "recall", query="mine"))
    await sink.append(ev("beta", "recall", query="theirs"))
    assert [e.payload["query"] for e in await sink.query("alpha")] == ["mine"]
    assert [e.payload["query"] for e in await sink.query("beta")] == ["theirs"]
    assert await sink.get("beta", a.event_id) is None
    assert await sink.get("alpha", a.event_id + 1000) is None


async def test_payloads_round_trip_with_nesting_and_unicode(sink):
    payload = {"items": [{"chunk_id": 1, "lanes": {"vector": 1}}], "note": "café ☕ 東京", "flag": True, "none": None, "n": 1.5}
    e = await sink.append(ev("alpha", "recall", **payload))
    back = await sink.get("alpha", e.event_id)
    assert dict(back.payload) == payload
    assert (back.ts, back.space, back.kind, back.schema_version) == ("2025-01-01T00:00:00.000Z", "alpha", "recall", 1)


async def test_since_accepts_any_rfc3339_form(sink):
    a = await sink.append(ev("alpha", "recall", "2025-01-01T12:00:00.000Z", query="noon"))
    await sink.append(ev("alpha", "recall", "2025-01-01T06:00:00.000Z", query="morning"))
    assert [e.event_id for e in await sink.query("alpha", since="2025-01-01T13:00:00+01:00")] == [a.event_id]
    assert len(await sink.query("alpha", since="2025-01-01")) == 2


async def test_after_id_is_a_stable_forward_cursor(sink):
    """Five events at most, so a small ring in a test fixture keeps them all."""
    ids = [(await sink.append(ev("alpha", "recall", f"2025-01-0{i}T00:00:00.000Z", n=i))).event_id for i in range(1, 5)]
    await sink.append(ev("beta", "recall", n=99))
    page = await sink.query("alpha", after_id=ids[0], limit=2)
    assert [e.event_id for e in page] == [ids[1], ids[2]]  # oldest first, strictly after the cursor
    rest = await sink.query("alpha", after_id=page[-1].event_id, limit=10)
    assert [e.event_id for e in rest] == [ids[3]]
    assert await sink.query("alpha", after_id=ids[3]) == []
    assert [e.event_id for e in await sink.query("alpha", after_id=0, kind="recall", limit=100)] == ids


async def test_dedup_key_is_atomic_per_space_and_conflicts_on_a_different_payload(sink):
    a = await sink.append(NewEvent("2025-01-01T00:00:00.000Z", "alpha", "agent", {"n": 1}, dedup_key="k1"))
    again = await sink.append(NewEvent("2025-01-01T00:00:01.000Z", "alpha", "agent", {"n": 1}, dedup_key="k1"))
    assert again.event_id == a.event_id
    with pytest.raises(DuplicateEvent) as err:
        await sink.append(NewEvent("2025-01-01T00:00:02.000Z", "alpha", "agent", {"n": 2}, dedup_key="k1"))
    assert err.value.existing.event_id == a.event_id
    other = await sink.append(NewEvent("2025-01-01T00:00:03.000Z", "beta", "agent", {"n": 2}, dedup_key="k1"))
    assert other.event_id != a.event_id, "keys are per space"
    b = await sink.append(NewEvent("2025-01-01T00:00:04.000Z", "alpha", "recall", {"n": 3}))
    c = await sink.append(NewEvent("2025-01-01T00:00:05.000Z", "alpha", "recall", {"n": 3}))
    assert b.event_id != c.event_id, "events without a key never deduplicate"
    assert (await sink.get("alpha", a.event_id)).dedup_key == "k1"


__all__ = [
    "test_events_come_back_newest_first_with_increasing_ids",
    "test_kind_since_and_limit_compose",
    "test_spaces_are_isolated",
    "test_payloads_round_trip_with_nesting_and_unicode",
    "test_since_accepts_any_rfc3339_form",
    "test_after_id_is_a_stable_forward_cursor",
    "test_dedup_key_is_atomic_per_space_and_conflicts_on_a_different_payload",
]
