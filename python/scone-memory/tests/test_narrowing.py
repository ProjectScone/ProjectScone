"""Kind, source prefix and created_at bounds narrow recall the way the
Rust engine does: on the candidates fusion produced, after the lanes,
before the limit. Runs over every backend the conftest reaches, because
each one must carry kind and source on its episodes."""

from __future__ import annotations

import pytest

from scone_memory import InvalidInput

ROWS = [
    ("note", "deploy runbook: rotate the staging keys first", None, "2024-01-10"),
    ("file", "deploy runbook: rotate the staging keys, then restart", "/ops/runbooks/deploy.md", "2024-02-10"),
    ("file", "deploy runbook draft: staging keys rotate weekly", "/ops/drafts/deploy_v2.md", "2024-03-10"),
    ("conversation", "user: where is the deploy runbook for staging keys?", "session-42", "2024-04-10"),
]


async def seeded(engine):
    ids = []
    for kind, text, source, day in ROWS:
        ids.append((await engine.remember("default", text, kind=kind, source=source, created_at=day)).episode_id)
    return ids


async def episodes(engine, **narrow):
    result = await engine.recall("default", "deploy runbook staging keys", limit=10, **narrow)
    return sorted({i.episode_id for i in result.items})


async def test_without_narrowing_every_episode_is_a_candidate(engine):
    ids = await seeded(engine)
    assert await episodes(engine) == ids


async def test_kind_keeps_only_that_kind(engine):
    ids = await seeded(engine)
    assert await episodes(engine, kind="file") == ids[1:3]
    # The filter runs before the limit: with room for one item and a query
    # the note wins outright, asking for the conversation still finds it.
    one = await engine.recall("default", "rotate the staging keys first", limit=1, kind="conversation")
    assert [i.episode_id for i in one.items] == [ids[3]]
    with pytest.raises(InvalidInput, match="kind"):
        await engine.recall("default", "deploy", kind="chat")


async def test_source_prefix_is_literal_text(engine):
    ids = await seeded(engine)
    assert await episodes(engine, source_prefix="/ops/runbooks/") == [ids[1]]
    assert await episodes(engine, source_prefix="/ops/") == ids[1:3]
    assert await episodes(engine, source_prefix="session-") == [ids[3]]
    assert await episodes(engine, source_prefix="%") == [], "a wildcard character is a character"
    assert await episodes(engine, source_prefix="") == ids[1:4], "an episode without a source matches no prefix"


async def test_since_and_until_are_inclusive_bounds_on_when_it_happened(engine):
    ids = await seeded(engine)
    assert await episodes(engine, since="2024-02-10") == ids[1:4]
    assert await episodes(engine, until="2024-02-10") == ids[0:2]
    assert await episodes(engine, since="2024-02-01", until="2024-03-31", kind="file") == ids[1:3], "filters combine with AND"
    with pytest.raises(InvalidInput):
        await engine.recall("default", "deploy", since="yesterday-ish")


async def test_narrowing_is_on_the_evidence(engine):
    await seeded(engine)
    await engine.recall("default", "deploy runbook", kind="file", since="2024-02-01")
    [event] = await engine.events.query("default", kind="recall", limit=1)
    assert event.payload["narrow"] == {"kind": "file", "source_prefix": None, "since": "2024-02-01T00:00:00.000Z", "until": None}
