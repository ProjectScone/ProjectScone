"""Stable, authorized transcript pages over real memory, never a live store."""

import base64
import json
import pytest
from scone_memory import Record

from ..api.test_conversations_api import client_for, configured, create, engine  # noqa: F401


async def seed(engine, sid, count, *, space="alpha"):
    ids = []
    for index in range(count):
        [added] = await engine.remember_many(space, [Record(f"Message {index}",
                                      metadata={"session_id": sid, "role": "user"},
                                      created_at="2026-09-06T10:00:00Z",
                                      dedup_key=f"{sid}:{index}")])
        ids.append(added.episode_id)
    return ids


async def test_default_transcript_includes_the_newest_message_beyond_the_page_limit(engine, tmp_path):
    app, _ = configured(engine, tmp_path / "sessions.db")
    async with client_for(app) as client:
        session = await create(client)
        ids = await seed(engine, session["session_id"], 203)
        page = (await client.get(f"/v1/conversations/{session['session_id']}/transcript")).json()
        assert [e["episode_id"] for e in page["episodes"]] == ids[-200:]
        assert page["has_more"] is True
        assert isinstance(page["next_before"], str)


async def test_older_cursor_survives_boundary_deletion_and_new_arrivals_without_duplicates(engine, tmp_path):
    app, _ = configured(engine, tmp_path / "sessions.db")
    async with client_for(app) as client:
        session = await create(client)
        sid = session["session_id"]
        ids = await seed(engine, sid, 7)
        await seed(engine, "another-session", 3)
        await seed(engine, sid, 3, space="beta")
        url = f"/v1/conversations/{sid}/transcript"
        newest = (await client.get(url, params={"limit": 3})).json()
        assert [e["episode_id"] for e in newest["episodes"]] == ids[4:]
        await engine.forget("alpha", ids[4])
        await engine.remember("alpha", "New arrival", metadata={"session_id": sid})
        older = (await client.get(url, params={"limit": 3, "before": newest["next_before"]})).json()
        assert [e["episode_id"] for e in older["episodes"]] == ids[1:4]
        oldest = (await client.get(url, params={"limit": 3, "before": older["next_before"]})).json()
        assert [e["episode_id"] for e in oldest["episodes"]] == ids[:1]
        assert oldest["has_more"] is False and oldest["next_before"] is None
        assert (await client.get(url, headers={"Authorization": "Bearer beta-key"})).status_code == 404
        assert (await client.get("/v1/conversations/capabilities")).json()["transcript_pagination"] is True


@pytest.mark.parametrize("params", [{"limit": 0}, {"limit": 201}, {"limit": -1}, {"before": "not-a-cursor"}, {"before": "a" * 1025}])
async def test_invalid_transcript_pagination_is_rejected(engine, tmp_path, params):
    app, _ = configured(engine, tmp_path / "sessions.db")
    async with client_for(app) as client:
        session = await create(client)
        response = await client.get(f"/v1/conversations/{session['session_id']}/transcript", params=params)
        assert response.status_code == 422


async def test_cursor_cannot_change_scope_or_accept_malformed_boundary_fields(engine, tmp_path):
    app, _ = configured(engine, tmp_path / "sessions.db")
    async with client_for(app) as client:
        session = await create(client)
        sid = session["session_id"]
        url = f"/v1/conversations/{sid}/transcript"
        for cursor in [
            [True, "alpha", sid, "2026-09-06T10:00:00.000Z", 1],
            [2, "alpha", sid, "2026-09-06T10:00:00.000Z", 1],
            [1, "beta", sid, "2026-09-06T10:00:00.000Z", 1],
            [1, "alpha", "another-session", "2026-09-06T10:00:00.000Z", 1],
            [1, "alpha", sid, "not-a-date", 1],
            [1, "alpha", sid, "2026-09-06T10:00:00.000Z", True],
            [1, "alpha", sid, "2026-09-06T10:00:00.000Z", -1],
            [1, "alpha", sid, "2026-09-06T10:00:00.000Z", 2**63],
            {"cursor": "invalid"}, [],
        ]:
            encoded = base64.urlsafe_b64encode(json.dumps(cursor).encode()).decode().rstrip("=")
            assert (await client.get(url, params={"before": encoded})).status_code == 422
