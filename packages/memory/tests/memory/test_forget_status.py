"""Removal status reads durable progress without driving cleanup."""
from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from scone_memory.api import create_app
from scone_memory.core.errors import InvalidInput, NotFound
from .test_forget_recovery import make_engine, NOW


@pytest.mark.parametrize("backend", ["memory", "sqlite"])
async def test_status_present_unknown_foreign_and_completed(tmp_path, backend):
    memory = await make_engine(tmp_path, backend)
    try:
        source = await memory.remember("alpha", "source to remove")
        present = await memory.forget_status("alpha", source.episode_id)
        assert present.model_dump() == {"episode_id": source.episode_id, "state": "present",
                                        "requested_at": None, "forgotten_at": None, "impact": None}
        for space, identity in [("bravo", source.episode_id), ("alpha", source.episode_id + 100)]:
            with pytest.raises(NotFound):
                await memory.forget_status(space, identity)
        await memory.forget("alpha", source.episode_id)
        done = await memory.forget_status("alpha", source.episode_id)
        assert done.state == "forgotten" and done.forgotten_at == NOW
        assert done.impact is None  # Tombstones do not retain the complete receipt.
    finally:
        await memory.close()


@pytest.mark.parametrize("backend", ["memory", "sqlite"])
@pytest.mark.parametrize("stage", ["delete_episode", "vectors", "clear_retirement"])
async def test_pending_status_is_read_only_even_after_rows_or_tombstone(tmp_path, monkeypatch, backend, stage):
    memory = await make_engine(tmp_path, backend)
    try:
        source = await memory.remember("alpha", "source with cleanup interrupted")
        expected = await memory.impact("alpha", source.episode_id)
        target, method = (memory.vectors, "delete") if stage == "vectors" else (memory.documents, stage)
        original = getattr(target, method)
        async def interrupted(*args):
            raise RuntimeError("cleanup interrupted")
        monkeypatch.setattr(target, method, interrupted)
        with pytest.raises(RuntimeError):
            await memory.forget("alpha", source.episode_id)
        before = await memory.documents.retirement("alpha", source.episode_id)
        status = await memory.forget_status("alpha", source.episode_id)
        assert status.state == "pending" and status.requested_at == NOW
        assert status.forgotten_at is None and status.impact == expected
        assert await memory.documents.retirement("alpha", source.episode_id) == before
        assert (await memory.documents.get_episode("alpha", source.episode_id) is not None) == (stage == "delete_episode")
        monkeypatch.setattr(target, method, original)
        await memory.forget("alpha", source.episode_id)
        assert (await memory.forget_status("alpha", source.episode_id)).state == "forgotten"
    finally:
        await memory.close()


@pytest.mark.parametrize("identity", [True, 0, -1, 2**63, "1"])
async def test_status_validates_identity(tmp_path, identity):
    memory = await make_engine(tmp_path, "memory")
    try:
        with pytest.raises(InvalidInput):
            await memory.forget_status("alpha", identity)
    finally:
        await memory.close()


async def test_http_status_capability_auth_and_read_role(tmp_path, monkeypatch):
    memory = await make_engine(tmp_path, "memory")
    source = await memory.remember("alpha", "retained")
    auth = lambda key: {"Authorization": f"Bearer {key}"}
    with TestClient(create_app(memory, {"owner": "alpha", "reader": "alpha", "foreign": "bravo"},
                               roles={"reader": "read"})) as client:
        path = f"/v1/episodes/{source.episode_id}/forget-status"
        assert client.get(path).status_code == 401
        assert client.get(path, headers=auth("foreign")).status_code == 404
        assert client.get(path, headers=auth("reader")).json()["state"] == "present"
        assert client.delete(f"/v1/episodes/{source.episode_id}", headers=auth("reader")).status_code == 403
        assert client.get("/v1/capabilities", headers=auth("reader")).json()["features"]["episodes.forget"] is True
        assert client.delete(f"/v1/episodes/{source.episode_id}", headers=auth("owner")).status_code == 200
        assert client.get(path, headers=auth("reader")).json()["state"] == "forgotten"
        for value in ["0", "-1", str(2**63)]:
            assert client.get(f"/v1/episodes/{value}/forget-status", headers=auth("reader")).status_code == 422
        monkeypatch.setattr(memory.documents, "record_retirement", None)
        assert client.get("/v1/capabilities", headers=auth("reader")).json()["features"]["episodes.forget"] is False
        assert client.get(path, headers=auth("reader")).status_code == 422


@pytest.mark.parametrize("method", ["get_episode", "tombstone", "retirement"])
@pytest.mark.parametrize("wrong_field", ["space", "episode_id"])
async def test_status_rejects_adapter_identity_mismatch(tmp_path, monkeypatch, method, wrong_field):
    from scone_memory.core.models import ForgetReceipt, Tombstone
    from scone_memory.core.retirement import Retirement
    memory = await make_engine(tmp_path, "memory")
    try:
        source = await memory.remember("alpha", "source")
        episode = await memory.episode("alpha", source.episode_id)
        rows = {
            "get_episode": episode,
            "tombstone": Tombstone(space="alpha", episode_id=source.episode_id, content_hash=episode.content_hash, forgotten_at=NOW),
            "retirement": Retirement(space="alpha", episode_id=source.episode_id, content_hash=episode.content_hash,
                                     requested_at=NOW, chunk_ids=(), receipt=ForgetReceipt(episode_id=source.episode_id, chunks=0)),
        }
        wrong = rows[method].model_copy(update={wrong_field: "bravo" if wrong_field == "space" else source.episode_id + 1})
        async def mismatched(*args):
            return wrong
        monkeypatch.setattr(memory.documents, method, mismatched)
        with pytest.raises(InvalidInput, match="identity"):
            await memory.forget_status("alpha", source.episode_id)
    finally:
        await memory.close()
