"""A failed file cleanup keeps the episode's durable list of targets."""
from pathlib import Path

import pytest

from scone_memory.backends.blobs import FileBlobStore


@pytest.mark.parametrize("stage", ["hold", "bytes"])
@pytest.mark.parametrize("after_write", [False, True])
async def test_unlink_resumes_after_each_filesystem_failure(tmp_path, monkeypatch, stage, after_write):
    store = FileBlobStore(tmp_path)
    first = await store.put("alpha", b"first payload", "text/plain")
    second = await store.put("alpha", b"second payload", "text/plain")
    shared = await store.put("alpha", b"shared across spaces", "text/plain")
    await store.put("bravo", b"shared across spaces", "text/plain")
    linked = await store.put("alpha", b"another episode needs this", "text/plain")
    for attachment in (first, second, shared, linked):
        await store.link("alpha", attachment.attachment_id, 1)
    await store.link("alpha", linked.attachment_id, 2)
    target = (store._held("alpha", first.attachment_id) if stage == "hold"
              else store._blob(first.attachment_id))
    original = Path.unlink
    failed = False

    def fail_once(path, *args, **kwargs):
        nonlocal failed
        if path != target or failed:
            return original(path, *args, **kwargs)
        failed = True
        if after_write:
            original(path, *args, **kwargs)
        raise OSError("injected cleanup interruption")

    monkeypatch.setattr(Path, "unlink", fail_once)
    with pytest.raises(OSError, match="injected cleanup interruption"):
        await store.unlink("alpha", 1)
    # Reconstruct the object: retry must use disk state, not process memory.
    store = FileBlobStore(tmp_path)
    await store.unlink("alpha", 1)
    assert not store._links("alpha", 1).exists()
    assert await store.for_episode("alpha", 1) == []
    assert await store.held("alpha") == [linked.attachment_id]
    for attachment in (first, second):
        assert not store._blob(attachment.attachment_id).exists()
    assert (await store.get("bravo", shared.attachment_id))[1] == b"shared across spaces"
    assert (await store.for_episode("alpha", 2)) == [linked]
    assert await store.unlink("alpha", 1) == []
