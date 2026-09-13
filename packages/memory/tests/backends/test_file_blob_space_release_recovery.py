"""File cleanup keeps durable targets until both bytes and metadata are removed."""
from pathlib import Path

import pytest

from scone_memory.backends.blobs import FileBlobStore


@pytest.mark.parametrize('stage', ['bytes', 'metadata'])
async def test_space_release_resumes_after_filesystem_failure(tmp_path, monkeypatch, stage):
    import os

    store = FileBlobStore(tmp_path / 'blobs')
    unique = await store.put('alpha', b'unique private bytes', 'text/plain')
    shared = await store.put('alpha', b'shared bytes', 'text/plain')
    await store.put('bravo', b'shared bytes', 'text/plain')
    await store.link('alpha', unique.attachment_id, 1)
    blob = store._blob(unique.attachment_id)
    if stage == 'bytes':
        original = Path.unlink

        def fail(path, *args, **kwargs):
            if path == blob:
                raise OSError('interrupted byte deletion')
            return original(path, *args, **kwargs)

        monkeypatch.setattr(Path, 'unlink', fail)
    else:
        original = os.unlink

        def fail(path, *args, **kwargs):
            if str(path).endswith('.json'):
                raise OSError('interrupted metadata deletion')
            return original(path, *args, **kwargs)

        monkeypatch.setattr(os, 'unlink', fail)
    with pytest.raises(OSError):
        await store.release_space('alpha')
    assert unique.attachment_id in await store.held('alpha')
    assert (await store.get('bravo', shared.attachment_id))[1] == b'shared bytes'
    monkeypatch.undo()
    reopened = FileBlobStore(tmp_path / 'blobs')
    await reopened.release_space('alpha')
    assert not blob.exists()
    assert await reopened.held('alpha') == []
    assert await reopened.linked('alpha') == set()
    assert (await reopened.get('bravo', shared.attachment_id))[1] == b'shared bytes'
    assert await reopened.release_space('alpha') == ([], [])


async def test_space_release_finishes_interrupted_source_byte_cleanup(tmp_path, monkeypatch):
    store = FileBlobStore(tmp_path / 'blobs')
    private = await store.put('alpha', b'interrupted source bytes', 'text/plain')
    shared = await store.put('alpha', b'shared source bytes', 'text/plain')
    await store.put('bravo', b'shared source bytes', 'text/plain')
    await store.link('alpha', private.attachment_id, 1)
    await store.link('alpha', shared.attachment_id, 1)
    blob = store._blob(private.attachment_id)
    original = Path.unlink

    def fail(path, *args, **kwargs):
        if path == blob:
            raise OSError('interrupted source unlink')
        return original(path, *args, **kwargs)

    monkeypatch.setattr(Path, 'unlink', fail)
    with pytest.raises(OSError):
        await store.unlink('alpha', 1)
    assert private.attachment_id not in await store.held('alpha')
    assert private.attachment_id in await store.linked('alpha')
    monkeypatch.undo()
    reopened = FileBlobStore(tmp_path / 'blobs')
    await reopened.release_space('alpha')
    assert not blob.exists()
    assert (await reopened.get('bravo', shared.attachment_id))[1] == b'shared source bytes'


async def test_corrupt_link_target_refuses_before_removing_any_bytes(tmp_path):
    import json
    from scone_memory.core.errors import InvalidInput

    store = FileBlobStore(tmp_path / 'blobs')
    private = await store.put('alpha', b'keep on invalid inventory', 'text/plain')
    await store.link('alpha', private.attachment_id, 1)
    store._links('alpha', 1).write_text(json.dumps([private.attachment_id, '../../outside']))
    with pytest.raises(InvalidInput, match='attachment'):
        await store.release_space('alpha')
    assert (await store.get('alpha', private.attachment_id))[1] == b'keep on invalid inventory'
