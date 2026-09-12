"""Real native HTTP compatibility from the requests-only Python client."""
import time

import pytest
from scone import SconeError
from native_server import native_server


def wait_for(predicate):
    deadline = time.monotonic() + 10
    while time.monotonic() < deadline:
        if predicate():
            return
        time.sleep(.02)
    pytest.fail('directory operation did not settle')


@pytest.mark.integration
def test_directory_client_native_lifecycle_and_passive_reopen(tmp_path):
    root = tmp_path/'notes'
    root.mkdir()
    for index in range(3):
        (root/f'note-{index}.txt').write_text(f'Local source document number {index}.')
    with native_server(tmp_path, 'directory_server.py') as client:
        sync = client.directory_sync(expected_space='alpha')
        collection = sync.collections()[0]
        assert sync.list().items == ()
        sync.start('first', collection=collection)
        wait_for(lambda: not sync.status('first').active_local)
        assert sync.status('first').status == 'completed'
        assert sync.request('first').spec.configuration == collection.configuration
        first = sync.results('first', limit=2)
        assert len(first.items) == 2 and first.next_after == 1
        assert len(sync.results('first', limit=2, after=first.next_after).items) == 1
        assert all(item.source.status == 'added' for item in first.items)
        before = (tmp_path/'parses').read_text()
        sync.start('unchanged', collection=collection)
        wait_for(lambda: not sync.status('unchanged').active_local)
        assert all(item.source.status == 'unchanged' for item in sync.results('unchanged').items)
        assert (tmp_path/'parses').read_text() == before
        (root/'note-0.txt').write_text('A changed local document revision.')
        (root/'note-1.txt').unlink()
        sync.start('changed', collection=collection, delete_missing=True)
        wait_for(lambda: not sync.status('changed').active_local)
        outcomes = {item.source.path: item.source.status for item in sync.results('changed').items}
        assert outcomes == {'note-0.txt': 'updated', 'note-2.txt': 'unchanged', 'note-1.txt': 'deleted'}
        (tmp_path/'pause').touch()
        (root/'note-0.txt').write_text('Recovery needs an explicitly resumed attempt.')
        sync.start('interrupted', collection=collection)
        wait_for(lambda: len((tmp_path/'parses').read_text().splitlines()) == 5)
    with native_server(tmp_path, 'directory_server.py') as client:
        sync = client.directory_sync(expected_space='alpha')
        idle = sync.status('interrupted')
        assert not idle.active_local and idle.outcome_unknown
        before = (tmp_path/'parses').read_text()
        sync.list()
        sync.collections()
        assert sync.start('interrupted', collection=collection) == idle
        assert (tmp_path/'parses').read_text() == before
        (tmp_path/'pause').unlink()
        sync.resume(idle)
        wait_for(lambda: not sync.status('interrupted').active_local)
        assert sync.status('interrupted').record.attempt == 2
        assert sync.status('interrupted').status == 'completed'
        with pytest.raises(SconeError):
            sync.resume(sync.status('interrupted'))
        (tmp_path/'pause').touch()
        (root/'note-0.txt').write_text('Cancel this next revision before indexing.')
        sync.start('cancel', collection=collection)
        wait_for(lambda: len((tmp_path/'parses').read_text().splitlines()) == 7)
        assert sync.cancel(sync.status('cancel')).record.cancel_requested_at is not None
        wait_for(lambda: not sync.status('cancel').active_local)
        assert sync.status('cancel').status == 'cancelled'
        (tmp_path/'pause').unlink()
        sync.resume(sync.status('cancel'))
        wait_for(lambda: not sync.status('cancel').active_local)
        assert sync.status('cancel').status == 'completed'
