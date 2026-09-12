"""Document recovery and retained provenance through the actual local HTTP API."""
import json
import time

import pytest

from scone import SconeError
from native_server import native_server


def wait_for(predicate):
    deadline = time.monotonic() + 8
    while time.monotonic() < deadline:
        if predicate():
            return
        time.sleep(0.01)
    pytest.fail('document condition did not settle')


@pytest.mark.integration
def test_document_restart_explicit_recovery_and_verified_result(tmp_path):
    content = b'# Local document\n\nAda studies stars.'
    with native_server(tmp_path, 'document_server.py') as client:
        jobs = client.document_jobs(expected_space='alpha')
        assert jobs.formats().formats['.md'].available
        original = jobs.upload(content, media_type='text/markdown')
        jobs.start('one', attachment_id=original.attachment_id, filename='chosen.md')
        wait_for(lambda: (tmp_path / 'parses.jsonl').exists())
        saved = jobs.request('one')
        jobs.status('one').match(saved)
    with native_server(tmp_path, 'document_server.py') as client:
        jobs = client.document_jobs(expected_space='alpha')
        paused = jobs.status('one')
        assert not paused.active_local and paused.outcome_unknown
        assert jobs.request('one') == saved
        jobs.start('one', attachment_id=original.attachment_id, filename='chosen.md')
        assert len((tmp_path / 'parses.jsonl').read_text().splitlines()) == 1
        with pytest.raises(SconeError):
            jobs.result('one')
        (tmp_path / 'release').touch()
        jobs.resume(saved)
        wait_for(lambda: jobs.status('one').status == 'completed' and not jobs.status('one').active_local)
        result = jobs.result('one')
        assert result.original.attachment_id == original.attachment_id
        assert result.filename == 'chosen.md' and result.format == 'md' and result.segments >= 1
        completed = jobs.request('one')
        assert jobs.resume(completed).status == 'completed'
        assert jobs.cancel(completed).status == 'completed'
        assert jobs.list().items[0].status == 'completed'
        assert len((tmp_path / 'parses.jsonl').read_text().splitlines()) == 2
        (tmp_path / 'release').unlink()
        jobs.start('cancel', attachment_id=original.attachment_id, filename='chosen.md')
        wait_for(lambda: len((tmp_path / 'parses.jsonl').read_text().splitlines()) == 3)
        cancelled = jobs.cancel(jobs.request('cancel'))
        assert cancelled.status == 'cancelled'
        assert all(row['filename'] == 'chosen.md' for row in map(json.loads, (tmp_path / 'parses.jsonl').read_text().splitlines()))
