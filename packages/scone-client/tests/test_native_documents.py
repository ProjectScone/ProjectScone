"""Document recovery and retained provenance through the actual local HTTP API."""
import json
import time
import shutil
import subprocess

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


@pytest.mark.integration
def test_video_selection_survives_native_host_restart(tmp_path):
    ffmpeg = shutil.which('ffmpeg')
    if not ffmpeg or not shutil.which('ffprobe'):
        pytest.skip('installed ffmpeg and ffprobe required')
    path = tmp_path / 'slides.mp4'
    subprocess.run([ffmpeg, '-v', 'error', '-f', 'lavfi', '-i', 'testsrc2=size=64x32:rate=1:duration=2',
                    '-an', '-c:v', 'libx264', str(path)], check=True)
    for first in [True, False]:
        with native_server(tmp_path, 'video_document_server.py') as client:
            jobs = client.document_jobs(expected_space='alpha')
            assert jobs.formats().video_ocr_available
            if first:
                original = jobs.upload(path.read_bytes(), media_type='application/octet-stream')
                jobs.start('slides', attachment_id=original.attachment_id, filename='slides.mp4', video_ocr=True)
                wait_for(lambda: jobs.status('slides').status == 'completed' and not jobs.status('slides').active_local)
            assert jobs.request('slides').spec.video_ocr
            result = jobs.result('slides')
            assert result.video_ocr and result.format == 'mp4' and result.segments == 2
            assert len((tmp_path / 'recognitions').read_text().splitlines()) == 2
