"""Durable local sync intent and historical results, without starting scans."""
from dataclasses import replace

import pytest

from scone_memory.agents.workflow import WorkflowError
from scone_memory.ingestion.directory_sync import DirectorySyncResult, SourceReceipt
from scone_memory.ingestion.source_scan import ScanIssue
from scone_memory.ingestion.directory_runs import DirectoryRunStore, SyncRunSpec

KEY = b'd' * 32
SPEC = SyncRunSpec(collection_id='team-notes', configuration='a' * 64, delete_missing=False)
RESULT = DirectorySyncResult('b' * 32, False, (
    SourceReceipt('private-Ω.txt', 'added', episode_id=12),
    SourceReceipt('gone.txt', 'deleted', previous_episode_id=9),
), (ScanIssue('unreadable.txt', 'unreadable'),), 2)


def store(tmp_path, **kwargs):
    return DirectoryRunStore(tmp_path / 'runs.sqlite', key=KEY, **kwargs)


def started(registry, run_id='scan-1', space='alpha'):
    request = registry.register(space, run_id, SPEC)
    return registry.start_attempt(space, run_id, expected_revision=request.revision)


def test_registration_is_immutable_idempotent_and_private_after_reopen(tmp_path):
    registry = store(tmp_path)
    first = registry.register('alpha', 'scan-1', SPEC)
    assert first.status == 'registered' and first.attempt == 0
    assert registry.register('alpha', 'scan-1', SPEC) == first
    with pytest.raises(WorkflowError, match='sync_request_conflict'):
        registry.register('alpha', 'scan-1', SPEC.model_copy(update={'delete_missing': True}))
    registry.close()
    raw = (tmp_path / 'runs.sqlite').read_bytes()
    assert b'team-notes' not in raw and b'scan-1' not in raw and b'alpha' not in raw
    registry = store(tmp_path)
    try:
        assert registry.get('alpha', 'scan-1') == first
        assert registry.get('bravo', 'scan-1') is None
    finally:
        registry.close()


def test_attempts_require_explicit_resume_and_current_control_revision(tmp_path):
    registry = store(tmp_path)
    try:
        first = started(registry)
        assert first.status == 'running' and first.attempt == first.revision == 1
        with pytest.raises(WorkflowError, match='sync_request_conflict'):
            registry.start_attempt('alpha', 'scan-1', expected_revision=0, resume=True)
        with pytest.raises(WorkflowError, match='sync_resume_required'):
            registry.start_attempt('alpha', 'scan-1', expected_revision=1)
        resumed = registry.start_attempt('alpha', 'scan-1', expected_revision=1, resume=True)
        assert resumed.attempt == 2 and resumed.revision == 2
    finally:
        registry.close()


def test_result_and_paged_source_outcomes_survive_reopen_without_raw_paths(tmp_path):
    registry = store(tmp_path)
    first = started(registry)
    finished = registry.finish('alpha', 'scan-1', expected_revision=first.revision, result=RESULT)
    assert finished.status == 'partial' and finished.outcome_count == 3
    assert finished.source_count == 2 and finished.issue_count == 1 and finished.skipped == 2
    registry.close()
    assert 'private-Ω.txt'.encode() not in (tmp_path / 'runs.sqlite').read_bytes()
    registry = store(tmp_path)
    try:
        page = registry.outcomes('alpha', 'scan-1', limit=2)
        assert [row.source.path for row in page.items] == ['private-Ω.txt', 'gone.txt']
        assert page.next_after == 1
        last = registry.outcomes('alpha', 'scan-1', limit=2, after=page.next_after)
        assert last.items[0].issue.code == 'unreadable' and last.next_after is None
        assert registry.get('alpha', 'scan-1') == finished
        with pytest.raises(WorkflowError, match='sync_not_found'):
            registry.outcomes('bravo', 'scan-1')
        with pytest.raises(WorkflowError, match='sync_result_terminal'):
            registry.start_attempt('alpha', 'scan-1', expected_revision=finished.revision, resume=True)
    finally:
        registry.close()


def test_failed_result_persistence_never_exposes_partial_receipts(tmp_path, monkeypatch):
    registry = store(tmp_path)
    try:
        first = started(registry)
        seal = registry._storage._seal
        def broken(token, payload):
            if token.endswith(':o:000001'):
                raise OSError('synthetic receipt write failure')
            return seal(token, payload)
        monkeypatch.setattr(registry._storage, '_seal', broken)
        with pytest.raises(OSError):
            registry.finish('alpha', 'scan-1', expected_revision=first.revision, result=RESULT)
        assert registry.get('alpha', 'scan-1') == first
        with pytest.raises(WorkflowError, match='sync_result_unavailable'):
            registry.outcomes('alpha', 'scan-1')
        monkeypatch.setattr(registry._storage, '_seal', seal)
        assert registry.finish('alpha', 'scan-1', expected_revision=first.revision, result=RESULT).outcome_count == 3
    finally:
        registry.close()


def test_cancel_intent_prevents_a_late_success_and_resume_is_explicit(tmp_path):
    registry = store(tmp_path)
    try:
        first = started(registry)
        cancelled = registry.request_cancel('alpha', 'scan-1', expected_revision=first.revision)
        assert cancelled.cancel_requested_at is not None and cancelled.status == 'running'
        with pytest.raises(WorkflowError, match='sync_request_conflict'):
            registry.finish('alpha', 'scan-1', expected_revision=first.revision, result=RESULT)
        with pytest.raises(WorkflowError, match='sync_cancelled'):
            registry.finish('alpha', 'scan-1', expected_revision=cancelled.revision, result=RESULT)
        stopped = registry.fail('alpha', 'scan-1', expected_revision=cancelled.revision, error_code='cancelled')
        assert stopped.status == 'cancelled'
        resumed = registry.start_attempt('alpha', 'scan-1', expected_revision=stopped.revision, resume=True)
        assert resumed.cancel_requested_at is None and resumed.attempt == 2
    finally:
        registry.close()


def test_two_registries_cannot_overwrite_a_newer_attempt(tmp_path):
    one, two = store(tmp_path), store(tmp_path)
    try:
        request = one.register('alpha', 'scan-1', SPEC)
        first = one.start_attempt('alpha', 'scan-1', expected_revision=request.revision)
        with pytest.raises(WorkflowError, match='sync_request_conflict'):
            two.start_attempt('alpha', 'scan-1', expected_revision=request.revision)
        assert two.get('alpha', 'scan-1') == first
    finally:
        one.close()
        two.close()


def test_history_cursor_is_space_bound_and_does_not_count_result_rows_as_runs(tmp_path):
    registry = store(tmp_path, max_runs=2)
    try:
        first = started(registry)
        registry.finish('alpha', 'scan-1', expected_revision=first.revision, result=RESULT)
        registry.register('alpha', 'scan-2', SPEC)
        page = registry.list('alpha', limit=1)
        second = registry.list('alpha', limit=1, after=page.next_after)
        assert {page.items[0].run_id, second.items[0].run_id} == {'scan-1', 'scan-2'}
        assert second.next_after is None
        with pytest.raises(WorkflowError):
            registry.list('bravo', after=page.next_after)
        with pytest.raises(WorkflowError, match='sync_store_limit'):
            registry.register('alpha', 'scan-3', SPEC)
    finally:
        registry.close()


def test_missing_outcome_is_not_a_short_successful_page(tmp_path):
    registry = store(tmp_path)
    try:
        first = started(registry)
        registry.finish('alpha', 'scan-1', expected_revision=first.revision, result=RESULT)
        token = registry._token('alpha', 'scan-1') + ':o:000001'
        with registry._storage._access(write=True) as db:
            db.execute('DELETE FROM directory_runs WHERE token=?', (token,))
        with pytest.raises(WorkflowError, match='sync_key_or_integrity'):
            registry.outcomes('alpha', 'scan-1', limit=2)
    finally:
        registry.close()


def test_wrong_key_and_row_substitution_are_refused(tmp_path):
    registry = store(tmp_path)
    try:
        registry.register('alpha', 'scan-1', SPEC)
        registry.register('alpha', 'scan-2', SPEC)
        with pytest.raises(WorkflowError, match='sync_key_or_integrity'):
            DirectoryRunStore(tmp_path / 'runs.sqlite', key=b'x' * 32)
        with registry._storage._access(write=True) as db:
            db.execute('UPDATE directory_runs SET payload=(SELECT payload FROM directory_runs WHERE token=?) WHERE token=?',
                (registry._token('alpha', 'scan-1'), registry._token('alpha', 'scan-2')))
        with pytest.raises(WorkflowError, match='sync_key_or_integrity'):
            registry.get('alpha', 'scan-2')
    finally:
        registry.close()


def test_empty_complete_result_and_attempt_budget(tmp_path):
    registry = store(tmp_path)
    try:
        first = started(registry)
        done = registry.finish('alpha', 'scan-1', expected_revision=first.revision,
                               result=replace(RESULT, complete=True, receipts=(), issues=(), skipped=0))
        assert done.status == 'completed' and registry.outcomes('alpha', 'scan-1').items == ()
        request = registry.register('alpha', 'scan-2', SPEC.model_copy(update={'max_attempts': 1}))
        request = registry.start_attempt('alpha', 'scan-2', expected_revision=request.revision)
        with pytest.raises(WorkflowError, match='sync_attempt_limit'):
            registry.start_attempt('alpha', 'scan-2', expected_revision=request.revision, resume=True)
    finally:
        registry.close()


def test_cancelled_registration_requires_explicit_resume_before_first_scan(tmp_path):
    registry = store(tmp_path)
    try:
        request = registry.register('alpha', 'scan-1', SPEC)
        cancelled = registry.request_cancel('alpha', 'scan-1', expected_revision=request.revision)
        with pytest.raises(WorkflowError, match='sync_cancelled'):
            registry.start_attempt('alpha', 'scan-1', expected_revision=cancelled.revision)
        assert registry.start_attempt('alpha', 'scan-1', expected_revision=cancelled.revision, resume=True).attempt == 1
    finally:
        registry.close()


@pytest.mark.parametrize('receipts', [
    (SourceReceipt('missing-id.txt', 'added'),),
    (SourceReceipt('../outside.txt', 'added', episode_id=12),),
    (SourceReceipt('same.txt', 'added', episode_id=12), SourceReceipt('same.txt', 'deleted', previous_episode_id=12)),
    (SourceReceipt('same.txt', 'updated', episode_id=12, previous_episode_id=12),),
    (SourceReceipt('old.txt', 'deleted'),),
])
def test_invalid_source_outcomes_cannot_publish_a_completed_result(tmp_path, receipts):
    registry = store(tmp_path)
    try:
        first = started(registry)
        with pytest.raises((ValueError, WorkflowError)):
            registry.finish('alpha', 'scan-1', expected_revision=first.revision,
                            result=replace(RESULT, complete=True, receipts=receipts, issues=()))
        assert registry.get('alpha', 'scan-1') == first
    finally:
        registry.close()


def test_invalid_unicode_scan_issue_is_retained_as_an_explicit_escaped_diagnostic(tmp_path):
    import json
    registry = store(tmp_path)
    try:
        first = started(registry)
        issue = ScanIssue('broken-\udcff.txt', 'unsafe_path')
        done = registry.finish('alpha', 'scan-1', expected_revision=first.revision,
                               result=replace(RESULT, receipts=(), issues=(issue,)))
        assert done.status == 'partial'
        saved = registry.outcomes('alpha', 'scan-1').items[0].issue
        assert saved.path_escaped and json.loads(saved.path) == issue.path
        assert saved.code == 'unsafe_path'
    finally:
        registry.close()


def test_real_invalid_byte_filename_remains_a_persistable_scan_issue(tmp_path):
    import errno
    import json
    import os
    from scone_memory.ingestion.source_scan import DirectoryScanner
    root = tmp_path / 'sources'
    root.mkdir()
    try:
        descriptor = os.open(os.fsencode(root) + b'/broken-\xff.txt', os.O_CREAT | os.O_WRONLY, 0o600)
    except OSError as error:
        if error.errno in (errno.EILSEQ, errno.EINVAL):
            pytest.skip('this filesystem refuses non-Unicode filenames')
        raise
    os.close(descriptor)
    snapshot = DirectoryScanner(root).scan()
    assert snapshot.issues and not snapshot.files
    registry = store(tmp_path)
    try:
        first = started(registry)
        registry.finish('alpha', 'scan-1', expected_revision=first.revision,
                        result=replace(RESULT, receipts=(), issues=snapshot.issues))
        issue = registry.outcomes('alpha', 'scan-1').items[0].issue
        assert issue.path_escaped and json.loads(issue.path) == snapshot.issues[0].path
    finally:
        registry.close()


from .test_directory_sync import env, runner  # noqa: F401


async def test_native_source_lifecycle_produces_persistent_historical_results(env):
    from scone_memory.core.errors import Gone
    memory, root, temporary = env
    registry = store(temporary)
    sync = runner(env)
    first_episode = None
    try:
        for number, expected in enumerate(('added', 'updated', 'unchanged', 'deleted')):
            if number in (0, 1):
                (root / 'report.txt').write_text(f'Source revision {number}')
            elif number == 3:
                (root / 'report.txt').unlink()
            run_id = f'run-{number}'
            spec = SPEC.model_copy(update={'delete_missing': number == 3})
            request = registry.register('alpha', run_id, spec)
            request = registry.start_attempt('alpha', run_id, expected_revision=request.revision)
            result = await sync.synchronize(delete_missing=spec.delete_missing)
            done = registry.finish('alpha', run_id, expected_revision=request.revision, result=result)
            assert done.status == 'completed'
            receipt = registry.outcomes('alpha', run_id).items[0].source
            assert receipt.status == expected
            if first_episode is None:
                first_episode = receipt.episode_id
        with pytest.raises(Gone):
            await memory.episode('alpha', first_episode)
        assert registry.outcomes('alpha', 'run-0').items[0].source.status == 'added'
    finally:
        registry.close()
