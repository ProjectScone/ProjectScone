"""Typed directory operations bind intent, scope, revisions and historical pages."""
import pytest
from scone import Scone, SconeError, SyncCollection, SyncStatus
from stub_server import StubScone

STAMP = '2026-09-12T12:00:00+00:00'
COLLECTION = {'collection_id': 'notes', 'label': 'Local notes', 'allow_delete_missing': True, 'configuration': 'a'*64}
RECORD = {'space': 'alpha', 'run_id': 'scan', 'spec': {'collection_id': 'notes', 'configuration': 'a'*64,
    'delete_missing': False, 'deadline_s': 300.0, 'max_attempts': 3}, 'created_at': STAMP, 'revision': 1,
    'attempt': 1, 'status': 'running', 'last_started_at': STAMP, 'cancel_requested_at': None,
    'finished_at': None, 'error_code': None, 'collection_instance': None,
    'source_count': 0, 'issue_count': 0, 'outcome_count': 0, 'skipped': 0}
STATUS = {'record': RECORD, 'status': 'running', 'active_local': True, 'active_elsewhere': False, 'outcome_unknown': False}


@pytest.fixture
def fixture():
    with StubScone() as server, Scone(server.base_url, 'key') as client:
        server.route('GET', '/v1/capabilities', 200, {'schema_version': 1, 'implementation': 'python', 'features': {'documents.sync': True}})
        server.route('GET', '/v1/status', 200, {'space': 'alpha'})
        server.route('GET', '/v1/sync-collections', 200, {'items': [COLLECTION]})
        server.route('GET', '/v1/sync-runs/scan', 200, STATUS)
        yield server, client.directory_sync(expected_space='alpha')


def complete():
    return {**STATUS, 'active_local': False, 'status': 'partial', 'record': {**RECORD, 'status': 'partial',
        'finished_at': STAMP, 'collection_instance': 'b'*32, 'source_count': 1, 'issue_count': 1, 'outcome_count': 2}}


def test_discovery_and_start_preserve_selected_configuration_without_implicit_resume(fixture):
    server, sync = fixture
    collection = sync.collections()[0]
    idle = {**STATUS, 'status': 'interrupted', 'active_local': False, 'outcome_unknown': True}
    server.route('POST', '/v1/sync-runs', 202, idle)
    assert sync.start('scan', collection=collection).status == 'interrupted'
    writes = [request for request in server.requests if request.method == 'POST']
    assert len(writes) == 1 and writes[0].json == {'run_id': 'scan', 'collection_id': 'notes',
        'delete_missing': False, 'expected_configuration': 'a'*64}


@pytest.mark.parametrize('change', [{'space': 'beta'}, {'run_id': 'other'}, {'revision': 0}, {'attempt': True},
    {'outcome_count': 1}, {'finished_at': STAMP}, {'last_started_at': None}, {'collection_instance': 'b'*32}])
def test_status_rejects_inconsistent_control_records(change):
    with pytest.raises(SconeError):
        SyncStatus.from_json({**STATUS, 'record': {**RECORD, **change}}, expected_space='alpha', run_id='scan')


def test_unsupported_or_foreign_space_refuses_before_writes(fixture):
    server, sync = fixture
    collection = SyncCollection.from_json(COLLECTION)
    server.route('GET', '/v1/status', 200, {'space': 'beta'})
    with pytest.raises(SconeError):
        sync.start('scan', collection=collection)
    server.route('GET', '/v1/capabilities', 200, {'schema_version': 1, 'implementation': 'rust', 'features': {}})
    with pytest.raises(SconeError):
        sync.collections()
    assert all(request.method == 'GET' for request in server.requests)


def test_cancel_and_resume_require_actual_control_transitions(fixture):
    server, sync = fixture
    prior = sync.status('scan')
    server.route('POST', '/v1/sync-runs/scan/cancel', 202, STATUS)
    with pytest.raises(SconeError):
        sync.cancel(prior)
    cancelled = {**STATUS, 'record': {**RECORD, 'revision': 2, 'cancel_requested_at': STAMP}}
    server.route('POST', '/v1/sync-runs/scan/cancel', 202, cancelled)
    assert sync.cancel(prior).record.cancel_requested_at == STAMP
    idle = {**STATUS, 'active_local': False, 'status': 'interrupted', 'outcome_unknown': True}
    server.route('GET', '/v1/sync-runs/scan', 200, idle)
    prior = sync.status('scan')
    server.route('POST', '/v1/sync-runs/scan/resume', 202, idle)
    with pytest.raises(SconeError):
        sync.resume(prior)
    server.route('POST', '/v1/sync-runs/scan/resume', 202, {**STATUS, 'record': {**RECORD, 'revision': 2, 'attempt': 2}})
    assert sync.resume(prior).record.attempt == 2
    assert all(request.json == {'expected_revision': 1} for request in server.requests if request.method == 'POST')


def test_results_bind_envelope_and_preserve_diagnostic_paths_and_large_ids(fixture):
    server, sync = fixture
    server.route('GET', '/v1/sync-runs/scan', 200, complete())
    source = {'index': 0, 'source': {'path': '\ufeffnotes/😀.txt', 'status': 'added', 'episode_id': 2**62,
                                  'previous_episode_id': None, 'code': None}, 'issue': None}
    issue = {'index': 1, 'source': None, 'issue': {'path': '"bad\\udcff"', 'path_escaped': True, 'code': 'invalid_path'}}
    body = {'space': 'alpha', 'run_id': 'scan', 'items': [source, issue], 'next_after': None}
    server.route('GET', '/v1/sync-runs/scan/result', 200, body)
    result = sync.results('scan')
    assert result.items[0].source.episode_id == 2**62
    assert result.items[1].issue.path_escaped
    for change in ({'space': 'beta'}, {'run_id': 'other'}, {'items': [issue, source]}, {'items': [source]}, {'next_after': 1}):
        server.route('GET', '/v1/sync-runs/scan/result', 200, {**body, **change})
        with pytest.raises(SconeError):
            sync.results('scan')


def test_history_rejects_duplicates_and_bad_cursors(fixture):
    server, sync = fixture
    server.route('GET', '/v1/sync-runs', 200, {'items': [STATUS], 'next_after': None})
    assert sync.list().items[0].record.run_id == 'scan'
    server.route('GET', '/v1/sync-runs', 200, {'items': [STATUS, STATUS], 'next_after': None})
    with pytest.raises(SconeError):
        sync.list()
    before = len(server.requests)
    for kwargs in ({'limit': True}, {'after': 'bad'}, {'limit': 101}):
        with pytest.raises(SconeError):
            sync.list(**kwargs)
    assert len(server.requests) == before


def test_invalid_start_and_labels_do_not_silently_coerce(fixture):
    server, sync = fixture
    collection = SyncCollection.from_json({**COLLECTION, 'label': 'Notes\ud800'})
    assert collection.label == 'Notes\ud800'
    for kwargs in ({'run_id': '../bad'}, {'run_id': 'scan', 'delete_missing': 1}, {'run_id': '.', 'delete_missing': False}):
        with pytest.raises(SconeError):
            sync.start(collection=collection, **kwargs)
    assert not server.requests


@pytest.mark.parametrize('deadline', [True, 0, 3601, float('inf'), float('nan'), 10**10000], ids=['boolean', 'zero', 'over-limit', 'infinite', 'nan', 'huge'])
def test_invalid_deadlines_fail_as_client_errors(deadline):
    with pytest.raises(SconeError):
        SyncStatus.from_json({**STATUS, 'record': {**RECORD, 'spec': {**RECORD['spec'], 'deadline_s': deadline}}}, expected_space='alpha')
