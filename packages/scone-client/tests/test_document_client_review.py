"""Independent acknowledgement checks for durable document controls."""
import pytest
from scone import SconeError
from test_document_jobs import fixture, REQUEST, SPEC, STATUS


@pytest.mark.parametrize('change', [
    {'parser_revision': 'substituted-parser'},
    {'pdf_ocr': {'mode': 'all_pages', 'reading_order': 'provider'}},
])
def test_control_rechecks_immutable_request_after_post(fixture, change):
    server, jobs = fixture
    before = jobs.request('one')
    del server.routes[('GET', '/v1/document-jobs/one/request')]
    server.enqueue(200, REQUEST)
    server.enqueue(200, {**REQUEST, 'spec': {**SPEC, **change}, 'revision': 2, 'attempt': 2})
    server.route('POST', '/v1/document-jobs/one/resume', 202,
        {**STATUS, 'revision': 2, 'attempt': 2, 'status': 'running', 'active_local': True, 'outcome_unknown': False})
    with pytest.raises(SconeError):
        jobs.resume(before)
    assert sum(row.method == 'POST' for row in server.requests) == 1


@pytest.mark.parametrize('state', ['registered', 'sources_invalid', 'cancelled'])
def test_resume_refuses_nonadmitted_acknowledgement(fixture, state):
    server, jobs = fixture
    before = jobs.request('one')
    server.route('POST', '/v1/document-jobs/one/resume', 202,
        {**STATUS, 'revision': 2, 'attempt': 2, 'status': state, 'active_local': False})
    with pytest.raises(SconeError):
        jobs.resume(before)


@pytest.mark.parametrize('operation', ['resume', 'cancel'])
def test_completed_control_is_acknowledged_without_new_attempt(fixture, operation):
    server, jobs = fixture
    before = jobs.request('one')
    completed = {**STATUS, 'status': 'completed', 'completed_steps': ['extract', 'index'],
                 'inflight': None, 'outcome_unknown': False}
    server.route('POST', '/v1/document-jobs/one/' + operation, 200, completed)
    result = getattr(jobs, operation)(before)
    assert result.status == 'completed' and result.attempt == before.attempt
    assert sum(row.method == 'POST' for row in server.requests) == 1
