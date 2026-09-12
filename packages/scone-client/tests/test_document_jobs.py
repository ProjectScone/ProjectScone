"""Durable import identity, explicit CAS control, and original-backed results."""
import pytest
from scone import Scone, SconeError
from scone.document_models import DocumentRequest, DocumentStatus, PdfOcr
from stub_server import StubScone

LIMITS = {'max_input_bytes': 26214400, 'max_text_bytes': 2000000, 'max_segments': 20000,
          'max_archive_entries': 10000, 'max_archive_bytes': 100000000, 'timeout_seconds': 30.0}
SPEC = {'attachment_id': 'a'*64, 'filename': 'report.pdf', 'parser_revision': 'parser-1',
        'limits': LIMITS, 'pdf_ocr': None, 'deadline_s': 120.0, 'max_attempts': 3}
REQUEST = {'space': 'alpha', 'import_id': 'one', 'spec': SPEC, 'created_at': '2026-09-12T10:00:00Z',
           'revision': 1, 'attempt': 1, 'last_started_at': '2026-09-12T10:00:01Z', 'cancel_requested_at': None}
STATUS = {'space': 'alpha', 'import_id': 'one', 'filename': SPEC['filename'], 'attachment_id': SPEC['attachment_id'],
          'created_at': '2026-09-12T10:00:00+00:00', 'revision': 1, 'attempt': 1, 'max_attempts': 3,
          'status': 'interrupted', 'active_local': False, 'completed_steps': ['extract'],
          'inflight': None, 'outcome_unknown': True, 'error_class': None}
RESULT = {'space': 'alpha', 'import_id': 'one', 'filename': 'report.pdf', 'format': 'pdf', 'segments': 2,
          'original': {'attachment_id': 'a'*64, 'media_type': 'application/pdf', 'bytes': 100, 'filename': 'first.pdf'},
          'manifest': {'attachment_id': 'b'*64, 'media_type': 'application/json', 'bytes': 500, 'filename': 'manifest.json'},
          'added': {'episode_id': 1, 'deduplicated': False, 'chunks': 2, 'outcome': 'accepted', 'replaced': None, 'reason': None}}


@pytest.fixture
def fixture():
    with StubScone() as server, Scone(server.base_url, 'key') as client:
        server.route('GET', '/v1/capabilities', 200, {'schema_version': 1, 'implementation': 'python',
            'features': {'documents.jobs': True, 'documents.files': True, 'episodes.attachments': True}})
        server.route('GET', '/v1/status', 200, {'space': 'alpha'})
        server.route('GET', '/v1/document-jobs/one/request', 200, REQUEST)
        server.route('GET', '/v1/document-jobs/one', 200, STATUS)
        yield server, client.document_jobs(expected_space='alpha')


def test_start_does_not_implicitly_resume_existing_import(fixture):
    server, jobs = fixture
    server.route('POST', '/v1/document-jobs', 202, STATUS)
    assert jobs.start('one', attachment_id='a'*64, filename='report.pdf').status == 'interrupted'
    posts = [r for r in server.requests if r.method == 'POST']
    assert len(posts) == 1 and posts[0].path == '/v1/document-jobs'


def test_start_refuses_substituted_ocr_request(fixture):
    server, jobs = fixture
    server.route('POST', '/v1/document-jobs', 202, STATUS)
    with pytest.raises(SconeError, match='acknowledgement'):
        jobs.start('one', attachment_id='a'*64, filename='report.pdf', pdf_ocr=PdfOcr('all_pages', 'columns_ltr'))


def test_resume_unknown_document_attempt_is_explicit_and_uses_exact_revision(fixture):
    server, jobs = fixture
    before = jobs.request('one')
    resumed = {**STATUS, 'status': 'running', 'active_local': True, 'revision': 2, 'attempt': 2, 'outcome_unknown': False}
    server.route('POST', '/v1/document-jobs/one/resume', 202, resumed)
    del server.routes[('GET', '/v1/document-jobs/one/request')]
    server.enqueue(200, REQUEST)
    server.enqueue(200, {**REQUEST, 'revision': 2, 'attempt': 2})
    assert jobs.resume(before).attempt == 2
    posts = [r for r in server.requests if r.method == 'POST']
    assert len(posts) == 1 and posts[0].json == {'expected_revision': 1}
    server.route('GET', '/v1/document-jobs/one/request', 200, REQUEST)
    server.route('POST', '/v1/document-jobs/one/resume', 503, {'error': 'unavailable'})
    with pytest.raises(SconeError):
        jobs.resume(before)
    assert len([r for r in server.requests if r.method == 'POST']) == 2


def test_stale_or_replaced_request_is_rejected_before_control(fixture):
    server, jobs = fixture
    before = jobs.request('one')
    for changes in ({'revision': 2}, {'spec': {**SPEC, 'filename': 'changed.pdf'}}):
        server.route('GET', '/v1/document-jobs/one/request', 200, {**REQUEST, **changes})
        with pytest.raises(SconeError):
            jobs.cancel(before)
    assert all(r.method == 'GET' for r in server.requests)


def test_result_is_readable_without_resume_at_attempt_ceiling(fixture):
    server, jobs = fixture
    server.route('GET', '/v1/document-jobs/one/request', 200, {**REQUEST, 'attempt': 3, 'revision': 3})
    server.route('GET', '/v1/document-jobs/one', 200, {**STATUS, 'status': 'verification_unavailable', 'attempt': 3, 'revision': 3})
    server.route('GET', '/v1/document-jobs/one/result', 200, RESULT)
    result = jobs.result('one')
    assert result.original.filename == 'first.pdf' and result.filename == 'report.pdf'
    assert all(r.method == 'GET' for r in server.requests)


@pytest.mark.parametrize('change', [{'filename': 'other.pdf'}, {'space': 'beta'}, {'import_id': 'two'},
    {'pdf_ocr': {'mode': 'all_pages', 'reading_order': 'provider'}}, {'segments': True},
    {'original': {**RESULT['original'], 'attachment_id': 'c'*64}},
    {'manifest': {**RESULT['manifest'], 'media_type': 'text/plain'}}])
def test_result_refuses_malformed_or_unbound_receipts(fixture, change):
    server, jobs = fixture
    server.route('GET', '/v1/document-jobs/one/result', 200, {**RESULT, **change})
    with pytest.raises(SconeError):
        jobs.result('one')


@pytest.mark.parametrize('change', [{'revision': True}, {'revision': 0}, {'attempt': 2**31},
    {'completed_steps': ['index']}, {'completed_steps': ['extract'], 'inflight': 'extract'},
    {'status': 'completed', 'completed_steps': ['extract']}, {'active_local': 'false'}])
def test_status_rejects_invalid_revision_and_stage_shapes(change):
    with pytest.raises(SconeError):
        DocumentStatus.from_json({**STATUS, **change}, expected_space='alpha')


def test_invalid_ocr_and_filename_fail_before_io(fixture):
    server, jobs = fixture
    for name in ('x\0.pdf', 'é'*511+'.pdf', '\ud800.pdf'):
        with pytest.raises(SconeError):
            jobs.start('one', attachment_id='a'*64, filename=name)
    with pytest.raises(SconeError):
        jobs.start('one', attachment_id='a'*64, filename='data.csv', pdf_ocr=PdfOcr('all_pages', 'provider'))
    assert not server.requests


def test_formats_keep_ocr_availability_separate_from_jobs(fixture):
    server, jobs = fixture
    server.route('GET', '/v1/documents/formats', 200, {'max_input_bytes': 1024,
        'formats': {'.pdf': {'available': True, 'parser': 'pdf-text', 'requires': 'pdf extra'}},
        'pdf_ocr': {'available': False, 'modes': ['missing_text', 'all_pages'],
                    'reading_orders': ['provider', 'columns_ltr', 'columns_rtl']}})
    formats = jobs.formats()
    assert formats.formats['.pdf'].available and not formats.pdf_ocr_available
    assert all(r.method == 'GET' for r in server.requests)


def test_upload_preserves_bytes_and_checks_digest_before_start(fixture):
    import hashlib
    server, jobs = fixture
    content = 'A Unicode document: é'.encode()
    expected = hashlib.sha256(content).hexdigest()
    server.route('POST', '/v1/attachments', 200, {'attachment_id': expected, 'bytes': len(content),
        'media_type': 'text/plain', 'filename': 'first-upload.txt'})
    stored = jobs.upload(content, media_type='text/plain')
    assert stored.attachment_id == expected
    posts = [r for r in server.requests if r.method == 'POST']
    assert len(posts) == 1 and posts[0].body == content and posts[0].headers['content-type'] == 'text/plain'
    server.route('POST', '/v1/attachments', 200, {**RESULT['original'], 'bytes': len(content)})
    with pytest.raises(SconeError, match='upload acknowledgement'):
        jobs.upload(content, media_type='text/plain')
    assert len([r for r in server.requests if r.method == 'POST']) == 2


def test_upload_rejects_header_injection_and_empty_content_locally(fixture):
    server, jobs = fixture
    for content, media in ((b'', 'text/plain'), (b'x', 'text/plain\r\nX-Test: value'), ('text', 'text/plain')):
        with pytest.raises(SconeError):
            jobs.upload(content, media_type=media)
    assert not server.requests
