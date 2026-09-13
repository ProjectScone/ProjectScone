"""Video extraction choices survive acknowledgements and immutable controls."""
import pytest

from scone import SconeError
from scone.document_models import DocumentFormats
from test_document_jobs import REQUEST, RESULT, SPEC, STATUS, fixture


def selected(server, *, retained=True):
    request = {**REQUEST, 'spec': {**SPEC, 'filename': 'slides.mp4', 'video_ocr': retained}}
    server.route('GET', '/v1/document-jobs/one/request', 200, request)
    server.route('POST', '/v1/document-jobs', 202, {**STATUS, 'filename': 'slides.mp4'})
    return request


def test_start_sends_and_retains_video_choice(fixture):
    server, jobs = fixture
    selected(server)
    jobs.start('one', attachment_id='a' * 64, filename='slides.mp4', video_ocr=True)
    assert jobs.request('one').spec.video_ocr is True
    posts = [request for request in server.requests if request.method == 'POST']
    assert posts[0].json['video_ocr'] is True


def test_start_refuses_silently_ignored_video_choice(fixture):
    server, jobs = fixture
    selected(server, retained=False)
    with pytest.raises(SconeError, match='acknowledgement'):
        jobs.start('one', attachment_id='a' * 64, filename='slides.mp4', video_ocr=True)


def test_result_must_match_video_choice(fixture):
    server, jobs = fixture
    selected(server)
    result = {**RESULT, 'filename': 'slides.mp4', 'format': 'mp4', 'video_ocr': True}
    server.route('GET', '/v1/document-jobs/one/result', 200, result)
    assert jobs.result('one').video_ocr is True
    server.route('GET', '/v1/document-jobs/one/result', 200, {**result, 'video_ocr': False})
    with pytest.raises(SconeError, match='binding'):
        jobs.result('one')


@pytest.mark.parametrize('choice,name', [(1, 'slides.mp4'), ('true', 'slides.mp4'), (None, 'slides.mp4'),
                                        (True, 'source.ts'), (True, 'audio.wav')])
def test_invalid_video_choice_refuses_before_io(fixture, choice, name):
    server, jobs = fixture
    with pytest.raises(SconeError):
        jobs.start('one', attachment_id='a' * 64, filename=name, video_ocr=choice)
    assert not server.requests


def test_discovery_keeps_video_and_audio_separate_and_supports_older_hosts():
    body = {'formats': {}, 'max_input_bytes': 1024,
        'pdf_ocr': {'available': False, 'modes': ['all_pages'], 'reading_orders': ['provider']}}
    assert DocumentFormats.from_json(body).video_ocr_available is False
    formats = DocumentFormats.from_json({**body, 'video_ocr': {'available': True,
        'selection': {'video_ocr': True}, 'extensions': ['.mp4'],
        'extraction': 'sampled-frame-text', 'includes_audio': False}})
    assert formats.video_ocr_available is True and formats.video_ocr_extensions == ('.mp4',)


def test_visual_only_result_requires_explicit_video_choice_and_zero_chunks(fixture):
    server, jobs = fixture
    selected(server)
    result = {**RESULT, 'filename': 'slides.mp4', 'format': 'mp4', 'video_ocr': True,
              'segments': 0, 'added': {**RESULT['added'], 'chunks': 0}}
    server.route('GET', '/v1/document-jobs/one/result', 200, result)
    assert jobs.result('one').segments == 0
    server.route('GET', '/v1/document-jobs/one/result', 200,
                 {**result, 'added': {**result['added'], 'chunks': 1}})
    with pytest.raises(SconeError):
        jobs.result('one')
    selected(server, retained=False)
    server.route('GET', '/v1/document-jobs/one/result', 200, {**result, 'video_ocr': False})
    with pytest.raises(SconeError):
        jobs.result('one')
