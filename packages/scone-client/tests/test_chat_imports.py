"""The chat-import client sends a bounded request, checks the acknowledgement, and refuses what the server would."""
import hashlib

import pytest

from scone import Scone, SconeError
from stub_server import StubScone

EXPORT = b"13/03/2024, 14:05 - Alice: The harbour closes in November\n"
ATTACHMENT = hashlib.sha256(EXPORT).hexdigest()
RECEIPT = {'platform': 'whatsapp', 'chat': 'Harbour crew', 'filename': 'Harbour crew.txt', 'attachment_id': ATTACHMENT,
           'messages': 1, 'stored': 1, 'duplicates': 0, 'failed': 0, 'episodes': 1, 'sessions': 1, 'session_gap_seconds': 21600,
           'speakers': 1, 'first_at': '2024-03-13T14:05:00.000Z', 'last_at': '2024-03-13T14:05:00.000Z', 'messages_unread': 0,
           'system_messages': 0, 'unparsed_lines': 0, 'media_only_messages': 0, 'attachments_skipped': 0,
           'date_order': 'day-first', 'date_order_told': False, 'time_zone': 'UTC (assumed)', 'failed_reason': None}


@pytest.fixture
def fixture():
    with StubScone() as server, Scone(server.base_url, 'key') as client:
        server.route('GET', '/v1/capabilities', 200, {'schema_version': 1, 'implementation': 'python',
            'features': {'chats.imports': True, 'documents.files': True, 'episodes.attachments': True}})
        server.route('GET', '/v1/status', 200, {'space': 'alpha'})
        yield server, client.chat_imports(expected_space='alpha')


def test_upload_then_import_sends_a_bounded_request_and_reads_the_receipt(fixture):
    server, chats = fixture
    server.route('POST', '/v1/attachments', 200, {'attachment_id': ATTACHMENT, 'bytes': len(EXPORT), 'media_type': 'application/octet-stream',
                                                   'filename': 'Harbour crew.txt', 'created_at': '2024-03-13T14:05:00.000Z'})
    server.route('POST', '/v1/chat-imports', 200, RECEIPT)
    stored = chats.upload(EXPORT, filename='Harbour crew.txt')
    posted = next(row for row in server.requests if row.method == 'POST' and row.path == '/v1/attachments')
    assert posted.headers.get('x-filename', posted.headers.get('X-Filename')) == 'Harbour crew.txt', "the name travels with the bytes"
    receipt = chats.import_export(stored.attachment_id, time_zone='Europe/London', gap_seconds=3600, metadata={'user_id': 'mark'})
    assert receipt.stored == 1 and receipt.chat == 'Harbour crew' and receipt.date_order == 'day-first' and not receipt.date_order_told
    sent = next(row for row in server.requests if row.method == 'POST' and row.path == '/v1/chat-imports').json
    assert sent == {'attachment_id': ATTACHMENT, 'time_zone': 'Europe/London', 'gap_seconds': 3600, 'metadata': {'user_id': 'mark'}}


def test_the_client_refuses_bad_arguments_before_sending_and_bad_receipts_after(fixture):
    server, chats = fixture
    for kwargs in ({'date_order': 'sideways'}, {'gap_seconds': -1}, {'gap_seconds': 36 * 3600 + 1}, {'gap_seconds': True},
                   {'metadata': {'a': 1}}, {'metadata': {f'k{i}': 'v' for i in range(10)}}):
        with pytest.raises(SconeError):
            chats.import_export(ATTACHMENT, **kwargs)
    assert not any(row.path == '/v1/chat-imports' for row in server.requests)
    with pytest.raises(SconeError):
        chats.import_export('not a digest')
    server.route('POST', '/v1/chat-imports', 200, {**RECEIPT, 'stored': 5})
    with pytest.raises(SconeError, match='counts'):
        chats.import_export(ATTACHMENT)
    server.route('POST', '/v1/chat-imports', 200, {**RECEIPT, 'attachment_id': 'f' * 64})
    with pytest.raises(SconeError, match='acknowledgement'):
        chats.import_export(ATTACHMENT)
    server.route('POST', '/v1/chat-imports', 200, {**RECEIPT, 'platform': 'irc'})
    with pytest.raises(SconeError, match='platform'):
        chats.import_export(ATTACHMENT)


def test_a_server_without_the_capability_is_not_asked(fixture):
    server, chats = fixture
    server.route('GET', '/v1/capabilities', 200, {'schema_version': 1, 'implementation': 'python', 'features': {'documents.files': True}})
    with pytest.raises(SconeError):
        chats.import_export(ATTACHMENT)
    assert not any(row.path == '/v1/chat-imports' for row in server.requests)
