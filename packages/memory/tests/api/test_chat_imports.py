"""An uploaded chat export becomes conversation memories over HTTP, with the same receipt the command prints."""
import httpx
import pytest

from scone_memory import HashEmbedder, InMemoryDocumentStore, InMemoryVectorIndex, MemoryEngine
from scone_memory.api import create_app

WHATSAPP = ("13/03/2024, 14:05 - Messages and calls are end-to-end encrypted.\n"
            "13/03/2024, 14:05 - Alice: The harbour closes in November\n"
            "13/03/2024, 14:06 - Bob: <Media omitted>\n"
            "14/03/2024, 09:03 - Bob: Noted, moving the boat\n").encode()
QUIET = b"01/02/2024, 10:00 - Alice: hi\n"


@pytest.fixture
async def service():
    engine = await MemoryEngine(InMemoryDocumentStore(), InMemoryVectorIndex(), HashEmbedder()).open()
    app = create_app(engine, {'write': 'alpha', 'read': 'alpha'}, roles={'read': 'read'})
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url='http://test',
                                 headers={'authorization': 'Bearer write'}) as client:
        yield client, engine
    await engine.close()


async def upload(client, raw, filename):
    stored = await client.post('/v1/attachments', content=raw, headers={'content-type': 'application/octet-stream', 'x-filename': filename})
    assert stored.status_code == 200, stored.text
    return stored.json()['attachment_id']


async def test_an_uploaded_export_is_imported_with_the_receipt_and_a_second_import_adds_nothing(service):
    client, engine = service
    assert (await client.get('/v1/capabilities')).json()['features']['chats.imports'] is True
    attachment_id = await upload(client, WHATSAPP, 'Harbour crew.txt')
    imported = await client.post('/v1/chat-imports', json={'attachment_id': attachment_id, 'time_zone': 'Europe/London',
                                                           'metadata': {'user_id': 'mark'}})
    assert imported.status_code == 200, imported.text
    receipt = imported.json()
    assert receipt['platform'] == 'whatsapp' and receipt['chat'] == 'Harbour crew' and receipt['filename'] == 'Harbour crew.txt'
    assert (receipt['stored'], receipt['messages'], receipt['sessions'], receipt['episodes']) == (2, 2, 2, 2)
    assert receipt['system_messages'] == 1 and receipt['media_only_messages'] == 1 and receipt['time_zone'] == 'Europe/London'
    assert receipt['first_at'] == '2024-03-13T14:05:00.000Z' and receipt['attachment_id'] == attachment_id and 'episode_ids' not in receipt
    recalled = await client.get('/v1/recall', params={'q': 'harbour closing'})
    assert recalled.status_code == 200, recalled.text
    items = recalled.json()['items']
    hit = next((item for item in items if 'harbour closes' in item['text'].lower()), None)
    assert hit is not None, items
    episode = (await client.get(f"/v1/episodes/{hit['episode_id']}")).json()
    assert episode['kind'] == 'conversation' and episode['metadata']['user_id'] == 'mark' and episode['metadata']['platform'] == 'whatsapp'
    again = await client.post('/v1/chat-imports', json={'attachment_id': attachment_id, 'time_zone': 'Europe/London'})
    assert again.status_code == 200 and (again.json()['stored'], again.json()['duplicates']) == (0, 2)


async def test_the_route_refuses_what_it_cannot_import_and_says_why(service):
    client, engine = service
    quiet = await upload(client, QUIET, 'quiet.txt')
    undecided = await client.post('/v1/chat-imports', json={'attachment_id': quiet})
    assert undecided.status_code == 422 and 'day-first or month-first' in undecided.text, "a refusal from the reader is 422, as every InvalidInput is"
    told = await client.post('/v1/chat-imports', json={'attachment_id': quiet, 'date_order': 'month-first'})
    assert told.status_code == 200 and told.json()['date_order_told'] is True and told.json()['first_at'] == '2024-01-02T10:00:00.000Z'
    assert (await client.post('/v1/chat-imports', json={'attachment_id': quiet, 'gap_seconds': -1})).status_code == 400
    assert (await client.post('/v1/chat-imports', json={'attachment_id': quiet, 'metadata': {'speaker': 'me'}})).status_code == 422
    assert (await client.post('/v1/chat-imports', json={'attachment_id': 'x' * 64})).status_code == 400
    assert (await client.post('/v1/chat-imports', content=b'{"attachment_id": "' + b'a' * 9000 + b'"}',
                              headers={'content-type': 'application/json'})).status_code == 413
    table = await upload(client, b'a,b\n1,2\n', 'table.csv')
    refused = await client.post('/v1/chat-imports', json={'attachment_id': table})
    assert refused.status_code == 422 and 'must be a WhatsApp' in refused.text
    unnamed = await client.post('/v1/attachments', content=WHATSAPP, headers={'content-type': 'application/octet-stream'})
    nameless = await client.post('/v1/chat-imports', json={'attachment_id': unnamed.json()['attachment_id']})
    assert nameless.status_code == 422 and 'filename' in nameless.text, "an upload without a name needs one in the request"
    named = await client.post('/v1/chat-imports', json={'attachment_id': unnamed.json()['attachment_id'], 'filename': 'crew.txt'})
    assert named.status_code == 200 and named.json()['filename'] == 'crew.txt'
    missing = await client.post('/v1/chat-imports', json={'attachment_id': 'b' * 64})
    assert missing.status_code == 404
    assert (await client.post('/v1/chat-imports', json={'attachment_id': quiet, 'date_order': 'month-first'},
                              headers={'authorization': 'Bearer read'})).status_code == 403
