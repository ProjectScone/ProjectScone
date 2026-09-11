"""Actual parser children feed authenticated indexing and source retrieval."""
import httpx
import pytest

from scone_memory import HashEmbedder, InMemoryDocumentStore, InMemoryVectorIndex, MemoryEngine
from scone_memory.api import create_app


@pytest.fixture
async def service():
    engine = await MemoryEngine(InMemoryDocumentStore(), InMemoryVectorIndex(), HashEmbedder()).open()
    app = create_app(engine, {'write': 'alpha', 'read': 'alpha', 'other': 'beta'}, roles={'read': 'read'})
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url='http://test',
                                 headers={'authorization': 'Bearer write'}) as client:
        yield client, engine
    await engine.close()


@pytest.mark.parametrize(('filename', 'raw', 'locator'), [
    ('launch.json', b'{"schedule":{"launch":"Friday"}}', '/schedule/launch'),
    ('launch.csv', b'task,day\nlaunch,Friday\n', 'row:2'),
    ('launch.html', b'<title>Launch</title><p>Friday launch</p>', 'line:'),
    ('launch.md', b'# Launch\nFriday\n', 'line:'),
    ('launch.ipynb', b'{"nbformat":4,"cells":[{"cell_type":"markdown","source":"Friday launch"}]}', '#/cells/0/source'),
])
async def test_upload_index_recall_cite_repeat_and_delete(service, filename, raw, locator):
    client, engine = service
    upload = await client.post('/v1/attachments', content=raw,
                               headers={'content-type': 'application/octet-stream', 'x-filename': filename})
    assert upload.status_code == 200, upload.text
    body = {'attachment_id': upload.json()['attachment_id']}
    indexed = await client.post('/v1/documents', json=body)
    assert indexed.status_code == 200, indexed.text
    episode_id = indexed.json()['added']['episode_id']
    repeated = await client.post('/v1/documents', json=body)
    assert repeated.json()['added']['episode_id'] == episode_id
    assert repeated.json()['added']['deduplicated'] is True
    recalled = (await client.get('/v1/recall', params={'q': 'Friday'})).json()['items']
    chunk = next(item for item in recalled if item['episode_id'] == episode_id)
    evidence_path = f'/v1/episodes/{episode_id}/document'
    evidence = await client.get(evidence_path, params={'chunk_id': chunk['chunk_id']},
                                 headers={'authorization': 'Bearer read'})
    assert evidence.status_code == 200, evidence.text
    assert any(locator in item['locator'] for item in evidence.json()['segments'])
    assert (await client.get(evidence.json()['download_path'])).content == raw
    assert (await client.get(evidence_path, headers={'authorization': 'Bearer other'})).status_code == 404
    assert (await client.post('/v1/documents', json=body, headers={'authorization': 'Bearer read'})).status_code == 403
    assert (await client.delete(f'/v1/episodes/{episode_id}')).status_code == 200
    assert (await client.get(evidence_path)).status_code == 410
    assert (await engine.documents.counts('alpha')).episodes == 0


async def test_document_boundary_rejects_oversize_extra_fields_and_reports_formats(service):
    client, _ = service
    response = await client.get('/v1/documents/formats')
    assert response.status_code == 200
    formats = response.json()['formats']
    for suffix in ('.docx', '.xlsx', '.pptx', '.json', '.csv', '.eml', '.epub', '.ipynb'):
        assert suffix in formats
    assert (await client.post('/v1/documents', content=b'x'*4097)).status_code == 413
    assert (await client.post('/v1/documents', json={'attachment_id': 'a'*64, 'url': 'http://example.com'})).status_code == 400
    assert (await client.post('/v1/documents', json={'attachment_id': 'a'*64})).status_code == 404


async def test_office_annotation_roles_survive_indexing_and_source_citations(service):
    from ..ingestion.test_office_revisions import odt

    client, engine = service
    raw = odt('''<text:p>Revenue grew<office:annotation xmlns:dc="http://purl.org/dc/elements/1.1/">
        <dc:creator>Alice</dc:creator><text:p>Verify this estimate.</text:p></office:annotation> 4%.</text:p>''')
    uploaded = await client.post('/v1/attachments', content=raw,
        headers={'content-type': 'application/octet-stream', 'x-filename': 'report.odt'})
    assert uploaded.status_code == 200, uploaded.text
    indexed = await client.post('/v1/documents', json={'attachment_id': uploaded.json()['attachment_id']})
    assert indexed.status_code == 200, indexed.text
    episode_id = indexed.json()['added']['episode_id']
    episode = await engine.episode('alpha', episode_id)
    assert episode.content == 'Revenue grew 4%.\n\nVerify this estimate.'
    evidence = await client.get(f'/v1/episodes/{episode_id}/document',
        headers={'authorization': 'Bearer read'})
    assert evidence.status_code == 200, evidence.text
    main, comment = evidence.json()['segments']
    assert main['text'] == 'Revenue grew 4%.'
    assert comment['locator'] == 'paragraph:1/comment:1'
    assert comment['metadata'] == {
        'content_role': 'comment', 'parent_locator': 'paragraph:1', 'author': 'Alice',
    }
    assert (await client.get(evidence.json()['download_path'])).content == raw


async def test_docx_note_citation_retains_relocated_source_part_and_original(service):
    from ..ingestion.test_docx_parts import document, relation
    from ..ingestion.test_office_formats import W

    client, engine = service
    raw = document('<w:p><w:r><w:t>Revenue grew 4%.</w:t><w:footnoteReference w:id="2"/></w:r></w:p>',
        relationships=relation('footnotes', '../notes/financial.xml'), extra={
            'notes/financial.xml': f'<w:footnotes xmlns:w="{W}"><w:footnote w:id="2"><w:p><w:r><w:t>Unaudited café estimate.</w:t></w:r></w:p></w:footnote></w:footnotes>',
        })
    uploaded = await client.post('/v1/attachments', content=raw,
        headers={'content-type': 'application/octet-stream', 'x-filename': 'report.docx'})
    assert uploaded.status_code == 200, uploaded.text
    indexed = await client.post('/v1/documents', json={'attachment_id': uploaded.json()['attachment_id']})
    assert indexed.status_code == 200, indexed.text
    episode_id = indexed.json()['added']['episode_id']
    episode = await engine.episode('alpha', episode_id)
    assert episode.content == 'Revenue grew 4%.\n\nUnaudited café estimate.'
    recalled = (await client.get('/v1/recall', params={'q': 'Unaudited café estimate'})).json()['items']
    chunk = next(item for item in recalled if item['episode_id'] == episode_id)
    evidence = await client.get(f'/v1/episodes/{episode_id}/document',
        params={'chunk_id': chunk['chunk_id']}, headers={'authorization': 'Bearer read'})
    assert evidence.status_code == 200, evidence.text
    note = next(segment for segment in evidence.json()['segments'] if segment['metadata'].get('content_role') == 'footnote')
    assert note['text'] == 'Unaudited café estimate.'
    assert note['locator'] == 'paragraph:1/footnote:2/paragraph:1'
    assert note['metadata'] == {
        'member': 'notes/financial.xml', 'content_role': 'footnote',
        'parent_locator': 'paragraph:1', 'note_id': '2',
    }
    assert (await client.get(evidence.json()['download_path'])).content == raw


@pytest.mark.parametrize('first_name', [None, 'plan.txt'])
async def test_extraction_filename_overrides_first_upload_label_without_rewriting_original(service, first_name):
    client, _ = service
    raw = b'task,day\nlaunch,Friday\n'
    headers = {'content-type': 'application/octet-stream'}
    if first_name is not None:
        headers['x-filename'] = first_name
    uploaded = await client.post('/v1/attachments', content=raw, headers=headers)
    attachment_id = uploaded.json()['attachment_id']
    body = {'attachment_id': attachment_id, 'filename': 'plan.csv'}
    indexed = await client.post('/v1/documents', json=body)
    assert indexed.status_code == 200, indexed.text
    assert indexed.json()['format'] == 'csv'
    assert indexed.json()['filename'] == 'plan.csv'
    assert indexed.json()['original']['filename'] == first_name
    episode_id = indexed.json()['added']['episode_id']
    evidence = (await client.get(f'/v1/episodes/{episode_id}/document')).json()
    assert evidence['filename'] == 'plan.csv'
    assert evidence['original']['filename'] == first_name
    assert [segment['locator'] for segment in evidence['segments']] == ['row:2']
    assert (await client.get(evidence['download_path'])).content == raw
    repeated = await client.post('/v1/documents', json=body)
    assert repeated.json()['added']['episode_id'] == episode_id
    text = await client.post('/v1/documents', json={**body, 'filename': 'plan.txt'})
    assert text.status_code == 200, text.text
    assert text.json()['format'] == 'txt'
    assert text.json()['added']['episode_id'] != episode_id
    assert (await client.post('/v1/documents', json=body, headers={'authorization': 'Bearer other'})).status_code == 404


@pytest.mark.parametrize('filename', ['', 42, 'x'*1025, '\x00.csv', '\ud800.csv', 'é'*513 + '.csv'])
async def test_explicit_extraction_filename_is_validated(service, filename):
    client, _ = service
    uploaded = await client.post('/v1/attachments', content=b'a,b\n1,2',
                                 headers={'content-type': 'text/csv', 'x-filename': 'a.csv'})
    assert uploaded.status_code == 200, uploaded.text
    import json
    response = await client.post('/v1/documents', content=json.dumps({
        'attachment_id': uploaded.json()['attachment_id'], 'filename': filename}),
        headers={'content-type': 'application/json'})
    assert response.status_code in (400, 422)
