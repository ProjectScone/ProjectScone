"""Whole-space movement requires explicit authority over both spaces."""
import pytest
from fastapi.testclient import TestClient

from scone_memory import HashEmbedder, InMemoryDocumentStore, InMemoryVectorIndex, MemoryEngine
from scone_memory.api import create_app


@pytest.fixture
async def setup():
    memory = await MemoryEngine(InMemoryDocumentStore(), InMemoryVectorIndex(), HashEmbedder()).open()
    evidence = await memory.attach('source', b'merge evidence', media_type='text/plain')
    await memory.remember('source', 'Incoming source', attachment_ids=[evidence.attachment_id])
    await memory.remember('target', 'Original destination')
    app = create_app(memory, {'source-key': 'source', 'target-key': 'target', 'other-key': 'other'},
                     roles={'source-key': 'full', 'target-key': 'full', 'other-key': 'full'})
    yield memory, app, evidence
    await memory.close()


@pytest.mark.parametrize('header', [None, 'Basic target-key', 'Bearer unknown', 'Bearer other-key'])
@pytest.mark.parametrize('preview', [False, True])
async def test_missing_or_wrong_destination_authority_cannot_merge_or_preview(setup, header, preview):
    memory, app, _ = setup
    headers = {'Authorization': 'Bearer source-key'}
    if header is not None:
        headers['X-Scone-Destination-Authorization'] = header
    with TestClient(app) as client:
        response = client.post('/v1/spaces/source/merge', headers=headers,
                               json={'into': 'target', 'confirm': 'source', 'preview': preview})
    assert response.status_code in (401, 403)
    assert await memory.space_deleted('source') is None
    assert (await memory.documents.counts('target')).episodes == 1
    assert await memory.blobs.held('target') == []


@pytest.mark.parametrize('role', ['read', 'write', 'review'])
async def test_destination_ledger_movement_requires_full_role(setup, role):
    memory, app, _ = setup
    app.state.roles['target-key'] = role
    with TestClient(app) as client:
        response = client.post('/v1/spaces/source/merge',
            headers={'Authorization': 'Bearer source-key', 'X-Scone-Destination-Authorization': 'Bearer target-key'},
            json={'into': 'target', 'confirm': 'source'})
    assert response.status_code == 403
    assert await memory.space_deleted('source') is None
    assert (await memory.documents.counts('target')).episodes == 1


async def test_two_full_keys_move_verified_evidence_and_return_receipt(setup):
    memory, app, evidence = setup
    headers = {'Authorization': 'Bearer source-key', 'X-Scone-Destination-Authorization': 'Bearer target-key'}
    with TestClient(app) as client:
        preview = client.post('/v1/spaces/source/merge', headers=headers, json={'into': 'target', 'preview': True})
        assert preview.status_code == 200 and preview.json()['attachments'] == 1 and not preview.json()['moved']
        result = client.post('/v1/spaces/source/merge', headers=headers, json={'into': 'target', 'confirm': 'source'})
    assert result.status_code == 200 and result.json()['moved'] and result.json()['attachments'] == 1
    assert await memory.attachment('target', evidence.attachment_id) == (evidence, b'merge evidence')


@pytest.mark.parametrize('change', ['source_revoke', 'target_revoke', 'target_scope', 'target_role'])
async def test_authority_change_during_copy_prevents_source_closure(setup, monkeypatch, change):
    memory, app, _ = setup
    imported = memory.import_records
    async def revoke(*args, **kwargs):
        result = await imported(*args, **kwargs)
        if change == 'source_revoke':
            del app.state.keys['source-key']
        elif change == 'target_revoke':
            del app.state.keys['target-key']
        elif change == 'target_scope':
            app.state.keys['target-key'] = 'other'
        else:
            app.state.roles['target-key'] = 'read'
        return result
    monkeypatch.setattr(memory, 'import_records', revoke)
    with TestClient(app) as client:
        response = client.post('/v1/spaces/source/merge',
            headers={'Authorization': 'Bearer source-key', 'X-Scone-Destination-Authorization': 'Bearer target-key'},
            json={'into': 'target', 'confirm': 'source'})
    assert response.status_code in (401, 403)
    assert await memory.space_deleted('source') is None
    assert (await memory.documents.counts('source')).episodes == 1
