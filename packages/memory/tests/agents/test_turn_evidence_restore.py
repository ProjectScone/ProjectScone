"""Retained packets survive restart only while their original sources do."""
import pytest

from .test_evidence_tool_loop import binding
from .test_task_workflow import memory


async def test_restoration_reuses_exact_packet_without_search(memory):
    await memory.remember('alpha', 'Juniper uses Polaris.', metadata={'team': 'blue'})
    tools = binding(memory)
    saved = await tools.prepare('search_memory', {'query': 'Juniper', 'limit': 5})
    assert saved.evidence_ids
    async def forbidden(*args, **kwargs):
        pytest.fail('restoration reran a search')
    tools.run = forbidden
    restored = await tools.restore(saved.payload, saved.source_digest)
    assert restored.payload == saved.payload and restored.evidence_ids == saved.evidence_ids
    assert restored.source_digest == saved.source_digest and await restored.validate()


@pytest.mark.parametrize('change', ['delete', 'revision', 'payload', 'digest'])
async def test_restoration_refuses_changed_evidence(memory, change):
    episode = await memory.remember('alpha', 'Juniper uses Polaris.', metadata={'team': 'blue'})
    tools = binding(memory)
    saved = await tools.prepare('search_memory', {'query': 'Juniper', 'limit': 5})
    if change == 'delete':
        await memory.documents.delete_episode('alpha', episode.episode_id)
    elif change == 'revision':
        await memory.remember('alpha', 'Another document.', metadata={'team': 'blue'})
    payload = saved.payload.replace('Polaris', 'Changed') if change == 'payload' else saved.payload
    digest = '0' * 64 if change == 'digest' else saved.source_digest
    with pytest.raises(ValueError):
        await tools.restore(payload, digest)


async def test_restoration_storage_outage_is_not_an_unavailable_substitute(memory):
    await memory.remember('alpha', 'Juniper uses Polaris.', metadata={'team': 'blue'})
    tools = binding(memory)
    saved = await tools.prepare('search_memory', {'query': 'Juniper', 'limit': 5})
    original = memory.documents.get_chunks
    async def unavailable(*args, **kwargs):
        raise OSError('private store failure')
    memory.documents.get_chunks = unavailable
    try:
        with pytest.raises(OSError):
            await tools.restore(saved.payload, saved.source_digest)
    finally:
        memory.documents.get_chunks = original
