"""Archived conversations rebuild ordinary vectors without capture privileges."""
import asyncio
from dataclasses import replace

import pytest

from scone_memory import HashEmbedder, InMemoryDocumentStore, InMemoryVectorIndex, MemoryEngine
from scone_memory.core.errors import InvalidInput
from .test_deferred_capture import message


async def engine(*, vectors=None, clock=None):
    options = {'clock': clock} if clock else {}
    return await MemoryEngine(InMemoryDocumentStore(), vectors or InMemoryVectorIndex(), HashEmbedder(), **options).open()


@pytest.mark.parametrize('indexed', [False, True])
@pytest.mark.parametrize('target_space', ['alpha', 'beta'])
async def test_exported_deferred_capture_imports_with_provenance_and_key_identity(indexed, target_space):
    source, target = await engine(), await engine()
    try:
        captured = await source.capture_text('alpha', replace(message(), tags=('launch',)))
        if indexed:
            await source.index_captured_text('alpha', captured.episode_id)
        await source.assert_fact('alpha', 'Lyra', 'owner', 'Leo', valid_from='2026-01-01T00:00:00Z',
                                 source_episode_id=captured.episode_id, quote='Lyra launch owner is Leo.')
        rows = [row async for row in source.export('alpha')]
        exported = next(row for row in rows if row['type'] == 'episode')
        await target.remember(target_space, 'An unrelated target source.')
        result = await target.import_records(target_space, rows)
        assert result.episodes == 1 and result.facts == 1
        retained = await target.documents.episode_by_hash(target_space, exported['content_hash'])
        assert retained is not None and retained.episode_id != captured.episode_id
        assert retained.content == message().content and retained.source == 'session'
        assert retained.tags == ('launch',)
        assert retained.metadata == {**message().metadata, 'chunking': 'length'}
        assert (await target.documents.list_facts(target_space, include_closed=True))[0].source_episode_id == retained.episode_id
        assert (await target.recall(target_space, 'Lyra launch', lanes=('vector',))).items[0].episode_id == retained.episode_id
        assert not await target.documents.inflight()
        again = await target.import_records(target_space, rows)
        assert again.episodes == 0 and again.deduplicated == 1
        # Normalization must not mutate the archive or source memory metadata.
        assert exported['metadata']['capture_indexing'] == 'deferred'
        assert (await source.documents.get_episode('alpha', captured.episode_id)).metadata['capture_key'] == 'one'
    finally:
        await source.close()
        await target.close()


async def test_deferred_archive_preserves_expiry_and_target_tombstones():
    now = ['2026-10-03T10:00:00Z']
    source, target = await engine(clock=lambda: now[0]), await engine(clock=lambda: now[0])
    try:
        captured = await source.capture_text('alpha', replace(message(), forget_after='2026-10-03T10:01:00Z'))
        rows = [row async for row in source.export('alpha')]
        await target.import_records('alpha', rows)
        retained = (await target.documents.recent_episodes('alpha', 10))[0]
        assert retained.metadata['forget_after'] == '2026-10-03T10:01:00.000Z'
        await target.forget('alpha', retained.episode_id)
        assert (await target.import_records('alpha', rows)).tombstoned == 1
        now[0] = '2026-10-03T10:02:00Z'
        skipped = await target.import_records('beta', rows)
        assert skipped.past_forget_after == 1 and skipped.episodes == 0
        assert await source.documents.get_episode('alpha', captured.episode_id)
    finally:
        await source.close()
        await target.close()


async def test_archive_import_does_not_inherit_pending_capture_forget_exemption():
    class HeldVectors(InMemoryVectorIndex):
        def __init__(self):
            super().__init__()
            self.entered, self.release = asyncio.Event(), asyncio.Event()

        async def upsert(self, points):
            self.entered.set()
            await self.release.wait()
            await super().upsert(points)

    source, vectors = await engine(), HeldVectors()
    target = await engine(vectors=vectors)
    importing = None
    try:
        await source.capture_text('alpha', message())
        rows = [row async for row in source.export('alpha')]
        importing = asyncio.create_task(target.import_records('alpha', rows))
        await asyncio.wait_for(vectors.entered.wait(), 1)
        retained = (await target.documents.recent_episodes('alpha', 10))[0]
        with pytest.raises(InvalidInput, match='indexing is unfinished'):
            await target.forget('alpha', retained.episode_id)
        vectors.release.set()
        assert (await importing).episodes == 1
        assert await vectors.ids('alpha')
    finally:
        vectors.release.set()
        if importing is not None:
            await asyncio.gather(importing, return_exceptions=True)
        await source.close()
        await target.close()
