"""Retention selects oldest instants, regardless of timestamp spelling."""
import pytest


@pytest.mark.parametrize('first,second,oldest_index', [
    ('2026-01-01T00:30:00-02:00', '2026-01-01T01:00:00+02:00', 1),
    ('2026-01-01T01:00:00+01:00', '2026-01-01T00:00:00Z', 0),
])
async def test_expiry_limit_uses_instants_then_episode_ids(engine, first, second, oldest_index):
    episodes = []
    for index, when in enumerate((first, second)):
        engine.clock = lambda when=when: when
        episodes.append(await engine.remember('alpha', f'retention source {index}', kind='note'))
    engine.clock = lambda: '2026-03-01T00:00:00Z'
    preview = await engine.expire('alpha', {'note':1}, limit=1, dry_run=True)
    assert preview.remaining == 2 and not preview.forgotten
    report = await engine.expire('alpha', {'note':1}, limit=1)
    assert report.forgotten == [episodes[oldest_index].episode_id]
    assert report.remaining == 1
    assert await engine.tombstone('alpha', episodes[oldest_index].episode_id) is not None
    assert (await engine.episode('alpha', episodes[1-oldest_index].episode_id)).content == f'retention source {1-oldest_index}'
