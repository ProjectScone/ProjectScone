"""Compare backend lifecycles against independently tracked logical memory.

Numeric IDs and rank scores are backend-local. The corpus is small enough
that an untruncated recall must return every eligible single-chunk episode.
Removing a delete, a scope predicate, or a revision bump breaks these tests.
"""

from collections import Counter
from dataclasses import dataclass
import random

import pytest

from scone_memory import HashEmbedder, InMemoryDocumentStore, InMemoryVectorIndex, MemoryEngine
from scone_memory.core.errors import NotFound
from scone_memory.testing import Clock


@dataclass(frozen=True)
class Note:
    content: str
    created_at: str
    tags: tuple[str, ...]
    user_id: str
    source: str

    def kwargs(self):
        return {
            "created_at": self.created_at,
            "tags": self.tags,
            "metadata": {"user_id": self.user_id},
            "source": self.source,
        }


async def assert_snapshot(engines, live, revisions):
    for space in ("alpha", "beta"):
        notes = list(live[space].values())
        byte_count = sum(len(note.content.encode("utf-8")) for note in notes)
        tags = dict(Counter(tag for note in notes for tag in note.tags))
        scenarios = (
            {},
            {"as_of": "2024-03-01T00:00:00.000Z"},
            {"tags": ["work", "approved"]},
            {"where": {"user_id": "alice"}},
            {"tags": ["work"], "where": {"user_id": "bob"}, "as_of": "2024-03-01T00:00:00.000Z"},
            {"where": {"user_id": "absent"}},
        )
        for candidate in engines:
            status = await candidate.status(space)
            assert (status.episodes, status.chunks, status.bytes, status.revision) == (
                len(notes), len(notes), byte_count, revisions[space],
            )
            assert await candidate.tags(space) == tags
            for filters in scenarios:
                eligible = {
                    note.content: note for note in notes
                    if (not filters.get("as_of") or note.created_at <= filters["as_of"])
                    and set(filters.get("tags", ())) <= set(note.tags)
                    and (not filters.get("where") or note.user_id == filters["where"]["user_id"])
                }
                result = await candidate.recall(space, "memory", limit=50, **filters)
                assert result.degraded == []
                assert len(result.items) == len(eligible)
                assert {item.text for item in result.items} == set(eligible)
                assert result.space_bytes == byte_count
                assert result.returned_bytes == sum(len(text.encode("utf-8")) for text in eligible)
                for item in result.items:
                    note = eligible[item.text]
                    assert (item.created_at, item.tags, item.metadata, item.source) == (
                        note.created_at, note.tags, {"user_id": note.user_id}, note.source,
                    )
                    stored = await candidate.episode(space, item.episode_id)
                    assert (stored.space, stored.content) == (space, note.content)


@pytest.mark.parametrize("seed", [7, 19, 43])
async def test_seeded_episode_lifecycle_matches_reference_and_oracle(engine, seed):
    reference = await MemoryEngine(
        InMemoryDocumentStore(), InMemoryVectorIndex(), HashEmbedder(),
        chunk_target=200, clock=Clock(),
    ).open()
    engines = (reference, engine)
    rng = random.Random(seed)
    notes = [
        Note(
            content=f"memory {i} café planning",
            created_at=f"2024-0{1 + i % 4}-01T00:00:00.000Z",
            tags=("work", "approved") if i % 3 == 0 else ("work",) if i % 3 == 1 else ("home",),
            user_id="alice" if i % 2 == 0 else "bob",
            source=f"note://{i}",
        )
        for i in range(8)
    ]
    live = {"alpha": {}, "beta": {}}
    revisions = {"alpha": 0, "beta": 0}
    ids = [{"alpha": {}, "beta": {}} for _ in engines]
    initial = [(space, note) for space in live for note in notes]
    rng.shuffle(initial)
    for space, note in initial:
        for candidate, handles in zip(engines, ids):
            added = await candidate.remember(space, note.content, **note.kwargs())
            assert not added.deduplicated
            handles[space][note.content] = added.episode_id
        live[space][note.content] = note
        revisions[space] += 1
    await assert_snapshot(engines, live, revisions)

    # Each seed changes the order and victims, while always exercising
    # duplicates, deletes, reinsertion, both spaces, and empty final stores.
    for cycle in range(4):
        space = ("alpha", "beta")[cycle % 2]
        victims = rng.sample(list(live[space].values()), 3)
        for note in victims:
            for candidate, handles in zip(engines, ids):
                again = await candidate.remember(space, note.content, **note.kwargs())
                assert again.deduplicated
                assert again.episode_id == handles[space][note.content]
                await candidate.forget(space, handles[space].pop(note.content))
            del live[space][note.content]
            revisions[space] += 1
        await assert_snapshot(engines, live, revisions)
        rng.shuffle(victims)
        for note in victims:
            for candidate, handles in zip(engines, ids):
                added = await candidate.remember(space, note.content, **note.kwargs())
                assert not added.deduplicated
                handles[space][note.content] = added.episode_id
            live[space][note.content] = note
            revisions[space] += 1
        await assert_snapshot(engines, live, revisions)

    for space in live:
        for candidate, handles in zip(engines, ids):
            for episode_id in handles[space].values():
                await candidate.forget(space, episode_id)
                with pytest.raises(NotFound):
                    await candidate.forget(space, episode_id)
        revisions[space] += len(live[space])
        live[space].clear()
    await assert_snapshot(engines, live, revisions)
