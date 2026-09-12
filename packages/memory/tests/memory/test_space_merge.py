"""Moving one space into another, and emptying one without deleting it.

A space is a container, and containers get merged and cleared. Both are
destructive in the way that matters — somebody's memory moves or goes —
so both preview first and both need the name said out loud.

A merge is not a new kind of write. It is the archive read out of one
space and into another, so everything that makes an import honest holds:
identity is re-derived for the space it lands in, what was forgotten
there stays forgotten, and the claims arrive with their history.
"""

from __future__ import annotations

import pytest

from scone_memory import HashEmbedder, InMemoryDocumentStore, InMemoryVectorIndex, MemoryEngine
from scone_memory.core.errors import InvalidInput
from scone_memory.testing import Clock

pytestmark = pytest.mark.asyncio
DAY = "2024-01-01T00:00:00Z"


async def memory() -> MemoryEngine:
    return await MemoryEngine(InMemoryDocumentStore(), InMemoryVectorIndex(), HashEmbedder(),
                              clock=Clock("2025-06-01T00:00:00.000Z")).open()


async def filled() -> MemoryEngine:
    engine = await memory()
    note = await engine.remember("old", "The harbour crane was repainted in March.")
    await engine.assert_fact("old", "crane", "painted_in", "March", valid_from=DAY,
                             source_episode_id=note.episode_id, quote="The harbour crane was repainted")
    await engine.remember("new", "The Lisbon office opened in March.")
    return engine


async def test_a_merge_previews_what_it_would_move():
    engine = await filled()
    preview = await engine.merge_space("old", into="new", preview=True)
    assert preview.episodes == 1 and preview.facts == 1 and preview.moved is False
    assert (await engine.documents.counts("old")).episodes == 1, "nothing moved yet"
    assert (await engine.documents.counts("new")).episodes == 1


async def test_a_merge_moves_what_it_previewed_and_leaves_the_old_space_empty():
    engine = await filled()
    receipt = await engine.merge_space("old", into="new", confirm="old")
    assert receipt.moved is True and receipt.episodes == 1 and receipt.facts == 1
    assert (await engine.documents.counts("new")).episodes == 2
    assert (await engine.documents.counts("old")).episodes == 0


async def test_what_moved_is_recalled_in_the_space_it_moved_into():
    engine = await filled()
    await engine.merge_space("old", into="new", confirm="old")
    found = await engine.recall("new", "harbour crane")
    assert found.items and "harbour crane" in found.items[0].text
    assert [f.object for f in await engine.documents.list_facts("new", include_closed=True)] == ["March"]


async def test_a_merge_without_the_name_said_out_loud_moves_nothing():
    engine = await filled()
    with pytest.raises(InvalidInput, match="confirm"):
        await engine.merge_space("old", into="new")
    assert (await engine.documents.counts("old")).episodes == 1


async def test_a_space_is_not_merged_into_itself():
    engine = await filled()
    with pytest.raises(InvalidInput, match="itself"):
        await engine.merge_space("old", into="old", confirm="old")


async def test_a_merged_space_is_closed_for_good_and_says_so():
    """Everything it held is somewhere else now. Leaving the name open
    would invite somebody to write into a space whose contents have moved,
    and find them missing."""
    engine = await filled()
    await engine.merge_space("old", into="new", confirm="old")
    assert await engine.space_deleted("old") is not None


async def test_a_merge_previews_and_moves_from_the_command_line():
    import io

    from scone_memory.runtime.cli import build_parser, run

    engine = await filled()
    out = io.StringIO()
    assert await run(build_parser().parse_args(["--space", "old", "merge-space", "--into", "new", "--dry-run"]),
                     engine, io.StringIO(""), out) == 0
    assert "would move" in out.getvalue() and (await engine.documents.counts("old")).episodes == 1

    out = io.StringIO()
    assert await run(build_parser().parse_args(
        ["--space", "old", "merge-space", "--into", "new", "--confirm", "old"]),
        engine, io.StringIO(""), out) == 0
    assert (await engine.documents.counts("new")).episodes == 2


async def test_the_command_line_refuses_a_merge_nobody_confirmed():
    import io

    from scone_memory.runtime.cli import build_parser, run

    engine = await filled()
    out = io.StringIO()
    code = await run(build_parser().parse_args(["--space", "old", "merge-space", "--into", "new"]),
                     engine, io.StringIO(""), out)
    assert code == 2 and "confirm" in out.getvalue()
    assert (await engine.documents.counts("old")).episodes == 1


async def test_a_merge_over_http_needs_the_full_role_as_a_deletion_does():
    from fastapi.testclient import TestClient

    from scone_memory.api import create_app

    engine = await filled()
    app = create_app(engine, {"key-w": "old", "key-a": "old"}, roles={"key-w": "write"})
    with TestClient(app) as client:
        refused = client.post("/v1/spaces/old/merge", json={"into": "new", "confirm": "old"},
                              headers={"Authorization": "Bearer key-w"})
        assert refused.status_code == 403, "moving a whole space is not an ordinary write"
        said = client.post("/v1/spaces/old/merge", json={"into": "new", "confirm": "old"},
                           headers={"Authorization": "Bearer key-a"})
        assert said.status_code == 200 and said.json()["moved"] is True
