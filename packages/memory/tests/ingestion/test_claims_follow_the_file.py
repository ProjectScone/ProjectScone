"""A claim read from a file holds while the file says it.

`replace` and `sync` store a changed file as an update: the old episode
is forgotten and the new one stored, and the new one's claims are read.
The old episode's claims stood, by forget's contract, so the ledger held
what the file used to say beside what it says now: a module that dropped
an import still imported it, a function that was removed was still
defined. This closes, on replacement, the extracted claims the new
content no longer makes -- with a reason naming the file -- and leaves
alone what it restates, what a person stated, and everything a plain
forget touches.
"""

from __future__ import annotations

import pytest

from scone_memory import (HashEmbedder, InMemoryDocumentStore, InMemoryEventLog, InMemoryVectorIndex,
                          MemoryEngine)
from scone_memory.ingestion.records import Record
from scone_memory.ingestion.sync import sync_directory

pytestmark = pytest.mark.asyncio

V1 = "import os\nimport re\n\ndef a():\n    return 1\n"
V2 = "import os\nimport json\n\ndef a():\n    return 2\n\ndef b():\n    return a()\n"
PATH = "pkg/mod.py"


async def engine(**options):
    return await MemoryEngine(InMemoryDocumentStore(), InMemoryVectorIndex(), HashEmbedder(),
                              code_graph=True, events=InMemoryEventLog(), **options).open()


async def closures(memory) -> list[str]:
    """Why each claim closed, as the event says it -- never "manual" here,
    since no person decided these."""
    return [str(event.payload.get("reason_kind")) for event in await memory.events.query("s", kind="fact_close")]


async def stored(memory, content, key="mod"):
    return await memory.replace("s", Record(content=content, kind="file", source=PATH, dedup_key=key))


def triples(facts):
    return sorted((f.subject, f.predicate, f.object) for f in facts)


async def test_a_claim_the_new_version_no_longer_makes_is_closed():
    memory = await engine()
    try:
        await stored(memory, V1)
        updated = await stored(memory, V2)
        held = triples(await memory.facts("s"))
        assert (PATH, "imports", "re") not in held
        assert (PATH, "imports", "json") in held and (PATH, "defines", f"{PATH}:b") in held
        closed = [f for f in await memory.facts("s", status="closed")]
        assert triples(closed) == [(PATH, "imports", "re")]
        assert closed[0].closed_reason == f"no longer stated by {PATH}"
        assert updated.claims_closed == 1
        assert await closures(memory) == ["source_changed"]
    finally:
        await memory.close()


async def test_a_claim_the_new_version_restates_is_one_fact_still_holding():
    memory = await engine()
    try:
        await stored(memory, V1)
        await stored(memory, V2)
        held = [f for f in await memory.facts("s") if (f.subject, f.predicate, f.object) == (PATH, "imports", "os")]
        assert len(held) == 1 and held[0].status == "active"
        assert not [f for f in await memory.facts("s", status="closed") if f.object == "os"]
    finally:
        await memory.close()


async def test_what_a_person_stated_about_the_old_episode_stands():
    memory = await engine()
    try:
        first = await stored(memory, V1)
        await memory.assert_fact("s", PATH, "owned_by", "team-a", source_episode_id=first.added.episode_id)
        await stored(memory, V2)
        assert (PATH, "owned_by", "team-a") in triples(await memory.facts("s"))
    finally:
        await memory.close()


async def test_a_first_store_closes_nothing_and_the_receipt_says_so():
    memory = await engine()
    try:
        first = await stored(memory, V1)
        assert first.claims_closed == 0 and first.claims_unread is False
        assert await memory.facts("s", status="closed") == []
    finally:
        await memory.close()


async def test_forgetting_alone_leaves_the_claims_standing():
    """Forget's contract is untouched: the data goes, the claims stay."""
    memory = await engine()
    try:
        first = await stored(memory, V1)
        await memory.forget("s", first.added.episode_id)
        assert (PATH, "imports", "re") in triples(await memory.facts("s"))
    finally:
        await memory.close()


async def test_a_store_that_cannot_read_claims_by_episode_says_so_instead_of_guessing():
    memory = await engine()
    try:
        await stored(memory, V1)
        memory.documents.facts_for_graph = None  # type: ignore[method-assign]
        updated = await stored(memory, V2)
        assert updated.claims_closed is None and updated.claims_unread is False
        assert (PATH, "imports", "re") in triples(await memory.facts("s")), "nothing was closed on a guess"
    finally:
        await memory.close()


async def test_an_old_episode_with_more_claims_than_can_be_read_is_disclosed(monkeypatch):
    from scone_memory.core import graph_read

    memory = await engine()
    try:
        await stored(memory, V1)
        monkeypatch.setattr(graph_read, "MAX_GRAPH_FACTS", 2)
        updated = await stored(memory, V2)
        assert updated.claims_unread is True
        assert isinstance(updated.claims_closed, int)
    finally:
        await memory.close()


# --- through sync ------------------------------------------------------------

async def synced(memory, root, **options):
    return await sync_directory(memory, "s", str(root), marker="tree", apply=True, **options)


async def test_sync_closes_what_a_changed_file_stopped_saying(tmp_path):
    (tmp_path / "pkg").mkdir()
    (tmp_path / "pkg" / "mod.py").write_text(V1, encoding="utf-8")
    memory = await engine()
    try:
        first = await synced(memory, tmp_path)
        assert "claims_closed" not in first.record(), "the receipt is unchanged when nothing was closed"
        (tmp_path / "pkg" / "mod.py").write_text(V2, encoding="utf-8")
        second = await synced(memory, tmp_path)
        assert second.claims_closed == 1 and second.record()["claims_closed"] == 1
        assert "1 claim(s) closed" in second.text()
        assert triples(await memory.facts("s", status="closed")) == [(PATH, "imports", "re")]
    finally:
        await memory.close()


async def test_sync_closes_every_claim_of_a_removed_file_only_when_asked_to_remove(tmp_path):
    (tmp_path / "pkg").mkdir()
    (tmp_path / "pkg" / "mod.py").write_text(V1, encoding="utf-8")
    (tmp_path / "pkg" / "other.py").write_text("import sys\n", encoding="utf-8")
    memory = await engine()
    try:
        await synced(memory, tmp_path)
        (tmp_path / "pkg" / "mod.py").unlink()
        kept = await synced(memory, tmp_path)
        assert kept.claims_closed == 0 and (PATH, "imports", "re") in triples(await memory.facts("s"))
        removed = await synced(memory, tmp_path, remove=True)
        assert removed.forgotten == 1 and removed.claims_closed == 3
        closed = await memory.facts("s", status="closed")
        assert triples(closed) == [(PATH, "defines", f"{PATH}:a"), (PATH, "imports", "os"), (PATH, "imports", "re")]
        assert {f.closed_reason for f in closed} == {f"{PATH} was removed"}
        assert await closures(memory) == ["source_removed"] * 3
        assert ("pkg/other.py", "imports", "sys") in triples(await memory.facts("s"))
    finally:
        await memory.close()
