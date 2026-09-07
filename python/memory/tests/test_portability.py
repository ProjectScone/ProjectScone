"""A dump moved into a space of another name keeps deduplicating there.
Identity is bound to the space name, so a default content hash exported
from space A means nothing in space B; import re-derives it for B, while
a keyed identity, which carries no content, is passed through as it was."""

from __future__ import annotations

import json

from scone_memory import HashEmbedder, InMemoryDocumentStore, InMemoryVectorIndex, MemoryEngine
from scone_memory.engine import Record, content_hash


async def moved(dump, into="archive"):
    target = await MemoryEngine(InMemoryDocumentStore(), InMemoryVectorIndex(), HashEmbedder()).open()
    await target.import_records(into, dump)
    return target


async def test_a_dump_moved_into_a_renamed_space_still_deduplicates_there():
    source = await MemoryEngine(InMemoryDocumentStore(), InMemoryVectorIndex(), HashEmbedder()).open()
    await source.remember("work", "the harbour crane was repainted")
    dump = [json.loads(json.dumps(r)) async for r in source.export("work")]
    assert dump[0]["space"] == "work" and dump[0]["content_hash"] == content_hash("work", "the harbour crane was repainted")

    target = await moved(dump, into="archive")
    [again] = await target.remember_many("archive", [Record(content="the harbour crane was repainted")])
    assert again.deduplicated, "the moved episode should be the same memory as the one remembered next"
    [stored] = await target.documents.recent_episodes("archive", 10)
    assert stored.content_hash == content_hash("archive", "the harbour crane was repainted")

    # Same-space move: nothing to re-derive, and still one memory.
    same = await moved(dump, into="work")
    [again] = await same.remember_many("work", [Record(content="the harbour crane was repainted")])
    assert again.deduplicated


async def test_a_keyed_identity_is_carried_as_the_source_made_it():
    source = await MemoryEngine(InMemoryDocumentStore(), InMemoryVectorIndex(), HashEmbedder()).open()
    await source.remember_many("work", [Record(content="ok", dedup_key="chat#1"), Record(content="ok", dedup_key="chat#2")])
    dump = [json.loads(json.dumps(r)) async for r in source.export("work")]
    keyed = {r["content_hash"] for r in dump}
    assert len(keyed) == 2 and content_hash("work", "ok") not in keyed

    target = await moved(dump, into="archive")
    stored = await target.documents.recent_episodes("archive", 10)
    assert {e.content_hash for e in stored} == keyed, "two turns stay two episodes, under the identities the source gave them"
    [retry] = await target.remember_many("archive", [Record(content="ok", dedup_key="chat#1")])
    assert not retry.deduplicated, "a renamed space cannot recover a keyed identity: the key is not in the dump"


async def test_a_dump_without_a_space_field_is_taken_as_it_is():
    source = await MemoryEngine(InMemoryDocumentStore(), InMemoryVectorIndex(), HashEmbedder()).open()
    await source.remember("work", "note")
    dump = [json.loads(json.dumps(r)) async for r in source.export("work")]
    for r in dump:
        r.pop("space")
    target = await moved(dump, into="archive")
    [stored] = await target.documents.recent_episodes("archive", 10)
    assert stored.content_hash == content_hash("work", "note"), "an older dump names no space, so its identity is passed through"
