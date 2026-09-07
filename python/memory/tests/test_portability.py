"""A dump moved into a space of another name keeps deduplicating there.
Identity is bound to the space name, so a default content hash exported
from space A means nothing in space B; import re-derives it for B, while
a keyed identity, which carries no content, is passed through as it was."""

from __future__ import annotations

import json

from scone_memory import HashEmbedder, InMemoryDocumentStore, InMemoryVectorIndex, MemoryEngine
from scone_memory.memory.engine import Record, content_hash


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


async def test_links_travel_in_a_dump_with_their_ends_and_sources_remapped():
    """A relation is part of the ledger's history, so it moves with it.
    Ids are store-local: a link's ends and its source are remapped through
    the ids the import produced, a link to a fact not in the dump is
    dropped and counted, and importing the same dump twice adds nothing."""
    source = await MemoryEngine(InMemoryDocumentStore(), InMemoryVectorIndex(), HashEmbedder()).open()
    episode = await source.remember("default", "Acme is headquartered in Lisbon, near the river.")
    works = await source.assert_fact("default", "mark", "works_at", "Acme")
    based = await source.assert_fact("default", "Acme", "based_in", "Lisbon", source_episode_id=episode.episode_id, quote="headquartered in Lisbon")
    derived = await source.assert_fact("default", "mark", "works_in", "Lisbon", derived_from=[works.fact_id, based.fact_id])
    await source.link_facts("default", works.fact_id, based.fact_id, "supports", source_episode_id=episode.episode_id, quote="headquartered in Lisbon")
    dump = [record async for record in source.export("default")]
    links = [r for r in dump if r["type"] == "fact_link"]
    assert sorted(l["kind"] for l in links) == ["derived_from", "derived_from", "supports"]
    assert all({"from_fact", "to_fact", "kind", "created_at", "source_episode_id", "quote"} <= set(l) for l in links)

    # The target already holds an unrelated episode, so every id shifts.
    target = await MemoryEngine(InMemoryDocumentStore(), InMemoryVectorIndex(), HashEmbedder()).open()
    await target.remember("default", "something else entirely")
    await target.assert_fact("default", "zed", "is", "first")
    summary = await target.import_records("default", dump)
    assert (summary.facts, summary.links, summary.links_skipped) == (3, 3, 0)
    facts = {(f.subject, f.predicate): f for f in await target.facts("default", include_closed=True)}
    moved = await target.fact_links("default", facts[("mark", "works_in")].fact_id)
    assert sorted((l.from_fact, l.to_fact, l.kind) for l in moved) == sorted([
        (facts[("mark", "works_in")].fact_id, facts[("mark", "works_at")].fact_id, "derived_from"),
        (facts[("mark", "works_in")].fact_id, facts[("acme", "based_in")].fact_id, "derived_from"),
    ])
    [supports] = [l for l in await target.fact_links("default", facts[("acme", "based_in")].fact_id) if l.kind == "supports"]
    moved_episode = next(e for e in await target.documents.recent_episodes("default", 10) if "headquartered" in e.content)
    assert supports.source_episode_id == moved_episode.episode_id != episode.episode_id, "the source follows the episode's new id"
    assert supports.quote == "headquartered in Lisbon"

    again = await target.import_records("default", dump)
    assert (again.facts, again.links, again.links_skipped) == (0, 0, 3), "the same dump twice adds nothing"
    assert len(await target.fact_links("default", facts[("mark", "works_in")].fact_id)) == 2

    orphan = [r for r in dump if r["type"] != "fact" or r["subject"] != "mark" or r["predicate"] != "works_at"]
    fresh = await MemoryEngine(InMemoryDocumentStore(), InMemoryVectorIndex(), HashEmbedder()).open()
    partial = await fresh.import_records("default", orphan)
    assert (partial.facts, partial.links, partial.links_skipped) == (2, 1, 2), "both links that named the missing fact are dropped, not pointed elsewhere"
