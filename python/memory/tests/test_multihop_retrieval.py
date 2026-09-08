"""Offline, source-verified bounded traversal, exercised on both local stores."""
from __future__ import annotations

from unittest.mock import AsyncMock

import pytest

from scone_memory import HashEmbedder, InMemoryDocumentStore, InMemoryVectorIndex, MemoryEngine
from scone_memory.backends.sqlite import SqliteDocumentStore
from scone_memory.core.models import RecallResult
from scone_memory.core.ports import NewFact, NewFactLink, TextFilter
from scone_memory.retrieval.multihop import MultiHopLimits, expand_multihop

STAMP = "2025-01-01T00:00:00Z"


@pytest.fixture(params=["memory", "sqlite"])
async def engine(request, tmp_path):
    store = InMemoryDocumentStore() if request.param == "memory" else SqliteDocumentStore(tmp_path / "hops.db")
    instance = await MemoryEngine(store, InMemoryVectorIndex(), HashEmbedder()).open()
    yield instance
    await instance.close()


async def fact(engine, subject, predicate, obj, *, space="alpha", at=STAMP, **episode_options):
    quote = f"{subject} {predicate} {obj}."
    source = await engine.remember(space, quote, created_at=at, **episode_options)
    return await engine.documents.insert_fact(NewFact(space=space, subject=subject, predicate=predicate,
        object=obj, valid_from=at, source_episode_id=source.episode_id, quote=quote))


async def chain(engine, **episode_options):
    return [await fact(engine, "Aster", "depends on", "Beacon", **episode_options),
            await fact(engine, "Beacon", "owned by", "Cedar", **episode_options),
            await fact(engine, "Cedar", "located in", "Denver", **episode_options)]


async def link(engine, left, right, kind="supports", *, source=None, at=STAMP, quote=None):
    return await engine.documents.insert_fact_link(NewFactLink(space="alpha", from_fact=left.fact_id,
        to_fact=right.fact_id, kind=kind, created_at=at,
        source_episode_id=source if source is not None else left.source_episode_id,
        quote=quote if quote is not None else left.quote))


async def test_three_source_chain_is_read_only_and_has_exact_directed_path(engine):
    facts = await chain(engine)
    revision = await engine.documents.revision("alpha")
    engine.documents.list_facts = AsyncMock(side_effect=AssertionError("no ledger scan"))
    engine.documents.fact_links = AsyncMock(side_effect=AssertionError("no unbounded links"))
    result = await expand_multihop(engine.documents, "alpha", seeds=RecallResult(facts=[facts[0]]))
    assert result.facts == facts
    assert {f.source_episode_id for f in result.facts} == {f.source_episode_id for f in facts}
    assert [e.kind for e in result.edges] == ["subject_object", "subject_object"]
    assert result.paths[-1].fact_ids == [f.fact_id for f in facts]
    assert result.paths[-1].directions == ["forward", "forward"]
    assert result.coverage.complete
    assert await engine.documents.revision("alpha") == revision


async def test_stored_contradiction_retains_direction_and_never_selects_winner(engine):
    left = await fact(engine, "Aster", "is", "green")
    right = await fact(engine, "Aster", "is", "red")
    stored = await link(engine, right, left, "contradicts")
    result = await expand_multihop(engine.documents, "alpha", seed_fact_ids=[left.fact_id])
    edge = result.edges[0]
    assert edge.kind == "contradicts"
    assert (edge.from_fact, edge.to_fact, edge.link_id) == (right.fact_id, left.fact_id, stored.link_id)
    assert result.paths[-1].directions == ["reverse"]
    assert {f.object for f in result.facts} == {"green", "red"}


@pytest.mark.parametrize("scope", [TextFilter(tags=("public",)), TextFilter(where={"team": "blue"}),
    TextFilter(kind="note"), TextFilter(source_prefix="public/"), TextFilter(since="2025-01-01T00:00:00Z"),
    TextFilter(until="2025-02-01T00:00:00Z")])
async def test_scope_applies_at_every_fact_and_seed(engine, scope):
    first = await fact(engine, "Aster", "depends on", "Beacon", tags=["public"], metadata={"team": "blue"},
                       kind="note", source="public/a")
    hidden_at = "2024-01-01T00:00:00Z" if scope.since else "2025-03-01T00:00:00Z"
    hidden = await fact(engine, "Beacon", "secret", "hidden", at=hidden_at, kind="conversation", source="private/a")
    result = await expand_multihop(engine.documents, "alpha", seed_fact_ids=[first.fact_id], scope=scope)
    assert [f.fact_id for f in result.facts] == [first.fact_id]
    assert "hidden" not in result.model_dump_json()
    result = await expand_multihop(engine.documents, "alpha", seed_fact_ids=[hidden.fact_id], scope=scope)
    assert result.facts == []


async def test_link_source_scope_and_quote_are_independently_verified(engine):
    left, right, _ = await chain(engine, tags=["public"])
    # Avoid a subject/object chain so only the stored relation could expand.
    right = await fact(engine, "Other", "is", "value", tags=["public"])
    hidden = await engine.remember("alpha", "secret link justification", tags=["private"])
    await link(engine, left, right, source=hidden.episode_id, quote="secret link justification")
    result = await expand_multihop(engine.documents, "alpha", seed_fact_ids=[left.fact_id], scope=TextFilter(tags=("public",)))
    assert right.fact_id not in [f.fact_id for f in result.facts]
    assert "secret" not in result.model_dump_json()


async def test_current_asof_history_and_future_link(engine):
    first = await fact(engine, "Aster", "depends on", "Beacon")
    past = await fact(engine, "Beacon", "located in", "Paris")
    await engine.documents.update_fact(past.model_copy(update={"status": "closed", "valid_until": "2025-06-01T00:00:00Z"}))
    current = await fact(engine, "Beacon", "located in", "Denver", at="2025-06-01T00:00:00Z")
    distant = await fact(engine, "Other", "is", "late")
    await link(engine, first, distant, at="2025-08-01T00:00:00Z")
    current_result = await expand_multihop(engine.documents, "alpha", seed_fact_ids=[first.fact_id])
    assert past.fact_id not in [f.fact_id for f in current_result.facts]
    assert current.fact_id in [f.fact_id for f in current_result.facts]
    past_result = await expand_multihop(engine.documents, "alpha", seed_fact_ids=[first.fact_id], scope=TextFilter(as_of="2025-03-01"))
    assert [f.fact_id for f in past_result.facts] == [first.fact_id, past.fact_id]
    history = await expand_multihop(engine.documents, "alpha", seed_fact_ids=[first.fact_id], include_history=True)
    assert {past.fact_id, current.fact_id}.issubset({f.fact_id for f in history.facts})


@pytest.mark.parametrize("change", [{"status": "proposed"}, {"status": "declined"}, {"excluded_reason": "excluded"},
    {"quote": "invented quotation"}, {"source_episode_id": None}])
async def test_untrusted_fact_cannot_be_a_seed_or_bridge(engine, change):
    facts = await chain(engine)
    changed = facts[1].model_copy(update=change)
    # SQLite update_fact intentionally doesn't update quote/source; insert a changed record directly.
    data = changed.model_dump(exclude={"fact_id"})
    await engine.documents.update_fact(facts[1].model_copy(update={"excluded_reason": "replaced"}))
    bad = await engine.documents.insert_fact(NewFact(**data))
    result = await expand_multihop(engine.documents, "alpha", seed_fact_ids=[facts[0].fact_id, bad.fact_id])
    assert [f.fact_id for f in result.facts] == [facts[0].fact_id]


async def test_deleted_source_dangling_links_cycles_and_crossspace(engine):
    facts = await chain(engine)
    await link(engine, facts[2], facts[0])
    foreign = await fact(engine, "Denver", "secret", "other tenant", space="beta")
    await engine.documents.insert_fact_link(NewFactLink(space="alpha", from_fact=facts[0].fact_id,
        to_fact=foreign.fact_id, kind="supports", created_at=STAMP, source_episode_id=facts[0].source_episode_id, quote=facts[0].quote))
    result = await expand_multihop(engine.documents, "alpha", seed_fact_ids=[facts[0].fact_id, foreign.fact_id])
    assert {f.fact_id for f in result.facts} == {f.fact_id for f in facts}
    assert all(len(path.fact_ids) == len(set(path.fact_ids)) for path in result.paths)
    assert result.counts.store_calls <= 30
    await engine.forget("alpha", facts[1].source_episode_id)
    after = await expand_multihop(engine.documents, "alpha", seed_fact_ids=[facts[1].fact_id])
    assert after.facts == []


@pytest.mark.parametrize("limits,reason", [(MultiHopLimits(max_hops=1), "max_hops"),
    (MultiHopLimits(max_nodes=1), "max_nodes"), (MultiHopLimits(max_edges=1), "max_edges"),
    (MultiHopLimits(max_store_calls=1), "max_store_calls"), (MultiHopLimits(max_candidates=1), "max_candidates"),
    (MultiHopLimits(max_bytes=1024), "max_bytes")])
async def test_limits_are_hard_and_coverage_is_visible(engine, limits, reason):
    facts = await chain(engine)
    result = await expand_multihop(engine.documents, "alpha", seed_fact_ids=[facts[0].fact_id], limits=limits)
    assert result.coverage.truncated and not result.coverage.complete
    assert reason in result.coverage.reasons
    assert len(result.facts) <= limits.max_nodes
    assert len(result.edges) <= limits.max_edges
    assert result.counts.store_calls <= limits.max_store_calls
    assert result.counts.candidates <= limits.max_candidates
    assert len(result.model_dump_json().encode()) <= limits.max_bytes


async def test_unsupported_backend_is_seed_only_and_never_scans(engine):
    facts = await chain(engine)

    class BasicReader:
        get_fact = engine.documents.get_fact
        get_episode = engine.documents.get_episode

    result = await expand_multihop(BasicReader(), "alpha", seed_fact_ids=[facts[0].fact_id])
    assert result.facts == [facts[0]]
    assert not result.coverage.complete
    assert "unsupported_bounded_links" in result.coverage.reasons
    assert "unsupported_bounded_subjects" in result.coverage.reasons


async def test_stale_recall_seed_is_rejected(engine):
    facts = await chain(engine)
    stale = facts[0].model_copy(update={"object": "forged"})
    result = await expand_multihop(engine.documents, "alpha", seeds=RecallResult(facts=[stale]))
    assert result.facts == []


async def test_exact_entity_join_does_not_use_semantic_similarity(engine):
    first = await fact(engine, "Aster", "depends on", "Beacon")
    await fact(engine, "beacon", "located in", "Paris")
    await fact(engine, "Beacon project", "located in", "Rome")
    result = await expand_multihop(engine.documents, "alpha", seed_fact_ids=[first.fact_id])
    assert result.facts == [first]
    assert result.edges == []


async def test_metadata_conditions_and_excluded_session_apply_to_bridges(engine):
    from scone_memory.retrieval.filters import parse_filter

    first = await fact(engine, "Aster", "depends on", "Beacon", metadata={"priority": "10"})
    middle = await fact(engine, "Beacon", "owned by", "Cedar", metadata={"priority": "1", "session_id": "blocked"})
    await fact(engine, "Cedar", "located in", "Denver", metadata={"priority": "10"})
    for options in ({"scope": TextFilter(conditions=parse_filter({"field": "priority", "above": 5}))},
                    {"exclude_session_id": "blocked"}):
        result = await expand_multihop(engine.documents, "alpha", seed_fact_ids=[first.fact_id, middle.fact_id], **options)
        assert result.facts == [first]


@pytest.mark.parametrize("mode", ["missing", "bad_quote", "future_source", "wrong_space"])
async def test_link_provenance_blocks_expansion(engine, mode):
    first = await fact(engine, "Aster", "is", "green")
    second = await fact(engine, "Other", "is", "red")
    source = await engine.remember("beta" if mode == "wrong_space" else "alpha", "Aster contradicts Other.",
                                   created_at="2035-01-01" if mode == "future_source" else STAMP)
    await link(engine, first, second, "contradicts", source=source.episode_id,
               quote="forged" if mode == "bad_quote" else "Aster contradicts Other.")
    if mode == "missing":
        await engine.forget("alpha", source.episode_id)
    result = await expand_multihop(engine.documents, "alpha", seed_fact_ids=[first.fact_id])
    assert result.facts == [first]
    assert result.edges == []
    assert result.counts.rejected >= 1


async def test_scope_applies_to_link_sources_all_filter_dimensions(engine):
    from scone_memory.retrieval.filters import parse_filter

    first = await fact(engine, "Aster", "is", "green", metadata={"priority": "10", "team": "blue"},
                       tags=["public"], source="public/a")
    other = await fact(engine, "Other", "is", "red", metadata={"priority": "10", "team": "blue"},
                       tags=["public"], source="public/b")
    source = await engine.remember("alpha", "Aster contradicts Other.", kind="conversation", source="private/a", created_at="2025-03-01")
    await link(engine, first, other, "contradicts", source=source.episode_id, quote="Aster contradicts Other.")
    for scope in (TextFilter(tags=("public",)), TextFilter(where={"team": "blue"}), TextFilter(kind="note"),
                  TextFilter(source_prefix="public/"), TextFilter(until="2025-02-01"),
                  TextFilter(conditions=parse_filter({"field": "priority", "above": 5}))):
        result = await expand_multihop(engine.documents, "alpha", seed_fact_ids=[first.fact_id], scope=scope)
        assert result.facts == [first]
        assert result.edges == []


async def test_asof_uses_instants_and_exact_end_is_excluded(engine):
    first = await fact(engine, "Aster", "depends on", "Beacon", at="2025-01-01T01:00:00+01:00")
    old = await fact(engine, "Beacon", "is", "old", at="2025-01-01T01:00:00+01:00")
    await engine.documents.update_fact(old.model_copy(update={"status": "closed", "valid_until": "2025-01-02T01:00:00+01:00"}))
    early = await expand_multihop(engine.documents, "alpha", seed_fact_ids=[first.fact_id], scope=TextFilter(as_of="2025-01-01T00:00:00Z"))
    assert {f.fact_id for f in early.facts} == {first.fact_id, old.fact_id}
    ended = await expand_multihop(engine.documents, "alpha", seed_fact_ids=[first.fact_id], scope=TextFilter(as_of="2025-01-02T00:00:00Z"))
    assert ended.facts == [first]


async def test_dense_fanout_reports_postfilter_window_and_bounded_work(engine):
    first = await fact(engine, "Aster", "depends on", "Beacon", tags=["public"])
    for number in range(20):
        await fact(engine, "Beacon", "candidate", str(number), tags=["private"])
    eligible = await fact(engine, "Beacon", "candidate", "eligible", tags=["public"])
    reader = engine.documents.facts_by_subject
    engine.documents.facts_by_subject = AsyncMock(wraps=reader)
    result = await expand_multihop(engine.documents, "alpha", seed_fact_ids=[first.fact_id], scope=TextFilter(tags=("public",)),
                                  limits=MultiHopLimits(per_node_limit=3))
    assert eligible.fact_id not in [f.fact_id for f in result.facts]
    assert result.coverage.truncated and "candidate_window" in result.coverage.reasons
    assert result.counts.candidates <= 5
    assert result.counts.store_calls <= 16
    engine.documents.facts_by_subject.assert_awaited_once_with("alpha", "Beacon", 4)


async def test_output_byte_budget_includes_unicode_paths_and_counters(engine):
    first = await fact(engine, "Aster", "depends on", "Beacon")
    await fact(engine, "Beacon", "description", "雪" * 1000)
    for budget in (512, 1024, 2048, 8192):
        result = await expand_multihop(engine.documents, "alpha", seed_fact_ids=[first.fact_id], limits=MultiHopLimits(max_bytes=budget))
        assert len(result.model_dump_json().encode()) <= budget
        assert result.counts.output_bytes == len(result.model_dump_json().encode())
        retained_ids = {f.fact_id for f in result.facts}
        assert all(set(path.fact_ids) <= retained_ids for path in result.paths)
        assert all({edge.from_fact, edge.to_fact} <= retained_ids for edge in result.edges)


async def test_bounded_store_lookup_preserves_order_directions_duplicates_and_cleanup(engine):
    store = engine.documents
    facts = await chain(engine)
    first = await link(engine, facts[2], facts[0], "supports")
    second = await link(engine, facts[0], facts[2], "contradicts")
    self_link = await link(engine, facts[0], facts[0], "supports")
    assert await link(engine, facts[2], facts[0], "supports") == first
    assert await store.fact_links_from("alpha", facts[0].fact_id, 1000) == [first, second, self_link]
    assert await store.fact_links_from("alpha", facts[0].fact_id, 1) == [first]
    assert await store.fact_links_from("beta", facts[0].fact_id, 1000) == []
    assert await store.fact_links_from("alpha", facts[0].fact_id, -1) == []
    assert await store.facts_by_subject("alpha", "Beacon", 1) == [facts[1]]
    assert await store.facts_by_subject("beta", "Beacon", 10) == []
    assert await store.facts_by_subject("alpha", "Beacon", 0) == []
    await store.delete_space("alpha", STAMP)
    assert await store.fact_links_from("alpha", facts[0].fact_id, 1000) == []
    assert await store.facts_by_subject("alpha", "Beacon", 1000) == []
    fresh = await store.insert_fact(NewFact(space="alpha", subject="Beacon", predicate="is", object="new", valid_from=STAMP))
    assert await store.facts_by_subject("alpha", "Beacon", 1000) == [fresh]


async def test_memory_indices_never_scan_global_or_full_adjacency_maps():
    store = InMemoryDocumentStore()
    desired = await store.insert_fact(NewFact(space="alpha", subject="Beacon", predicate="is", object="first", valid_from=STAMP))
    wanted_link = await store.insert_fact_link(NewFactLink(space="alpha", from_fact=1, to_fact=2, kind="supports", created_at=STAMP))
    for number in range(1000):
        await store.insert_fact(NewFact(space="alpha", subject=f"unrelated{number}", predicate="is", object="other", valid_from=STAMP))
        await store.insert_fact_link(NewFactLink(space="alpha", from_fact=1000 + number, to_fact=10000 + number,
                                               kind="supports", created_at=STAMP))
    for number in range(1000):
        await store.insert_fact_link(NewFactLink(space="alpha", from_fact=1, to_fact=20000 + number, kind="supports", created_at=STAMP))

    class NoScan(dict):
        def __iter__(self):
            raise AssertionError("no global iteration")

        def values(self):
            raise AssertionError("no global values")

        def items(self):
            raise AssertionError("no global items")

    class CountReads(NoScan):
        reads = 0

        def __getitem__(self, key):
            self.reads += 1
            return super().__getitem__(key)

    store._facts = CountReads(store._facts)
    store._links = CountReads(store._links)
    assert await store.facts_by_subject("alpha", "Beacon", 4) == [desired]
    assert store._facts.reads == 1
    assert (await store.fact_links_from("alpha", 1, 4))[0] == wanted_link
    assert store._links.reads == 4
    assert await store.insert_fact_link(NewFactLink(space="alpha", from_fact=1, to_fact=2, kind="supports", created_at=STAMP)) == wanted_link


async def test_sqlite_candidate_work_is_independent_of_unrelated_rows(tmp_path):
    store = SqliteDocumentStore(tmp_path / "steps.db")
    await store.insert_fact(NewFact(space="alpha", subject="Beacon", predicate="is", object="first", valid_from=STAMP))
    await store.insert_fact_link(NewFactLink(space="alpha", from_fact=1, to_fact=2, kind="supports", created_at=STAMP))

    async def count_steps():
        steps = 0

        def progress():
            nonlocal steps
            steps += 1
            return 0

        store.conn.set_progress_handler(progress, 1)
        try:
            assert len(await store.facts_by_subject("alpha", "Beacon", 4)) == 1
            assert len(await store.fact_links_from("alpha", 1, 4)) == 1
        finally:
            store.conn.set_progress_handler(None, 0)
        return steps

    await count_steps()  # Warm SQLite statement preparation before comparing VM work.
    before = await count_steps()
    for number in range(1000):
        await store.insert_fact(NewFact(space="alpha", subject=f"unrelated{number}", predicate="is", object="other", valid_from=STAMP))
        await store.insert_fact_link(NewFactLink(space="alpha", from_fact=1000 + number, to_fact=10000 + number,
                                               kind="supports", created_at=STAMP))
    after = await count_steps()
    assert after <= before + 10, (before, after)
    store.conn.close()


async def test_empty_source_prefix_does_not_admit_missing_source_label(engine):
    first = await fact(engine, "Aster", "is", "green")
    result = await expand_multihop(engine.documents, "alpha", seed_fact_ids=[first.fact_id], scope=TextFilter(source_prefix=""))
    assert result.facts == []


async def test_final_revalidation_prunes_deleted_source_and_every_dependent_path(engine):
    facts = await chain(engine)
    read = engine.documents.get_fact
    deleted = False

    async def interleaved(space, fact_id):
        nonlocal deleted
        if fact_id == facts[1].fact_id and not deleted:
            deleted = True
            await engine.documents.delete_episode("alpha", facts[0].source_episode_id)
        return await read(space, fact_id)

    engine.documents.get_fact = interleaved
    result = await expand_multihop(engine.documents, "alpha", seed_fact_ids=[facts[0].fact_id])
    assert result.facts == []
    assert result.edges == []
    assert result.paths == []
    assert "stale_evidence" in result.coverage.reasons
    assert facts[0].quote not in result.model_dump_json()


async def test_final_revalidation_prunes_changed_fact_and_link_source(engine):
    first = await fact(engine, "Aster", "is", "green")
    second = await fact(engine, "Other", "is", "red")
    source = await engine.remember("alpha", "Aster contradicts Other.", created_at=STAMP)
    await link(engine, first, second, "contradicts", source=source.episode_id, quote="Aster contradicts Other.")
    read = engine.documents.facts_by_subject
    changed = False

    async def interleaved(space, subject, limit):
        nonlocal changed
        if not changed:
            changed = True
            await engine.documents.delete_episode("alpha", source.episode_id)
        return await read(space, subject, limit)

    engine.documents.facts_by_subject = interleaved
    result = await expand_multihop(engine.documents, "alpha", seed_fact_ids=[first.fact_id])
    assert result.facts == [first]
    assert result.edges == []
    assert [path.fact_ids for path in result.paths] == [[first.fact_id]]
    assert "stale_evidence" in result.coverage.reasons


async def test_unchecked_final_records_are_omitted_when_work_budget_runs_out(engine):
    facts = await chain(engine)
    result = await expand_multihop(engine.documents, "alpha", seed_fact_ids=[facts[0].fact_id],
                                  limits=MultiHopLimits(max_store_calls=12))
    assert result.counts.store_calls <= 12
    assert result.facts == []
    assert not result.coverage.complete
    assert "max_store_calls" in result.coverage.reasons


async def test_final_revalidation_rejects_changed_link_record(engine):
    first = await fact(engine, "Aster", "is", "green")
    second = await fact(engine, "Other", "is", "red")
    relation = await link(engine, first, second)
    read = engine.documents.get_fact_link

    async def changed(space, link_id):
        stored = await read(space, link_id)
        return stored.model_copy(update={"kind": "contradicts"}) if stored is not None else None

    engine.documents.get_fact_link = changed
    result = await expand_multihop(engine.documents, "alpha", seed_fact_ids=[first.fact_id])
    assert result.facts == [first]
    assert result.edges == []
    assert "stale_evidence" in result.coverage.reasons
    assert await read("beta", relation.link_id) is None


async def test_revision_guard_rejects_engine_deletion_during_final_source_checks(engine):
    facts = await chain(engine)
    read = engine.documents.get_episode
    first_reads = 0
    deleted = False

    async def interleaved(space, episode_id):
        nonlocal first_reads, deleted
        if episode_id == facts[0].source_episode_id:
            first_reads += 1
        if first_reads >= 2 and episode_id == facts[1].source_episode_id and not deleted:
            deleted = True
            await engine.documents.delete_episode("alpha", facts[0].source_episode_id)
            await engine.documents.bump_revision("alpha")
        return await read(space, episode_id)

    engine.documents.get_episode = interleaved
    result = await expand_multihop(engine.documents, "alpha", seed_fact_ids=[facts[0].fact_id])
    assert result.facts == []
    assert result.paths == []
    assert "stale_evidence" in result.coverage.reasons
