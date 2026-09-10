"""An optional fact index stays compatible with validated legacy scan recall."""

import asyncio

import pytest

from scone_memory import HashEmbedder, InMemoryDocumentStore, InMemoryVectorIndex, MemoryEngine
from scone_memory.backends.sqlite import SqliteDocumentStore
from scone_memory.core.ports import TextFilter

WHEN = "2026-09-01T00:00:00Z"


@pytest.fixture
async def memory():
    engine = await MemoryEngine(InMemoryDocumentStore(), InMemoryVectorIndex(), HashEmbedder(), clock=lambda: WHEN).open()
    yield engine
    await engine.close()


@pytest.fixture
async def sqlite_memory(tmp_path):
    engine = await MemoryEngine(SqliteDocumentStore(tmp_path / "facts.db"), InMemoryVectorIndex(), HashEmbedder(), clock=lambda: WHEN).open()
    yield engine
    await engine.close()


async def add_fact(memory, subject="Juniper", value="Polaris", *, team=None, confidence=1.0):
    source = None
    if team is not None:
        source = await memory.remember("alpha", f"{subject} calibration uses {value}.", metadata={"team": team},
                                       kind="file", source=f"manuals/{team}")
    return await memory.assert_fact("alpha", subject, "calibration", value, confidence=confidence,
        valid_from="2026-01-01T00:00:00Z", source_episode_id=source.episode_id if source else None)


async def test_sqlite_index_is_preferred_without_scanning_and_matches_legacy_ranking(sqlite_memory, monkeypatch):
    from scone_memory.retrieval.fact_search import IndexedFactSearch

    await add_fact(sqlite_memory, "Juniper", "Polaris", confidence=.6)
    await add_fact(sqlite_memory, "Juniper", "Meridian", confidence=.9)
    await add_fact(sqlite_memory, "Aster", "Meridian", confidence=1.0)
    expected = await sqlite_memory._scan_facts_for_query("alpha", "Juniper calibration", WHEN)
    assert isinstance(sqlite_memory.documents, IndexedFactSearch)

    async def forbidden(*args, **kwargs):
        pytest.fail("indexed recall scanned the full fact ledger")

    monkeypatch.setattr(sqlite_memory.documents, "list_facts", forbidden)
    result = await sqlite_memory.recall("alpha", "Juniper calibration")
    assert result.facts == expected and result.degraded == []


async def test_missing_index_preserves_unsourced_manual_facts_without_degradation(memory):
    fact = await add_fact(memory)
    result = await memory.recall("alpha", "Juniper calibration")
    assert result.facts == [fact] and result.degraded == []


async def test_valid_empty_index_result_does_not_trigger_scan(memory, monkeypatch):
    await add_fact(memory)

    async def empty(*args, **kwargs):
        return []

    async def forbidden(*args, **kwargs):
        pytest.fail("an empty index result is valid")

    monkeypatch.setattr(memory.documents, "search_facts", empty, raising=False)
    monkeypatch.setattr(memory.documents, "list_facts", forbidden)
    assert (await memory.recall("alpha", "Juniper calibration")).facts == []


@pytest.mark.parametrize("malformed", ["error", "tuple", "dictionary", "non_fact", "duplicate", "over_limit",
    "wrong_space", "wrong_status", "excluded", "future", "closed", "no_overlap", "stale", "mutated_type"])
async def test_invalid_index_output_falls_back_once_with_safe_diagnostic(memory, monkeypatch, malformed):
    retained = await add_fact(memory)
    copies = {
        "wrong_space": {"space": "beta"}, "wrong_status": {"status": "proposed"},
        "excluded": {"excluded_reason": "private-policy"}, "future": {"valid_from": "2027-01-01T00:00:00Z"},
        "closed": {"valid_until": "2026-02-01T00:00:00Z"},
        "no_overlap": {"subject": "Cedar", "predicate": "maintains", "object": "Beacon"},
        "stale": {"object": "PRIVATE fabricated value"}, "mutated_type": {"fact_id": True},
    }
    scans = []
    original = memory.documents.list_facts

    async def tracked(*args, **kwargs):
        scans.append(True)
        return await original(*args, **kwargs)

    async def invalid(*args, **kwargs):
        if malformed == "error":
            raise RuntimeError("PRIVATE credentials")
        if malformed == "tuple":
            return (retained,)
        if malformed == "dictionary":
            return {"facts": [retained]}
        if malformed == "non_fact":
            return [retained.model_dump()]
        if malformed == "duplicate":
            return [retained, retained]
        if malformed == "over_limit":
            return [retained] * 11
        return [retained.model_copy(update=copies[malformed])]

    monkeypatch.setattr(memory.documents, "search_facts", invalid, raising=False)
    monkeypatch.setattr(memory.documents, "list_facts", tracked)
    result = await memory.recall("alpha", "Juniper calibration")
    assert result.facts == [retained] and scans == [True]
    assert result.degraded == ["fact_index: unavailable"] and "PRIVATE" not in str(result)


async def test_scoped_index_rows_are_checked_again_before_delivery(memory, monkeypatch):
    blocked = await add_fact(memory, team="legal")
    allowed = await add_fact(memory, "Juniper science", team="science")
    seen_scope = []

    async def blocked_index(space, query, when, limit, scope=None):
        seen_scope.append(scope)
        return [blocked]

    monkeypatch.setattr(memory.documents, "search_facts", blocked_index, raising=False)
    result = await memory.recall("alpha", "Juniper calibration", where={"team": "science"}, kind="file", source_prefix="manuals/")
    assert result.facts == [allowed] and result.degraded == ["fact_index: unavailable"]
    assert seen_scope[0].where == {"team": "science"} and seen_scope[0].kind == "file"


async def test_sqlite_scope_is_applied_before_index_limit(sqlite_memory, monkeypatch):
    for number in range(12):
        await add_fact(sqlite_memory, f"Juniper {number}", team="legal")
    allowed = await add_fact(sqlite_memory, "Juniper science", team="science")

    async def forbidden(*args, **kwargs):
        pytest.fail("valid scoped index must not scan")

    monkeypatch.setattr(sqlite_memory.documents, "list_facts", forbidden)
    result = await sqlite_memory.recall("alpha", "Juniper calibration", where={"team": "science"})
    assert result.facts == [allowed] and result.degraded == []


async def test_index_fact_is_reread_and_excluded_change_falls_back(memory, monkeypatch):
    changed = await add_fact(memory)
    allowed = await add_fact(memory, "Juniper backup")

    async def stale_index(*args, **kwargs):
        snapshot = changed.model_copy(deep=True)
        await memory.exclude("alpha", changed.fact_id, "superseded source")
        return [snapshot]

    monkeypatch.setattr(memory.documents, "search_facts", stale_index, raising=False)
    result = await memory.recall("alpha", "Juniper calibration")
    assert result.facts == [allowed] and result.degraded == ["fact_index: unavailable"]


@pytest.mark.parametrize("stage", ["index", "point_read"])
async def test_external_cancellation_propagates_without_scan(memory, monkeypatch, stage):
    fact = await add_fact(memory)

    async def cancelled(*args, **kwargs):
        raise asyncio.CancelledError()

    async def indexed(*args, **kwargs):
        return [fact]

    async def forbidden(*args, **kwargs):
        pytest.fail("cancellation must not start fallback")

    monkeypatch.setattr(memory.documents, "search_facts", cancelled if stage == "index" else indexed, raising=False)
    if stage == "point_read":
        monkeypatch.setattr(memory.documents, "get_fact", cancelled)
    monkeypatch.setattr(memory.documents, "list_facts", forbidden)
    with pytest.raises(asyncio.CancelledError):
        await memory.recall("alpha", "Juniper calibration")


async def test_private_helper_call_without_diagnostic_list_remains_compatible(memory, monkeypatch):
    fact = await add_fact(memory)

    async def invalid(*args, **kwargs):
        return None

    monkeypatch.setattr(memory.documents, "search_facts", invalid, raising=False)
    assert await memory._facts_for_query("alpha", "Juniper", WHEN) == [fact]
    assert await memory._facts_for_query("alpha", "Juniper", WHEN, scope=TextFilter(kind="file")) == []


async def test_index_cannot_mutate_scope_used_for_engine_validation(memory, monkeypatch):
    blocked = await add_fact(memory, team="legal")
    allowed = await add_fact(memory, "Juniper science", team="science")

    async def mutate_scope(space, query, when, limit, scope=None):
        scope.where.clear()
        return [blocked]

    monkeypatch.setattr(memory.documents, "search_facts", mutate_scope, raising=False)
    result = await memory.recall("alpha", "Juniper calibration", where={"team": "science"})
    assert result.facts == [allowed] and result.degraded == ["fact_index: unavailable"]


async def test_point_read_cannot_mutate_the_index_snapshot_in_place(memory, monkeypatch):
    fact = await add_fact(memory)
    original = memory.documents.get_fact

    async def indexed(*args, **kwargs):
        return [await original("alpha", fact.fact_id)]

    async def mutate_before_return(space, fact_id):
        current = await original(space, fact_id)
        current.object = "Meridian"
        return current

    monkeypatch.setattr(memory.documents, "search_facts", indexed, raising=False)
    monkeypatch.setattr(memory.documents, "get_fact", mutate_before_return)
    result = await memory.recall("alpha", "Juniper calibration")
    assert result.facts[0].object == "Meridian" and result.degraded == ["fact_index: unavailable"]


async def test_dirty_sqlite_index_in_pinned_snapshot_falls_back_without_ending_transaction(sqlite_memory):
    import sqlite3

    fact = await add_fact(sqlite_memory)
    store = sqlite_memory.documents
    store.conn.execute("BEGIN")
    store.conn.execute("SELECT count(*) FROM facts").fetchone()
    other = sqlite3.connect(store.path)
    try:
        other.execute("UPDATE facts SET confidence=0.8")
        other.commit()
    finally:
        other.close()
    degraded = []
    try:
        result = await sqlite_memory._facts_for_query("alpha", "Juniper", WHEN, degraded=degraded)
        assert result == [fact] and result[0].confidence == 1.0
        assert degraded == ["fact_index: unavailable"] and store.conn.in_transaction
    finally:
        store.conn.rollback()


async def test_public_query_length_guard_precedes_optional_index(memory, monkeypatch):
    from scone_memory.core.errors import InvalidInput
    from scone_memory.memory.engine import MAX_QUERY

    calls = []

    async def indexed(space, query, when, limit, scope=None):
        calls.append(query)
        return []

    monkeypatch.setattr(memory.documents, "search_facts", indexed, raising=False)
    await memory.recall("alpha", "x" * MAX_QUERY)
    with pytest.raises(InvalidInput, match="query"):
        await memory.recall("alpha", "x" * (MAX_QUERY + 1))
    assert calls == ["x" * MAX_QUERY]
