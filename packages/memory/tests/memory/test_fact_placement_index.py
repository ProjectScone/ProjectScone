"""A store's placement index must answer exactly as a scan of the slot's whole history does.

``FactPlacementIndex.facts_placing`` exists so that placing a fact does not read every fact the subject and
predicate ever held. It may return more than is needed; it must never return less, or a fact is placed wrongly
and nothing says so. These tests write a few hundred facts straight into each store, in every shape a
timestamp is stored in and every status, and compare the index with the scan at many instants.

Run against every document store the session has: in-memory and SQLite always, the servers when their URL is
set.
"""
from __future__ import annotations

import random
from datetime import datetime, timedelta, timezone

import pytest

from scone_memory.core.models import Fact
from scone_memory.core.ports import FactPlacementIndex, LEDGER_STATUSES, NewFact
from scone_memory.core.timeutil import format_rfc3339, parse_rfc3339
from scone_memory.memory import fact_placement

SPACE, SUBJECT, PREDICATE = "plant", "pump-3", "reads"
BASE = datetime(2026, 3, 1, 12, 0, 0, tzinfo=timezone.utc)
OBJECTS = ("70", "80", "90", "open")


def written(instant: datetime, shape: int) -> str:
    """One instant in a shape a store may hold. Only shape 0 is the canonical one the engine writes."""
    if shape == 0:
        return format_rfc3339(instant)
    if shape == 1:  # microseconds and a Z
        return instant.strftime("%Y-%m-%dT%H:%M:%S.%f") + "Z"
    if shape == 2:  # an offset, so the text sorts nowhere near its instant
        return instant.astimezone(timezone(timedelta(hours=5, minutes=30))).isoformat()
    if shape == 3:  # whole seconds, as a hand-written clock returns
        return instant.replace(microsecond=0).strftime("%Y-%m-%dT%H:%M:%SZ")
    return instant.astimezone(timezone(timedelta(hours=-8))).isoformat()


def instants(rng: random.Random, count: int) -> list[datetime]:
    """Instants that collide and nearly collide: same instant, 1 microsecond, 400 microseconds and 1 ms apart."""
    out: list[datetime] = []
    for _ in range(count):
        base = BASE + timedelta(seconds=rng.randrange(0, 600))
        out.append(base + timedelta(microseconds=rng.choice((0, 0, 1, 400, 999, 1000, 1001, 500_000))))
    return out


async def fill(documents, rng: random.Random, canonical_only: bool) -> list[Fact]:
    """Facts of the slot under test, among others that must never be returned for it."""
    points = instants(rng, 90)
    facts: list[Fact] = []
    for number in range(260):
        start = rng.choice(points)
        # An end at the start itself is a claim cut short the instant it began; the ledger stores those.
        end = None if rng.random() < 0.25 else start + timedelta(microseconds=rng.choice((0, 1, 1000, 30_000_000, 240_000_000)))
        shape = 0 if canonical_only else rng.randrange(5)
        other = number % 7
        facts.append(await documents.insert_fact(NewFact(
            space=SPACE if other != 1 else "elsewhere",
            subject=SUBJECT if other != 2 else "pump-4",
            predicate=PREDICATE if other != 3 else "is",
            object=rng.choice(OBJECTS),
            valid_from=written(start, shape),
            valid_until=None if end is None else written(end, 0 if canonical_only else rng.randrange(5)),
            status=rng.choice(("active", "closed", "closed", "proposed", "declined")))))
    return facts


def needed(history: list[Fact], start: str, object: str | None, exclude_id: int | None) -> set[int]:
    """What the contract obliges an index to return, worked out from the whole history."""
    at = parse_rfc3339(start)
    rivals = [f for f in history if f.space == SPACE and f.subject == SUBJECT and f.predicate == PREDICATE
              and f.status in LEDGER_STATUSES and f.fact_id != exclude_id and (object is None or f.object == object)]
    covering = {f.fact_id for f in rivals if parse_rfc3339(f.valid_from) <= at
                and (f.valid_until is None or parse_rfc3339(f.valid_until) > at)}
    later = [f for f in rivals if parse_rfc3339(f.valid_from) > at]
    if later:
        covering.add(min(later, key=lambda f: (parse_rfc3339(f.valid_from), f.fact_id)).fact_id)
    return covering


class Scanning:
    """The same store with its placement index hidden, so ``place`` reads the slot's whole history."""

    def __init__(self, documents) -> None:
        self._documents = documents

    def __getattr__(self, name: str):
        if name == "facts_placing":
            raise AttributeError(name)
        return getattr(self._documents, name)


def summary(placement: fact_placement.Placement) -> tuple:
    return (sorted(f.fact_id for f in placement.covering), placement.restates and placement.restates.fact_id,
            placement.bound, placement.bound_reason, placement.bound_by,
            placement.resumes and placement.resumes.first.valid_from)


@pytest.mark.parametrize("canonical_only", [True, False], ids=["canonical timestamps", "every stored shape"])
async def test_the_index_returns_everything_a_scan_of_the_slot_would_need(ledger_engine, canonical_only):
    documents = ledger_engine.documents
    if not isinstance(documents, FactPlacementIndex):
        pytest.skip(f"{type(documents).__name__} has no placement index; the engine scans its slots")
    assert not isinstance(Scanning(documents), FactPlacementIndex), "the scan must really be a scan"
    rng = random.Random(20261009 + canonical_only)
    history = await fill(documents, rng, canonical_only)
    in_slot = {f.fact_id for f in history if (f.space, f.subject, f.predicate) == (SPACE, SUBJECT, PREDICATE)}
    probes = instants(rng, 70) + [BASE - timedelta(days=1), BASE + timedelta(days=1)]
    cases: list[tuple[str, str | None, int | None]] = []
    for probe in probes:
        start = written(probe, rng.randrange(5))
        object = rng.choice((None, None, *OBJECTS))
        cases.append((start, object, rng.choice((None, None, rng.choice(sorted(in_slot))))))
        # Leave out each fact the placement rests on in turn: the next one down has to take its place.
        cases.extend((start, object, fact_id) for fact_id in sorted(needed(history, start, object, None)))
    checked = 0
    for start, object, exclude_id in cases:
        found = await documents.facts_placing(SPACE, SUBJECT, PREDICATE, start, object=object, exclude_id=exclude_id)
        ids = [f.fact_id for f in found]
        assert len(ids) == len(set(ids)), "a fact is returned once"
        assert set(ids) <= in_slot, "only facts of the slot are returned"
        missing = needed(history, start, object, exclude_id) - set(ids)
        assert not missing, f"placing at {start} (object={object}, exclude={exclude_id}) needs facts {sorted(missing)}"
        for many_valued in (False, True):
            target = object or "70"
            indexed = await fact_placement.place(documents, SPACE, SUBJECT, PREDICATE, target, start, exclude_id,
                                                 many_valued=many_valued)
            scanned = await fact_placement.place(Scanning(documents), SPACE, SUBJECT, PREDICATE, target, start,
                                                 exclude_id, many_valued=many_valued)
            assert summary(indexed) == summary(scanned), f"placement at {start} differs from the scan's"
            checked += 1
    assert checked == 2 * len(cases) and len(cases) > 3 * len(probes)


async def test_an_index_reads_a_few_facts_where_a_scan_reads_the_slots_history(ledger_engine):
    """The point of the index: appending to a long history returns the open fact, not the history."""
    documents = ledger_engine.documents
    if not isinstance(documents, FactPlacementIndex):
        pytest.skip(f"{type(documents).__name__} has no placement index")
    for number in range(300):
        await ledger_engine.assert_fact(SPACE, SUBJECT, PREDICATE, str(number % 11),
                                        valid_from=format_rfc3339(BASE + timedelta(seconds=number)))
    assert len(await documents.facts_for(SPACE, SUBJECT, PREDICATE)) == 300
    at_the_end = await documents.facts_placing(SPACE, SUBJECT, PREDICATE, format_rfc3339(BASE + timedelta(seconds=300)))
    in_the_middle = await documents.facts_placing(SPACE, SUBJECT, PREDICATE,
                                                  format_rfc3339(BASE + timedelta(seconds=150, milliseconds=500)))
    assert len(at_the_end) == 1 and at_the_end[0].valid_until is None
    assert len(in_the_middle) <= 2, "the fact that covers the instant and the one that starts next"


async def test_facts_asserted_through_the_engine_are_placed_as_a_scan_places_them(ledger_engine, monkeypatch):
    """End to end, out of order and with restatements: one sequence replayed through the index and through the
    scan leaves the same ledger."""
    documents = ledger_engine.documents
    if not isinstance(documents, FactPlacementIndex):
        pytest.skip(f"{type(documents).__name__} has no placement index")
    rng = random.Random(4)
    steps = [(rng.randrange(0, 90), rng.choice(OBJECTS)) for _ in range(140)]

    async def replay(subject: str) -> list[tuple]:
        for second, value in steps:
            await ledger_engine.assert_fact(SPACE, subject, PREDICATE, value,
                                            valid_from=format_rfc3339(BASE + timedelta(seconds=second)))
        return sorted((f.valid_from, f.valid_until or "", f.object, f.status)
                      for f in await documents.facts_for(SPACE, subject, PREDICATE))

    indexed = await replay("indexed-pump")

    class NoStoreHasThis:  # with the index out of sight, ``place`` scans
        pass

    monkeypatch.setattr(fact_placement, "FactPlacementIndex", NoStoreHasThis)
    scanned = await replay("scanned-pump")
    assert indexed == scanned
    assert len(indexed) > 20 and sum(until == "" for _, until, _, _ in indexed) == 1


async def _strip_time_keys(documents, fact_ids: list[int]) -> None:
    """Leave these facts as a build before the time keys stored them."""
    if type(documents).__name__ == "MongoDocumentStore":
        from scone_memory.backends.mongo import _ENDS, _STARTS

        await documents.facts.update_many({"_id": {"$in": fact_ids}}, {"$unset": {_STARTS: "", _ENDS: ""}})
    else:
        from scone_memory.backends.elastic import FACT_ENDS, FACT_STARTS

        for fact_id in fact_ids:
            await documents.client.update(
                index=documents._idx("facts"), id=str(fact_id), refresh=True,
                script={"source": "ctx._source.remove(params.a); ctx._source.remove(params.b)",
                        "params": {"a": FACT_STARTS, "b": FACT_ENDS}})


async def test_facts_stored_before_the_time_keys_are_placed_rightly_and_keyed_on_open(ledger_engine):
    """MongoDB and Elasticsearch keep the keys as stored fields. A store written by an earlier build has facts
    without them: those must be returned as candidates until they are keyed, never skipped."""
    documents = ledger_engine.documents
    if not hasattr(documents, "_key_fact_times"):
        pytest.skip(f"{type(documents).__name__} keeps no stored time keys")
    rng = random.Random(77)
    history = await fill(documents, rng, canonical_only=False)
    unkeyed = [f.fact_id for f in history if f.fact_id % 2]
    await _strip_time_keys(documents, unkeyed)
    probes = [(written(probe, rng.randrange(5)), rng.choice((None, *OBJECTS))) for probe in instants(rng, 40)]

    async def check() -> int:
        returned = 0
        for start, object in probes:
            found = {f.fact_id for f in await documents.facts_placing(SPACE, SUBJECT, PREDICATE, start, object=object)}
            missing = needed(history, start, object, None) - found
            assert not missing, f"placing at {start} needs facts {sorted(missing)}"
            returned += len(found)
        return returned

    before = await check()
    assert await documents._key_fact_times() == len(unkeyed)
    assert await documents._key_fact_times() == 0, "keyed once"
    after = await check()
    assert after < before, "once keyed, facts that cannot matter are no longer returned"


def _plan_nodes(node: dict):
    yield node
    for child in (*node.get("Plans", ()), *node.get("inputStages", ()), *filter(None, [node.get("inputStage")])):
        yield from _plan_nodes(child)


async def _work_done(documents, start: str, object: str | None = None, predicate: str = PREDICATE) -> tuple[set[str], int]:
    """The indexes the store's own lookup used for ``start`` and how many rows or index entries it read, as the
    database reports them: -1 where it reports the plan and no counts. Skips a store that reports neither."""
    kind = type(documents).__name__
    used: set[str] = set()
    if kind == "SqliteDocumentStore":
        from scone_memory.backends.sqlite import _placing_queries

        slot = "space=? AND subject=? AND predicate=? AND " + ("" if object is None else "object=? AND ")
        for query, values in _placing_queries(SPACE, SUBJECT, predicate, start, object, None):
            (detail,) = [row["detail"] for row in documents.conn.execute("EXPLAIN QUERY PLAN " + query, values)]
            index, _, bound = detail.partition("USING INDEX ")[2].partition(" (")
            # SQLite reports no counts. The plan must bind the time key as well as the slot, or it reads the slot.
            assert bound.startswith(slot + "<expr>"), detail
            used.add(index)
        return used, -1
    if kind == "PostgresDocumentStore":
        from scone_memory.core.timeutil import canonical_floor

        names = (SPACE, SUBJECT, predicate) if object is None else (SPACE, SUBJECT, predicate, object)
        query, values = documents._placing_query(names, canonical_floor(start), None)
        (row,) = await documents._rows("EXPLAIN (ANALYZE, FORMAT JSON) " + query, values)
        nodes = list(_plan_nodes(row["QUERY PLAN"][0]["Plan"]))
        kinds = [node["Node Type"] for node in nodes]
        # A sort means the rows that start next were fetched and ordered, where the index holds them in order.
        assert not {"Sort", "Seq Scan", "Incremental Sort"} & set(kinds), kinds
        scans = [node for node in nodes if "Index Name" in node]
        assert len(scans) == 3, kinds
        read = sum(node["Actual Rows"] * node["Actual Loops"] + node.get("Rows Removed by Filter", 0)
                   + node.get("Rows Removed by Index Recheck", 0) for node in nodes if "Scan" in node["Node Type"])
        return {node["Index Name"] for node in scans}, int(read)
    if kind == "MongoDocumentStore":
        read = 0
        for cursor in documents._placing_finds(SPACE, SUBJECT, predicate, start, object, None):
            stats = (await cursor.explain())["executionStats"]
            read += stats["totalDocsExamined"]
            scans = [stage for stage in _plan_nodes(stats["executionStages"]) if stage["stage"] == "IXSCAN"]
            assert scans and not any(stage["stage"] == "COLLSCAN" for stage in _plan_nodes(stats["executionStages"]))
            used.update(stage["indexName"] for stage in scans)
        return used, read
    pytest.skip(f"{kind} does not report the work a query did")


async def test_the_database_reads_the_slot_indexes_and_not_the_slots_history(ledger_engine):
    """What each database says it did. A lookup that returns the right facts by reading every fact the subject
    holds passes every test above; this one fails it. Nothing here refreshes the table's statistics, which is
    the state a store is in while it is being filled. The counts are rows read, whether or not returned."""
    documents = ledger_engine.documents
    for number in range(400):
        await ledger_engine.assert_fact(SPACE, SUBJECT, PREDICATE, str(number % 11),
                                        valid_from=format_rfc3339(BASE + timedelta(seconds=number)))
        # The same subject under other predicates: an index on the subject alone reaches all of these too.
        await ledger_engine.assert_fact(SPACE, SUBJECT, f"note-{number % 5}", str(number),
                                        valid_from=format_rfc3339(BASE + timedelta(seconds=number)))
    used, at_the_end = await _work_done(documents, format_rfc3339(BASE + timedelta(seconds=400)))
    assert used == {name for name in used if name.endswith(("slot_starts", "slot_ends", "placing_starts", "placing_ends"))}
    assert len(used) == 2, used
    _, near_the_end = await _work_done(documents, format_rfc3339(BASE + timedelta(seconds=389, milliseconds=500)))
    _, in_the_middle = await _work_done(documents, format_rfc3339(BASE + timedelta(seconds=200, milliseconds=500)))
    if at_the_end >= 0:
        assert at_the_end <= 2, "an append reads the open fact"
        assert near_the_end <= 6 and in_the_middle <= 6, "a fact dated in the past reads the one it lands in and the next few"


async def test_a_predicate_holding_many_objects_reads_the_facts_of_one_object(ledger_engine):
    """A predicate that holds many objects at once (a camera that saw three hundred things, all still true) is
    placed among the facts of its own object. Without an index that names the object, every assert reads the
    three hundred."""
    documents = ledger_engine.documents
    if not isinstance(documents, FactPlacementIndex):
        pytest.skip(f"{type(documents).__name__} has no placement index")
    for number in range(300):
        await documents.insert_fact(NewFact(space=SPACE, subject=SUBJECT, predicate="saw", object=f"object-{number}",
                                            valid_from=format_rfc3339(BASE + timedelta(seconds=number)), status="active"))
    later = format_rfc3339(BASE + timedelta(seconds=900))
    assert len(await documents.facts_placing(SPACE, SUBJECT, "saw", later)) == 300
    (only,) = await documents.facts_placing(SPACE, SUBJECT, "saw", later, object="object-7")
    assert only.object == "object-7"
    assert await documents.facts_placing(SPACE, SUBJECT, "saw", later, object="object-300") == []
    # Rewritten to another object, the fact leaves the first object's facts and joins the second's.
    await documents.update_fact(only.model_copy(update={"object": "object-7b"}))
    assert await documents.facts_placing(SPACE, SUBJECT, "saw", later, object="object-7") == []
    assert [f.fact_id for f in await documents.facts_placing(SPACE, SUBJECT, "saw", later, object="object-7b")] == [only.fact_id]
    used, read = await _work_done(documents, later, object="object-9", predicate="saw")
    assert len(used) == 2 and all("object" in name for name in used), used
    assert read <= 2, "the one fact that names the object, and not the three hundred"


async def test_the_fact_that_starts_next_is_found_past_any_number_out_of_the_ledger(ledger_engine):
    """Proposed and declined facts are not rivals. However many of them start first, the first fact in the
    ledger after them is the one that starts next."""
    documents = ledger_engine.documents
    if not isinstance(documents, FactPlacementIndex):
        pytest.skip(f"{type(documents).__name__} has no placement index")
    for number in range(1, 23):
        await documents.insert_fact(NewFact(space=SPACE, subject=SUBJECT, predicate=PREDICATE, object="70",
                                            valid_from=format_rfc3339(BASE + timedelta(seconds=number)),
                                            valid_until=format_rfc3339(BASE + timedelta(seconds=number, milliseconds=1)),
                                            status="proposed" if number % 2 else "declined"))
    async def in_ledger(start: datetime, exclude_id: int | None = None) -> list[int]:
        found = await documents.facts_placing(SPACE, SUBJECT, PREDICATE, format_rfc3339(start), exclude_id=exclude_id)
        return [f.fact_id for f in found if f.in_ledger]

    for start in (BASE, BASE + timedelta(seconds=14, milliseconds=500)):
        assert await in_ledger(start) == []
    rival = await documents.insert_fact(NewFact(space=SPACE, subject=SUBJECT, predicate=PREDICATE, object="80",
                                                valid_from=format_rfc3339(BASE + timedelta(seconds=30)), status="active"))
    for start in (BASE, BASE + timedelta(seconds=14, milliseconds=500), BASE + timedelta(seconds=22, milliseconds=500)):
        assert await in_ledger(start) == [rival.fact_id]
    assert await in_ledger(BASE, exclude_id=rival.fact_id) == []
    if type(documents).__name__ == "PostgresDocumentStore":
        # Postgres reads the rows that start next a page at a time; each page begins where the last ended.
        from scone_memory.backends.postgres import _STARTS_NEXT_PAGE

        statements = []
        reading, documents._rows = documents._rows, lambda *args: statements.append(args) or reading(*args)
        assert await in_ledger(BASE) == [rival.fact_id]
        assert len(statements) == -(-23 // _STARTS_NEXT_PAGE), "22 facts out of the ledger and the one in it, in whole pages"


@pytest.mark.parametrize("analysed", [False, True], ids=["no statistics", "fresh statistics"])
async def test_postgres_walks_a_long_slot_in_order_whatever_its_planner_knows(ledger_engine, analysed):
    """Postgres chooses a plan by estimate and takes no hints. On a slot of thirty thousand facts it once read
    every later fact and sorted them to find the one that starts next: right answers, ten thousand rows read.
    The plan is checked at a size where the planner's choice costs something, with and without statistics."""
    documents = ledger_engine.documents
    if type(documents).__name__ != "PostgresDocumentStore":
        pytest.skip("a check of the Postgres planner")
    facts = f"{documents.schema}.facts"
    stamp = "to_char(timestamptz '2026-03-01T12:00:00Z' + n * interval '1 second', 'YYYY-MM-DD\"T\"HH24:MI:SS\".000Z\"')"
    await documents._rows(
        f"INSERT INTO {facts} (space, subject, predicate, object, confidence, valid_from, valid_until, status)"
        f" SELECT %s, %s, p, (n %% 11)::text, 1.0, {stamp}, {stamp.replace('n *', '(n + 1) *')}, 'closed'"
        " FROM generate_series(0, 29999) n, unnest(%s::text[]) p RETURNING NULL",
        (SPACE, SUBJECT, [PREDICATE, "note"]))
    if analysed:
        async with documents.pool.connection() as conn:
            await conn.execute(f"ANALYZE {facts}")
    assert len(await documents.facts_placing(SPACE, SUBJECT, PREDICATE, format_rfc3339(BASE + timedelta(seconds=15000.5)))) == 2
    for seconds in (30000, 29989.5, 15000.5, 10.5):
        used, read = await _work_done(documents, format_rfc3339(BASE + timedelta(seconds=seconds)))
        assert used <= {"facts_placing_starts", "facts_placing_ends"}, used
        assert read <= 6, f"placing {30000 - seconds:.0f} seconds back read {read} rows of the slot's 30,000"
