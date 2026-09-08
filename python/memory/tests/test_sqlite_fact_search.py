"""SQLite fact postings preserve lexical, temporal, and scoped scan behavior."""
from __future__ import annotations

from collections.abc import Iterator
import json
from pathlib import Path
import sqlite3
from typing import cast

import pytest

from scone_memory.backends.sqlite import SCHEMA, SCHEMA_VERSION, SqliteDocumentStore
from scone_memory.core.models import Fact
from scone_memory.core.ports import NewFact, TextFilter
from scone_memory.retrieval.lexical import tokenize

WHEN = "2025-01-01T00:00:00Z"


@pytest.fixture
def store(tmp_path: Path) -> Iterator[SqliteDocumentStore]:
    result = SqliteDocumentStore(tmp_path/"facts.db")
    yield result
    result.conn.close()


async def fact(store: SqliteDocumentStore, subject: str, obj: str = "value", *, space: str = "alpha",
               confidence: float = 1.0, episode: int | None = None) -> Fact:
    return await store.insert_fact(NewFact(space,subject,"relates",obj,"2024-01-01T00:00:00Z",
        confidence=confidence,source_episode_id=episode,quote="source quote" if episode else None))


def source(store: SqliteDocumentStore, number: int, *, space: str = "alpha", kind: str = "file",
           prefix: str = "manuals/a", tags: tuple[str,...] = ("blue",), metadata: dict[str,str] | None = None,
           created: str = "2024-01-01T00:00:00Z") -> None:
    store.conn.execute("INSERT INTO episodes VALUES (?,?,?,?,?,?,?,?,?,?)",(number,space,kind,"source quote",str(number),
        prefix,json.dumps(tags),json.dumps(metadata if metadata is not None else {"team":"blue","priority":"9"}),created,created))
    store.conn.commit()


async def oracle(store: SqliteDocumentStore, query: str, when: str = WHEN, limit: int = 100) -> list[Fact]:
    terms = set(tokenize(query))
    scored = [(len(terms & set(tokenize(f"{row.subject} {row.predicate} {row.object}"))),row)
        for row in await store.list_facts("alpha",True) if not row.excluded and row.holds_at(when)]
    return [row for score,row in sorted(scored,key=lambda pair:(-pair[0],-pair[1].confidence,pair[1].fact_id)) if score][:limit]


@pytest.mark.parametrize("query",["don't DON'T café STRASSE", "alpha alpha beta", "the and where", "value", "東京"])
async def test_exact_tokenizer_ranking_parity(store: SqliteDocumentStore,query: str) -> None:
    await fact(store,"don't café Straße",confidence=.1)
    await fact(store,"alpha alpha","beta",confidence=.8)
    await fact(store,"alpha","gamma",confidence=.9)
    await fact(store,"alpha","gamma",confidence=.9)
    await fact(store,"東京",space="other")
    assert await store.search_facts("alpha",query,WHEN,100) == await oracle(store,query)


async def test_time_offsets_microseconds_closed_and_excluded_match_scan(store: SqliteDocumentStore) -> None:
    rows = [await fact(store,"needle") for _ in range(6)]
    changes = [("closed","2024-12-31T18:00:00-06:00",None,None),
        ("active","2025-01-01T00:00:00.000001Z",None,None),
        ("closed","2024-01-01","2025-01-01T00:00:00Z",None),
        ("active","2024-01-01",None,"excluded"),("proposed","2024-01-01",None,None),
        ("active","2024-01-01","2025-01-01T00:00:00.000001Z",None)]
    for row,(status,start,end,excluded) in zip(rows,changes):
        store.conn.execute("UPDATE facts SET status=?,valid_from=?,valid_until=?,excluded_reason=? WHERE id=?",(status,start,end,excluded,row.fact_id))
    store.conn.commit()
    assert await store.search_facts("alpha","needle",WHEN,100) == await oracle(store,"needle")
    assert [row.fact_id for row in await store.search_facts("alpha","needle",WHEN,100)] == [rows[0].fact_id,rows[5].fact_id]


@pytest.mark.parametrize("blocked",["space","missing","tags","where","kind","prefix","since","until","as_of","conditions"])
async def test_scope_filtering_happens_before_limit(store: SqliteDocumentStore,blocked: str) -> None:
    from scone_memory.retrieval.filters import Condition
    source(store,1)
    source(store,2)
    if blocked == "space": store.conn.execute("UPDATE episodes SET space='other' WHERE id=1")
    if blocked == "missing": store.conn.execute("DELETE FROM episodes WHERE id=1")
    if blocked == "tags": store.conn.execute("UPDATE episodes SET tags='[]' WHERE id=1")
    if blocked == "where": store.conn.execute("UPDATE episodes SET metadata='{}' WHERE id=1")
    if blocked == "kind": store.conn.execute("UPDATE episodes SET kind='chat' WHERE id=1")
    if blocked == "prefix": store.conn.execute("UPDATE episodes SET source='manualX/a' WHERE id=1")
    if blocked == "since": store.conn.execute("UPDATE episodes SET created_at='2023-01-01T00:00:00Z' WHERE id=1")
    if blocked in ("until","as_of"): store.conn.execute("UPDATE episodes SET created_at='2026-01-01T00:00:00Z' WHERE id=1")
    if blocked == "conditions": store.conn.execute("UPDATE episodes SET metadata=? WHERE id=1",(json.dumps({"team":"blue","priority":" 2 "}),))
    store.conn.commit()
    await fact(store,"needle needle second",confidence=1,episode=1)
    wanted = await fact(store,"needle",confidence=.1,episode=2)
    scope = TextFilter(tags=("blue",),where={"team":"blue"},kind="file",source_prefix="manuals/",
        since="2024-01-01T00:00:00Z",until=WHEN,as_of=WHEN,conditions=Condition("priority","above",8.0))
    assert await store.search_facts("alpha","needle second",WHEN,1,scope) == [wanted]


async def test_unsourced_facts_only_allowed_without_scope(store: SqliteDocumentStore) -> None:
    wanted = await fact(store,"needle")
    assert await store.search_facts("alpha","needle",WHEN,1) == [wanted]
    assert await store.search_facts("alpha","needle",WHEN,1,TextFilter()) == []


async def test_raw_old_client_updates_deletes_and_reopen_are_healed(store: SqliteDocumentStore) -> None:
    row = await fact(store,"oldtoken")
    assert await store.search_facts("alpha","oldtoken",WHEN,1) == [row]
    old = sqlite3.connect(store.path)
    old.execute("UPDATE facts SET subject='newtoken' WHERE id=?",(row.fact_id,))
    old.commit()
    old.close()
    assert await store.search_facts("alpha","oldtoken",WHEN,1) == []
    assert (await store.search_facts("alpha","newtoken",WHEN,1))[0].fact_id == row.fact_id
    reopened = SqliteDocumentStore(store.path)
    try:
        assert (await reopened.search_facts("alpha","newtoken",WHEN,1))[0].fact_id == row.fact_id
        reopened.conn.execute("DELETE FROM facts WHERE id=?",(row.fact_id,))
        reopened.conn.commit()
        assert await store.search_facts("alpha","newtoken",WHEN,1) == []
        assert store.conn.execute("SELECT count(*) FROM fact_search_postings").fetchone()[0] == 0
    finally: reopened.conn.close()


async def test_space_move_and_erase_clean_old_postings(store: SqliteDocumentStore) -> None:
    row = await fact(store,"needle")
    await store.search_facts("alpha","needle",WHEN,1)
    store.conn.execute("UPDATE facts SET space='beta' WHERE id=?",(row.fact_id,))
    store.conn.commit()
    assert await store.search_facts("alpha","needle",WHEN,1) == []
    assert len(await store.search_facts("beta","needle",WHEN,1)) == 1
    await store.delete_space("beta",WHEN)
    assert store.conn.execute("SELECT count(*) FROM fact_search_postings").fetchone()[0] == 0
    assert store.conn.execute("SELECT count(*) FROM fact_search_dirty").fetchone()[0] == 0


async def test_existing_v11_backfills_once_without_semantic_version_change(tmp_path: Path) -> None:
    path = tmp_path/"old.db"
    old = sqlite3.connect(path)
    old.executescript(SCHEMA)
    old.execute("INSERT INTO meta VALUES ('schema_version',?)",(str(SCHEMA_VERSION),))
    old.execute("INSERT INTO facts(id,space,subject,predicate,object,confidence,valid_from,status) VALUES(1,'alpha','legacy','uses','disk',1,'2024-01-01','active')")
    old.commit()
    old.close()
    current = SqliteDocumentStore(path)
    try:
        assert (await current.search_facts("alpha","legacy",WHEN,1))[0].fact_id == 1
        assert current.conn.execute("SELECT value FROM meta WHERE key='schema_version'").fetchone()[0] == str(SCHEMA_VERSION)
        assert current.conn.execute("SELECT count(*) FROM fact_search_dirty").fetchone()[0] == 0
    finally: current.conn.close()
    again = SqliteDocumentStore(path)
    try: assert again.conn.execute("SELECT count(*) FROM fact_search_dirty").fetchone()[0] == 0
    finally: again.conn.close()


async def test_healing_is_space_local_and_does_not_commit_caller_transaction(store: SqliteDocumentStore) -> None:
    row = await fact(store,"oldtoken")
    await fact(store,"other",space="beta")
    await store.search_facts("alpha","oldtoken",WHEN,1)
    assert store.conn.execute("SELECT space FROM fact_search_dirty").fetchall()[0][0] == "beta"
    store.conn.execute("BEGIN")
    store.conn.execute("UPDATE facts SET subject='newtoken' WHERE id=?",(row.fact_id,))
    assert len(await store.search_facts("alpha","newtoken",WHEN,1)) == 1
    assert store.conn.in_transaction
    store.conn.rollback()
    assert len(await store.search_facts("alpha","oldtoken",WHEN,1)) == 1
    assert await store.search_facts("alpha","newtoken",WHEN,1) == []


async def test_cache_failure_rolls_back_only_its_savepoint(store: SqliteDocumentStore,monkeypatch: pytest.MonkeyPatch) -> None:
    import scone_memory.backends.sqlite_fact_search as module
    row = await fact(store,"oldtoken")
    await store.search_facts("alpha","oldtoken",WHEN,1)
    store.conn.execute("BEGIN")
    store.conn.execute("UPDATE facts SET subject='changed' WHERE id=?",(row.fact_id,))
    def failed(text: str) -> list[str]: raise RuntimeError("index failure")
    monkeypatch.setattr(module,"tokenize",failed)
    with pytest.raises(RuntimeError): await store.search_facts("alpha","changed",WHEN,1)
    assert store.conn.in_transaction
    assert store.conn.execute("SELECT subject FROM facts WHERE id=?",(row.fact_id,)).fetchone()[0] == "changed"
    store.conn.rollback()


async def test_warm_search_uses_posting_index_and_hydrates_only_results(store: SqliteDocumentStore,monkeypatch: pytest.MonkeyPatch) -> None:
    import scone_memory.backends.sqlite as sqlite_backend
    from scone_memory.backends.sqlite_fact_search import RANK_SQL
    for i in range(100): await fact(store,f"filler{i}")
    wanted = await fact(store,"needle")
    await store.search_facts("alpha","needle",WHEN,1)
    hydrate = sqlite_backend._fact
    calls: list[int] = []
    def counted(row: sqlite3.Row) -> Fact:
        calls.append(cast(int,row["id"]))
        return hydrate(row)
    monkeypatch.setattr(sqlite_backend,"_fact",counted)
    assert await store.search_facts("alpha","needle",WHEN,1) == [wanted]
    assert calls == [wanted.fact_id]
    plan = " ".join(str(row[3]) for row in store.conn.execute("EXPLAIN QUERY PLAN "+RANK_SQL,(json.dumps(["needle"]),"alpha","alpha")))
    assert "SEARCH p USING PRIMARY KEY (space=? AND term=?)" in plan
    assert "SEARCH f USING INTEGER PRIMARY KEY (rowid=?)" in plan
    assert "SCAN facts" not in plan
    from scone_memory.backends.sqlite_fact_search import HYDRATE_SQL
    hydrate_plan = " ".join(str(row[3]) for row in store.conn.execute(
        "EXPLAIN QUERY PLAN "+HYDRATE_SQL,(json.dumps([wanted.fact_id]),"alpha")))
    assert "SEARCH f USING INTEGER PRIMARY KEY (rowid=?)" in hydrate_plan


@pytest.mark.parametrize("limit",[0,101,True,1.0])
async def test_bounded_strict_limit(store: SqliteDocumentStore,limit: object) -> None:
    with pytest.raises(ValueError): await store.search_facts("alpha","needle",WHEN,cast(int,limit))


@pytest.mark.parametrize("damage", ["postings", "dirty", "update_trigger", "noop_trigger", "wrong_table", "wrong_index", "uppercase_table"])
async def test_incomplete_or_changed_derived_schema_rebuilds_before_trusting_marker(store: SqliteDocumentStore,damage: str) -> None:
    row = await fact(store,"oldtoken")
    assert await store.search_facts("alpha","oldtoken",WHEN,1) == [row]
    old = sqlite3.connect(store.path)
    if damage in ("postings","wrong_table","uppercase_table"):
        old.execute("DROP TABLE fact_search_postings")
        if damage == "wrong_table": old.execute("CREATE TABLE fact_search_postings(wrong_column TEXT)")
        if damage == "uppercase_table": old.execute("CREATE TABLE FACT_SEARCH_POSTINGS(wrong_column TEXT)")
    elif damage == "dirty":
        # Lose a pending update by dropping its queue after a trigger ran.
        old.execute("UPDATE facts SET subject='newtoken' WHERE id=?",(row.fact_id,))
        old.execute("DROP TABLE fact_search_dirty")
    elif damage in ("update_trigger","noop_trigger"):
        old.execute("DROP TRIGGER fact_search_update")
        if damage == "noop_trigger":
            old.execute("CREATE TRIGGER fact_search_update AFTER UPDATE ON facts BEGIN SELECT 1; END")
        old.execute("UPDATE facts SET subject='newtoken' WHERE id=?",(row.fact_id,))
    else:
        old.execute("DROP INDEX fact_search_by_fact")
        old.execute("CREATE INDEX fact_search_by_fact ON facts(confidence)")
    old.commit()
    old.close()
    repaired = SqliteDocumentStore(store.path)
    try:
        expected = "newtoken" if damage in ("dirty","update_trigger","noop_trigger") else "oldtoken"
        found = await repaired.search_facts("alpha",expected,WHEN,1)
        assert [item.fact_id for item in found] == [row.fact_id]
        if expected == "newtoken": assert await repaired.search_facts("alpha","oldtoken",WHEN,1) == []
        assert repaired.conn.execute("SELECT value FROM meta WHERE key='schema_version'").fetchone()[0] == str(SCHEMA_VERSION)
        # Repaired triggers must also track subsequent legacy writes.
        repaired.conn.execute("UPDATE facts SET subject='lasttoken' WHERE id=?",(row.fact_id,))
        repaired.conn.commit()
        assert (await repaired.search_facts("alpha","lasttoken",WHEN,1))[0].fact_id == row.fact_id
        assert await repaired.search_facts("alpha",expected,WHEN,1) == []
    finally: repaired.conn.close()


async def test_schema_repair_does_not_commit_caller_transaction(store: SqliteDocumentStore) -> None:
    from scone_memory.backends.sqlite_fact_search import initialize_fact_search
    row = await fact(store,"oldtoken")
    await store.search_facts("alpha","oldtoken",WHEN,1)
    store.conn.execute("DROP TRIGGER fact_search_update")
    store.conn.commit()
    store.conn.execute("BEGIN")
    store.conn.execute("UPDATE facts SET subject='pendingtoken' WHERE id=?",(row.fact_id,))
    initialize_fact_search(store.conn)
    assert store.conn.in_transaction
    assert (await store.search_facts("alpha","pendingtoken",WHEN,1))[0].fact_id == row.fact_id
    store.conn.rollback()
    assert store.conn.execute("SELECT subject FROM facts WHERE id=?",(row.fact_id,)).fetchone()[0] == "oldtoken"
    assert store.conn.execute("SELECT name FROM sqlite_master WHERE name='fact_search_update'").fetchone() is None


async def test_schema_repair_failure_rolls_back_derived_work_only(store: SqliteDocumentStore) -> None:
    from scone_memory.backends.sqlite_fact_search import initialize_fact_search
    row = await fact(store,"oldtoken")
    await store.search_facts("alpha","oldtoken",WHEN,1)
    store.conn.execute("DROP TRIGGER fact_search_update")
    store.conn.commit()
    store.conn.execute("BEGIN")
    store.conn.execute("UPDATE facts SET subject='pendingtoken' WHERE id=?",(row.fact_id,))
    def deny_index(action: int, arg1: str | None, arg2: str | None, database: str | None, trigger: str | None) -> int:
        return sqlite3.SQLITE_DENY if action == sqlite3.SQLITE_CREATE_INDEX else sqlite3.SQLITE_OK
    store.conn.set_authorizer(deny_index)
    try:
        with pytest.raises(sqlite3.DatabaseError): initialize_fact_search(store.conn)
    finally: store.conn.set_authorizer(None)
    assert store.conn.in_transaction
    assert store.conn.execute("SELECT subject FROM facts WHERE id=?",(row.fact_id,)).fetchone()[0] == "pendingtoken"
    assert store.conn.execute("SELECT name FROM sqlite_master WHERE name='fact_search_update'").fetchone() is None
    assert store.conn.execute("SELECT count(*) FROM fact_search_postings WHERE term='oldtoken'").fetchone()[0] == 1
    store.conn.rollback()
