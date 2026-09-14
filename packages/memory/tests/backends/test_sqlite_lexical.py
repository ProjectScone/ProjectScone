"""The SQLite text lane ranks our own tokens, kept current by triggers and rebuilt when the tokenizer moves."""

from __future__ import annotations

from scone_memory import HashEmbedder, InMemoryVectorIndex, MemoryEngine
from scone_memory.backends import SqliteDocumentStore
from scone_memory.backends import sqlite_lexical as module
from scone_memory.core.ports import TextFilter


async def test_rows_follow_writes_and_deletions_and_a_query_finds_a_part_of_an_unspaced_run(tmp_path):
    store = SqliteDocumentStore(str(tmp_path / "l.db"))
    engine = await MemoryEngine(store, InMemoryVectorIndex(), HashEmbedder()).open()
    kept = await engine.remember("s", "東京タワーは1958年に完成した。")
    await engine.remember("s", "Don't forget the résumé for the interview.")
    assert store.conn.execute("SELECT count(*) FROM chunk_lexical_dirty").fetchone()[0] == 2, "writes are marked, not tokenised yet"
    found = await store.search_terms("s", "東京", 5, TextFilter(), prefixes=())
    assert len(found) == 1 and store.conn.execute("SELECT count(*) FROM chunk_lexical_dirty").fetchone()[0] == 0
    assert [c for c, _ in await store.search_terms("s", "resume", 5, TextFilter(), prefixes=())] != [], "accents folded on both sides"
    assert [c for c, _ in await store.search_terms("s", "don't", 5, TextFilter(), prefixes=())] != [], "an apostrophe inside a token survives"
    await engine.forget("s", kept.episode_id)
    assert await store.search_terms("s", "東京", 5, TextFilter(), prefixes=()) == []
    assert store.conn.execute("SELECT count(*) FROM chunk_lexical").fetchone()[0] == 1, "the deleted chunk's row went with it"
    assert store.conn.execute("SELECT count(*) FROM chunk_lexical_fts WHERE chunk_lexical_fts MATCH '\"東京\"'").fetchone()[0] == 0


async def test_a_new_tokenizer_version_rebuilds_the_index_lazily(tmp_path, monkeypatch):
    path = str(tmp_path / "v.db")
    store = SqliteDocumentStore(path)
    engine = await MemoryEngine(store, InMemoryVectorIndex(), HashEmbedder()).open()
    await engine.remember("s", "The billing run went out late.")
    assert await store.search_terms("s", "billing", 5, TextFilter(), prefixes=()) != []
    await engine.close()
    monkeypatch.setattr(module, "_VERSION", module._VERSION + ";changed")
    reopened = SqliteDocumentStore(path)
    assert reopened.conn.execute("SELECT value FROM meta WHERE key=?", (module._VERSION_KEY,)).fetchone()[0].endswith(";changed")
    assert reopened.conn.execute("SELECT count(*) FROM chunk_lexical").fetchone()[0] == 0, "rebuilt whole: rows dropped"
    assert reopened.conn.execute("SELECT count(*) FROM chunk_lexical_dirty").fetchone()[0] == 1, "every chunk marked to be re-read"
    assert await reopened.search_terms("s", "billing", 5, TextFilter(), prefixes=()) != [], "and re-read at the next query"


def test_the_match_expression_folds_and_quotes_like_the_index():
    assert module.lexical_match("Résumé, don't", ("bill",)) == '"resume" OR "don\u02bct" OR "bill"*'
    assert module.lexical_match("   ") is None
