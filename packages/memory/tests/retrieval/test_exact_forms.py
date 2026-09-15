"""Among the passages holding a query word's family, the one holding the word itself ranks first when asked.

"billing" asked with the prefix ``bill`` is one signal, the family, and a
passage holding "bills" earns as much from it as one holding "billing".
With exact forms on, the family is still one term, counted once, but a
passage holding the query's own word has it weighed at that word's idf,
which is at least the family's: the rarer, more specific evidence. A
passage holding only a relative scores exactly as before, and a passage
holding only the exact word scores what the word alone would.
"""

from __future__ import annotations

import pytest

from scone_memory import InMemoryDocumentStore
from scone_memory.backends import SqliteDocumentStore
from scone_memory.core.ports import NewChunk, NewEpisode, TextFilter

EXACT = "The billing statement for the harbour office arrived in March."
RELATIVE = "Bills were paid at the harbour office."
NOISE = ["The garden shed needs a new roof before winter.", "Lisbon in March is mild and green.",
         "A kettle whistled in the kitchen upstairs.", "The ferry leaves the pier at seven.",
         "Snow closed the mountain road for a week.", "Her violin lesson moved to Thursday."]


@pytest.fixture(params=["memory", "sqlite"])
async def store(request, tmp_path):
    documents = InMemoryDocumentStore() if request.param == "memory" else SqliteDocumentStore(str(tmp_path / "x.db"))
    yield documents
    close = getattr(documents, "close", None)
    if close is not None:
        await close()


async def put(documents, texts: list[str]) -> list[int]:
    ids = []
    for number, text in enumerate(texts):
        episode = await documents.insert_episode(NewEpisode(space="s", kind="note", content=text, content_hash=f"h{number}",
                                                            source=None, tags=(), metadata={},
                                                            created_at="2026-09-15T00:00:00Z",
                                                            ingested_at="2026-09-15T00:00:00Z"))
        [chunk] = await documents.insert_chunks([NewChunk(episode_id=episode.episode_id, space="s", ordinal=0, start=0,
                                                           end=len(text.encode()), text=text,
                                                           created_at=episode.created_at)])
        ids.append(chunk.chunk_id)
    return ids


async def test_the_passage_holding_the_exact_word_outranks_a_relative_only_when_asked(store):
    exact, relative, *_ = await put(store, [EXACT, RELATIVE, *NOISE])
    plain = await store.search_terms("s", "billing", 5, TextFilter(), prefixes=("bill",))
    assert [chunk for chunk, _ in plain] == [relative, exact], "without it the shorter relative wins on length"
    preferred = await store.search_terms("s", "billing", 5, TextFilter(), prefixes=("bill",), exact_forms=True)
    assert [chunk for chunk, _ in preferred] == [exact, relative]


async def test_the_exact_form_is_counted_once_and_a_relative_scores_as_before(store):
    exact, relative, *_ = await put(store, [EXACT, RELATIVE, *NOISE])
    plain = dict(await store.search_terms("s", "billing", 5, TextFilter(), prefixes=("bill",)))
    preferred = dict(await store.search_terms("s", "billing", 5, TextFilter(), prefixes=("bill",), exact_forms=True))
    alone = dict(await store.search_text("s", "billing", 5, TextFilter()))
    assert preferred[exact] == pytest.approx(alone[exact], rel=1e-9), "the word's own score, not the word and its family"
    assert preferred[exact] > plain[exact]
    assert preferred[relative] == pytest.approx(plain[relative], rel=1e-12), "a passage without the word is untouched"


async def test_the_most_specific_form_a_passage_holds_sets_the_weight(store):
    """"bills" is common here and "billing" rare; a passage holding both is
    weighed at the rarer word's idf, as a passage of two "billing"s is."""
    both, twice, *_ = await put(store, ["bills billing", "billing billing", "bills paid", "bills due", "bills late",
                                        *NOISE, *NOISE])
    preferred = dict(await store.search_terms("s", "bills billing", 20, TextFilter(), prefixes=("bill",), exact_forms=True))
    alone = dict(await store.search_text("s", "billing", 20, TextFilter()))
    assert preferred[both] == pytest.approx(alone[twice], rel=1e-9)


async def test_other_query_words_and_a_family_no_query_word_is_in_keep_their_scores(store):
    exact, relative, *_ = await put(store, [EXACT, RELATIVE, *NOISE])
    plain = dict(await store.search_terms("s", "billing kettle", 5, TextFilter(), prefixes=("bill", "gard")))
    preferred = dict(await store.search_terms("s", "billing kettle", 5, TextFilter(), prefixes=("bill", "gard"),
                                              exact_forms=True))
    others = [chunk for chunk in plain if chunk not in (exact, relative)]
    assert len(others) == 2, "the kettle and the garden passages"
    assert all(preferred[chunk] == pytest.approx(plain[chunk], rel=1e-12) for chunk in others)
    assert preferred[relative] == pytest.approx(plain[relative], rel=1e-12) and preferred[exact] > plain[exact]


async def test_a_family_most_passages_hold_still_yields_to_the_exact_word(store):
    """FTS5 gives a phrase most rows hold a floor idf rather than a
    negative one; the exact word is still weighed at its own."""
    exact, *_ = await put(store, ["billing notice", "bills one", "bills two", "bills three", "bills four"])
    preferred = dict(await store.search_terms("s", "billing", 5, TextFilter(), prefixes=("bill",), exact_forms=True))
    alone = dict(await store.search_text("s", "billing", 5, TextFilter()))
    assert preferred[exact] == pytest.approx(alone[exact], rel=1e-9)
    assert max(preferred, key=preferred.__getitem__) == exact


async def test_sqlite_reads_each_family_once_and_scans_the_index_first(tmp_path):
    """Looked up per matching row, a prefix is expanded again for every row:
    seconds a query on 45,000 chunks. The plan scans the index once for the
    expression and twice per query word a family holds (the family with the
    word, and the word), and joins the rest by key."""
    documents = SqliteDocumentStore(str(tmp_path / "plan.db"))
    try:
        await put(documents, [EXACT, RELATIVE, "Travellers travelled.", "The traveller left.",
                              "We waited a while.", "Whilst they slept.", *NOISE])
        executed: list[str] = []
        documents.conn.set_trace_callback(executed.append)
        # Three families: with three, SQLite's own choice put the chunks first
        # and looked the whole expression up again for each of them.
        await documents.search_terms("s", "billing while travelling", 5, TextFilter(), prefixes=("bill", "whil", "travell"),
                                     exact_forms=True)
        documents.conn.set_trace_callback(None)
        [search] = [statement for statement in executed if "bm25(" in statement and "LIMIT" in statement]
        plan = [row[3] for row in documents.conn.execute("EXPLAIN QUERY PLAN " + search)]
        scans = [step for step in plan if "chunk_lexical_fts" in step]
        assert len(scans) == 7 and all("INDEX 0:M" in step and "=M" not in step for step in scans), plan
        assert plan.index(scans[-1]) < min(plan.index(step) for step in plan if step.startswith("SEARCH c ")), plan
    finally:
        await documents.close()


async def test_sqlite_adds_nothing_for_a_family_whose_only_member_is_the_query_word(tmp_path):
    from scone_memory.backends.sqlite_lexical import exact_form_rank

    documents = SqliteDocumentStore(str(tmp_path / "only.db"))
    try:
        await put(documents, [EXACT, *NOISE])
        await documents.search_text("s", "billing", 5, TextFilter())
        assert exact_form_rank(documents.conn, "billing", ("bill",)) == ("", "bm25(chunk_lexical_fts)", "", [])
    finally:
        await documents.close()
