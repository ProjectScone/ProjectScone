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

import math

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
        # Each query word is in one passage and each family in more, so every word moves.
        await put(documents, [EXACT, RELATIVE, "Travellers travelled.", "The traveller left.", "Travelling north today.",
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
        assert exact_form_rank(documents.conn, "billing", ("bill",)) == ("", "bm25(chunk_lexical_fts)", "", [], 0)
    finally:
        await documents.close()


def test_the_credit_can_pass_the_idf_gap_but_not_the_saturated_part():
    """The credit is the gap between the two idfs times BM25's count part,
    and that part reaches k1 + 1 for a short passage: a one-word passage
    gains more than the gap, and never more than 2.2 times it."""
    from scone_memory.retrieval.lexical import Bm25

    bm25 = Bm25()
    bm25.add(0, "billing")
    for number in range(1, 40):
        bm25.add(number, "bills paid late again today for the harbour office staff members")
    for number in range(40, 400):
        bm25.add(number, f"noise words number {number} kettle garden ferry lisbon violin snow road")
    plain = dict(bm25.search("billing", 5, prefixes=("bill",)))
    preferred = dict(bm25.search("billing", 5, prefixes=("bill",), exact_forms=True))
    idf = lambda df: math.log(1 + (400 - df + 0.5) / (df + 0.5))  # noqa: E731
    gap = idf(1) - idf(40)
    assert gap < preferred[0] - plain[0] <= (bm25.k1 + 1) * gap


async def test_a_family_every_passage_of_which_holds_the_word_moves_nothing(store):
    """Where no passage holds a relative without the word, the family is as
    rare as the word, so its idf is the word's and nothing moves. A passage
    holding "billing" and "bills" is one passage of the family, not two."""
    family = await put(store, ["billing bills harbour", "billing bills office", "billing bills march", "kettle harbour",
                               "garden shed roof kettle", "lisbon march mild kettle", "ferry pier seven",
                               "snow mountain road", "violin lesson thursday", "kitchen upstairs", "green fields"])
    plain = dict(await store.search_terms("s", "billing kettle", 10, TextFilter(), prefixes=("bill",)))
    preferred = dict(await store.search_terms("s", "billing kettle", 10, TextFilter(), prefixes=("bill",), exact_forms=True))
    assert preferred == pytest.approx(plain, rel=1e-12)
    assert [chunk for chunk in sorted(plain, key=lambda chunk: -plain[chunk])][:3] == family[:3]


def _words(count: int) -> list[str]:
    """Distinct stems the rules keep whole: "bako" for "bakoing" and "bakos"."""
    return [f"{first}{vowel}{last}o" for first in "bdfgkmnprtv" for vowel in "aeiu" for last in "kmnprt"][:count]


async def test_sqlite_answers_a_query_with_more_words_than_a_join_can_hold_and_says_how_many_it_left(tmp_path):
    """Each word moved reads two tables and SQLite joins at most 64: past
    MAX_EXACT_FORMS words, the rest keep their family's weight and the
    store says how many, rather than the lane failing."""
    from scone_memory.backends.sqlite_lexical import MAX_EXACT_FORMS
    from scone_memory.retrieval.stems import prefixes

    documents = SqliteDocumentStore(str(tmp_path / "many.db"))
    try:
        words = _words(40)
        await put(documents, [text for word in words for text in (f"notes about {word}ing today", f"{word}s were here")])
        query = " ".join(f"{word}ing" for word in words)
        found = await documents.search_terms("s", query, 5, TextFilter(), prefixes=prefixes(query.split()), exact_forms=True)
        assert len(found) == 5
        assert documents.exact_forms_cut("s") == 40 - MAX_EXACT_FORMS
        await documents.search_terms("s", query, 5, TextFilter(), prefixes=prefixes(query.split()))
        assert documents.exact_forms_cut("s") == 0, "a search without exact forms cuts nothing"
        await documents.search_terms("s", query, 5, TextFilter(), prefixes=prefixes(query.split()), exact_forms=True)
        assert await documents.search_text("s", "?!", 5, TextFilter()) == []
        assert documents.exact_forms_cut("s") == 0, "nor does a search with no words, read after one that cut"
    finally:
        await documents.close()


async def test_sqlite_joins_a_repeated_word_once_and_a_word_no_row_holds_not_at_all(tmp_path):
    from scone_memory.backends.sqlite_lexical import exact_form_rank

    documents = SqliteDocumentStore(str(tmp_path / "once.db"))
    try:
        await put(documents, ["billing notice", "bills one", "bills two", "kettle"])
        await documents.search_text("s", "kettle", 5, TextFilter())
        _, _, joins, _, _ = exact_form_rank(documents.conn, " ".join(["billing"] * 31), ("bill",))
        assert joins.count("LEFT JOIN") == 2
        assert exact_form_rank(documents.conn, "billings", ("bill",)) == ("", "bm25(chunk_lexical_fts)", "", [], 0)
    finally:
        await documents.close()


async def test_past_the_bound_the_rarest_words_keep_the_credit_and_recall_says_the_rest_did_not(tmp_path, monkeypatch):
    from scone_memory import InMemoryVectorIndex, HashEmbedder, MemoryEngine
    from scone_memory.backends import sqlite_lexical

    monkeypatch.setattr(sqlite_lexical, "MAX_EXACT_FORMS", 1)
    documents = SqliteDocumentStore(str(tmp_path / "cap.db"))
    engine = await MemoryEngine(documents, InMemoryVectorIndex(), HashEmbedder()).open()
    try:
        # "billing" is in one passage and "travelling" in two: "billing" is the rarer word.
        texts = [EXACT, RELATIVE, "Bills came again.", "Travelling north today.", "Travelling south later.",
                 "They travelled far.", *NOISE]
        for text in texts:
            await engine.remember("s", text)
        # The query names "travelling" first: the bound keeps the rarer word, not the first.
        found = await engine.recall("s", "travelling billing", limit=5, lanes=("text",))
        assert "text: 1 query word(s) past MAX_EXACT_FORMS kept their family's weight" in found.degraded
        preferred = dict(await documents.search_terms("s", "travelling billing", 10, TextFilter(),
                                                      prefixes=("travell", "bill"), exact_forms=True))
        plain = dict(await documents.search_terms("s", "travelling billing", 10, TextFilter(), prefixes=("travell", "bill")))
        ids = {text: chunk for chunk, text in documents.conn.execute("SELECT id, text FROM chunks")}
        assert preferred[ids[EXACT]] > plain[ids[EXACT]], "the rarer word keeps its credit"
        assert preferred[ids["Travelling north today."]] == pytest.approx(plain[ids["Travelling north today."]], rel=1e-12)
    finally:
        await engine.close()
