"""Synonyms a caller wrote down, applied to the text lane's query and said out loud.

The lexical lane finds the words a passage has. A passage that says
"automobile" is invisible to a query about a "car" unless someone wrote
down that the two are one; this is that list, and nothing more: no
model, no guessing, every expansion on the record so a reader can see
which words were added to which query and why.
"""

from __future__ import annotations

import pytest

from scone_memory import HashEmbedder, InMemoryDocumentStore, InMemoryVectorIndex, MemoryEngine
from scone_memory.core.errors import InvalidInput
from scone_memory.retrieval import synonyms as module
from scone_memory.retrieval.synonyms import Synonyms


GROUPS = [["car", "automobile", "motor car"], ["invoice", "bill"]]


def test_a_matched_word_adds_the_rest_of_its_group_and_says_so():
    found = Synonyms(GROUPS).expand("Where did I park the car?")
    assert found.matched == ("car",)
    assert found.added == ("automobile", "motor car")
    assert found.text_query == "Where did I park the car? automobile motor car"
    assert found.query == "Where did I park the car?" and found.capped is False


def test_matching_ignores_case_and_possessives_like_the_lane_does():
    found = Synonyms(GROUPS).expand("The Car's colour")
    assert found.matched == ("car",) and found.added == ("automobile", "motor car")


def test_a_phrase_matches_as_a_phrase_not_as_its_words():
    found = Synonyms(GROUPS).expand("a motor car in the drive")
    assert found.matched == ("motor car",) and found.added == ("car", "automobile")
    assert Synonyms(GROUPS).expand("the motor is loud").added == ()


def test_words_the_query_already_has_are_not_added_again():
    found = Synonyms(GROUPS).expand("car or automobile")
    assert found.matched == ("car", "automobile") and found.added == ("motor car",)


def test_a_query_with_no_synonyms_is_left_exactly_as_it_was():
    found = Synonyms(GROUPS).expand("Where is the ledger?")
    assert found.added == () and found.matched == () and found.text_query == "Where is the ledger?"


def test_added_terms_are_capped_and_the_offered_count_is_honest(monkeypatch):
    monkeypatch.setattr(module, "MAX_ADDED", 2)
    found = Synonyms([["car", "auto", "automobile", "motor", "vehicle"]]).expand("the car")
    assert found.added == ("auto", "automobile") and found.capped and found.offered == 4


def test_the_list_is_bounded_and_refuses_rather_than_cuts(monkeypatch):
    monkeypatch.setattr(module, "MAX_GROUPS", 1)
    with pytest.raises(InvalidInput, match="groups"):
        Synonyms(GROUPS)
    monkeypatch.setattr(module, "MAX_GROUPS", 100)
    monkeypatch.setattr(module, "MAX_PER_GROUP", 2)
    with pytest.raises(InvalidInput, match="terms"):
        Synonyms([["a", "b", "c"]])
    with pytest.raises(InvalidInput, match="characters"):
        Synonyms([["x" * (module.MAX_TERM_CHARS + 1), "y"]])
    with pytest.raises(InvalidInput, match="one term"):
        Synonyms([["alone"]])
    with pytest.raises(InvalidInput, match="nothing"):
        Synonyms([["the", "a"]]), "a term the tokenizer drops entirely is not a term"


def test_lines_are_groups_with_comments_and_blanks_ignored():
    text = "# vehicles\ncar, automobile , motor car\n\ninvoice,bill\n"
    parsed = Synonyms.from_lines(text)
    assert parsed.record() == {"groups": 2, "terms": 5}
    assert parsed.expand("the bill").added == ("invoice",)


def test_a_file_is_read_the_same_way(tmp_path):
    path = tmp_path / "synonyms.txt"
    path.write_text("car, automobile\n")
    assert Synonyms.from_file(path).expand("car").added == ("automobile",)
    with pytest.raises(InvalidInput, match="not found"):
        Synonyms.from_file(tmp_path / "missing.txt")


def test_the_record_of_an_expansion_is_content_light():
    found = Synonyms(GROUPS).expand("the car")
    assert found.record() == {"matched": ["car"], "added": ["automobile", "motor car"], "offered": 2, "capped": False}


async def test_recall_searches_the_text_lane_with_the_added_words_and_says_so():
    engine = await MemoryEngine(InMemoryDocumentStore(), InMemoryVectorIndex(), HashEmbedder(),
                                synonyms=Synonyms(GROUPS)).open()
    await engine.remember("s", "The automobile is parked on Elm Street behind the bakery.")
    await engine.remember("s", "The invoices for March were sent late.")
    found = await engine.recall("s", "car", limit=2)
    assert [item.text for item in found.items][0].startswith("The automobile")
    assert found.items[0].lanes.get("text") == 1, "the text lane found it through the added word"
    assert found.expansion == {"matched": ["car"], "added": ["automobile", "motor car"], "offered": 2, "capped": False}
    plain = await engine.recall("s", "ledger", limit=2)
    assert plain.expansion is None, "a query nothing expanded carries no expansion record"


async def test_without_a_list_recall_is_unchanged():
    engine = await MemoryEngine(InMemoryDocumentStore(), InMemoryVectorIndex(), HashEmbedder()).open()
    await engine.remember("s", "The automobile is parked on Elm Street behind the bakery.")
    found = await engine.recall("s", "car", limit=2)
    assert found.expansion is None
    assert all(item.lanes.get("text") is None for item in found.items)


async def test_the_expansion_is_on_the_recall_event():
    from scone_memory import InMemoryEventLog

    events = InMemoryEventLog()
    engine = await MemoryEngine(InMemoryDocumentStore(), InMemoryVectorIndex(), HashEmbedder(),
                                events=events, synonyms=Synonyms(GROUPS)).open()
    await engine.remember("s", "The automobile is parked.")
    await engine.recall("s", "car", limit=1)
    [event] = await engine.events.query("s", kind="recall")
    assert event.payload["synonyms"] == {"matched": 1, "added": 2, "capped": False}


def test_the_engine_refuses_a_list_that_is_not_synonyms():
    with pytest.raises(InvalidInput, match="Synonyms"):
        MemoryEngine(InMemoryDocumentStore(), InMemoryVectorIndex(), HashEmbedder(), synonyms=GROUPS)  # type: ignore[arg-type]


def test_the_longest_term_wins_where_a_phrase_begins_with_a_term():
    listed = Synonyms([["credit", "trust"], ["credit card", "charge card"]])
    found = listed.expand("my credit card")
    assert found.matched == ("credit card",) and found.added == ("charge card",)
