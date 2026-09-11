"""The lexical tokenizer keeps words whole in every script.

It used to match ``[a-z0-9]+`` after casefolding, so any letter outside
ASCII split a word: "Zürich" became "z" and "rich", a search for Zürich
matched documents about the rich, and Chinese or Japanese text produced no
tokens at all. These tests pin what a token is now.
"""
from __future__ import annotations

import unicodedata

import pytest

from scone_memory.retrieval.lexical import Bm25, tokenize


@pytest.mark.parametrize(("text", "tokens"), [
    ("Café résumé Zürich", ["café", "résumé", "zürich"]),
    ("Müller", ["müller"]),
    ("Ångström", ["ångström"]),
    ("naïve coöperate", ["naïve", "coöperate"]),
    ("STRASSE Straße", ["strasse", "strasse"]),
])
def test_accented_latin_words_stay_whole(text: str, tokens: list[str]) -> None:
    assert tokenize(text) == tokens


def test_composed_and_decomposed_spellings_give_the_same_token() -> None:
    assert tokenize("Caf\u00e9") == tokenize("Cafe\u0301") == ["caf\u00e9"]


@pytest.mark.parametrize(("text", "tokens"), [
    ("ＡＢＣ１２３", ["abc123"]),
    ("ﬁle", ["file"]),
])
def test_compatibility_forms_fold_to_their_plain_letters(text: str, tokens: list[str]) -> None:
    assert tokenize(text) == tokens


@pytest.mark.parametrize(("text", "tokens"), [
    ("हिन्दी भाषा", ["हिन्दी", "भाषा"]),
    ("مَرْحَبًا بك", ["مَرْحَبًا", "بك"]),
    ("Привет, мир", ["привет", "мир"]),
    ("Γειά σου", ["γειά", "σου"]),
])
def test_combining_marks_do_not_break_words(text: str, tokens: list[str]) -> None:
    assert tokenize(text) == tokens


@pytest.mark.parametrize(("text", "tokens"), [
    ("東京タワー", ["東京", "京タ", "タワ", "ワー"]),
    ("猫", ["猫"]),
    ("iPhone15を買った", ["iphone15", "を買", "買っ", "った"]),
    ("서울에서", ["서울", "울에", "에서"]),
    ("東京・大阪", ["東京", "大阪"]),
    # Thai vowel signs are combining marks: pairs are whole characters,
    # never a mark cut from the letter it sits on.
    ("สวัสดี", ["สวั", "วัส", "สดี"]),
])
def test_unspaced_scripts_index_overlapping_character_pairs(text: str, tokens: list[str]) -> None:
    assert tokenize(text) == tokens


def test_ascii_behaviour_is_unchanged() -> None:
    assert tokenize("The Cat's on THE mat, isn't it?") == ["cat's", "mat", "isn't"]
    assert tokenize("Alice works_at Acme, doesn't she? 2021") == ["alice", "works", "acme", "doesn't", "she", "2021"]


def test_a_typographic_apostrophe_reads_as_a_plain_one() -> None:
    assert tokenize("doesn’t") == tokenize("doesn't") == ["doesn't"]


def test_searching_an_accented_name_does_not_match_its_fragments() -> None:
    index = Bm25()
    index.add(1, "Offices in Zürich and Geneva")
    index.add(2, "The rich get richer")
    assert [doc for doc, _ in index.search("Zürich", limit=5)] == [1]


def test_chinese_and_japanese_text_can_be_found() -> None:
    index = Bm25()
    index.add(1, "昨日は東京タワーに行った")
    index.add(2, "大阪城を見た")
    index.add(3, "我住在里斯本")
    assert [doc for doc, _ in index.search("東京", limit=5)] == [1]
    assert [doc for doc, _ in index.search("里斯本", limit=5)] == [3]


def test_hash_vectors_name_the_tokenizer_that_made_them() -> None:
    # Hash vectors are bags of tokens. Vectors hashed under different token
    # rules must not look like the same embedder in status, events or dedup.
    from scone_memory.embedders.hash import HashEmbedder
    from scone_memory.retrieval.lexical import TOKENIZER_VERSION
    assert TOKENIZER_VERSION >= 2
    assert HashEmbedder(64).id == f"hash-64-t{TOKENIZER_VERSION}-u{unicodedata.unidata_version}"


@pytest.mark.parametrize("text", [
    "The Cat's on THE mat, isn't it?",
    "Alice works_at Acme, doesn't she? 2021",
    "rock'n'roll O'Brien 3.14 x86_64 ''quoted'' it's",
])
def test_plain_ascii_reads_the_same_on_either_path(text: str) -> None:
    # ASCII text takes a fast path. One accented word forces the full
    # Unicode path, and must not change how the rest of the text reads.
    assert tokenize(text + " é") == tokenize(text) + ["é"]
