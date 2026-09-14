"""A word's family in the lexical lane, by prefix, with no stemmer's guesses in the index.

"bills", "billing" and "billed" are one word to a reader and three to the
text lane. Rather than rewriting what is indexed, a query term that ends
in a known English suffix also searches as a prefix of its stem, so the
family is found, the index is untouched, and a language the rules do not
know is left exactly as it was.
"""

from __future__ import annotations

import pytest

from scone_memory.retrieval.stems import prefixes, stem


@pytest.mark.parametrize("token, expected", [
    ("bills", "bill"), ("billing", "bill"), ("billed", "bill"), ("invoices", "invoic"), ("invoice", "invoic"),
    ("policies", "polic"), ("running", "run"), ("stopped", "stop"), ("payments", "pay"), ("payment", "pay"),
    ("quickly", "quick"), ("darkness", "dark"), ("readers", "read"), ("meetings", "meet"), ("classes", "class"),
    ("decisions", "decis"),
])
def test_a_suffix_gives_a_stem_prefix_the_whole_family_starts_with(token, expected):
    assert stem(token) == expected


@pytest.mark.parametrize("token", ["bus", "this", "genius", "cars", "lisbon", "2024", "übung", "es", "ing", "red", "is"])
def test_short_odd_or_non_english_tokens_are_left_alone(token):
    assert stem(token) is None


def test_prefixes_are_the_distinct_stems_of_the_query_in_order_without_the_token_itself():
    assert prefixes(["bills", "billing", "lisbon", "invoices"]) == ["bill", "invoic"]
    assert prefixes(["bill"]) == [] and prefixes([]) == []
