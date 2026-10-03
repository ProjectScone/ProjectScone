"""The in-memory text lane scores only the documents that hold a query term.

A document holding none of the query's terms or family members scores zero
and is never returned, so the index keeps, for every term, the documents
that hold it and scores only those. These tests hold the ranking to one that
scores every document, on queries with repeated words, word families,
narrowed candidate lists and documents rewritten or removed, and on this
package's own documentation.
"""

from __future__ import annotations

from collections import Counter
import math
import random

from scone_memory.ingestion.chunker import chunk_spans
from scone_memory.retrieval.lexical import Bm25, fold_diacritics, tokenize
from scone_memory.retrieval.stems import prefixes as stem_prefixes

from ..paths import PACKAGE_ROOT


def every_document(index: Bm25, query: str, limit: int, allowed=None, prefixes=()) -> list[tuple[int, float]]:
    """BM25 as it reads when every document is scored."""
    docs = {doc_id: Counter(fold_diacritics(token) for token in tokenize(text)) for doc_id, text in index.texts.items()}
    df: Counter[str] = Counter()
    for counts in docs.values():
        df.update(counts.keys())
    terms = [fold_diacritics(token) for token in tokenize(query)]
    families = [(fold_diacritics(prefix), [term for term in df if term.startswith(fold_diacritics(prefix))])
                for prefix in prefixes]
    families = [(prefix, members) for prefix, members in families if members]
    covered = {term for term in terms if any(term.startswith(prefix) for prefix, _ in families)}
    terms = [term for term in terms if term not in covered]
    if (not terms and not families) or not docs:
        return []
    n = len(docs)
    lengths = {doc_id: sum(counts.values()) for doc_id, counts in docs.items()}
    avg_len = sum(lengths.values()) / n
    # A document holding two members of a family is one document of it.
    family_df = {prefix: sum(1 for counts in docs.values() if any(counts.get(member) for member in members))
                 for prefix, members in families}
    candidates = docs.keys() if allowed is None else [d for d in allowed if d in docs]
    scored = []
    for doc_id in candidates:
        counts = docs[doc_id]
        weighed = [(counts[term], df[term]) for term in terms if counts.get(term)]
        for prefix, members in families:
            tf = sum(counts.get(member, 0) for member in members)
            if tf:
                weighed.append((tf, family_df[prefix]))
        score = 0.0
        for tf, dfs in weighed:
            idf = math.log(1 + (n - dfs + 0.5) / (dfs + 0.5))
            score += idf * tf * (index.k1 + 1) / (tf + index.k1 * (1 - index.b + index.b * lengths[doc_id] / avg_len))
        if score > 0:
            scored.append((doc_id, score))
    scored.sort(key=lambda pair: (-pair[1], pair[0]))
    return scored[:limit]


class Recorded(Bm25):
    """A Bm25 that remembers the text of what it holds, for the reference."""

    def __init__(self) -> None:
        super().__init__()
        self.texts: dict[int, str] = {}

    def add(self, doc_id: int, text: str) -> None:
        super().add(doc_id, text)
        self.texts[doc_id] = text

    def remove(self, doc_id: int) -> None:
        super().remove(doc_id)
        self.texts.pop(doc_id, None)


def filled(texts: list[str]) -> Recorded:
    index = Recorded()
    for doc_id, text in enumerate(texts, start=1):
        index.add(doc_id, text)
    return index


BILLS = [
    "The billing run sends bills on the first; billed accounts are marked paid.",
    "A bill of lading travels with the shipment, not with the invoice.",
    "Invoices go out after billing closes; invoicing runs nightly.",
    "Caf\u00e9 receipts are filed under caf\u00e9, the accent folded away.",
    "Nothing here is about money at all, only the weather in Lisbon.",
    "bill bill bill bill: a document that repeats its one word.",
]


def test_the_ranking_is_the_one_every_document_scored_gives():
    index = filled(BILLS)
    for query, prefixes in (("billing invoices", ()), ("billing invoices", ("bill", "invoic")),
                            ("bill bill lading", ()), ("cafe receipts", ()), ("weather", ("weath",)),
                            ("unknown words only", ()), ("the and of", ()), ("billing", ("zzz",))):
        for limit in (1, 3, 10):
            assert index.search(query, limit, prefixes=prefixes) == every_document(index, query, limit, None, prefixes)


def test_a_narrowed_candidate_list_is_kept_in_its_own_order_and_repeats():
    index = filled(BILLS)
    for allowed in ([6, 2, 1], [1, 1, 3, 99], [], [5]):
        found = index.search("bill invoices", 10, allowed=allowed, prefixes=("bill",))
        assert found == every_document(index, "bill invoices", 10, allowed, ("bill",))


def test_a_rewritten_or_removed_document_is_scored_by_what_it_now_holds():
    index = filled(BILLS)
    index.add(2, "Rewritten: nothing about shipping or bills any more.")
    index.remove(6)
    index.remove(42)
    for query, prefixes in (("lading", ()), ("bill", ("bill",)), ("rewritten nothing", ())):
        assert index.search(query, 10, prefixes=prefixes) == every_document(index, query, 10, None, prefixes)
    assert index.search("lading", 10) == []


def test_a_term_no_document_holds_any_more_is_forgotten():
    index = filled(["alpha beta", "beta gamma"])
    index.remove(1)
    assert set(index._postings) == set(index._df) == {"beta", "gamma"}
    index.remove(2)
    assert index._postings == {} and not index._df


def test_this_packages_documentation_ranks_as_every_document_scored_gives():
    texts = []
    for path in sorted((PACKAGE_ROOT / "docs").rglob("*.md"))[:12]:
        content = path.read_text(encoding="utf-8")
        texts.extend(content[span.start:span.end] for span in chunk_spans(content, 700))
    index = filled(texts)
    chooser = random.Random(9)
    words = [token for text in texts for token in tokenize(text)]
    for _ in range(40):
        query = " ".join(chooser.choice(words) for _ in range(chooser.randint(1, 4)))
        found_prefixes = stem_prefixes(tokenize(query))
        assert index.search(query, 20, prefixes=found_prefixes) == every_document(index, query, 20, None, found_prefixes)
        assert index.search(query, 20) == every_document(index, query, 20)
