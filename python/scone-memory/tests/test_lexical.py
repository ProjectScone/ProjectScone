from scone_memory.lexical import Bm25, tokenize


def test_more_query_terms_rank_higher():
    index = Bm25()
    index.add(1, "the cat sat on the mat")
    index.add(2, "a cat and a dog sat together")
    index.add(3, "quarterly revenue exceeded expectations")
    ranked = [doc for doc, _ in index.search("cat dog sat", limit=10)]
    assert ranked[0] == 2
    assert 3 not in ranked


def test_removed_documents_stop_counting():
    index = Bm25()
    index.add(1, "lisbon flat")
    index.add(2, "porto flat")
    index.remove(1)
    assert [doc for doc, _ in index.search("lisbon", limit=5)] == []
    assert len(index) == 1


def test_tokenizer_drops_stopwords_and_case():
    assert tokenize("The Cat's on THE mat, isn't it?") == ["cat's", "mat", "isn't"]
