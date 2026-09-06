"""Any LangChain VectorStore as a vector index. The published contract runs
through the bridge over langchain_core's InMemoryVectorStore, filtered and
post-filtered (tests/test_contract.py); this file pins what the bridge
promises about the three things that vary per store: how vectors get in,
what happens when the unfiltered window runs out, and what a score means."""

from __future__ import annotations

import importlib.util
import math

import pytest

from scone_memory import HashEmbedder, InMemoryDocumentStore, MemoryEngine
from scone_memory.ports import VectorPoint

pytestmark = pytest.mark.skipif(importlib.util.find_spec("langchain_core") is None, reason="langchain-core not installed")


def bridge(**kwargs):
    from langchain_core.vectorstores import InMemoryVectorStore

    from scone_memory.backends import LangChainVectorIndex

    index = LangChainVectorIndex(**kwargs)
    return index.bind(InMemoryVectorStore(embedding=index.embeddings))


def point(chunk_id, vector, space="default", **meta):
    return VectorPoint(chunk_id, space, chunk_id, "2025-01-01T00:00:00.000Z", vector, metadata=meta)


async def test_vectors_reach_the_store_through_the_bridge_only():
    index = bridge(score="cosine_similarity")
    await index.ensure(2)
    await index.upsert([point(1, [1.0, 0.0]), point(2, [0.0, 1.0])])
    assert await index.search("default", [1.0, 0.0], 5) == [(1, pytest.approx(1.0)), (2, pytest.approx(0.0, abs=1e-9))]
    assert index.embeddings.pending == {}, "placeholders are released after the add"
    with pytest.raises(RuntimeError, match="through the bridge"):
        index.store.add_texts(["chunk:9"], ids=["9"])
    with pytest.raises(RuntimeError, match="by vector"):
        index.store.similarity_search("anything")


async def test_an_exhausted_window_is_refused_not_answered_short():
    from scone_memory.backends.langchain import WindowExhausted

    index = bridge(score="cosine_similarity", overfetch=2)
    await index.ensure(2)
    # Six out-of-scope points that all beat the in-scope one on similarity.
    await index.upsert([point(i, [1.0, 0.01 * i], space="other") for i in range(1, 7)] + [point(7, [0.0, 1.0])])
    with pytest.raises(WindowExhausted, match="filter_builder"):
        await index.search("default", [1.0, 0.0], 3)  # window 6: all six are "other"
    # A full window is fine when it already holds ``limit`` matches: nothing
    # past it can score higher than what is in it.
    full = bridge(score="cosine_similarity", overfetch=1)
    await full.ensure(2)
    await full.upsert([point(1, [1.0, 0.0]), point(2, [0.9, 0.1]), point(3, [0.8, 0.2]), point(4, [0.0, 1.0], space="other")])
    assert [cid for cid, _ in await full.search("default", [1.0, 0.0], 2)] == [1, 2]
    relaxed = bridge(score="cosine_similarity", overfetch=10)
    await relaxed.ensure(2)
    await relaxed.upsert([point(i, [1.0, 0.01 * i], space="other") for i in range(1, 7)] + [point(7, [0.0, 1.0])])
    assert [cid for cid, _ in await relaxed.search("default", [1.0, 0.0], 3)] == [7], "a wider window finds it and the store had nothing more"


async def test_the_engine_reports_a_refused_window_as_a_degraded_lane():
    index = bridge(score="cosine_similarity", overfetch=1)
    engine = await MemoryEngine(InMemoryDocumentStore(), index, HashEmbedder()).open()
    for i in range(5):
        await engine.remember("other", f"unrelated note {i} about the deploy runbook")
    await engine.remember("default", "the deploy runbook is in the ops wiki")
    result = await engine.recall("default", "deploy runbook", limit=1)
    assert result.degraded and "filter_builder" in result.degraded[0] and result.degraded[0].startswith("vectors:")
    assert [i.episode_id for i in result.items] == [6], "the lexical lane still answered"


async def test_a_filter_builder_removes_the_window():
    from scone_memory.backends.langchain import LangChainVectorIndex as Bridge

    def builder(space, as_of_ts, tags, where):
        return lambda doc: Bridge._matches(doc.metadata, space, as_of_ts, tags, where)

    index = bridge(score="cosine_similarity", overfetch=1, filter_builder=builder)
    await index.ensure(2)
    await index.upsert([point(i, [1.0, 0.01 * i], space="other") for i in range(1, 7)] + [point(7, [0.0, 1.0], owner="mark")])
    assert [cid for cid, _ in await index.search("default", [1.0, 0.0], 3)] == [7]
    assert await index.search("default", [1.0, 0.0], 3, where={"owner": "ana"}) == []


async def test_scores_are_converted_by_their_stated_meaning():
    from scone_memory.backends import LangChainVectorIndex

    index = LangChainVectorIndex(score="cosine_distance")
    assert index._similarity(0.25) == pytest.approx(0.75)
    assert LangChainVectorIndex(score="unit_l2_squared")._similarity(2.0) == pytest.approx(0.0)
    assert LangChainVectorIndex(score="cosine_similarity")._similarity(0.4) == 0.4
    assert math.isnan(LangChainVectorIndex(score="unknown")._similarity(0.4))
    with pytest.raises(ValueError, match="score must be"):
        LangChainVectorIndex(score="euclidean")


async def test_an_unknown_score_keeps_the_order_and_claims_no_similarity():
    index = bridge()  # score defaults to unknown
    engine = await MemoryEngine(InMemoryDocumentStore(), index, HashEmbedder(), similarity_floor=0.5).open()
    await engine.remember("default", "grocery list: eggs")
    await engine.remember("default", "the deploy runbook is in the ops wiki")
    result = await engine.recall("default", "deploy runbook")
    assert [i.episode_id for i in result.items][0] == 2, "the store's order still drives fusion"
    assert all(i.similarity is None for i in result.items), "no number is shown that the bridge cannot vouch for"
    assert result.top_similarity is None and result.low_confidence is None, "and no confidence is judged from it"
    order = await index.search("default", (await HashEmbedder().embed(["deploy runbook"]))[0], 5)
    assert [cid for cid, _ in order][0] == 2 and all(math.isnan(s) for _, s in order)


async def test_a_store_without_vector_search_is_refused_up_front():
    from scone_memory.backends import LangChainVectorIndex

    class NoSearch:
        pass

    with pytest.raises(TypeError, match="similarity_search_with_score_by_vector"):
        await LangChainVectorIndex(NoSearch()).ensure(2)
    with pytest.raises(ValueError, match="bind"):
        await LangChainVectorIndex().ensure(2)


@pytest.mark.skipif(importlib.util.find_spec("faiss") is None or importlib.util.find_spec("langchain_community") is None,
                    reason="faiss-cpu and langchain-community not installed")
async def test_faiss_replaces_a_point_without_deleting_what_is_not_there():
    """FAISS raises on deleting an unknown id where InMemoryVectorStore
    ignores it. A first write must not try to delete, and a rewrite of
    the same chunk must replace it."""
    import warnings

    from scone_memory.backends import LangChainVectorIndex

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        import faiss
        from langchain_community.docstore.in_memory import InMemoryDocstore
        from langchain_community.vectorstores import FAISS
        from langchain_community.vectorstores.utils import DistanceStrategy

    index = LangChainVectorIndex(score="cosine_similarity")
    index.bind(FAISS(embedding_function=index.embeddings, index=faiss.IndexFlatIP(2), docstore=InMemoryDocstore(),
                     index_to_docstore_id={}, distance_strategy=DistanceStrategy.MAX_INNER_PRODUCT, normalize_L2=True))
    await index.ensure(2)
    await index.upsert([point(1, [1.0, 0.0], owner="old")])
    await index.upsert([point(1, [0.0, 1.0], owner="new")])
    assert await index.search("default", [0.0, 1.0], 5, where={"owner": "new"}) == [(1, pytest.approx(1.0))]
    assert await index.search("default", [0.0, 1.0], 5, where={"owner": "old"}) == [], "replaced, not duplicated"
    assert index.store.index.ntotal == 1
