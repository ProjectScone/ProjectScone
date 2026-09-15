"""The north star sweep runs one embedder on both sides, hashed or a real model.

Every earlier row used the hashed-token embedder. With a real model the
sweep's cost is embedding, so both sides embed through wrappers over one
vector cache, and the result says what each side embedded and how long it
took. A chunk axis names characters or tokens, and each is the engine the
row says it is.
"""

from __future__ import annotations

import importlib.util
import io
import sys

import pytest

from scone_memory import HashEmbedder
from scone_memory.bench.comparative import CachedEmbedder
from scone_memory.ingestion.embedding_cache import InMemoryEmbeddingCache

from ..paths import PACKAGE_ROOT
from .test_comparative import ITEMS, Worded

#: As the script samples them: an item with no evidence session is not scored.
SCORED = [item for item in ITEMS if item.has_evidence]

SCRIPT = PACKAGE_ROOT / "benchmarks" / "northstar_defaults.py"
spec = importlib.util.spec_from_file_location("northstar_defaults_benchmark", SCRIPT)
assert spec is not None and spec.loader is not None
northstar = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = northstar
spec.loader.exec_module(northstar)


def test_a_chunk_axis_names_characters_or_tokens_and_refuses_anything_else():
    assert northstar.chunkings("700,2000,512t") == [northstar.Chunking(700, None), northstar.Chunking(2000, None),
                                                    northstar.Chunking(None, 512)]
    assert [c.label for c in northstar.chunkings("700,512t")] == ["chunk=700", "chunk_tokens=512"]
    for bad in ("700,big", "t512", "0", "-5t"):
        with pytest.raises(SystemExit, match="--chunks"):
            northstar.chunkings(bad)


async def test_each_chunking_builds_the_engine_its_row_names():
    by_characters = await northstar.open_engine(northstar.Chunking(2000, None), HashEmbedder())
    assert by_characters.chunk_target == 2000 and by_characters.chunk_tokens is None
    by_tokens = await northstar.open_engine(northstar.Chunking(None, 64), HashEmbedder())
    assert by_tokens.chunk_tokens == 64


def test_the_embedders_are_hashed_and_uncached_unless_asked(tmp_path):
    ours, theirs, cache = northstar.embedders("hash", model_cache=None, vector_cache=None, max_entries=10)
    assert type(ours) is HashEmbedder and type(theirs) is HashEmbedder and cache is None
    ours, theirs, cache = northstar.embedders("hash", model_cache=None, vector_cache=str(tmp_path / "v.sqlite"),
                                              max_entries=10)
    assert isinstance(ours, CachedEmbedder) and isinstance(theirs, CachedEmbedder) and ours is not theirs
    assert ours.cache is theirs.cache is cache and cache.record()["max_entries"] == 10


async def test_a_sweep_on_a_cached_model_embeds_each_text_once_and_says_what_each_side_cost():
    pytest.importorskip("llama_index.core")
    pytest.importorskip("llama_index.retrievers.bm25")
    cache = InMemoryEmbeddingCache(10_000)
    model = Worded()

    async def run():
        return await northstar.sweep(SCORED, chunkings=[northstar.Chunking(700, None), northstar.Chunking(None, 16)],
                                     fusions=["rank"], weights=[1.0, 0.5], diversities=[None], log=io.StringIO(),
                                     embedders=(CachedEmbedder(model, cache), CachedEmbedder(model, cache)), cache=cache)

    first = await run()
    assert first["engine_default_vector_weight"] == 1.0, "a model that is not hashed tokens keeps its full voice"
    assert "chunk_tokens=16 fusion=rank vector_weight=0.5 diversity=None" in first["rows"]
    assert first["embedding"]["embedder"] == "counting-model-v1"
    assert first["embedding"]["scone"]["embedded"] > 0 and first["embedding"]["llamaindex"]["embedded"] > 0
    assert first["embedding"]["cache"]["evicted"] == 0, "a cache that dropped vectors would embed a text twice"
    assert first["llamaindex_tokenizer"] == "counting-model-v1, a chunk's markers included"
    assert first["embedding"]["llamaindex"]["over_window"] == first["embedding"]["scone"]["over_window"] == 0, (
        "the model counts the reference's nodes, so none runs past its window")
    asked = sum(len(call) for call in model.calls)
    again = await run()
    assert sum(len(call) for call in model.calls) == asked, "the second run reads every vector it needs"
    assert again["embedding"]["scone"]["embedded"] == again["embedding"]["llamaindex"]["embedded"] == 0
    assert again["rows"] == first["rows"]
    wrapped = CachedEmbedder(model, cache)
    wrapped.asked = 7
    fresh = northstar.recount(wrapped)
    assert isinstance(fresh, CachedEmbedder) and fresh.asked == 0 and fresh.inner is model and fresh.cache is cache
    hashed = HashEmbedder()
    assert northstar.recount(hashed) is hashed
    checked = await northstar.check(SCORED, io.StringIO(), (CachedEmbedder(model, cache), CachedEmbedder(model, cache)), cache)
    assert checked["embedding"]["scone"]["embedded"] == checked["embedding"]["llamaindex"]["embedded"] == 0, (
        "compare() at defaults runs the same model through the same cache as the sweep")
    assert checked["embedding"]["scone"]["asked"] > 0 and checked["embedding"]["llamaindex"]["asked"] > 0, (
        "and both sides of it asked through the wrappers they were given")
    at_defaults = first["rows"]["chunk=700 fusion=rank vector_weight=1.0 diversity=None"]
    assert {key: checked["scone"][key] for key in at_defaults} == at_defaults
    reference = first["rows"]["llamaindex hybrid (BM25+vector, RRF, chunk 512 tokens)"]
    assert {key: checked["llamaindex"][key] for key in reference} == reference


async def test_the_hashed_sweep_names_its_embedder_and_carries_no_embedding_costs():
    pytest.importorskip("llama_index.core")
    pytest.importorskip("llama_index.retrievers.bm25")
    result = await northstar.sweep(ITEMS[:1], chunkings=[northstar.Chunking(700, None)], fusions=["rank"], weights=[0.01],
                                   diversities=[None], log=io.StringIO())
    assert result["engine_default_vector_weight"] == 0.01 and result["embedding"] == {"embedder": HashEmbedder().id}
    assert result["llamaindex_tokenizer"] == "tiktoken gpt-3.5-turbo (LlamaIndex's default)"


async def test_the_vector_cache_file_answers_the_reference_embedding_from_its_worker_thread(tmp_path):
    """LlamaIndex builds its index in a worker thread and embeds from there;
    SQLite refuses a connection used outside the thread that opened it."""
    import asyncio

    ours, theirs, cache = northstar.embedders("hash", model_cache=None, vector_cache=str(tmp_path / "v.sqlite"),
                                              max_entries=10)
    [vector] = await asyncio.to_thread(lambda: asyncio.run(theirs.embed(["blue binder"])))
    assert await ours.embed(["blue binder"]) == [vector] and ours.record()["reused"] == 1
    assert cache.record()["entries"] == 1
