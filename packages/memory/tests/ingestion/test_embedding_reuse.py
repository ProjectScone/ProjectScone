"""An unchanged chunk is never embedded twice: the cache answers it, the
receipt says so, and what it answers is checked before it is trusted."""
import io
import json

import pytest

from scone_memory import HashEmbedder, InMemoryDocumentStore, InMemoryVectorIndex, MemoryEngine
from scone_memory.core.errors import InvalidInput
from scone_memory.ingestion.embedding_cache import (InMemoryEmbeddingCache, SqliteEmbeddingCache, build_embedding_cache,
                                                    cache_key)
from scone_memory.ingestion.records import Record

pytestmark = pytest.mark.asyncio

PARAGRAPHS = [f"Paragraph {n}: the harbour closes to sailing boats every November, and the {n}th light is lit." * 6
              for n in range(6)]
DOCUMENT = "\n\n".join(PARAGRAPHS)


class CountingEmbedder(HashEmbedder):
    """Every text it was asked for, in order."""

    def __init__(self):
        super().__init__()
        self.texts: list[str] = []

    async def embed(self, texts):
        self.texts.extend(texts)
        return await super().embed(texts)


async def open_memory(cache, embedder=None, **settings):
    return await MemoryEngine(InMemoryDocumentStore(), InMemoryVectorIndex(), embedder or CountingEmbedder(),
                              embedding_cache=cache, chunk_target=300, **settings).open()


async def test_a_replaced_record_embeds_only_the_chunks_that_changed():
    cache = InMemoryEmbeddingCache()
    memory = await open_memory(cache)
    embedder = memory.embedder
    first = await memory.replace("s", Record(DOCUMENT, kind="file", source="notes.md", dedup_key="doc"))
    stored = first.added.chunks
    assert stored >= 4 and first.added.embeddings_reused == 0 and len(embedder.texts) == stored
    embedder.texts.clear()
    changed = DOCUMENT.replace("Paragraph 3:", "Paragraph 3 (revised):")
    second = await memory.replace("s", Record(changed, kind="file", source="notes.md", dedup_key="doc"))
    assert second.outcome == "updated" and second.added.chunks == stored
    assert len(embedder.texts) == stored - second.added.embeddings_reused, "the embedder saw only the misses"
    assert 0 < len(embedder.texts) < stored, "some chunks changed, most did not"
    assert all("revised" in text or "Paragraph 3" in text or "Paragraph 4" in text for text in embedder.texts), \
        "what was embedded again is what the edit touched"
    found = await memory.recall("s", "Paragraph 3 revised", limit=1)
    assert found.items and "revised" in found.items[0].text
    assert cache.record()["reused"] == second.added.embeddings_reused and cache.record()["evicted"] == 0
    await memory.close()


async def test_without_a_cache_every_chunk_is_embedded_and_the_receipt_says_none_reused():
    memory = await open_memory(None)
    first = await memory.replace("s", Record(DOCUMENT, kind="file", source="notes.md", dedup_key="doc"))
    memory.embedder.texts.clear()
    second = await memory.replace("s", Record(DOCUMENT + "\n\nOne more line.", kind="file", source="notes.md", dedup_key="doc"))
    assert second.added.embeddings_reused == 0 and len(memory.embedder.texts) == second.added.chunks >= first.added.chunks
    await memory.close()


async def test_a_file_cache_serves_a_later_process(tmp_path):
    path = tmp_path / "vectors.sqlite"
    cache = SqliteEmbeddingCache(path)
    memory = await open_memory(cache)
    first = await memory.replace("s", Record(DOCUMENT, kind="file", source="notes.md", dedup_key="doc"))
    await memory.close()
    cache.close()
    # A new engine, a new embedder instance, the same file: nothing is embedded.
    later = SqliteEmbeddingCache(path)
    memory = await open_memory(later)
    again = await memory.replace("s", Record(DOCUMENT, kind="file", source="notes.md", dedup_key="doc"))
    assert again.added.embeddings_reused == first.added.chunks == again.added.chunks
    assert memory.embedder.texts == [] and later.record()["entries"] == first.added.chunks
    assert later.record()["path"] == str(path) and later.record()["store"] == "sqlite"
    await memory.close()
    later.close()


async def test_the_key_is_the_embedder_and_the_exact_text():
    embedder = CountingEmbedder()
    assert cache_key(embedder.id, embedder.dim, "a text") != cache_key(embedder.id, embedder.dim, "a text ")
    assert cache_key(embedder.id, embedder.dim, "a text") != cache_key("other-model", embedder.dim, "a text")
    assert cache_key(embedder.id, embedder.dim, "a text") != cache_key(embedder.id, embedder.dim + 1, "a text")
    # A vector is of a text, not of a space: the same document stored in
    # another space is served from the cache whole.
    cache = InMemoryEmbeddingCache()
    memory = await open_memory(cache)
    first = await memory.remember("s", DOCUMENT, kind="file", source="one.md")
    memory.embedder.texts.clear()
    elsewhere = await memory.remember("t", DOCUMENT, kind="file", source="one.md")
    assert elsewhere.embeddings_reused == first.chunks and memory.embedder.texts == []
    await memory.close()
    # Contextual embeddings prefix the source into the text, so the same
    # content under another name is another key and is embedded again.
    memory = await open_memory(InMemoryEmbeddingCache(), contextual_embeddings=True)
    first = await memory.remember("s", DOCUMENT, kind="file", source="one.md")
    memory.embedder.texts.clear()
    second = await memory.remember("t", DOCUMENT, kind="file", source="two.md")
    assert second.embeddings_reused == 0 and len(memory.embedder.texts) == second.chunks == first.chunks
    await memory.close()


def test_the_bound_drops_the_least_recently_used_and_says_so():
    cache = InMemoryEmbeddingCache(max_entries=2)
    cache.keep({"a": [1.0, 0.0], "b": [0.0, 1.0]}, 2)
    assert cache.take(["a"], 2) == {"a": [1.0, 0.0]}, "a is now the most recently used"
    cache.keep({"c": [0.5, 0.5]}, 2)
    assert cache.take(["a", "b", "c"], 2) == {"a": [1.0, 0.0], "c": [0.5, 0.5]}, "b, least recently used, went"
    assert cache.record() == {"store": "memory", "entries": 2, "max_entries": 2, "reused": 3, "kept": 3, "evicted": 1}
    with pytest.raises(InvalidInput):
        InMemoryEmbeddingCache(max_entries=0)


def test_the_file_bound_drops_the_least_recently_used_too(tmp_path):
    cache = SqliteEmbeddingCache(tmp_path / "v.sqlite", max_entries=2)
    cache.keep({"a": [1.0, 0.0], "b": [0.0, 1.0]}, 2)
    cache.take(["a"], 2)
    cache.keep({"c": [0.5, 0.5]}, 2)
    assert set(cache.take(["a", "b", "c"], 2)) == {"a", "c"} and cache.record()["evicted"] == 1
    cache.close()


def test_a_vector_read_back_is_checked_before_it_is_trusted(tmp_path):
    cache = SqliteEmbeddingCache(tmp_path / "v.sqlite")
    cache.keep({"k": [1.0, 2.0, 3.0]}, 3)
    assert cache.take(["k"], 4) == {}, "a width the embedder does not have is not a vector for it"
    assert cache.record()["entries"] == 0, "and the row is gone rather than served again"
    cache.keep({"k": [1.0, 2.0, 3.0]}, 3)
    cache._conn.execute("UPDATE vectors SET vector = ? WHERE key = 'k'", (b"\x01\x02\x03",))
    cache._conn.commit()
    assert cache.take(["k"], 3) == {} and cache.record()["entries"] == 0, "bytes that are not doubles are not a vector"
    with pytest.raises(InvalidInput):
        cache.keep({"n": [float("nan"), 0.0, 0.0]}, 3)
    with pytest.raises(InvalidInput):
        InMemoryEmbeddingCache().keep({"w": [1.0]}, 2)
    cache.close()


def test_the_setting_names_the_cache_or_none(tmp_path):
    assert build_embedding_cache(None) is None and build_embedding_cache("  ") is None
    assert isinstance(build_embedding_cache("memory"), InMemoryEmbeddingCache)
    built = build_embedding_cache(str(tmp_path / "cache.sqlite"))
    assert isinstance(built, SqliteEmbeddingCache) and (tmp_path / "cache.sqlite").exists()
    built.close()
    from scone_memory.runtime.config import Settings

    assert Settings.from_env({"SCONE_EMBEDDING_CACHE": "memory"}).embedding_cache == "memory"
    assert Settings.from_env({}).embedding_cache is None


async def test_map_and_sync_receipts_say_how_many_embeddings_were_reused(tmp_path):
    from scone_memory.runtime.cli import build_parser, run

    root = tmp_path / "repo"
    root.mkdir()
    (root / "a.py").write_text("def a():\n    return 1\n\n\n" + "\n".join(f"def f{n}():\n    return {n}\n" for n in range(20)),
                               encoding="utf-8")
    (root / "b.py").write_text("def b():\n    return 2\n", encoding="utf-8")
    memory = await open_memory(InMemoryEmbeddingCache())
    out = io.StringIO()
    code = await run(build_parser().parse_args(["--json", "map", str(root)]), memory, io.StringIO(""), out)
    first = json.loads(out.getvalue())
    assert code == 0 and first["read"] == 2 and first["embeddings_reused"] == 0
    # A change the same length as what it replaces: the chunks after it
    # keep their text, which is what the cache answers.
    (root / "a.py").write_text((root / "a.py").read_text(encoding="utf-8").replace("return 1\n", "return 9\n"), encoding="utf-8")
    out = io.StringIO()
    code = await run(build_parser().parse_args(["--json", "map", str(root)]), memory, io.StringIO(""), out)
    second = json.loads(out.getvalue())
    assert code == 0 and second["updated"] == 1 and second["deduplicated"] == 1
    assert second["embeddings_reused"] >= 1, "a.py's unchanged chunks were not embedded again"
    out = io.StringIO()
    code = await run(build_parser().parse_args(["map", str(root)]), memory, io.StringIO(""), out)
    assert code == 0 and "already here" in out.getvalue() and "reused" not in out.getvalue(), "nothing changed: nothing said"
    # sync's receipt carries the same count.
    from scone_memory.ingestion.sync import sync_directory

    (root / "a.py").write_text((root / "a.py").read_text(encoding="utf-8").replace("return 9\n", "return 8\n"), encoding="utf-8")
    receipt = await sync_directory(memory, "default", root, marker="repo", suffixes=(".py",), apply=True, remove=False)
    assert receipt.added == 2 and receipt.embeddings_reused >= 1, "a sync stores under its own marker: new files, cached chunks"
    assert f"{receipt.embeddings_reused} chunk embedding(s) reused" in receipt.text()
    assert receipt.record()["embeddings_reused"] == receipt.embeddings_reused
    await memory.close()
