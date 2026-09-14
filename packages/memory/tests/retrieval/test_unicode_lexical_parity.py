"""A query in an unspaced script finds its passage in every store the same way.

Our tokenizer cuts unspaced scripts into character grams so a query for
part of a word can find it; a store whose own index tokenizes the text
another way would find it in memory and lose it on disk. This pins that
the in-memory and SQLite text lanes agree on Japanese, Chinese, Thai and
accented Latin queries.
"""

from __future__ import annotations

import pytest

from scone_memory import HashEmbedder, InMemoryDocumentStore, InMemoryVectorIndex, MemoryEngine
from scone_memory.backends import SqliteDocumentStore

PASSAGES = {
    "japanese": ("東京タワーは1958年に完成した。", "東京タワー"),
    "chinese": ("北京大学成立于1898年。", "北京大学"),
    "thai": ("กรุงเทพมหานครเป็นเมืองหลวงของประเทศไทย", "กรุงเทพ"),
    "accented": ("La reunión sobre facturación fue ayer en Málaga.", "facturacion"),
    "korean": ("서울특별시는 대한민국의 수도이다.", "서울"),
}


@pytest.fixture(params=["memory", "sqlite"])
async def engine(request, tmp_path):
    documents = InMemoryDocumentStore() if request.param == "memory" else SqliteDocumentStore(str(tmp_path / "u.db"))
    engine = await MemoryEngine(documents, InMemoryVectorIndex(), HashEmbedder()).open()
    for text, _ in PASSAGES.values():
        await engine.remember("s", text)
    await engine.remember("s", "An unrelated line about a garden shed.")
    yield engine
    close = getattr(documents, "close", None)
    if close is not None:
        result = close()
        if hasattr(result, "__await__"):
            await result


@pytest.mark.parametrize("script", list(PASSAGES))
async def test_the_text_lane_finds_an_unspaced_or_accented_query_in_every_store(engine, script):
    text, query = PASSAGES[script]
    found = await engine.recall("s", query, limit=2)
    hit = next((item for item in found.items if item.text == text), None)
    assert hit is not None and hit.lanes.get("text") is not None, (script, [(i.text, i.lanes) for i in found.items])
