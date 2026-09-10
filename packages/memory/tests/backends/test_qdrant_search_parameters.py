"""An explicit HNSW search effort preserves Qdrant defaults unless configured."""
from types import SimpleNamespace

import pytest

pytest.importorskip("qdrant_client")
from qdrant_client import models

from scone_memory.backends.qdrant import QdrantVectorIndex


class QueryClient:
    def __init__(self) -> None:
        self.requests: list[dict[str, object]] = []

    async def collection_exists(self, collection: str) -> bool:
        return True

    async def get_collection(self, collection: str) -> SimpleNamespace:
        return SimpleNamespace(
            config=SimpleNamespace(params=SimpleNamespace(vectors=models.VectorParams(size=3, distance=models.Distance.COSINE))),
            payload_schema={key: SimpleNamespace(data_type=value) for key, value in
                            {"space": "keyword", "tags": "keyword", "created_ts": "float"}.items()})

    async def query_points(self, collection: str, **kwargs: object) -> SimpleNamespace:
        self.requests.append({"collection": collection, **kwargs})
        return SimpleNamespace(points=[SimpleNamespace(id=7, score=0.75)])


@pytest.mark.parametrize("hnsw_ef", [None, 1, 128, 256])
async def test_search_applies_only_explicit_hnsw_effort_and_keeps_scope(hnsw_ef: int | None) -> None:
    client = QueryClient()
    index = QdrantVectorIndex(collection="test", client=client, hnsw_ef=hnsw_ef)
    await index.ensure(3)
    result = await index.search("alpha", [1.0, 0.0, 0.0], 10,
                                tags=("image",), where={"entity_id": "entity7"})
    assert result == [(7, 0.75)]
    request = client.requests[0]
    assert request["collection"] == "test"
    assert request["limit"] == 10 and request["with_payload"] is False
    if hnsw_ef is None:
        assert "search_params" not in request
    else:
        params = request["search_params"]
        assert getattr(params, "hnsw_ef") == hnsw_ef
        assert getattr(params, "exact") is False
    conditions = getattr(request["query_filter"], "must")
    assert [(condition.key, condition.match.value) for condition in conditions] == [
        ("space", "alpha"), ("tags", "image"), ("meta.entity_id", "entity7")]


@pytest.mark.parametrize("hnsw_ef", [0, -1, True, False, "128", 1.5])
def test_invalid_hnsw_effort_is_rejected_before_client_requests(hnsw_ef: object) -> None:
    client = QueryClient()
    with pytest.raises(ValueError, match="hnsw_ef"):
        QdrantVectorIndex(client=client, hnsw_ef=hnsw_ef)  # type: ignore[arg-type]
    assert client.requests == []
