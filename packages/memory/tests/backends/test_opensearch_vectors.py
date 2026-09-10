"""HTTP request-contract tests; these do not emulate ANN or benchmark a server."""
import json
from collections.abc import Callable

import httpx
import pytest

from scone_memory.core.ports import VectorPoint


def test_native_adapter_is_exported():
    from scone_memory import backends

    assert hasattr(backends, "OpenSearchVectorIndex")


def point(chunk_id: int = 1, vector: tuple[float, ...] = (1.0, 0.0)) -> VectorPoint:
    return VectorPoint(chunk_id, "alpha", 7, "2026-01-01T00:00:00Z", vector,
                       ("image", "private"), {"a.b": "x:y", "a": "b:x:y"})


def client_with(handler: Callable[[httpx.Request], httpx.Response]) -> httpx.AsyncClient:
    return httpx.AsyncClient(transport=httpx.MockTransport(handler))


async def test_create_native_mapping_and_validate_existing_dimensions():
    from scone_memory.backends.opensearch import OpenSearchVectorIndex

    requests: list[httpx.Request] = []
    mapping: dict[str, object] = {}

    def handle(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        if request.url.path.endswith("/_settings"):
            return httpx.Response(200, json={"vectors": {"settings": {"index": {"knn": "true"}}}})
        if request.method == "GET":
            if not mapping:
                return httpx.Response(404, json={})
            return httpx.Response(200, json={"vectors": {"mappings": mapping}})
        body = json.loads(request.content)
        assert body["settings"]["index"]["knn"] is True
        mapping.update(body["mappings"])
        return httpx.Response(200, json={"acknowledged": True})

    async with client_with(handle) as client:
        index = OpenSearchVectorIndex("http://localhost:9200", index="vectors", client=client)
        await index.ensure(2)
        assert mapping["properties"]["embedding"] == {  # type: ignore[index]
            "type": "knn_vector", "dimension": 2,
            "method": {"name": "hnsw", "engine": "lucene", "space_type": "cosinesimil"},
        }
        await index.ensure(2)
        with pytest.raises(ValueError, match="dimensions"):
            await index.ensure(3)
        assert sum(r.method == "PUT" for r in requests) == 1


async def test_bulk_bounds_transport_and_preserves_exact_metadata_pairs():
    from scone_memory.backends.opensearch import OpenSearchVectorIndex

    batches: list[bytes] = []

    def handle(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/vectors/_bulk"
        assert request.url.params["refresh"] == "true"
        assert request.headers["content-type"] == "application/x-ndjson"
        batches.append(request.content)
        count = len(request.content.splitlines()) // 2
        return httpx.Response(200, json={"errors": False, "items": [{"index": {"status": 201}}] * count})

    async with client_with(handle) as client:
        index = OpenSearchVectorIndex("http://localhost:9200", index="vectors", client=client, batch_size=2)
        index.dim = 2
        await index.upsert([point(i) for i in range(5)])
        assert [len(batch.splitlines()) for batch in batches] == [4, 4, 2]
        assert all(batch.endswith(b"\n") for batch in batches)
        document = json.loads(batches[0].splitlines()[1])
        assert document["metadata_pairs"] == ['["a.b","x:y"]', '["a","b:x:y"]']
        assert document["created_ts"] == 1767225600.0


async def test_filtered_query_returns_cosine_with_stable_ties():
    from scone_memory.backends.opensearch import OpenSearchVectorIndex

    def handle(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        assert body["size"] == 4
        query = body["query"]["knn"]["embedding"]
        assert query["vector"] == [1.0, 0.0]
        assert query["k"] == 4
        assert query["filter"]["bool"]["filter"] == [
            {"term": {"space": "alpha"}}, {"range": {"created_ts": {"lte": 1767225600.0}}},
            {"term": {"tags": "image"}}, {"term": {"tags": "private"}},
            {"term": {"metadata_pairs": '["a.b","x:y"]'}},
        ]
        return httpx.Response(200, json={"hits": {"hits": [
            {"_source": {"chunk_id": 3}, "_score": 0.5},
            {"_source": {"chunk_id": 2}, "_score": 1.0},
            {"_source": {"chunk_id": 1}, "_score": 0.5},
            {"_source": {"chunk_id": 4}, "_score": 0.0},
        ]}})

    async with client_with(handle) as client:
        index = OpenSearchVectorIndex("http://localhost:9200", client=client)
        index.dim = 2
        assert await index.search("alpha", [1.0, 0.0], 4, "2026-01-01T00:00:00Z",
                                  ("image", "private"), {"a.b": "x:y"}) == [(2, 1.0), (1, 0.0), (3, 0.0), (4, -1.0)]


@pytest.mark.parametrize("vector", [(0.0, 0.0), (1.0,), (float("nan"), 0.0), (float("inf"), 0.0)])
async def test_bad_vectors_rejected_before_writing(vector: tuple[float, ...]):
    from scone_memory.backends.opensearch import OpenSearchVectorIndex

    def handle(request: httpx.Request) -> httpx.Response:
        pytest.fail("invalid batch must not reach transport")

    async with client_with(handle) as client:
        index = OpenSearchVectorIndex("http://localhost:9200", client=client, batch_size=1)
        index.dim = 2
        with pytest.raises(ValueError):
            await index.upsert([point(), point(2, vector)])
        with pytest.raises(ValueError):
            await index.search("alpha", vector, 2)


async def test_partial_bulk_failure_is_not_reported_as_success():
    from scone_memory.backends.opensearch import OpenSearchVectorIndex

    async with client_with(lambda request: httpx.Response(200, json={"errors": True, "items": [
        {"index": {"status": 400, "error": {"type": "mapper_parsing_exception"}}}]})) as client:
        index = OpenSearchVectorIndex("http://localhost:9200", client=client)
        index.dim = 2
        with pytest.raises(RuntimeError, match="bulk"):
            await index.upsert([point()])


async def test_delete_batches_missing_ids_and_close_ownership():
    from scone_memory.backends.opensearch import OpenSearchVectorIndex

    requests: list[httpx.Request] = []

    def handle(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        if request.url.path.endswith("_bulk"):
            return httpx.Response(200, json={"errors": True, "items": [{"delete": {"status": 404}}]})
        return httpx.Response(200, json={"deleted": 2, "failures": []})

    async with client_with(handle) as client:
        index = OpenSearchVectorIndex("http://localhost:9200", index="vectors", client=client, batch_size=1)
        await index.delete([1, 2])
        assert len(requests) == 2
        assert json.loads(requests[0].content)["delete"]["_id"] == "1"
        await index.delete_space("alpha")
        assert json.loads(requests[-1].content) == {"query": {"term": {"space": "alpha"}}}
        await index.clear()
        assert json.loads(requests[-1].content) == {"query": {"match_all": {}}}
        await index.close()
        assert not client.is_closed
    owned = OpenSearchVectorIndex("http://localhost:9200")
    await owned.close()
    assert owned.client.is_closed


@pytest.mark.parametrize("url", ["https://user:secret@localhost:9200", "file:///tmp/index", "http://localhost:9200/?q=x"])
def test_rejects_ambiguous_or_embedded_credentials(url: str):
    from scone_memory.backends.opensearch import OpenSearchVectorIndex

    with pytest.raises(ValueError):
        OpenSearchVectorIndex(url)


async def test_delete_respects_byte_limit_as_well_as_count():
    from scone_memory.backends.opensearch import OpenSearchVectorIndex

    sizes: list[int] = []

    def handle(request: httpx.Request) -> httpx.Response:
        sizes.append(len(request.content))
        return httpx.Response(200, json={"items": [
            {"delete": {"status": 200}} for _ in request.content.splitlines()]})

    async with client_with(handle) as client:
        index = OpenSearchVectorIndex("http://localhost:9200", client=client, max_batch_bytes=1024)
        await index.delete(list(range(100)))
        assert max(sizes) <= 1024
        assert len(sizes) > 1


async def test_bulk_byte_limit_and_oversized_document_preflight():
    from scone_memory.backends.opensearch import OpenSearchVectorIndex

    sizes: list[int] = []

    def handle(request: httpx.Request) -> httpx.Response:
        sizes.append(len(request.content))
        return httpx.Response(200, json={"items": [
            {"index": {"status": 201}} for _ in request.content.splitlines()[::2]]})

    async with client_with(handle) as client:
        index = OpenSearchVectorIndex("http://localhost:9200", client=client, max_batch_bytes=1024)
        index.dim = 2
        await index.upsert([point(i) for i in range(10)])
        assert max(sizes) <= 1024
        assert len(sizes) > 1
        sizes.clear()
        enormous = VectorPoint(2, "alpha", 7, "2026-01-01T00:00:00Z", [1.0, 0.0], metadata={"huge": "x" * 2000})
        with pytest.raises(ValueError, match="max_batch_bytes"):
            await index.upsert([point(), enormous])
        assert sizes == []


@pytest.mark.parametrize("change", ["dimension", "missing", "engine", "metadata"])
async def test_existing_incompatible_schema_is_refused(change: str):
    from scone_memory.backends.opensearch import OpenSearchVectorIndex

    mapping = {"_meta": {"scone_vectors": 1}, "properties": {
        "chunk_id": {"type": "long"}, "episode_id": {"type": "long"},
        "space": {"type": "keyword"}, "created_ts": {"type": "double"},
        "tags": {"type": "keyword"}, "metadata_pairs": {"type": "keyword"},
        "embedding": {"type": "knn_vector", "dimension": 2,
            "method": {"name": "hnsw", "engine": "lucene", "space_type": "cosinesimil"}},
    }}
    if change == "missing":
        del mapping["properties"]["embedding"]
    elif change == "dimension":
        mapping["properties"]["embedding"]["dimension"] = 3
    elif change == "engine":
        mapping["properties"]["embedding"]["method"]["engine"] = "faiss"
    else:
        mapping["properties"]["metadata_pairs"]["type"] = "text"
    async with client_with(lambda request: httpx.Response(200, json={"scone_vectors": {"mappings": mapping}})) as client:
        index = OpenSearchVectorIndex("http://localhost:9200", client=client)
        index.dim = 2
        with pytest.raises(ValueError):
            await index.ensure(2)
        assert index.dim is None


@pytest.mark.parametrize("body", [
    {"timed_out": True, "hits": {"hits": []}},
    {"_shards": {"failed": 1}, "hits": {"hits": []}},
])
async def test_partial_search_failure_is_not_silent(body: dict[str, object]):
    from scone_memory.backends.opensearch import OpenSearchVectorIndex

    async with client_with(lambda request: httpx.Response(200, json=body)) as client:
        index = OpenSearchVectorIndex("http://localhost:9200", client=client)
        index.dim = 2
        with pytest.raises(RuntimeError, match="partial"):
            await index.search("alpha", [1.0, 0.0], 2)


async def test_drop_resets_dimensions_and_uninitialized_search_is_refused():
    from scone_memory.backends.opensearch import OpenSearchVectorIndex

    async with client_with(lambda request: httpx.Response(404, json={})) as client:
        index = OpenSearchVectorIndex("http://localhost:9200", client=client)
        index.dim = 2
        await index.drop()
        assert index.dim is None
        with pytest.raises(ValueError, match="ensure"):
            await index.search("alpha", [1.0, 0.0], 2)


async def test_zero_limit_does_not_call_server():
    from scone_memory.backends.opensearch import OpenSearchVectorIndex

    def handle(request: httpx.Request) -> httpx.Response:
        pytest.fail("zero limit must not reach transport")

    async with client_with(handle) as client:
        index = OpenSearchVectorIndex("http://localhost:9200", client=client)
        index.dim = 2
        assert await index.search("alpha", [1.0, 0.0], 0) == []
        for limit in (-1, 10001):
            with pytest.raises(ValueError, match="limit"):
                await index.search("alpha", [1.0, 0.0], limit)


@pytest.mark.opensearch
async def test_live_local_opensearch_round_trip():
    """Opt-in server integration, not a throughput or recall-quality benchmark."""
    import os
    from urllib.parse import urlsplit
    from uuid import uuid4

    from scone_memory.backends.opensearch import OpenSearchVectorIndex

    url = os.environ.get("SCONE_TEST_OPENSEARCH_URL")
    if not url:
        pytest.skip("SCONE_TEST_OPENSEARCH_URL is unset; no live OpenSearch integration")
    if urlsplit(url).hostname not in ("localhost", "127.0.0.1", "::1"):
        pytest.skip("this integration test only connects to locally managed loopback services")
    index = OpenSearchVectorIndex(url, index=f"scone_test_vectors_{uuid4().hex}",
        username=os.environ.get("SCONE_TEST_OPENSEARCH_USERNAME"),
        password=os.environ.get("SCONE_TEST_OPENSEARCH_PASSWORD"))
    try:
        await index.ensure(2)
        await index.ensure(2)
        with pytest.raises(ValueError, match="dimensions"):
            await index.ensure(3)
        await index.ensure(2)
        await index.upsert([
            point(1), point(2, (0.0, 1.0)),
            VectorPoint(3, "beta", 7, "2026-01-01T00:00:00Z", [1.0, 0.0], ("image", "private"), {"a.b": "x:y"}),
            VectorPoint(4, "alpha", 7, "2026-01-02T00:00:00Z", [1.0, 0.0], ("image", "private"), {"a.b": "x:y"}),
            VectorPoint(5, "alpha", 7, "2026-01-01T00:00:00Z", [1.0, 0.0], ("image",), {"a.b": "x:y"}),
            VectorPoint(6, "alpha", 7, "2026-01-01T00:00:00Z", [1.0, 0.0], ("image", "private"), {"a.b": "other"}),
        ])
        hits = await index.search("alpha", [1.0, 0.0], 10, "2026-01-01T00:00:00Z",
                                  ("image", "private"), {"a.b": "x:y"})
        assert [chunk_id for chunk_id, _ in hits] == [1, 2]
        assert hits[0][1] == pytest.approx(1.0, abs=1e-5)
        assert hits[1][1] == pytest.approx(0.0, abs=1e-5)
        await index.upsert([point(2, (1.0, 0.0))])
        await index.delete([1, 9999])
        await index.delete_space("beta")
        assert await index.search("beta", [1.0, 0.0], 10) == []
        hits = await index.search("alpha", [1.0, 0.0], 10)
        assert 1 not in [chunk_id for chunk_id, _ in hits]
        assert dict(hits)[2] == pytest.approx(1.0, abs=1e-5)
        await index.clear()
        assert await index.search("alpha", [1.0, 0.0], 10) == []
    finally:
        try:
            await index.drop()
        finally:
            await index.close()


async def test_sigv4_domain_auth_preserves_refresh_and_signs_injected_client_requests():
    from scone_memory.backends.opensearch import OpenSearchVectorIndex
    from scone_memory.providers.aws_auth import AwsSigV4Auth

    requests: list[httpx.Request] = []

    def handle(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        assert request.headers["authorization"].startswith("AWS4-HMAC-SHA256 ")
        if request.url.path.endswith("_bulk"):
            assert request.method == "PUT", "AWS ignores URL parameters on signed POST"
            assert request.url.params["refresh"] == "true"
            return httpx.Response(200, json={"items": [{"index": {"status": 201}}]})
        return httpx.Response(200, json={"deleted": 1, "failures": [], "_shards": {"failed": 0}})

    auth = AwsSigV4Auth("AKIDEXAMPLE", "example-secret", region="us-east-1")
    async with httpx.AsyncClient(auth=auth, transport=httpx.MockTransport(handle)) as client:
        index = OpenSearchVectorIndex("https://domain.example.test", client=client)
        index.dim = 2
        await index.upsert([point()])
        await index.clear()
        assert [request.url.path for request in requests][-2:] == ["/scone_vectors/_delete_by_query", "/scone_vectors/_refresh"]
    owned = OpenSearchVectorIndex("https://domain.example.test", auth=auth)
    try:
        assert owned.client.auth is auth
    finally:
        await owned.close()


def test_opensearch_rejects_ambiguous_auth_and_serverless_mapping():
    from scone_memory.backends.opensearch import OpenSearchVectorIndex
    from scone_memory.providers.aws_auth import AwsSigV4Auth

    auth = AwsSigV4Auth("AKIDEXAMPLE", "example-secret", region="us-east-1")
    with pytest.raises(ValueError, match="auth"):
        OpenSearchVectorIndex("https://domain.example.test", auth=auth, username="u", password="p")
    serverless = AwsSigV4Auth("AKIDEXAMPLE", "example-secret", region="us-east-1", service="aoss")
    with pytest.raises(ValueError, match="Serverless"):
        OpenSearchVectorIndex("https://collection.example.test", auth=serverless)


def compatible_mapping() -> dict[str, object]:
    return {"_meta": {"scone_vectors": 1}, "properties": {
        "chunk_id": {"type": "long"}, "episode_id": {"type": "long"},
        "space": {"type": "keyword"}, "created_ts": {"type": "double"},
        "tags": {"type": "keyword"}, "metadata_pairs": {"type": "keyword"},
        "embedding": {"type": "knn_vector", "dimension": 2,
            "method": {"name": "hnsw", "engine": "lucene", "space_type": "cosinesimil"}},
    }}


@pytest.mark.parametrize("field,changes", [
    ("metadata_pairs", {"normalizer": "lowercase"}),
    ("metadata_pairs", {"ignore_above": 8}),
    ("tags", {"normalizer": "trim"}),
    ("space", {"index": False, "doc_values": False}),
    ("created_ts", {"index": False, "doc_values": False}),
    ("embedding", {"data_type": "byte"}),
    ("_source", {"enabled": False}),
    ("_source", {"excludes": ["chunk_*"]}),
    ("_source", {"includes": ["episode_id"]}),
])
async def test_reopen_rejects_mapping_modifiers_that_break_exact_retrieval(field, changes):
    from scone_memory.backends.opensearch import OpenSearchVectorIndex

    mapping = compatible_mapping()
    if field == "_source":
        mapping[field] = changes
    else:
        mapping["properties"][field].update(changes)
    requests = []

    def handle(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        if request.url.path.endswith("/_settings"):
            return httpx.Response(200, json={"scone_vectors": {"settings": {"index": {"knn": "true"}}}})
        return httpx.Response(200, json={"scone_vectors": {"mappings": mapping}})

    async with client_with(handle) as client:
        index = OpenSearchVectorIndex("http://localhost:9200", client=client)
        with pytest.raises(ValueError, match="mapping|source|searchable"):
            await index.ensure(2)
        assert index.dim is None
        assert all(request.method == "GET" for request in requests)


async def test_reopen_accepts_explicit_defaults_and_unrelated_mapping_options():
    from scone_memory.backends.opensearch import OpenSearchVectorIndex

    mapping = compatible_mapping()
    mapping["_source"] = {"enabled": True, "includes": ["chunk_*", "other"], "excludes": ["embedding"]}
    properties = mapping["properties"]
    properties["metadata_pairs"].update(normalizer=None, ignore_above=2147483647, store=True)
    properties["space"].update(index=False, doc_values=True)
    properties["embedding"].update(data_type="float")
    properties["unrelated"] = {"type": "text", "analyzer": "english"}

    def handle(request: httpx.Request) -> httpx.Response:
        if request.url.path.endswith("/_settings"):
            return httpx.Response(200, json={"scone_vectors": {"settings": {"index": {"knn": "true"}}}})
        return httpx.Response(200, json={"scone_vectors": {"mappings": mapping}})

    async with client_with(handle) as client:
        index = OpenSearchVectorIndex("http://localhost:9200", client=client)
        await index.ensure(2)
        assert index.dim == 2


@pytest.mark.parametrize("knn", [None, False, "false"])
async def test_reopen_refuses_index_with_ann_disabled(knn):
    from scone_memory.backends.opensearch import OpenSearchVectorIndex

    def handle(request: httpx.Request) -> httpx.Response:
        assert request.method == "GET"
        if request.url.path.endswith("/_settings"):
            settings = {} if knn is None else {"knn": knn}
            return httpx.Response(200, json={"scone_vectors": {"settings": {"index": settings}}})
        return httpx.Response(200, json={"scone_vectors": {"mappings": compatible_mapping()}})

    async with client_with(handle) as client:
        index = OpenSearchVectorIndex("http://localhost:9200", client=client)
        with pytest.raises(ValueError, match="knn"):
            await index.ensure(2)
        assert index.dim is None
