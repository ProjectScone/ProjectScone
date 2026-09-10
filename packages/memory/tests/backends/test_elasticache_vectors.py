"""Contract fixtures follow Valkey FT.INFO/FT.SEARCH RESP2 documentation.

These tests do not establish live ElastiCache compatibility or performance.
"""
from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from typing import cast

import pytest

from scone_memory.backends.elasticache import ElastiCacheVectorIndex
from scone_memory.core.ports import VectorPoint


def schema(dim: int = 2) -> dict[str, object]:
    return {
        "index_name": "scone_vectors_idx",
        "index_definition": ["key_type", "HASH", "prefixes", ["scone_vectors:"]],
        "attributes": [
            ["identifier", field, "attribute", field, "type", "TAG", "SEPARATOR", ",", "CASESENSITIVE", 1]
            for field in ("space", "tags", "meta")
        ] + [
            ["identifier", "created_ts", "attribute", "created_ts", "type", "NUMERIC"],
            ["identifier", "embedding", "attribute", "embedding", "type", "VECTOR", "index",
             ["dimensions", dim, "distance_metric", "COSINE", "data_type", "FLOAT32",
              "algorithm", ["name", "FLAT"]]],
        ],
        "backfill_in_progress": "0", "hash_indexing_failures": 0,
    }


class Client:
    def __init__(self) -> None:
        self.info = schema()
        self.version = "8.2.1"
        self.exists = True
        self.calls: list[tuple[str | bytes | int | float, ...]] = []
        self.result: object = [0]
        self.closed = False
        self.active = 0
        self.peak = 0
        self.keys: list[str] = []
        self.failure: Exception | None = None

    async def execute_command(self, *args: str | bytes | int | float) -> object:
        self.calls.append(args)
        command = args[0]
        if command == "INFO":
            return {"valkey_version": self.version}
        if command == "COMMAND INFO":
            return {str(name).lower(): {"name": name} for name in args[1:]}
        if command == "FT._LIST":
            return [b"scone_vectors_idx"] if self.exists else []
        if command == "FT.INFO":
            return self.info
        if command == "FT.CREATE":
            self.exists = True
            return b"OK"
        if command == "FT.SEARCH":
            return self.result
        if command in {"HSET", "DEL"}:
            self.active += 1
            self.peak = max(self.peak, self.active)
            try:
                await asyncio.sleep(0)
                if self.failure:
                    raise self.failure
            finally:
                self.active -= 1
        return 1

    async def scan_keys(self, pattern: str, count: int) -> AsyncIterator[str]:
        for key in self.keys:
            yield key

    async def aclose(self) -> None:
        self.closed = True


def adapter(client: Client, *, batch_size: int = 64, algorithm: str = "FLAT") -> ElastiCacheVectorIndex:
    return ElastiCacheVectorIndex("redis://127.0.0.1:6379", client=client, batch_size=batch_size, algorithm=algorithm)


async def test_create_checks_capabilities_and_native_schema() -> None:
    client = Client()
    client.exists = False
    index = adapter(client)
    await index.ensure(2)
    create = next(call for call in client.calls if call[0] == "FT.CREATE")
    assert "FLAT" in create and "CASESENSITIVE" in create
    assert "\x1f" not in create
    assert any(call[0] == "FT.INFO" for call in client.calls)


@pytest.mark.parametrize("version", ["7.2.0", "8.1.9", "nonsense"])
async def test_unsupported_engine_fails_before_creation(version: str) -> None:
    client = Client()
    client.version = version
    with pytest.raises(ValueError, match="Valkey"):
        await adapter(client).ensure(2)
    assert not any(call[0] == "FT.CREATE" for call in client.calls)


@pytest.mark.parametrize("field,replacement", [
    ("index_definition", ["key_type", "HASH", "prefixes", ["foreign:"]]),
    ("attributes", schema(3)["attributes"]),
    ("attributes", []),
    ("hash_indexing_failures", 1),
    ("backfill_in_progress", "1"),
])
async def test_rejects_incompatible_or_unready_schema(field: str, replacement: object) -> None:
    client = Client()
    client.info[field] = replacement
    with pytest.raises((ValueError, RuntimeError)):
        await adapter(client).ensure(2)


async def test_filtered_query_encodes_literals_and_converts_cosine_distance() -> None:
    client = Client()
    index = adapter(client)
    await index.ensure(2)
    point = VectorPoint(7, "Team A", 1, "2026-01-01T00:00:00Z", [1., 0.], ("a,b",), {"a=b": "c"})
    await index.upsert([point])
    call = next(call for call in client.calls if call[0] == "HSET")
    fields = dict(zip(call[2::2], call[3::2]))
    client.result = [1, b"scone_vectors:7", [b"chunk_id", b"7", b"space", fields["space"], b"__embedding_score", b"0.25"]]
    assert await index.search("Team A", [1., 0.], 3, "2026-01-02T00:00:00Z", ("a,b",), {"a=b": "c"}) == [(7, .75)]
    query = str(client.calls[-1][2])
    for field in ("space", "tags", "meta"):
        literal = fields[field]
        assert isinstance(literal, str)
        assert f"@{field}:{{{literal}}}" in query
    assert "@created_ts:[-inf " in query and "KNN 3" in query
    assert "Team A" not in query and "a,b" not in query


async def test_rejects_foreign_scope_and_invalid_response() -> None:
    client = Client()
    index = adapter(client)
    await index.ensure(2)
    for response in ([1, "scone_vectors:2", ["chunk_id", "2", "space", "foreign", "__embedding_score", ".1"]],
                     [1, "scone_vectors:2"], [1, "foreign:2", []]):
        client.result = response
        with pytest.raises(ValueError):
            await index.search("alpha", [1., 0.], 3)


async def test_batches_are_bounded_and_delete_is_cluster_safe() -> None:
    client = Client()
    index = adapter(client, batch_size=3)
    await index.ensure(2)
    await index.upsert([VectorPoint(i, "a", i, "2026-01-01T00:00:00Z", [1., 0.]) for i in range(10)])
    await index.delete(list(range(10)))
    assert client.peak == 3 and client.active == 0
    assert all(len(call) == 2 for call in client.calls if call[0] == "DEL")
    client.failure = RuntimeError("write failed")
    with pytest.raises(RuntimeError, match="write failed"):
        await index.delete(list(range(10)))
    assert client.active == 0


async def test_drop_deletes_own_keys_without_redis_stack_dd() -> None:
    client = Client()
    client.keys = ["scone_vectors:1", "scone_vectors:2"]
    index = adapter(client)
    await index.ensure(2)
    await index.drop()
    assert ("FT.DROPINDEX", "scone_vectors_idx") in client.calls
    assert ("DEL", "scone_vectors:1") in client.calls
    assert not any("DD" in call for call in client.calls)
    await index.close()
    assert client.closed is False


@pytest.mark.parametrize("url", ["redis://cache.example:6379", "rediss://user:secret@cache.example:6379", "https://cache.example", "rediss://cache.example?ssl_cert_reqs=none"])
def test_rejects_insecure_or_credential_bearing_urls(url: str) -> None:
    with pytest.raises(ValueError):
        ElastiCacheVectorIndex(url, client=Client())


async def test_filter_limit_and_vector_validation_precede_io() -> None:
    client = Client()
    index = adapter(client)
    await index.ensure(2)
    count = len(client.calls)
    with pytest.raises(ValueError):
        await index.search("alpha", [1., 0.], 1, tags=tuple(str(i) for i in range(17)))
    with pytest.raises(ValueError):
        await index.upsert([VectorPoint(1, "a", 1, "2026-01-01T00:00:00Z", [1e100, 0.])])
    assert len(client.calls) == count


async def test_schema_rejects_case_alias_metric_and_algorithm_differences() -> None:
    baseline = schema()
    for before, after in [("COSINE", "IP"), ("FLOAT32", "FLOAT64"), ("FLAT", "HNSW"), ("CASESENSITIVE", "IGNORED")]:
        client = Client()
        # Round-trip replacements keep the documented nested response shape.
        import json
        client.info = json.loads(json.dumps(baseline).replace(before, after))
        with pytest.raises(ValueError):
            await adapter(client).ensure(2)


async def test_injected_capability_acl_errors_propagate_without_creation() -> None:
    pytest.importorskip("redis", reason="Redis SDK is optional; required for its response exception")
    from redis.exceptions import ResponseError

    class Denied(Client):
        async def execute_command(self, *args: str | bytes | int | float) -> object:
            if args[0] == "FT._LIST":
                raise ResponseError("NOPERM forbidden")
            return await super().execute_command(*args)

    client = Denied()
    with pytest.raises(ResponseError, match="NOPERM"):
        await adapter(client).ensure(2)
    assert not any(call[0] == "FT.CREATE" for call in client.calls)


async def test_encoding_keeps_metadata_separators_case_and_entity_tags_distinct() -> None:
    client = Client()
    index = adapter(client)
    await index.ensure(2)
    metadata = {"team": "a\x1fb,c}|@space:{foreign", "a=b": "c"}
    points = [VectorPoint(1, "A", 1, "2026-01-01T00:00:00Z", [1., 0.], ("__entity__:deadbeef", "Tag", "tag"), metadata),
              VectorPoint(2, "a", 2, "2026-01-01T00:00:00Z", [1., 0.], (), {"a": "b=c"})]
    await index.upsert(points)
    calls = [call for call in client.calls if call[0] == "HSET"]
    fields = [dict(zip(call[2::2], call[3::2])) for call in calls]
    assert fields[0]["space"] != fields[1]["space"]
    assert fields[1]["meta"] not in str(fields[0]["meta"]).split(",")
    assert len(set(str(fields[0]["tags"]).split(","))) == 3
    await index.search("A", [1., 0.], 10, tags=points[0].tags, where=metadata)
    query = str(client.calls[-1][2])
    for value in str(fields[0]["tags"]).split(",") + str(fields[0]["meta"]).split(","):
        assert "{" + value + "}" in query
    assert "foreign" not in query and "\x1f" not in query


async def test_cancellation_drains_outstanding_writes() -> None:
    started = asyncio.Event()

    class Blocking(Client):
        async def execute_command(self, *args: str | bytes | int | float) -> object:
            if args[0] != "DEL":
                return await super().execute_command(*args)
            self.active += 1
            started.set()
            try:
                await asyncio.Event().wait()
            finally:
                self.active -= 1
            return 1

    client = Blocking()
    task = asyncio.create_task(adapter(client, batch_size=3).delete([1, 2, 3, 4]))
    await started.wait()
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert client.active == 0


async def test_owned_cluster_uses_verified_tls_and_explicit_coordinator(monkeypatch: pytest.MonkeyPatch) -> None:
    pytest.importorskip("redis", reason="Redis SDK is optional; required for owned transport checks")
    import redis.asyncio

    created: list[SdkClient] = []

    class SdkClient(Client):
        def __init__(self, **kwargs: object) -> None:
            super().__init__()
            self.options = kwargs
            self.routing: list[dict[str, object]] = []
            self.node = object()
            created.append(self)

        async def initialize(self) -> SdkClient:
            return self

        def get_default_node(self) -> object:
            return self.node

        async def execute_command(self, *args: str | bytes | int | float, **kwargs: object) -> object:
            self.routing.append(kwargs)
            return await super().execute_command(*args)

    monkeypatch.setattr(redis.asyncio, "RedisCluster", SdkClient)
    index = ElastiCacheVectorIndex("rediss://cache.example:6379", username="app", password="test-secret")
    client = created[0]
    assert not client.calls  # No connection from construction.
    assert client.options["ssl"] is True
    assert client.options["ssl_check_hostname"] is True
    assert client.options["ssl_cert_reqs"] == "required"
    assert client.options["read_from_replicas"] is False
    await index.ensure(2)
    assert all(options == {"target_nodes": client.node} for options in client.routing)
    await index.delete([1, 2])
    assert client.routing[-2:] == [{}, {}]
    await index.close()
    await index.close()
    assert client.closed


async def test_resp_bytes_schema_and_stable_ties() -> None:
    def resp(value: object) -> object:
        if isinstance(value, str):
            return value.encode()
        if isinstance(value, dict):
            return [resp(item) for pair in value.items() for item in pair]
        if isinstance(value, list):
            return [resp(item) for item in value]
        return value

    class BytesClient(Client):
        async def execute_command(self, *args: str | bytes | int | float) -> object:
            result = await super().execute_command(*args)
            return resp(result) if args[0] == "FT.INFO" else result

    client = BytesClient()
    index = adapter(client)
    await index.ensure(2)
    client.result = [2, b"scone_vectors:2", [b"chunk_id", b"2", b"space", b"x61", b"__embedding_score", b"1"],
                     b"scone_vectors:1", [b"chunk_id", b"1", b"space", b"x61", b"__embedding_score", b"1"]]
    assert await index.search("a", [1., 0.], 2) == [(1, 0.), (2, 0.)]


async def test_repeated_chunk_in_batch_preserves_last_write() -> None:
    class Writes(Client):
        def __init__(self) -> None:
            super().__init__()
            self.values: dict[str, object] = {}

        async def execute_command(self, *args: str | bytes | int | float) -> object:
            if args[0] == "HSET":
                fields = dict(zip(args[2::2], args[3::2]))
                if fields["space"] == "x6f6c64":  # old
                    await asyncio.sleep(.01)
                self.values[str(args[1])] = fields["space"]
            return await super().execute_command(*args)

    client = Writes()
    index = adapter(client)
    await index.ensure(2)
    await index.upsert([VectorPoint(1, space, 1, "2026-01-01T00:00:00Z", [1., 0.]) for space in ("old", "new")])
    assert client.values["scone_vectors:1"] == "x6e6577"


async def test_hnsw_is_explicit_and_rejects_flat_schema() -> None:
    import json
    client = Client()
    index = adapter(client, algorithm="HNSW")
    with pytest.raises(ValueError):
        await index.ensure(2)
    client.info = json.loads(json.dumps(client.info).replace("FLAT", "HNSW"))
    await index.ensure(2)


async def test_missing_search_module_is_not_silently_accepted() -> None:
    class NoSearch(Client):
        async def execute_command(self, *args: str | bytes | int | float) -> object:
            if args[0] == "COMMAND INFO":
                return [None] * 5
            return await super().execute_command(*args)

    with pytest.raises(ValueError, match="Search commands"):
        await adapter(NoSearch()).ensure(2)


@pytest.mark.parametrize("batch_size", [1.5, True, "2", None])
def test_batch_size_requires_exact_integer(batch_size: object) -> None:
    with pytest.raises(ValueError, match="batch_size"):
        # Deliberately violate the annotation to exercise runtime validation.
        ElastiCacheVectorIndex("redis://127.0.0.1:6379", client=Client(),
                               batch_size=cast(int, batch_size))


@pytest.mark.parametrize("timeout", [True, False, "30", None])
def test_timeout_requires_number_excluding_bool(timeout: object) -> None:
    with pytest.raises(ValueError, match="timeout"):
        ElastiCacheVectorIndex("redis://127.0.0.1:6379", client=Client(),
                               timeout=cast(float, timeout))


@pytest.mark.parametrize("cluster_mode", [1, 0, "false", None])
def test_cluster_mode_requires_boolean(cluster_mode: object) -> None:
    with pytest.raises(ValueError, match="cluster_mode"):
        ElastiCacheVectorIndex("redis://127.0.0.1:6379", client=Client(),
                               cluster_mode=cast(bool, cluster_mode))


async def test_search_and_upsert_require_ensure_before_io() -> None:
    client = Client()
    index = adapter(client)
    with pytest.raises(ValueError, match="ensure"):
        await index.search("a", [1., 0.], 1)
    with pytest.raises(ValueError, match="ensure"):
        await index.upsert([VectorPoint(1, "a", 1, "2026-01-01T00:00:00Z", [1., 0.])])
    assert not client.calls
