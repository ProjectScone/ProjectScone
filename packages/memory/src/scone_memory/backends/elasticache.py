"""Explicit Valkey 8.2+ vector-search adapter for node-based ElastiCache.

Uses the Valkey Search command contract, not Redis Stack's schema or DROPINDEX
extensions. FLAT is exact; HNSW is opt-in approximate search. Index updates are
asynchronous on the server, so writes do not promise immediate search visibility.
No endpoint discovery or AWS credential discovery is performed.

Protocol references: https://valkey.io/commands/ft.info/ and
https://docs.aws.amazon.com/AmazonElastiCache/latest/dg/search-features-limits.html
"""
from __future__ import annotations

import asyncio
import ipaddress
import json
import math
import re
import struct
from collections.abc import AsyncIterator, Mapping, Sequence
from typing import Protocol
from urllib.parse import urlsplit

from ..core.ports import VectorPoint
from ..core.timeutil import epoch_seconds
from .validation import validate_vector

CommandArg = str | bytes | int | float
_COMMANDS = ("FT.CREATE", "FT.SEARCH", "FT.INFO", "FT._LIST", "FT.DROPINDEX")


class ValkeyClient(Protocol):
    """Borrowed transports must route keyed commands and scan all primaries."""

    async def execute_command(self, *args: CommandArg) -> object: ...
    def scan_keys(self, pattern: str, count: int) -> AsyncIterator[str]: ...
    async def aclose(self) -> None: ...


class _RedisTransport:
    def __init__(self, host: str, port: int, *, secure: bool, username: str | None,
                 password: str | None, cluster_mode: bool, timeout: float,
                 ca_certs: str | None, connections: int) -> None:
        try:
            from redis.asyncio import Redis, RedisCluster
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise ImportError("ElastiCacheVectorIndex requires scone-memory[redis]") from exc
        self._cluster_client: RedisCluster | None = None
        self._client: Redis | RedisCluster
        if cluster_mode:
            self._client = RedisCluster(
                host=host, port=port, username=username, password=password,
                ssl=secure, ssl_cert_reqs="required", ssl_check_hostname=True,
                ssl_ca_certs=ca_certs, socket_timeout=timeout,
                socket_connect_timeout=timeout, max_connections=connections,
                read_from_replicas=False, decode_responses=False, protocol=2,
            )
            self._cluster_client = self._client
        else:
            self._client = Redis(
                host=host, port=port, username=username, password=password,
                ssl=secure, ssl_cert_reqs="required", ssl_check_hostname=True,
                ssl_ca_certs=ca_certs, socket_timeout=timeout,
                socket_connect_timeout=timeout, max_connections=connections,
                decode_responses=False, protocol=2,
            )

    async def execute_command(self, *args: CommandArg) -> object:
        # FT.* index names are not Redis data keys. Query one coordinator; the
        # search service fans out. HSET/DEL use the SDK's ordinary slot routing.
        if self._cluster_client is not None and args[0] not in {"HSET", "DEL"}:
            await self._cluster_client.initialize()
            coordinator = self._cluster_client.get_default_node()
            if coordinator is None:
                raise RuntimeError("Valkey cluster has no coordinator")
            # A concrete node also bypasses COMMAND GETKEYS discovery for search
            # module commands unknown to the installed SDK's command policies.
            return await self._cluster_client.execute_command(*args, target_nodes=coordinator)
        return await self._client.execute_command(*args)

    async def scan_keys(self, pattern: str, count: int) -> AsyncIterator[str]:
        async for key in self._client.scan_iter(match=pattern, count=count):
            yield _text(key)

    async def aclose(self) -> None:
        await self._client.aclose()


def _text(value: object) -> str:
    if isinstance(value, bytes):
        return value.decode("utf-8")
    if isinstance(value, str):
        return value
    if isinstance(value, (int, float)):
        return str(value)
    raise ValueError("invalid Valkey response scalar")


def _array(value: object) -> list[object]:
    if not isinstance(value, (list, tuple)):
        raise ValueError("invalid Valkey response array")
    return list(value)


def _mapping(value: object) -> dict[str, object]:
    if isinstance(value, dict):
        return {_text(key).lower(): item for key, item in value.items()}
    values = _array(value)
    if len(values) % 2:
        raise ValueError("invalid Valkey response pairs")
    return {_text(values[i]).lower(): values[i + 1] for i in range(0, len(values), 2)}


def _literal(value: str) -> str:
    # Nonempty ASCII encoding has no query operators, delimiters, whitespace,
    # case-folding ambiguity, or collisions with empty strings.
    return "x" + value.encode("utf-8").hex()


def _metadata(key: str, value: str) -> str:
    return _literal(json.dumps([key, value], ensure_ascii=False, separators=(",", ":")))


def _endpoint(url: str) -> tuple[str, int, bool, bool]:
    parsed = urlsplit(url)
    if (parsed.scheme not in {"redis", "rediss"} or not parsed.hostname
            or parsed.username is not None or parsed.password is not None
            or parsed.query or parsed.fragment or parsed.path not in {"", "/", "/0"}):
        raise ValueError("use an explicit redis(s) endpoint without URL credentials or options")
    host = parsed.hostname
    try:
        loopback = ipaddress.ip_address(host).is_loopback
    except ValueError:
        loopback = host == "localhost"
    secure = parsed.scheme == "rediss"
    if not loopback and not secure:
        raise ValueError("remote ElastiCache endpoints require verified TLS (rediss://)")
    return host, parsed.port or 6379, secure, loopback


class ElastiCacheVectorIndex:
    name = "elasticache"

    def __init__(self, url: str, *, prefix: str = "scone_vectors",
                 username: str | None = None, password: str | None = None,
                 cluster_mode: bool = True, algorithm: str = "FLAT",
                 batch_size: int = 64, timeout: float = 30.0,
                 ca_certs: str | None = None, client: ValkeyClient | None = None) -> None:
        host, port, secure, loopback = _endpoint(url)
        if not re.fullmatch(r"[A-Za-z0-9_-]{1,128}", prefix):
            raise ValueError("prefix must contain 1-128 ASCII letters, digits, '_' or '-'")
        if algorithm not in {"FLAT", "HNSW"}:
            raise ValueError("algorithm must be FLAT or HNSW")
        if type(batch_size) is not int or not 1 <= batch_size <= 1024:
            raise ValueError("batch_size must be between 1 and 1024")
        if (isinstance(timeout, bool) or not isinstance(timeout, (int, float))
                or not math.isfinite(timeout) or not 0 < timeout <= 60):
            raise ValueError("timeout must be between 0 and 60 seconds")
        if not isinstance(cluster_mode, bool):
            raise ValueError("cluster_mode must be a boolean")
        if not loopback and (not username or not password):
            raise ValueError("remote ElastiCache requires explicit username and password")
        if bool(username) != bool(password):
            raise ValueError("username and password must be supplied together")
        self.prefix = prefix
        self.index = f"{prefix}_idx"
        self.algorithm = algorithm
        self.batch_size = batch_size
        self.timeout = timeout
        self.dim: int | None = None
        self._owned = client is None
        self._closed = False
        self._client: ValkeyClient = client if client is not None else _RedisTransport(
            host, port, secure=secure, username=username, password=password,
            cluster_mode=cluster_mode, timeout=timeout, ca_certs=ca_certs,
            connections=batch_size + 2,
        )

    def _key(self, chunk_id: int) -> str:
        if isinstance(chunk_id, bool) or not isinstance(chunk_id, int) or chunk_id < 0:
            raise ValueError("chunk_id must be a nonnegative integer")
        return f"{self.prefix}:{chunk_id}"

    def _vector(self, vector: Sequence[float]) -> bytes:
        if self._closed:
            raise RuntimeError("ElastiCache vector index is closed")
        validate_vector(vector, self.dim)
        try:
            return struct.pack(f"<{len(vector)}f", *vector)
        except (OverflowError, struct.error) as exc:
            raise ValueError("vector values must fit FLOAT32") from exc

    async def ensure(self, dim: int) -> None:
        if self._closed:
            raise RuntimeError("ElastiCache vector index is closed")
        if isinstance(dim, bool) or not isinstance(dim, int) or not 1 <= dim <= 32768:
            raise ValueError("Valkey vector dimensions must be between 1 and 32768")
        self.dim = None
        info = _mapping(await self._client.execute_command("INFO", "server"))
        version = _text(info.get("valkey_version", ""))
        match = re.match(r"^(\d+)\.(\d+)(?:\.|$)", version)
        if not match or (int(match[1]), int(match[2])) < (8, 2):
            raise ValueError("ElastiCache vector search requires Valkey 8.2 or newer")
        commands = await self._client.execute_command("COMMAND INFO", *_COMMANDS)
        if isinstance(commands, dict):
            available = _mapping(commands)
            supported = all(available.get(name.lower()) for name in _COMMANDS)
        else:
            available_commands = _array(commands)
            supported = len(available_commands) == len(_COMMANDS) and all(available_commands)
        if not supported:
            raise ValueError("server does not provide required Valkey Search commands")
        indexes = _array(await self._client.execute_command("FT._LIST"))
        if self.index not in {_text(item) for item in indexes}:
            fields: list[CommandArg] = []
            for field in ("space", "tags", "meta"):
                fields.extend((field, "TAG", "SEPARATOR", ",", "CASESENSITIVE"))
            fields.extend(("created_ts", "NUMERIC", "embedding", "VECTOR", self.algorithm,
                           6, "TYPE", "FLOAT32", "DIM", dim, "DISTANCE_METRIC", "COSINE"))
            # ACL, OOM and connection failures propagate. Never reinterpret an
            # arbitrary server error as a missing index.
            await self._client.execute_command("FT.CREATE", self.index, "ON", "HASH",
                                               "PREFIX", 1, f"{self.prefix}:", "SCHEMA", *fields)
        self._check_schema(await self._client.execute_command("FT.INFO", self.index), dim)
        self.dim = dim

    def _check_schema(self, response: object, dim: int) -> None:
        info = _mapping(response)
        definition = _mapping(info.get("index_definition"))
        if (_text(definition.get("key_type")) != "HASH"
                or [_text(p) for p in _array(definition.get("prefixes"))] != [f"{self.prefix}:"]):
            raise ValueError("Valkey index has incompatible ownership prefix or key type")
        attrs = [_mapping(item) for item in _array(info.get("attributes"))]
        fields = {_text(item.get("attribute")): item for item in attrs}
        if len(fields) != len(attrs) or set(fields) != {"space", "tags", "meta", "created_ts", "embedding"}:
            raise ValueError("Valkey index has incompatible fields")
        for field, attr in fields.items():
            if _text(attr.get("identifier")) != field:
                raise ValueError("Valkey index has incompatible field aliases")
            expected_type = "VECTOR" if field == "embedding" else "NUMERIC" if field == "created_ts" else "TAG"
            if _text(attr.get("type")) != expected_type:
                raise ValueError("Valkey index has incompatible field types")
            if expected_type == "TAG" and (attr.get("separator") not in {",", b","}
                                           or _text(attr.get("casesensitive")) != "1"):
                raise ValueError("Valkey index must use case-sensitive comma-separated tags")
        vector = _mapping(fields["embedding"].get("index"))
        if (_text(vector.get("dimensions")) != str(dim)
                or _text(vector.get("distance_metric")) != "COSINE"
                or _text(vector.get("data_type")) != "FLOAT32"
                or _text(_mapping(vector.get("algorithm")).get("name")) != self.algorithm):
            raise ValueError("Valkey index has incompatible vector dimensions or metric/algorithm")
        if _text(info.get("backfill_in_progress", "0")) != "0":
            raise RuntimeError("Valkey index is backfilling; retry ensure after it completes")
        if int(_text(info.get("hash_indexing_failures", "0"))) != 0:
            raise RuntimeError("Valkey reports indexing failures; inspect the index before use")

    async def _batch(self, commands: Sequence[tuple[CommandArg, ...]]) -> None:
        # Each command targets exactly one key. All tasks are drained on failure
        # or cancellation before control returns to the caller.
        tasks = [asyncio.create_task(self._client.execute_command(*command)) for command in commands]
        try:
            await asyncio.gather(*tasks)
        finally:
            for task in tasks:
                if not task.done():
                    task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)

    def _record(self, point: VectorPoint) -> tuple[CommandArg, ...]:
        vector = self._vector(point.vector)
        fields = (_literal(point.space), ",".join(_literal(tag) for tag in point.tags),
                  ",".join(_metadata(key, value) for key, value in point.metadata.items()))
        if any(len(value) > 10000 for value in fields):
            raise ValueError("encoded tag field exceeds Valkey's 10000-byte limit")
        return ("HSET", self._key(point.chunk_id), "chunk_id", point.chunk_id,
                "space", fields[0], "tags", fields[1], "meta", fields[2],
                "created_ts", epoch_seconds(point.created_at), "embedding", vector)

    async def upsert(self, points: Sequence[VectorPoint]) -> None:
        # Validate the entire input before the first write, without retaining a
        # second unbounded copy of all vector payloads.
        for point in points:
            self._record(point)
        for start in range(0, len(points), self.batch_size):
            # Concurrent writes to the same key can finish out of order. Retain
            # the final value in each batch; batches themselves remain ordered.
            records = {point.chunk_id: self._record(point)
                       for point in points[start:start + self.batch_size]}
            await self._batch(list(records.values()))

    async def search(self, space: str, vector: Sequence[float], limit: int,
                     as_of: str | None = None, tags: tuple[str, ...] = (),
                     where: Mapping[str, str] | None = None) -> list[tuple[int, float]]:
        packed = self._vector(vector)
        if isinstance(limit, bool) or not isinstance(limit, int) or limit < 0:
            raise ValueError("limit must be a nonnegative integer")
        if limit == 0:
            return []
        clauses = [f"@space:{{{_literal(space)}}}"]
        clauses.extend(f"@tags:{{{_literal(tag)}}}" for tag in tags)
        clauses.extend(f"@meta:{{{_metadata(key, value)}}}" for key, value in (where or {}).items())
        if as_of is not None:
            clauses.append(f"@created_ts:[-inf {epoch_seconds(as_of)!r}]")
        if len(clauses) > 16:
            raise ValueError("Valkey 8.2 supports at most 16 filter terms per query")
        query = f"({' '.join(clauses)})=>[KNN {limit} @embedding $vec]"
        response = _array(await self._client.execute_command(
            "FT.SEARCH", self.index, query, "PARAMS", 2, "vec", packed,
            "RETURN", 3, "chunk_id", "space", "__embedding_score",
            "SORTBY", "__embedding_score", "ASC", "LIMIT", 0, limit,
            "TIMEOUT", max(1, int(self.timeout * 1000)), "DIALECT", 2,
        ))
        if not response or len(response) % 2 != 1:
            raise ValueError("invalid Valkey search response")
        ranked: list[tuple[int, float]] = []
        seen: set[int] = set()
        for offset in range(1, len(response), 2):
            fields = _mapping(response[offset + 1])
            chunk_id = int(_text(fields.get("chunk_id")))
            if (_text(response[offset]) != self._key(chunk_id)
                    or _text(fields.get("space")) != _literal(space) or chunk_id in seen):
                raise ValueError("Valkey search returned invalid or foreign-scope record")
            distance = float(_text(fields.get("__embedding_score")))
            if not math.isfinite(distance) or not -1e-5 <= distance <= 2.00001:
                raise ValueError("invalid Valkey cosine distance")
            seen.add(chunk_id)
            ranked.append((chunk_id, 1.0 - distance))
        if len(ranked) > limit:
            raise ValueError("Valkey search exceeded requested limit")
        return sorted(ranked, key=lambda item: (-item[1], item[0]))

    async def delete(self, chunk_ids: Sequence[int]) -> None:
        if self._closed:
            raise RuntimeError("ElastiCache vector index is closed")
        for chunk_id in chunk_ids:
            self._key(chunk_id)
        for start in range(0, len(chunk_ids), self.batch_size):
            await self._batch([("DEL", self._key(chunk_id)) for chunk_id in chunk_ids[start:start + self.batch_size]])

    async def drop(self) -> None:
        """Remove this adapter's hashes and index; requires exclusive writers."""
        if self.dim is None or self._closed:
            raise RuntimeError("ensure the owned index before dropping it")
        # Recheck ownership before destructive operations, including after reuse.
        self._check_schema(await self._client.execute_command("FT.INFO", self.index), self.dim)
        batch: list[tuple[CommandArg, ...]] = []
        async for key in self._client.scan_keys(f"{self.prefix}:*", self.batch_size):
            if not re.fullmatch(re.escape(self.prefix) + r":\d+", key):
                raise ValueError("unexpected key in owned Valkey namespace")
            batch.append(("DEL", key))
            if len(batch) == self.batch_size:
                await self._batch(batch)
                batch.clear()
        await self._batch(batch)
        await self._client.execute_command("FT.DROPINDEX", self.index)
        self.dim = None

    async def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        if self._owned:
            await self._client.aclose()
