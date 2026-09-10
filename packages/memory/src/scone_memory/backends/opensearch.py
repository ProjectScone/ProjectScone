"""Native REST vectors for self-managed OpenSearch 2.4+ or managed AWS domains.

Lucene HNSW uses ``knn_vector`` and filters inside ``query.knn``. Cosine
scores are (1 + cosine) / 2, converted back to the VectorIndex contract.
Metadata pairs are encoded keyword values, avoiding dynamic field mappings.

The pooled async HTTP client verifies TLS and ignores environment proxies.
Writes refresh by default for immediate recall. Bulk requests are bounded by
both document count and bytes, but are not atomic across documents/batches.
Injected clients remain owned by the caller. No models or cloud clients load.
For IAM-protected managed AWS domains, pass an explicit AwsSigV4Auth (service
``es``); basic auth is also available for domains configured to accept it.
Serverless's different mapping/API contract is not supported by this adapter.

API references:
https://docs.opensearch.org/latest/vector-search/filter-search-knn/efficient-knn-filtering/
https://docs.opensearch.org/latest/api-reference/document-apis/bulk/
https://docs.opensearch.org/docs/2.18/search-plugins/knn/approximate-knn/
"""
from __future__ import annotations

import json
import math
import re
from collections.abc import Iterable, Mapping, Sequence
from typing import TYPE_CHECKING, cast
from urllib.parse import urlsplit

from ..core.ports import VectorPoint
from ..core.timeutil import epoch_seconds
from .validation import validate_vector

if TYPE_CHECKING:
    import httpx


def _object(value: object) -> Mapping[str, object]:
    if not isinstance(value, dict) or not all(isinstance(key, str) for key in value):
        raise ValueError("OpenSearch returned an invalid object")
    return cast(Mapping[str, object], value)


def _array(value: object) -> list[object]:
    if not isinstance(value, list):
        raise ValueError("OpenSearch returned an invalid array")
    return cast(list[object], value)


def _json(value: object) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"), allow_nan=False)


def _pair(key: str, value: str) -> str:
    return _json([key, value])


class OpenSearchVectorIndex:
    name = "opensearch"

    def __init__(
        self, url: str, index: str = "scone_vectors", *,
        username: str | None = None, password: str | None = None,
        client: httpx.AsyncClient | None = None, auth: httpx.Auth | None = None, refresh: bool = True,
        batch_size: int = 256, max_batch_bytes: int = 5 * 1024 * 1024,
        timeout: float = 30.0,
    ) -> None:
        parsed = urlsplit(url)
        if (parsed.scheme not in ("http", "https") or not parsed.hostname
                or parsed.username is not None or parsed.password is not None
                or parsed.query or parsed.fragment):
            raise ValueError("OpenSearch URL must be an HTTP(S) endpoint without credentials, query, or fragment")
        if not re.fullmatch(r"[a-z0-9][a-z0-9_-]{0,254}", index):
            raise ValueError("OpenSearch index must be a concrete lowercase index name")
        if (username is None) != (password is None):
            raise ValueError("OpenSearch username and password must be configured together")
        if type(batch_size) is not int or not 1 <= batch_size <= 10000:
            raise ValueError("OpenSearch batch_size must be between 1 and 10000")
        if type(max_batch_bytes) is not int or not 1024 <= max_batch_bytes <= 100 * 1024 * 1024:
            raise ValueError("OpenSearch max_batch_bytes must be between 1024 and 104857600")
        if not math.isfinite(timeout) or timeout <= 0:
            raise ValueError("OpenSearch timeout must be positive and finite")
        try:
            import httpx
        except ImportError as error:  # pragma: no cover
            raise ImportError("OpenSearchVectorIndex needs pip install 'scone-memory[opensearch]'") from error
        from ..providers.aws_auth import AwsSigV4Auth

        if auth is not None and (username is not None or client is not None):
            raise ValueError("OpenSearch auth cannot be combined with basic credentials or an injected client; configure the client's auth directly")
        if client is not None and username is not None:
            raise ValueError("OpenSearch injected client must configure its own auth")
        signing_auth = auth if client is None else client.auth
        if isinstance(signing_auth, AwsSigV4Auth):
            if signing_auth.service != "es":
                raise ValueError("OpenSearch Serverless is not supported by this Lucene vector adapter")
            if parsed.scheme != "https":
                raise ValueError("OpenSearch AWS auth requires HTTPS")
            if client is not None and client.follow_redirects:
                raise ValueError("OpenSearch AWS auth requires redirects disabled")
        self._aws_signed = isinstance(signing_auth, AwsSigV4Auth)
        self.client = client if client is not None else httpx.AsyncClient(
            auth=auth if auth is not None else ((username, password) if username is not None and password is not None else None),
            timeout=httpx.Timeout(timeout), trust_env=False,
            limits=httpx.Limits(max_connections=10, max_keepalive_connections=10),
        )
        self._owns_client = client is None
        self.url = url.rstrip("/")
        self.index = index
        self.refresh = refresh
        self.batch_size = batch_size
        self.max_batch_bytes = max_batch_bytes
        self.dim: int | None = None

    def _url(self, suffix: str = "") -> str:
        return f"{self.url}/{self.index}{suffix}"

    async def ensure(self, dim: int) -> None:
        self.dim = None
        if type(dim) is not int or not 1 <= dim <= 16000:
            raise ValueError("OpenSearch dimensions must be between 1 and 16000")
        response = await self.client.get(self._url("/_mapping"))
        if response.status_code == 404:
            response = await self.client.put(self._url(), json={
                "settings": {"index": {"knn": True}},
                "mappings": self._mapping(dim),
            })
            if response.status_code == 400:
                error = _object(response.json()).get("error")
                if isinstance(error, dict) and error.get("type") == "resource_already_exists_exception":
                    response = await self.client.get(self._url("/_mapping"))
                    response.raise_for_status()
                    self._check_mapping(response.json(), dim)
                else:
                    response.raise_for_status()
            else:
                response.raise_for_status()
        else:
            response.raise_for_status()
            self._check_mapping(response.json(), dim)
        response = await self.client.get(self._url("/_settings"))
        response.raise_for_status()
        entry = _object(_object(response.json()).get(self.index))
        settings = _object(_object(entry.get("settings")).get("index"))
        if settings.get("knn") not in (True, "true"):
            raise ValueError("OpenSearch index.knn must be enabled for vector search")
        self.dim = dim

    @staticmethod
    def _mapping(dim: int) -> dict[str, object]:
        return {"dynamic": "strict", "_meta": {"scone_vectors": 1}, "properties": {
            "chunk_id": {"type": "long"}, "episode_id": {"type": "long"},
            "space": {"type": "keyword"}, "created_ts": {"type": "double"},
            "tags": {"type": "keyword"}, "metadata_pairs": {"type": "keyword"},
            "embedding": {"type": "knn_vector", "dimension": dim, "method": {
                "name": "hnsw", "engine": "lucene", "space_type": "cosinesimil",
            }},
        }}

    def _check_mapping(self, raw: object, dim: int) -> None:
        entry = _object(_object(raw).get(self.index))
        mappings = _object(entry.get("mappings"))
        properties = _object(mappings.get("properties"))
        embedding = _object(properties.get("embedding"))
        if embedding.get("dimension") != dim:
            raise ValueError(f"OpenSearch index {self.index!r} dimensions {embedding.get('dimension')} differ from {dim}")
        expected = _object(self._mapping(dim)["properties"])
        for name, definition in expected.items():
            existing = _object(properties.get(name))
            for key, value in _object(definition).items():
                if key == "method":
                    method = _object(existing.get("method"))
                    if any(method.get(k) != v for k, v in _object(value).items()):
                        raise ValueError("OpenSearch embedding requires Lucene HNSW cosine mapping")
                elif existing.get(key) != value:
                    raise ValueError(f"OpenSearch field {name!r} has an incompatible mapping")
        if _object(mappings.get("_meta")).get("scone_vectors") != 1:
            raise ValueError("OpenSearch index has an incompatible metadata schema")
        if embedding.get("data_type", "float") != "float":
            raise ValueError("OpenSearch embedding mapping requires float vectors")
        for name in ("space", "tags", "metadata_pairs", "created_ts"):
            field = _object(properties.get(name))
            if field.get("index", True) is False and field.get("doc_values", True) is False:
                raise ValueError(f"OpenSearch field {name!r} must remain searchable")
            if name != "created_ts" and (field.get("normalizer") is not None
                    or field.get("ignore_above", 2147483647) != 2147483647):
                raise ValueError(f"OpenSearch field {name!r} mapping must preserve exact keyword values")
        self._check_source(_object(mappings.get("_source", {})))

    @staticmethod
    def _check_source(source: Mapping[str, object]) -> None:
        def matches(patterns: object) -> bool:
            values = [patterns] if isinstance(patterns, str) else _array(patterns)
            for pattern in values:
                if not isinstance(pattern, str):
                    raise ValueError("OpenSearch source mapping has invalid field patterns")
                expression = re.escape(pattern).replace(r"\*", ".*").replace(r"\?", ".")
                if re.fullmatch(expression, "chunk_id"):
                    return True
            return False

        includes = source.get("includes", [])
        if (source.get("enabled", True) is False or matches(source.get("excludes", []))
                or bool(includes) and not matches(includes)):
            raise ValueError("OpenSearch source mapping must retain chunk_id")

    def _validate(self, vector: Sequence[float]) -> None:
        validate_vector(vector, self.dim)
        if not any(value != 0 for value in vector):
            raise ValueError("OpenSearch cosine vectors must be nonzero")

    def _document(self, point: VectorPoint) -> bytes:
        return (_json({"index": {"_id": str(point.chunk_id)}}) + "\n" + _json({
            "chunk_id": point.chunk_id, "episode_id": point.episode_id, "space": point.space,
            "created_ts": epoch_seconds(point.created_at), "embedding": list(point.vector),
            "tags": list(point.tags), "metadata_pairs": [_pair(k, v) for k, v in point.metadata.items()],
        }) + "\n").encode("utf-8")

    async def upsert(self, points: Sequence[VectorPoint]) -> None:
        # Validate every row before the first mutation, without retaining a
        # second serialized copy of the entire caller-owned sequence.
        for point in points:
            self._validate(point.vector)
            if len(self._document(point)) > self.max_batch_bytes:
                raise ValueError("OpenSearch document exceeds max_batch_bytes")
        await self._send_batches(self._document(point) for point in points)

    async def _send_batches(self, documents: Iterable[bytes], *, deleting: bool = False) -> None:
        batch: list[bytes] = []
        size = 0
        for document in documents:
            if batch and (len(batch) >= self.batch_size or size + len(document) > self.max_batch_bytes):
                await self._bulk(batch, deleting=deleting)
                batch, size = [], 0
            batch.append(document)
            size += len(document)
        if batch:
            await self._bulk(batch, deleting=deleting)

    async def _bulk(self, actions: Sequence[bytes], *, deleting: bool = False) -> None:
        # AWS ignores URL parameters on signed POSTs. PUT is a supported
        # bulk endpoint and preserves refresh=true without an extra request.
        response = await self.client.request("PUT" if self._aws_signed else "POST", self._url("/_bulk"),
            params={"refresh": str(self.refresh).lower()}, content=b"".join(actions),
            headers={"Content-Type": "application/x-ndjson"})
        response.raise_for_status()
        body = _object(response.json())
        items = _array(body.get("items"))
        if len(items) != len(actions):
            raise RuntimeError("OpenSearch bulk response has missing results")
        for item in items:
            result = _object(_object(item).get("delete" if deleting else "index"))
            status = result.get("status")
            if type(status) is not int or not (200 <= status < 300 or deleting and status == 404):
                # Avoid echoing document payloads or server error text to callers.
                raise RuntimeError(f"OpenSearch bulk operation failed with status {status}; some documents may have succeeded")

    async def search(self, space: str, vector: Sequence[float], limit: int,
                     as_of: str | None = None, tags: tuple[str, ...] = (),
                     where: Mapping[str, str] | None = None) -> list[tuple[int, float]]:
        self._validate(vector)
        if type(limit) is not int or not 0 <= limit <= 10000:
            raise ValueError("OpenSearch limit must be between 0 and 10000")
        if limit == 0:
            return []
        filters: list[dict[str, object]] = [{"term": {"space": space}}]
        if as_of is not None:
            filters.append({"range": {"created_ts": {"lte": epoch_seconds(as_of)}}})
        filters.extend({"term": {"tags": tag}} for tag in tags)
        filters.extend({"term": {"metadata_pairs": _pair(k, v)}} for k, v in (where or {}).items())
        response = await self.client.post(self._url("/_search"), json={
            "size": limit, "_source": ["chunk_id"],
            "query": {"knn": {"embedding": {"vector": list(vector), "k": limit,
                       "filter": {"bool": {"filter": filters}}}}},
        })
        response.raise_for_status()
        body = _object(response.json())
        if body.get("timed_out") is True or _object(body.get("_shards", {})).get("failed", 0) != 0:
            raise RuntimeError("OpenSearch search returned partial results")
        ranked: list[tuple[int, float]] = []
        for raw_hit in _array(_object(body.get("hits")).get("hits")):
            hit = _object(raw_hit)
            chunk_id = _object(hit.get("_source")).get("chunk_id")
            score = hit.get("_score")
            if type(chunk_id) is not int or not isinstance(score, (int, float)) or not math.isfinite(score):
                raise ValueError("OpenSearch returned invalid vector search results")
            ranked.append((chunk_id, max(-1.0, min(1.0, 2.0 * score - 1.0))))
        return sorted(ranked, key=lambda pair: (-pair[1], pair[0]))

    async def delete(self, chunk_ids: Sequence[int]) -> None:
        def action(chunk_id: int) -> bytes:
            return (_json({"delete": {"_id": str(chunk_id)}}) + "\n").encode()

        for chunk_id in chunk_ids:
            if len(action(chunk_id)) > self.max_batch_bytes:
                raise ValueError("OpenSearch delete action exceeds max_batch_bytes")
        await self._send_batches((action(chunk_id) for chunk_id in chunk_ids), deleting=True)

    async def _delete_query(self, query: Mapping[str, object]) -> None:
        response = await self.client.post(self._url("/_delete_by_query"),
            params={"refresh": "true"}, json={"query": query})
        response.raise_for_status()
        result = _object(response.json())
        if result.get("failures") or result.get("timed_out") or result.get("version_conflicts"):
            raise RuntimeError("OpenSearch delete_by_query failed; some documents may have been deleted")
        if self._aws_signed:
            # delete_by_query only supports POST, whose refresh query
            # parameter AWS ignores; explicitly refresh before returning.
            response = await self.client.post(self._url("/_refresh"))
            response.raise_for_status()
            shards = _object(_object(response.json()).get("_shards"))
            if shards.get("failed", 0) != 0:
                raise RuntimeError("OpenSearch refresh returned partial results")

    async def delete_space(self, space: str) -> None:
        await self._delete_query({"term": {"space": space}})

    async def clear(self) -> None:
        await self._delete_query({"match_all": {}})

    async def drop(self) -> None:
        response = await self.client.delete(self._url())
        if response.status_code != 404:
            response.raise_for_status()
        self.dim = None

    async def close(self) -> None:
        if self._owns_client:
            await self.client.aclose()
