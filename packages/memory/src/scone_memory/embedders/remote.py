from __future__ import annotations

import math
import hashlib
import logging
import time
from typing import TYPE_CHECKING, Optional, Sequence

from ..ingestion.vectors import validated_vectors

if TYPE_CHECKING:
    import httpx


class RemoteEmbedder:
    """OpenAI-compatible ``/embeddings`` over httpx (Ollama, OpenAI, vLLM).

    ``dim`` is learned from the first response, so the caller does not
    need to know the model's width up front.

    ``embed`` and ``embed_queries`` apply their explicit document/query
    prefixes independently. Both are part of the embedder identity; use a
    fresh index when changing either. No instruction is inferred from a name.
    """

    def __init__(
        self,
        base_url: str,
        model: str,
        api_key: Optional[str] = None,
        dim: Optional[int] = None,
        timeout: float = 60.0,
        *,
        query_prefix: str = "",
        document_prefix: str = "",
        transport: httpx.AsyncBaseTransport | None = None,
        trust_env: bool = True,
    ) -> None:
        try:
            import httpx
        except ImportError as e:  # pragma: no cover
            raise ImportError("RemoteEmbedder needs httpx: pip install 'scone-memory[remote-embed]'") from e
        self._httpx = httpx
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.api_key = api_key
        self.id = f"remote:{model}"
        if not isinstance(query_prefix, str):
            raise ValueError("query_prefix must be text")
        if not isinstance(document_prefix, str):
            raise ValueError("document_prefix must be text")
        if query_prefix:
            self.id += ":query:" + hashlib.sha256(query_prefix.encode()).hexdigest()[:16]
        self.query_prefix = query_prefix
        if document_prefix:
            self.id += ":document:" + hashlib.sha256(document_prefix.encode()).hexdigest()[:16]
        self.document_prefix = document_prefix
        self._transport = transport
        self._trust_env = trust_env
        self.dim = dim or 0
        self.timeout = timeout
        self._client: httpx.AsyncClient | None = None
        self._closed = False

    async def close(self) -> None:
        """Release pooled connections when the owning engine shuts down."""
        self._closed = True
        if self._client is not None:
            await self._client.aclose()

    def query_cache_text(self, text: str) -> str | None:
        encoded = self.query_prefix + text
        if not encoded.startswith(self.document_prefix):
            return None
        return encoded[len(self.document_prefix):]

    async def embed_queries(self, texts: Sequence[str]) -> list[list[float]]:
        return await self._observe_embed([self.query_prefix + text for text in texts])

    async def embed(self, texts: Sequence[str]) -> list[list[float]]:
        return await self._observe_embed([self.document_prefix + text for text in texts])

    async def _observe_embed(self, texts: Sequence[str]) -> list[list[float]]:
        from ..observability.turn_performance import observe, provider_name
        started, outcome = time.perf_counter(), 'cancelled'
        try:
            vectors = await self._embed(texts)
            outcome = 'completed'
            return vectors
        except Exception:
            outcome = 'failed'
            raise
        finally:
            observe('embedding', elapsed_ms=(time.perf_counter() - started) * 1000,
                    outcome=outcome, model=self.model, provider=provider_name(self.base_url))
            logging.getLogger(__name__).info('embedding_call.finished', extra={
                'event': 'embedding_call.finished', 'model_name': self.model,
                'reference_count': len(texts), 'outcome': outcome,
                'elapsed_ms': round((time.perf_counter() - started) * 1000, 3)})

    async def _embed(self, texts: Sequence[str]) -> list[list[float]]:
        if self._closed:
            raise RuntimeError('remote embedder is closed')
        if not texts:
            return []
        headers = {"content-type": "application/json"}
        if self.api_key:
            headers["authorization"] = f"Bearer {self.api_key}"
        if self._client is None:
            self._client = self._httpx.AsyncClient(timeout=self.timeout, transport=self._transport,
                                                   trust_env=self._trust_env)
        response = await self._client.post(
            f"{self.base_url}/embeddings",
            json={"model": self.model, "input": list(texts)},
            headers=headers,
        )
        if response.status_code >= 400:
            raise RuntimeError(f"embedding server returned {response.status_code}")
        packet = response.json()
        data = packet.get("data") if isinstance(packet, dict) else None
        if not isinstance(data, list) or len(data) != len(texts):
            raise RuntimeError("embedding server must return one vector per input")
        if (any(not isinstance(item, dict) or type(item.get("index")) is not int for item in data)
                or sorted(item["index"] for item in data) != list(range(len(texts)))):
            raise RuntimeError("embedding server returned invalid input indices")
        raw = [item.get("embedding") for item in sorted(data, key=lambda item: item["index"])]
        vectors = [_unit(vector) for vector in validated_vectors(raw, len(texts), self.dim)]
        if not self.dim:
            self.dim = len(vectors[0])
        return vectors


def _unit(vec: Sequence[float]) -> list[float]:
    norm = math.sqrt(sum(v * v for v in vec))
    return [v / norm for v in vec] if norm else list(vec)
