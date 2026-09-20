from __future__ import annotations

import math
import hashlib
from typing import Optional, Sequence, TYPE_CHECKING

from ..ingestion.vectors import validated_vectors

if TYPE_CHECKING:
    import httpx


class RemoteEmbedder:
    """OpenAI-compatible ``/embeddings`` over httpx (Ollama, OpenAI, vLLM).

    ``dim`` is learned from the first response, so the caller does not
    need to know the model's width up front.

    ``embed`` encodes documents unchanged. ``embed_queries`` prepends an
    explicit ``query_prefix`` for instruction-aware retrieval models. The
    prefix is part of the embedder identity; use a fresh index when changing
    it. No provider-specific instruction is silently inferred from a name.
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
        if query_prefix:
            self.id += ":query:" + hashlib.sha256(query_prefix.encode()).hexdigest()[:16]
        self.query_prefix = query_prefix
        self._transport = transport
        self._trust_env = trust_env
        self.dim = dim or 0
        self.timeout = timeout

    async def embed_queries(self, texts: Sequence[str]) -> list[list[float]]:
        return await self.embed([self.query_prefix + text for text in texts])

    async def embed(self, texts: Sequence[str]) -> list[list[float]]:
        if not texts:
            return []
        headers = {"content-type": "application/json"}
        if self.api_key:
            headers["authorization"] = f"Bearer {self.api_key}"
        async with self._httpx.AsyncClient(timeout=self.timeout, transport=self._transport,
                                         trust_env=self._trust_env) as client:
            response = await client.post(
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
