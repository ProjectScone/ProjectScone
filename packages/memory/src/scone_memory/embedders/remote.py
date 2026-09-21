from __future__ import annotations

import math
import logging
import time
from typing import TYPE_CHECKING, Optional, Sequence

if TYPE_CHECKING:
    import httpx


class RemoteEmbedder:
    """OpenAI-compatible ``/embeddings`` over httpx (Ollama, OpenAI, vLLM).

    ``dim`` is learned from the first response, so the caller does not
    need to know the model's width up front.
    """

    def __init__(
        self,
        base_url: str,
        model: str,
        api_key: Optional[str] = None,
        dim: Optional[int] = None,
        timeout: float = 60.0,
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
        self.dim = dim or 0
        self.timeout = timeout
        self._client: httpx.AsyncClient | None = None
        self._closed = False

    async def close(self) -> None:
        """Release pooled connections when the owning engine shuts down."""
        self._closed = True
        if self._client is not None:
            await self._client.aclose()

    async def embed(self, texts: Sequence[str]) -> list[list[float]]:
        started, outcome = time.perf_counter(), 'cancelled'
        try:
            vectors = await self._embed(texts)
            outcome = 'completed'
            return vectors
        except Exception:
            outcome = 'failed'
            raise
        finally:
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
            self._client = self._httpx.AsyncClient(timeout=self.timeout, trust_env=False)
        response = await self._client.post(
            f"{self.base_url}/embeddings",
            json={"model": self.model, "input": list(texts)},
            headers=headers,
        )
        if response.status_code >= 400:
            raise RuntimeError(f"embedding server returned {response.status_code}: {response.text[:200]}")
        data = response.json().get("data") or []
        if len(data) != len(texts):
            raise RuntimeError(f"embedding server returned {len(data)} vectors for {len(texts)} inputs")
        vectors = [_unit(item["embedding"]) for item in sorted(data, key=lambda d: d.get("index", 0))]
        if not self.dim:
            self.dim = len(vectors[0])
        return vectors


def _unit(vec: Sequence[float]) -> list[float]:
    norm = math.sqrt(sum(v * v for v in vec))
    return [v / norm for v in vec] if norm else list(vec)
