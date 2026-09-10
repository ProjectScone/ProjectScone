from __future__ import annotations

import asyncio
from typing import Optional, Sequence

from ..providers._onnx import prepare_onnx_runtime

MODELS = {
    "bge-small-en-v1.5": ("BAAI/bge-small-en-v1.5", 384),
    "bge-base-en-v1.5": ("BAAI/bge-base-en-v1.5", 768),
    "nomic-embed-text-v1.5": ("nomic-ai/nomic-embed-text-v1.5", 768),
}


class LocalEmbedder:
    """ONNX embedding in-process through fastembed. Same model names and
    widths as the Rust core, so a vector made here matches one made there."""

    def __init__(self, name: str = "bge-small-en-v1.5", cache_dir: Optional[str] = None) -> None:
        try:
            prepare_onnx_runtime()
            from fastembed import TextEmbedding
        except ImportError as e:  # pragma: no cover
            raise ImportError("LocalEmbedder needs fastembed: pip install 'scone-memory[local-embed]'") from e
        if name not in MODELS:
            raise ValueError(f"unknown embed model {name!r}; known: {sorted(MODELS)}")
        hf_name, dim = MODELS[name]
        self._model = TextEmbedding(model_name=hf_name, cache_dir=cache_dir)
        self.id = name
        self.dim = dim

    async def embed(self, texts: Sequence[str]) -> list[list[float]]:
        if not texts:
            return []
        loop = asyncio.get_running_loop()
        vectors = await loop.run_in_executor(None, lambda: list(self._model.embed(list(texts))))
        return [[float(x) for x in v] for v in vectors]
