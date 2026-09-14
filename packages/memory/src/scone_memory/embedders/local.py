from __future__ import annotations

import asyncio
from typing import Optional, Sequence

from ..providers._onnx import prepare_onnx_runtime

MODELS = {
    "bge-small-en-v1.5": ("BAAI/bge-small-en-v1.5", 384),
    "bge-base-en-v1.5": ("BAAI/bge-base-en-v1.5", 768),
    "nomic-embed-text-v1.5": ("nomic-ai/nomic-embed-text-v1.5", 768),
}
#: The tokens a model reads, where its card states it. A model left out
#: declares no window, so an embedding budget cannot be set for it.
MAX_INPUT_TOKENS = {"bge-small-en-v1.5": 512, "bge-base-en-v1.5": 512}


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
        self.max_input_tokens = MAX_INPUT_TOKENS.get(name)
        self._counter: object | None = None

    def count_tokens(self, text: str) -> int:
        """The tokens this model reads for ``text``, counted past its window.

        The model's own tokenizer truncates at the window, which is exactly
        what a count must see past, so an untruncated copy counts."""
        if self._counter is None:
            from tokenizers import Tokenizer

            counter = Tokenizer.from_str(self._model.model.tokenizer.to_str())  # type: ignore[attr-defined]
            counter.no_truncation()
            counter.no_padding()
            self._counter = counter
        return len(self._counter.encode(text).ids)  # type: ignore[attr-defined]

    async def embed(self, texts: Sequence[str]) -> list[list[float]]:
        if not texts:
            return []
        loop = asyncio.get_running_loop()
        vectors = await loop.run_in_executor(None, lambda: list(self._model.embed(list(texts))))
        return [[float(x) for x in v] for v in vectors]
