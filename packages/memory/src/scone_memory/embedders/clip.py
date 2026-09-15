"""A CLIP-style image embedder through sentence-transformers, when it is installed.

CLIP checkpoints published for sentence-transformers (``clip-ViT-B-32`` and
its siblings) embed a PIL image and a string with one ``encode`` call, into
one space. The package, torch and the model download are not dependencies of
this framework: nothing imports them until this adapter is built, and building
it without them raises ``InvalidInput`` naming what to install.
"""

from __future__ import annotations

import asyncio
import importlib
import importlib.util
from io import BytesIO
from typing import Any, Sequence

from ..core.errors import InvalidInput

PACKAGE = "sentence_transformers"
DEFAULT_MODEL = "clip-ViT-B-32"


class ClipImageEmbedder:
    """Image bytes and texts into a CLIP model's shared space. See the module."""

    def __init__(self, model: str = DEFAULT_MODEL, *, device: str | None = None) -> None:
        if importlib.util.find_spec(PACKAGE) is None:
            raise InvalidInput(f"ClipImageEmbedder needs sentence-transformers and its CLIP model {model!r}: "
                               "pip install sentence-transformers (it brings torch; the model downloads on first use)")
        module = importlib.import_module(PACKAGE)
        self._model: Any = module.SentenceTransformer(model, device=device)
        self.id = f"clip-st:{model}"
        # A CLIP checkpoint may not declare its width; the model answers it.
        [probe] = self._encode(["width"])
        self.dim = len(probe)

    def _encode(self, items: Sequence[object]) -> list[list[float]]:
        rows = self._model.encode(list(items), normalize_embeddings=True, convert_to_numpy=True,
                                  show_progress_bar=False)
        return [[float(x) for x in row.tolist()] for row in rows]

    async def embed_images(self, images: Sequence[bytes]) -> list[list[float]]:
        from PIL import Image, UnidentifiedImageError

        opened = []
        for data in images:
            try:
                picture = Image.open(BytesIO(data))
                opened.append(picture.convert("RGB"))
            except (UnidentifiedImageError, OSError) as error:
                raise InvalidInput(f"image lane input is not an image the model can read: {error}") from error
        return await asyncio.to_thread(self._encode, opened)

    async def embed_texts(self, texts: Sequence[str]) -> list[list[float]]:
        return await asyncio.to_thread(self._encode, list(texts))
