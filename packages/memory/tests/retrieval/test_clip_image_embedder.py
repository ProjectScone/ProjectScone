"""The optional CLIP-style adapter: imported only when its package is there.

``ClipImageEmbedder`` runs a sentence-transformers CLIP checkpoint, which
embeds images and text into one space. The package is not a dependency: an
engine that never asks for the adapter never imports it, and one that asks
without the package installed is told what to install. The real model is
exercised only where the package is installed; here the glue is tested
against a stand-in module that records what it was given.
"""

from __future__ import annotations

import importlib.util
import os
import subprocess
import sys
import types
from io import BytesIO
from pathlib import Path

import pytest
from PIL import Image

from scone_memory.core.errors import InvalidInput
from scone_memory.core.ports import ImageEmbedder

INSTALLED = importlib.util.find_spec("sentence_transformers") is not None


def picture(color: str) -> bytes:
    out = BytesIO()
    Image.new("RGB", (8, 8), color).save(out, format="PNG")
    return out.getvalue()


def test_importing_the_embedders_does_not_import_the_model_package():
    # A fresh interpreter: this process may already hold either module, imported by another test.
    script = ("import sys, scone_memory.embedders; "
              "print(sorted(m for m in ('scone_memory.embedders.clip', 'sentence_transformers') if m in sys.modules))")
    source = str(Path(__file__).resolve().parents[2] / "src")
    run = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True, timeout=120,
                         env={**os.environ, "PYTHONPATH": source})
    assert run.returncode == 0, run.stderr
    assert run.stdout.strip() == "[]"


@pytest.mark.skipif(INSTALLED, reason="sentence-transformers is installed; the refusal cannot be observed")
def test_asking_for_clip_without_its_package_says_what_to_install():
    from scone_memory.embedders import ClipImageEmbedder

    with pytest.raises(InvalidInput, match="pip install sentence-transformers"):
        ClipImageEmbedder()


class _Model:
    made: list[tuple[str, object]] = []

    def __init__(self, name: str, device: object = None) -> None:
        _Model.made.append((name, device))
        self.seen: list[object] = []

    def encode(self, items, normalize_embeddings=False, convert_to_numpy=True, show_progress_bar=False):
        assert normalize_embeddings is True, "vectors are normalised by the model call"
        self.seen.append(list(items))
        out = []
        for item in items:
            if isinstance(item, str):
                out.append(_Row([0.0, 1.0, 0.0]))
            else:
                assert isinstance(item, Image.Image) and item.mode == "RGB"
                out.append(_Row([1.0, 0.0, 0.0]))
        return out


class _Row(list):
    def tolist(self):
        return list(self)


async def test_the_adapter_embeds_image_bytes_and_texts_with_one_model(monkeypatch):
    fake = types.ModuleType("sentence_transformers")
    fake.SentenceTransformer = _Model  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "sentence_transformers", fake)
    monkeypatch.setattr(importlib.util, "find_spec", lambda name, *a: object() if name == "sentence_transformers" else None)
    from scone_memory.embedders.clip import ClipImageEmbedder

    _Model.made.clear()
    clip = ClipImageEmbedder("clip-ViT-B-32", device="cpu")
    assert isinstance(clip, ImageEmbedder)
    assert _Model.made == [("clip-ViT-B-32", "cpu")]
    assert clip.id == "clip-st:clip-ViT-B-32" and clip.dim == 3
    gray = BytesIO()
    Image.new("L", (8, 8), 128).save(gray, format="PNG")
    assert await clip.embed_images([picture("red"), gray.getvalue()]) == [[1.0, 0.0, 0.0], [1.0, 0.0, 0.0]]
    assert await clip.embed_texts(["a red square"]) == [[0.0, 1.0, 0.0]]
    assert await clip.embed_images([]) == [] and await clip.embed_texts([]) == []
    with pytest.raises(InvalidInput, match="not an image"):
        await clip.embed_images([b"not an image"])


@pytest.mark.skipif(not INSTALLED, reason="sentence-transformers is not installed; the real CLIP model is not run")
async def test_the_real_clip_model_puts_a_caption_nearer_its_image():  # pragma: no cover - needs the package and a download
    from scone_memory.embedders import ClipImageEmbedder

    clip = ClipImageEmbedder()
    [red, blue] = await clip.embed_images([picture("red"), picture("blue")])
    [text] = await clip.embed_texts(["a plain red square"])
    dot = lambda a, b: sum(x * y for x, y in zip(a, b))  # noqa: E731
    assert dot(text, red) > dot(text, blue)
