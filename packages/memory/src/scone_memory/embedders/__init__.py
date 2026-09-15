"""Embedders: text in, unit vectors out.

``HashEmbedder`` is deterministic and dependency-free, for tests and for
environments where a model cannot be loaded; related sentences only
score close when they share words, so it is a lexical stand-in, not a
semantic one. ``RemoteEmbedder`` speaks the OpenAI embeddings shape,
which Ollama, OpenAI, and most hosted models serve. ``LocalEmbedder``
runs the same bge-small model the Rust core defaults to, through
fastembed, so vectors agree across the two stacks.

The image lane's embedders put images and text queries into one space:
``HashImageEmbedder`` is deterministic and needs no model (it hashes bytes
and maps phrases, so it proves plumbing, not sight); ``ClipImageEmbedder``
runs a CLIP checkpoint through sentence-transformers, imported only when
asked for.
"""

from .hash import HashEmbedder
from .image_hash import HashImageEmbedder

__all__ = ["HashEmbedder", "HashImageEmbedder", "RemoteEmbedder", "LocalEmbedder", "ClipImageEmbedder"]


def __getattr__(name: str):
    if name == "RemoteEmbedder":
        from .remote import RemoteEmbedder

        return RemoteEmbedder
    if name == "LocalEmbedder":
        from .local import LocalEmbedder

        return LocalEmbedder
    if name == "ClipImageEmbedder":
        from .clip import ClipImageEmbedder

        return ClipImageEmbedder
    raise AttributeError(name)
