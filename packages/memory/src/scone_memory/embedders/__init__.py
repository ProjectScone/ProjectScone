"""Embedders: text in, unit vectors out.

``HashEmbedder`` is deterministic and dependency-free, for tests and for
environments where a model cannot be loaded; related sentences only
score close when they share words, so it is a lexical stand-in, not a
semantic one. ``RemoteEmbedder`` speaks the OpenAI embeddings shape,
which Ollama, OpenAI, and most hosted models serve. ``LocalEmbedder``
runs the same bge-small model the Rust core defaults to, through
fastembed, so vectors agree across the two stacks.
"""

from .hash import HashEmbedder

__all__ = ["HashEmbedder", "RemoteEmbedder", "LocalEmbedder"]


def __getattr__(name: str):
    if name == "RemoteEmbedder":
        from .remote import RemoteEmbedder

        return RemoteEmbedder
    if name == "LocalEmbedder":
        from .local import LocalEmbedder

        return LocalEmbedder
    raise AttributeError(name)
