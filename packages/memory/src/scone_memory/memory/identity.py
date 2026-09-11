"""How the memory decides that two names are the same entity.

This is the seed of the entity layer. For now it holds the identity rule
that every join shares; resolution of genuinely different spellings into
one entity builds on it and never replaces it.
"""

from __future__ import annotations

from ..core.validation import entity_key

__all__ = ["entity_key"]
