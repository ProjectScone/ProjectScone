"""What `import scone_memory` gives you.

This package is meant to be used as a library, so its front door is part
of the product: a caller who cannot catch the exception an engine raises,
or name the type it returns, is being asked to import from somewhere that
looks private and can move.
"""

from __future__ import annotations

import importlib

import pytest

import scone_memory


def test_every_promised_name_is_actually_there():
    """__all__ is a promise. A name listed and not importable breaks
    `from scone_memory import *` and every editor that reads it."""
    missing = [name for name in scone_memory.__all__ if not hasattr(scone_memory, name)]
    assert missing == []


@pytest.mark.parametrize("name", ["Attachment", "InMemoryBlobStore", "FileBlobStore",
                                  "Conflict", "BatchDecision", "DecisionOutcome"])
def test_what_the_engine_returns_and_raises_can_be_named(name):
    """attach() returns an Attachment, decide() returns a BatchDecision of
    DecisionOutcomes and raises Conflict when the space moved, and a
    server chooses a blob store. Every one of those has to be nameable
    without reaching into a submodule."""
    assert name in scone_memory.__all__
    assert getattr(scone_memory, name) is not None


def test_the_public_names_are_the_ones_the_modules_define():
    """No shadowing: what the package hands out is what the module that
    owns it defines, not a copy that drifted."""
    for module, names in (
        ("scone_memory.core.models", ["Attachment", "BatchDecision", "DecisionOutcome", "Episode", "Fact"]),
        ("scone_memory.backends.blobs", ["InMemoryBlobStore", "FileBlobStore"]),
        ("scone_memory.core.errors", ["Conflict", "InvalidInput", "NotFound"]),
    ):
        owner = importlib.import_module(module)
        for name in names:
            assert getattr(scone_memory, name) is getattr(owner, name), f"{name} is not {module}'s"
