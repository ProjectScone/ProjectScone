"""SCONE_SEMANTIC_TURN lets served voice sessions wait out a pause mid-clause; off by default."""

from __future__ import annotations

import asyncio

import pytest

from scone_memory import HashEmbedder, InMemoryDocumentStore, InMemoryVectorIndex, MemoryEngine
from scone_memory.core.errors import InvalidInput
from scone_memory.runtime.config import Settings


def test_off_by_default_and_read_as_a_flag():
    assert Settings.from_env({}).semantic_turn is False
    assert Settings.from_env({"SCONE_SEMANTIC_TURN": "1"}).semantic_turn is True
    assert Settings.from_env({"SCONE_SEMANTIC_TURN": "off"}).semantic_turn is False
    with pytest.raises(InvalidInput, match="SCONE_SEMANTIC_TURN"):
        Settings.from_env({"SCONE_SEMANTIC_TURN": "sometimes"})


class _Stop(Exception):
    pass


@pytest.mark.parametrize("flag, expected", [(None, False), ("1", True)])
def test_serve_hands_the_setting_to_the_conversation_service(tmp_path, monkeypatch, flag, expected):
    from scone_memory.api import __main__ as serve

    captured: dict[str, object] = {}

    def create(*args, **options):
        captured.update(options)
        raise _Stop()

    monkeypatch.setattr("scone_memory.api.conversations.create_conversation_app", create)
    env = {"SCONE_API_KEY": "solo", "SCONE_CONVERSATIONS_JOURNAL": str(tmp_path / "sessions.db")}
    if flag is not None:
        env["SCONE_SEMANTIC_TURN"] = flag
    engine = asyncio.run(MemoryEngine(InMemoryDocumentStore(), InMemoryVectorIndex(), HashEmbedder()).open())
    with pytest.raises(_Stop):
        serve.build_app(Settings.from_env(env), engine)
    assert captured["semantic_turn"] is expected
