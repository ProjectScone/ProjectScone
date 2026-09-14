"""The command line says when a narrowed recall's window was full."""
from __future__ import annotations

import io
import json
import math

import pytest

from scone_memory import InMemoryDocumentStore, InMemoryVectorIndex, MemoryEngine
from scone_memory.runtime.cli import build_parser, run

QUERY = "what went out"
FITS = "manifest shipped"


class Planted:
    id = "planted-v1"
    dim = 4

    async def embed(self, texts):
        return [[0.9, math.sqrt(1 - 0.81), 0.0, 0.0] if FITS in t else
                [1.0, 0.0, 0.0, 0.0] if (QUERY in t or "draft note" in t) else [0.0, 0.0, 1.0, 0.0] for t in texts]


@pytest.fixture
async def engine():
    memory = await MemoryEngine(InMemoryDocumentStore(), InMemoryVectorIndex(), Planted()).open()
    for n in range(30):
        await memory.remember("default", f"draft note {n} about weather", metadata={"status": "draft"}, source=f"drafts/{n}")
    await memory.remember("default", FITS, metadata={"status": "published"}, source="release/plan")
    yield memory
    await memory.close()


async def recall(engine, *arguments: str) -> str:
    out = io.StringIO()
    code = await run(build_parser().parse_args(["recall", QUERY, *arguments]), engine, io.StringIO(""), out)
    assert code == 0, out.getvalue()
    return out.getvalue()


async def test_a_full_window_is_noted_on_the_line(engine, monkeypatch):
    monkeypatch.setattr(InMemoryVectorIndex, "narrows_conditions", False)
    text = await recall(engine, "--conditions", json.dumps({"field": "status", "is": "published"}), "--candidate-limit", "20")
    assert FITS not in text
    assert "note: the narrowing removed 20 candidate(s)" in text and "window of 20 was full" in text, text


async def test_a_narrowed_recall_that_found_its_answer_carries_no_note(engine):
    text = await recall(engine, "--conditions", json.dumps({"field": "status", "is": "published"}))
    assert FITS in text and "note: the narrowing" not in text
