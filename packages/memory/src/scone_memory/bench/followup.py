"""Second-turn recall with follow-up queries off and on, over two-turn pairs.

Each pair is a first question and a second turn, each with the one
passage that answers it. The passages of every pair, and distractors
that share a follow-up's wording with other subjects, are stored once.
A turn is a hit at 5 when its passage is among what the surface put in
front of the model: the references of ``MemoryContext`` (limit 5), or
the episodes ``recall_context`` supplied (limit 5, a budget every
passage fits). The first turn is asked alone, so follow-up queries must
leave it exactly as it was; the second is asked after the first and a
neutral assistant reply. No model runs: ``carry`` is the rule, and a
rewrite needs a model this bench does not have.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ..backends import InMemoryDocumentStore, InMemoryVectorIndex
from ..core.errors import InvalidInput
from ..embedders.hash import HashEmbedder
from ..integrations.chat import recall_context
from ..memory.engine import MemoryEngine
from ..realtime.context import MemoryContext

SPACE = "followup-bench"
#: What the assistant said between the turns; it names nothing.
REPLY = "Here is what memory holds."
SURFACES = ("context", "chat")
MODES = ("off", "carry")


@dataclass
class Tally:
    pairs: int = 0
    #: surface -> mode -> turn ("first", "second") -> hits at 5
    hits: dict[str, dict[str, dict[str, int]]] = field(default_factory=lambda: {
        surface: {mode: {"first": 0, "second": 0} for mode in MODES} for surface in SURFACES})
    #: Pairs whose second turn carry found and off did not, and the reverse.
    gained: list[str] = field(default_factory=list)
    lost: list[str] = field(default_factory=list)
    #: Pairs whose second turn had a carried query searched (context surface).
    carried: list[str] = field(default_factory=list)


def load_pairs(path: Path) -> dict[str, Any]:
    data = json.loads(path.read_text())
    if not isinstance(data, dict) or data.get("schema_version") != 1:
        raise InvalidInput("a follow-up pairs file has schema_version 1")
    passages = data["passages"]
    for pair in data["pairs"]:
        for turn in pair["turns"]:
            if turn["gold"] not in passages:
                raise InvalidInput(f"pair {pair['id']}: its gold passage is not in the passages")
    return data


async def _hit(engine: MemoryEngine, surface: str, mode: str, messages: list[dict[str, object]],
               episode: int) -> tuple[bool, bool]:
    """Whether the passage was supplied, and whether a follow-up query was searched."""
    if surface == "context":
        context = MemoryContext(engine, SPACE, "followup-bench-session", limit=5, followup_queries=mode)
        _, receipt = await context.prepare(messages)
        applied = bool(receipt.get("followup", {}).get("applied"))
        return episode in {reference["episode_id"] for reference in receipt["references"]}, applied
    _, supplied = await recall_context(engine, SPACE, messages, limit=5, budget=64_000, followup=mode)
    return episode in supplied.episode_ids, bool((supplied.followup or {}).get("applied"))


async def measure(path: Path, *, split: str | None = None) -> Tally:
    """Hits at 5 per surface, mode and turn over the pairs of ``split`` (all when None)."""
    data = load_pairs(path)
    engine = await MemoryEngine(InMemoryDocumentStore(), InMemoryVectorIndex(), HashEmbedder()).open()
    episodes = {text: (await engine.remember(SPACE, text)).episode_id for text in data["passages"]}
    tally = Tally()
    for pair in data["pairs"]:
        if split is not None and pair["split"] != split:
            continue
        tally.pairs += 1
        first, second = pair["turns"]
        alone = [{"role": "user", "content": first["question"]}]
        after = [*alone, {"role": "assistant", "content": REPLY}, {"role": "user", "content": second["question"]}]
        for surface in SURFACES:
            found: dict[str, bool] = {}
            for mode in MODES:
                row = tally.hits[surface][mode]
                row["first"] += (await _hit(engine, surface, mode, alone, episodes[first["gold"]]))[0]
                found[mode], applied = await _hit(engine, surface, mode, after, episodes[second["gold"]])
                row["second"] += found[mode]
                if surface == "context" and applied:
                    tally.carried.append(pair["id"])
            if surface == "context" and found["carry"] != found["off"]:
                (tally.gained if found["carry"] else tally.lost).append(pair["id"])
    return tally


def report(tally: Tally) -> str:
    lines = [f"pairs: {tally.pairs}", "surface  mode   first R@5   second R@5"]
    for surface in SURFACES:
        for mode in MODES:
            row = tally.hits[surface][mode]
            lines.append(f"{surface:<8} {mode:<6} {row['first']:>2}/{tally.pairs}       {row['second']:>2}/{tally.pairs}")
    lines.append(f"context second turn gained: {', '.join(tally.gained) or '-'}; lost: {', '.join(tally.lost) or '-'}")
    lines.append(f"context second turns searched with a carried query: {len(tally.carried)}/{tally.pairs}")
    return "\n".join(lines)
