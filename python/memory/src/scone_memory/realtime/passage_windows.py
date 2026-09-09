"""Bounded passage continuity for conversation context, without new retrieval scores."""
from __future__ import annotations

from dataclasses import dataclass

from ..core.models import RecallItem
from ..integrations.read_memory import ReadMemoryArgs, ReadMemoryError, read_memory
from ..memory.engine import MemoryEngine
from ..retrieval.evidence_graph import MAX_CHUNKS
from ..retrieval.recall_scope import RecallScope


@dataclass
class PassageWindows:
    items: list[RecallItem]
    anchors: dict[int, list[int]]
    revision: int
    failed: bool


async def passage_windows(memory: MemoryEngine, space: str, seeds: list[RecallItem], *,
                          radius: int, scope: RecallScope, session_id: str) -> PassageWindows:
    """Read at most twenty windows; caller owns deadline, packing and final validation.

    Anchors retain priority. New chunks alternate across anchors, nearest first,
    and keep the reader's zero score rather than inheriting retrieval relevance.
    """
    revision = await memory.documents.revision(space)
    windows: list[tuple[int, list[RecallItem]]] = []
    failed = False
    for seed in seeds[:20]:
        try:
            result = await read_memory(memory, space,
                ReadMemoryArgs(chunk_id=seed.chunk_id, before=radius, after=radius), scope, session_id)
            raw_items = result['items']
            if not isinstance(raw_items, list):
                raise ReadMemoryError()
            items = [RecallItem.model_validate(item, strict=True) for item in raw_items]
            anchor = next(item for item in items if item.chunk_id == seed.chunk_id)
            if (anchor.episode_id, anchor.text, anchor.source, anchor.created_at) != (
                    seed.episode_id, seed.text, seed.source, seed.created_at):
                raise ReadMemoryError()
            center = items.index(anchor)
            ordered = sorted(enumerate(items), key=lambda pair: (abs(pair[0] - center), -pair[0]))
            windows.append((seed.chunk_id, [item for _, item in ordered if item.chunk_id != seed.chunk_id]))
        except Exception:
            failed = True
    if await memory.documents.revision(space) != revision:
        raise ReadMemoryError()
    selected = list(seeds)
    seen = {item.chunk_id for item in seeds}
    anchors: dict[int, list[int]] = {}
    for offset in range(radius * 2):
        for seed_id, window in windows:
            if offset >= len(window):
                continue
            item = window[offset]
            if item.chunk_id in anchors:
                anchors[item.chunk_id].append(seed_id)
            if item.chunk_id in seen or len(selected) >= MAX_CHUNKS:
                continue
            selected.append(item)
            seen.add(item.chunk_id)
            anchors[item.chunk_id] = [seed_id]
    return PassageWindows(selected, anchors, revision, failed)
