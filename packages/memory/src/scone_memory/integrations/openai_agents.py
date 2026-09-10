"""scone-memory as a Session for the OpenAI Agents SDK runner. Install
with ``pip install 'scone-memory[openai-agents]'``.

The runner stores every input item and model output in the session and
reads them back before each turn, so ``get_items`` must return exactly
what was added. A plain ``{"role", "content": str}`` message is stored as
its text (recall reads well over it); any other item is stored as JSON.
Both come back verbatim.
"""

from __future__ import annotations

from typing import Any, Optional

try:
    from agents.memory.session import SessionABC
except ImportError as e:  # pragma: no cover - exercised only without the extra
    raise ImportError("scone_memory.integrations.openai_agents needs openai-agents: pip install 'scone-memory[openai-agents]'") from e

from ..memory.engine import MemoryEngine
from ..memory.sync import SyncMemoryEngine
from .turns import Turn, next_seq, read_turn, turn_records


def item_turn(item: Any) -> Turn:
    if isinstance(item, dict) and set(item) == {"role", "content"} and isinstance(item["content"], str) and isinstance(item["role"], str):
        return Turn(item["role"], item["content"])
    role = item.get("role", "item") if isinstance(item, dict) else "item"
    return Turn(str(role), None, item)


def turn_item(turn: Turn) -> Any:
    if turn.text is not None:
        return {"role": turn.role, "content": turn.text}
    return turn.payload


class SconeSession(SessionABC):
    """One session's items, one episode each, under ``session_id`` in the
    space; ``extra`` metadata (``user_id``, ``agent_id``) goes on every
    item so the conversation is findable by scope."""

    def __init__(self, memory: MemoryEngine | SyncMemoryEngine, space: str, session_id: str, extra: dict[str, str] | None = None) -> None:
        self.session_id = session_id
        self.engine = memory.engine if isinstance(memory, SyncMemoryEngine) else memory
        self.space = space
        self.extra = dict(extra or {})

    def _where(self) -> dict[str, str]:
        return {"session_id": self.session_id}

    async def get_items(self, limit: Optional[int] = None) -> list[Any]:
        episodes = await self.engine.episodes(self.space, self._where(), limit=limit)
        return [turn_item(read_turn(e)) for e in episodes]

    async def add_items(self, items: list[Any]) -> None:
        if not items:
            return
        start = next_seq(await self.engine.episodes(self.space, self._where()))
        await self.engine.remember_many(self.space, turn_records(self.session_id, [item_turn(i) for i in items], start, self.extra))

    async def pop_item(self) -> Optional[Any]:
        episodes = await self.engine.episodes(self.space, self._where(), limit=1)
        if not episodes:
            return None
        await self.engine.forget(self.space, episodes[-1].episode_id)
        return turn_item(read_turn(episodes[-1]))

    async def clear_session(self) -> None:
        for episode in await self.engine.episodes(self.space, self._where()):
            await self.engine.forget(self.space, episode.episode_id)


__all__ = ["SconeSession", "item_turn", "turn_item"]
