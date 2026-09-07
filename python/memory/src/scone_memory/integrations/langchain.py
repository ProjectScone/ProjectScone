"""scone-memory for LangChain: a retriever over recall and a chat message
history over a session's turns.

Both take a ``SyncMemoryEngine`` (LangChain calls the sync methods from
plain code and the async ones from a running loop; the sync engine serves
both). Install with ``pip install 'scone-memory[langchain]'``.
"""

from __future__ import annotations

from typing import Any, Optional, Sequence

try:
    from langchain_core.chat_history import BaseChatMessageHistory
    from langchain_core.documents import Document
    from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage, message_to_dict, messages_from_dict
    from langchain_core.retrievers import BaseRetriever
except ImportError as e:  # pragma: no cover - exercised only without the extra
    raise ImportError("scone_memory.integrations.langchain needs langchain-core: pip install 'scone-memory[langchain]'") from e

from ..models import RecallResult
from ..sync import SyncMemoryEngine
from .turns import Turn, item_metadata, next_seq, read_turn, turn_records

PLAIN = {"human": HumanMessage, "ai": AIMessage, "system": SystemMessage}


def message_turn(message: BaseMessage) -> Turn:
    """A human/ai/system message that is nothing but its text is stored as
    that text; anything else (tool calls, ids, extra kwargs, structured or
    blank content) is stored verbatim. "Nothing but its text" is checked by
    rebuilding the message from the text and comparing, so no field is
    forgotten."""
    if message.type in PLAIN and isinstance(message.content, str) and message.content.strip():
        if message_to_dict(PLAIN[message.type](content=message.content)) == message_to_dict(message):
            return Turn(message.type, message.content)
    return Turn(message.type, None, message_to_dict(message))


def turn_message(turn: Turn) -> BaseMessage:
    if turn.text is not None and turn.role in PLAIN:
        return PLAIN[turn.role](content=turn.text)
    if turn.payload is not None:
        [message] = messages_from_dict([turn.payload])
        return message
    return HumanMessage(content=turn.text or "")


def facts_document(result: RecallResult) -> Optional[Document]:
    if not result.facts:
        return None
    lines = [f"{f.subject} {f.predicate} {f.object} (since {f.valid_from[:10]})" for f in result.facts]
    return Document(page_content="\n".join(lines), metadata={"kind": "facts", "fact_ids": [f.fact_id for f in result.facts]})


def documents(result: RecallResult, include_facts: bool) -> list[Document]:
    docs = [Document(page_content=i.text, metadata=item_metadata(i)) for i in result.items]
    if include_facts and (facts := facts_document(result)) is not None:
        docs.insert(0, facts)
    return docs


class SconeRetriever(BaseRetriever):
    """``invoke(query)`` returns recall items as Documents, best first, with
    the item's provenance and scores in ``metadata``. With
    ``include_facts`` the facts that hold at ``as_of`` come first as one
    Document of ``kind: facts``."""

    memory: Any
    space: str
    limit: int = 5
    tags: list[str] = []
    where: dict[str, str] = {}
    as_of: Optional[str] = None
    include_facts: bool = False

    model_config = {"arbitrary_types_allowed": True}

    def _recall_kwargs(self) -> dict:
        return {"limit": self.limit, "tags": self.tags, "where": self.where, "as_of": self.as_of}

    def _get_relevant_documents(self, query: str, *, run_manager: Any = None) -> list[Document]:
        result = _sync(self.memory).recall(self.space, query, **self._recall_kwargs())
        return documents(result, self.include_facts)

    async def _aget_relevant_documents(self, query: str, *, run_manager: Any = None) -> list[Document]:
        result = await _async(self.memory).recall(self.space, query, **self._recall_kwargs())
        return documents(result, self.include_facts)


class SconeChatMessageHistory(BaseChatMessageHistory):
    """A session's turns, stored one episode each so they read back in
    order and the whole transcript is recallable like any other memory.
    ``extra`` metadata (``user_id``, ``agent_id``) is stamped on every turn
    so scoped recall can find this conversation."""

    def __init__(self, memory: Any, space: str, session_id: str, extra: dict[str, str] | None = None) -> None:
        self.memory = memory
        self.space = space
        self.session_id = session_id
        self.extra = dict(extra or {})

    def _where(self) -> dict[str, str]:
        return {"session_id": self.session_id}

    @property
    def messages(self) -> list[BaseMessage]:
        return [turn_message(read_turn(e)) for e in _sync(self.memory).episodes(self.space, self._where())]

    async def aget_messages(self) -> list[BaseMessage]:
        return [turn_message(read_turn(e)) for e in await _async(self.memory).episodes(self.space, self._where())]

    def add_messages(self, messages: Sequence[BaseMessage]) -> None:
        if not messages:
            return
        engine = _sync(self.memory)
        start = next_seq(engine.episodes(self.space, self._where()))
        engine.remember_many(self.space, turn_records(self.session_id, [message_turn(m) for m in messages], start, self.extra))

    async def aadd_messages(self, messages: Sequence[BaseMessage]) -> None:
        if not messages:
            return
        engine = _async(self.memory)
        start = next_seq(await engine.episodes(self.space, self._where()))
        await engine.remember_many(self.space, turn_records(self.session_id, [message_turn(m) for m in messages], start, self.extra))

    def clear(self) -> None:
        engine = _sync(self.memory)
        for episode in engine.episodes(self.space, self._where()):
            engine.forget(self.space, episode.episode_id)

    async def aclear(self) -> None:
        engine = _async(self.memory)
        for episode in await engine.episodes(self.space, self._where()):
            await engine.forget(self.space, episode.episode_id)


def _sync(memory: Any) -> SyncMemoryEngine:
    if isinstance(memory, SyncMemoryEngine):
        return memory
    raise TypeError("LangChain's sync methods need a SyncMemoryEngine; wrap the engine with SyncMemoryEngine(engine)")


def _async(memory: Any):
    return memory.engine if isinstance(memory, SyncMemoryEngine) else memory


__all__ = ["SconeRetriever", "SconeChatMessageHistory", "message_turn", "turn_message", "documents", "facts_document"]
