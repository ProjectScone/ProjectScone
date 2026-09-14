"""scone-memory as a LlamaIndex retriever and chat store. Install with
``pip install 'scone-memory[llamaindex]'``."""

from __future__ import annotations

from typing import Any, Optional, Sequence, cast

try:
    from llama_index.core.base.llms.types import ChatMessage, MessageRole, TextBlock
    from llama_index.core.retrievers import BaseRetriever
    from llama_index.core.schema import NodeWithScore, QueryBundle, TextNode
    from llama_index.core.storage.chat_store.base import BaseChatStore
except ImportError as e:  # pragma: no cover - exercised only without the extra
    raise ImportError("scone_memory.integrations.llamaindex needs llama-index-core: pip install 'scone-memory[llamaindex]'") from e

from pydantic import Field, PrivateAttr

from ..core.models import Episode, RecallResult
from ..memory.sync import SyncMemoryEngine
from .turns import Turn, item_metadata, next_seq, read_turn, turn_records
from ..core.errors import InvalidInput
from ..retrieval.query_formulation import formulate_query
from ..retrieval.reranking import validate_candidate_limit


def nodes(result: RecallResult) -> list[NodeWithScore]:
    """One node per recall item in engine order. Node score remains the
    normalized fusion score, not similarity or reranker confidence; optional
    rerank_score is separate metadata. The node id names the original chunk."""
    return [
        NodeWithScore(node=TextNode(id_=f"scone-chunk-{i.chunk_id}", text=i.text, metadata=item_metadata(i)), score=i.score)
        for i in result.items
    ]


class SconeRetriever(BaseRetriever):
    def __init__(
        self,
        memory: Any,
        space: str,
        limit: int = 5,
        tags: Sequence[str] = (),
        where: Optional[dict[str, str]] = None,
        as_of: Optional[str] = None,
        candidate_limit: int | None = None,
        rerank: bool = True,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.memory = memory
        self.space = space
        self.limit = limit
        self.tags = list(tags)
        self.where = dict(where or {})
        self.as_of = as_of
        self.candidate_limit = validate_candidate_limit(candidate_limit)
        if type(rerank) is not bool:
            raise InvalidInput("rerank must be a boolean")
        self.rerank = rerank

    def _kwargs(self) -> dict:
        return {"limit": self.limit, "tags": self.tags, "where": self.where, "as_of": self.as_of,
                "candidate_limit": self.candidate_limit, "rerank": self.rerank}

    def _retrieve(self, query_bundle: QueryBundle) -> list[NodeWithScore]:
        if not isinstance(self.memory, SyncMemoryEngine):
            raise TypeError("retrieve() needs a SyncMemoryEngine; use aretrieve() with an async engine")
        return nodes(self.memory.recall(self.space, formulate_query(query_bundle.query_str).text, **self._kwargs()))

    async def _aretrieve(self, query_bundle: QueryBundle) -> list[NodeWithScore]:
        engine = self.memory.engine if isinstance(self.memory, SyncMemoryEngine) else self.memory
        return nodes(await engine.recall(self.space, formulate_query(query_bundle.query_str).text, **self._kwargs()))


def message_turn(message: ChatMessage) -> Turn:
    """A plain text message as its text, so recall over the conversation reads well;
    any other message (tool calls, several blocks, extra fields) as its JSON, whole."""
    role = message.role.value if isinstance(message.role, MessageRole) else str(message.role)
    first = message.blocks[0] if message.blocks else None
    if not message.additional_kwargs and len(message.blocks) == 1 and isinstance(first, TextBlock):
        return Turn(role, first.text)
    return Turn(role, None, message.model_dump(mode="json"))


def turn_message(turn: Turn) -> ChatMessage:
    if turn.payload is not None:
        return ChatMessage.model_validate(turn.payload)
    return ChatMessage(role=MessageRole(turn.role), content=turn.text)


def _in_order(episodes: list[Episode]) -> list[Episode]:
    return sorted(episodes, key=lambda episode: int(episode.metadata.get("seq", "0")))


class SconeChatStore(BaseChatStore):
    """A LlamaIndex chat store over Scone turns: a key is a session, each message one
    episode with ``session_id``, ``role`` and ``seq``, so a conversation reads back in
    order, is recallable like any other memory, and is shared with the LangChain history
    and the OpenAI Agents session of the same session id. ``extra`` metadata (``user_id``,
    ``agent_id``) is stamped on every message. The sync methods need a SyncMemoryEngine;
    the async ones take either."""

    space: str
    extra: dict[str, str] = Field(default_factory=dict)
    _memory: Any = PrivateAttr()

    def __init__(self, memory: Any, space: str, extra: Optional[dict[str, str]] = None, **kwargs: Any) -> None:
        # BaseChatStore is a pydantic model; its fields are declared here, not in its typed __init__.
        cast(Any, super()).__init__(space=space, extra=dict(extra or {}), **kwargs)
        self._memory = memory

    @classmethod
    def class_name(cls) -> str:
        return "SconeChatStore"

    def _where(self, key: str) -> dict[str, str]:
        return {"session_id": key}

    def _records(self, key: str, messages: Sequence[ChatMessage], start: int) -> list:
        return turn_records(key, [message_turn(message) for message in messages], start, self.extra)

    def get_messages(self, key: str) -> list[ChatMessage]:
        return [turn_message(read_turn(e)) for e in _in_order(_sync(self._memory).episodes(self.space, self._where(key)))]

    def set_messages(self, key: str, messages: list[ChatMessage]) -> None:
        engine = _sync(self._memory)
        stored = engine.episodes(self.space, self._where(key))
        # Positions continue past the replaced messages, so no deduplication key is used twice.
        start = next_seq(stored)
        for episode in stored:
            engine.forget(self.space, episode.episode_id)
        if messages:
            engine.remember_many(self.space, self._records(key, messages, start))

    def add_message(self, key: str, message: ChatMessage) -> None:
        engine = _sync(self._memory)
        start = next_seq(engine.episodes(self.space, self._where(key)))
        engine.remember_many(self.space, self._records(key, [message], start))

    def delete_messages(self, key: str) -> Optional[list[ChatMessage]]:
        engine = _sync(self._memory)
        stored = _in_order(engine.episodes(self.space, self._where(key)))
        if not stored:
            return None
        for episode in stored:
            engine.forget(self.space, episode.episode_id)
        return [turn_message(read_turn(e)) for e in stored]

    def delete_message(self, key: str, idx: int) -> Optional[ChatMessage]:
        engine = _sync(self._memory)
        stored = _in_order(engine.episodes(self.space, self._where(key)))
        if not -len(stored) <= idx < len(stored):
            return None
        engine.forget(self.space, stored[idx].episode_id)
        return turn_message(read_turn(stored[idx]))

    def delete_last_message(self, key: str) -> Optional[ChatMessage]:
        return self.delete_message(key, -1)

    def get_keys(self) -> list[str]:
        return sorted(_sync(self._memory).scopes(self.space).get("session_id", {}))

    async def aget_messages(self, key: str) -> list[ChatMessage]:
        stored = await _async(self._memory).episodes(self.space, self._where(key))
        return [turn_message(read_turn(e)) for e in _in_order(stored)]

    async def aset_messages(self, key: str, messages: list[ChatMessage]) -> None:
        engine = _async(self._memory)
        stored = await engine.episodes(self.space, self._where(key))
        start = next_seq(stored)
        for episode in stored:
            await engine.forget(self.space, episode.episode_id)
        if messages:
            await engine.remember_many(self.space, self._records(key, messages, start))

    async def async_add_message(self, key: str, message: ChatMessage) -> None:
        engine = _async(self._memory)
        start = next_seq(await engine.episodes(self.space, self._where(key)))
        await engine.remember_many(self.space, self._records(key, [message], start))

    async def adelete_messages(self, key: str) -> Optional[list[ChatMessage]]:
        engine = _async(self._memory)
        stored = _in_order(await engine.episodes(self.space, self._where(key)))
        if not stored:
            return None
        for episode in stored:
            await engine.forget(self.space, episode.episode_id)
        return [turn_message(read_turn(e)) for e in stored]

    async def adelete_message(self, key: str, idx: int) -> Optional[ChatMessage]:
        engine = _async(self._memory)
        stored = _in_order(await engine.episodes(self.space, self._where(key)))
        if not -len(stored) <= idx < len(stored):
            return None
        await engine.forget(self.space, stored[idx].episode_id)
        return turn_message(read_turn(stored[idx]))

    async def adelete_last_message(self, key: str) -> Optional[ChatMessage]:
        return await self.adelete_message(key, -1)

    async def aget_keys(self) -> list[str]:
        return sorted((await _async(self._memory).scopes(self.space)).get("session_id", {}))


def _sync(memory: Any) -> SyncMemoryEngine:
    if isinstance(memory, SyncMemoryEngine):
        return memory
    raise TypeError("the chat store's sync methods need a SyncMemoryEngine; wrap the engine with SyncMemoryEngine(engine)")


def _async(memory: Any) -> Any:
    return memory.engine if isinstance(memory, SyncMemoryEngine) else memory


__all__ = ["SconeRetriever", "SconeChatStore", "message_turn", "turn_message", "nodes"]
