"""scone-memory as a LlamaIndex retriever. Install with
``pip install 'scone-memory[llamaindex]'``."""

from __future__ import annotations

from typing import Any, Optional, Sequence

try:
    from llama_index.core.retrievers import BaseRetriever
    from llama_index.core.schema import NodeWithScore, QueryBundle, TextNode
except ImportError as e:  # pragma: no cover - exercised only without the extra
    raise ImportError("scone_memory.integrations.llamaindex needs llama-index-core: pip install 'scone-memory[llamaindex]'") from e

from ..models import RecallResult
from ..sync import SyncMemoryEngine
from .turns import item_metadata


def nodes(result: RecallResult) -> list[NodeWithScore]:
    """One node per recall item, best first, scored by the item's rank
    score (1.0 for the top item; not a similarity). The node id names the
    chunk so the same chunk retrieved twice is the same node."""
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
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.memory = memory
        self.space = space
        self.limit = limit
        self.tags = list(tags)
        self.where = dict(where or {})
        self.as_of = as_of

    def _kwargs(self) -> dict:
        return {"limit": self.limit, "tags": self.tags, "where": self.where, "as_of": self.as_of}

    def _retrieve(self, query_bundle: QueryBundle) -> list[NodeWithScore]:
        if not isinstance(self.memory, SyncMemoryEngine):
            raise TypeError("retrieve() needs a SyncMemoryEngine; use aretrieve() with an async engine")
        return nodes(self.memory.recall(self.space, query_bundle.query_str, **self._kwargs()))

    async def _aretrieve(self, query_bundle: QueryBundle) -> list[NodeWithScore]:
        engine = self.memory.engine if isinstance(self.memory, SyncMemoryEngine) else self.memory
        return nodes(await engine.recall(self.space, query_bundle.query_str, **self._kwargs()))


__all__ = ["SconeRetriever", "nodes"]
