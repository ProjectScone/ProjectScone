"""Optional LlamaIndex retrieval composed with LangChain runnable stages.

Import this module explicitly when both framework extras are installed. The
core memory package does not import frameworks or start tracing through it.
"""
from __future__ import annotations

try:
    from langchain_core.documents import Document
    from langchain_core.runnables import Runnable, RunnableLambda
    from llama_index.core.schema import MetadataMode, NodeWithScore
    from langsmith.run_helpers import tracing_context
except ImportError as error:  # pragma: no cover - exercised without optional extras
    raise ImportError(
        "scone_memory.integrations.composition requires scone-memory[langchain,llamaindex]"
    ) from error

from ..memory.engine import MemoryEngine
from .llamaindex import SconeRetriever


def documents_from_nodes(nodes: list[NodeWithScore]) -> list[Document]:
    """Preserve source text and provenance across framework representations."""
    return [Document(page_content=node.node.get_content(metadata_mode=MetadataMode.NONE),
                     metadata=dict(node.node.metadata)) for node in nodes]


def build_retrieval_workflow(
    memory: MemoryEngine, space: str, *, where: dict[str, str], limit: int = 5,
) -> Runnable[str, list[Document]]:
    """The application authorizes the fixed space and metadata scope first."""
    retriever = SconeRetriever(memory, space, where=where, limit=limit)

    async def retrieve(query: str) -> list[NodeWithScore]:
        return await retriever.aretrieve(query)

    return RunnableLambda(retrieve) | RunnableLambda(documents_from_nodes)


async def retrieve_without_tracing(workflow: Runnable[str, list[Document]], query: str) -> list[Document]:
    """Override hosted tracing for this invocation without changing environment settings."""
    with tracing_context(enabled=False, parent=False):
        return await workflow.ainvoke(query, config={"callbacks": []})


# Earlier callers used the deployment-oriented name; tracing control is the contract.
retrieve_locally = retrieve_without_tracing

__all__ = ["documents_from_nodes", "build_retrieval_workflow", "retrieve_without_tracing", "retrieve_locally"]
