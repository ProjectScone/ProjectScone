"""Adapters that let frameworks use scone-memory as their memory.

Each module imports its framework lazily and raises a clear ImportError
with the extra to install when it is missing:

- ``scone_memory.integrations.langchain``: a ``BaseRetriever`` over recall
  and a ``BaseChatMessageHistory`` over a session's turns (langchain-core).
- ``scone_memory.integrations.llamaindex``: a ``BaseRetriever`` returning
  ``NodeWithScore`` (llama-index-core).
- ``scone_memory.integrations.openai_agents``: a ``Session`` for the OpenAI
  Agents SDK runner (openai-agents).

The framework-free parts, how a recall item becomes a document and how a
turn is stored and read back losslessly, live in ``turns`` and are what
the unit tests cover without any framework installed.
"""
