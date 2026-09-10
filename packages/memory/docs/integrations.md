# Framework integrations and reranking

[Package overview](../README.md) · [Architecture](../ARCHITECTURE.md)

Examples below run from `packages/memory/` unless a section names another working directory.

## LlamaIndex and LangChain workflows

The optional `llamaindex` and `langchain` extras can coexist over one Scone
engine. `scone_memory.integrations.llamaindex.SconeRetriever` returns scored
nodes; `scone_memory.integrations.langchain.SconeRetriever` returns documents,
and `SconeChatMessageHistory` provides explicit session history.

The packaged [composition API](../src/scone_memory/integrations/composition.py) runs
LlamaIndex retrieval inside a LangChain Runnable workflow while preserving
source text, chunk/episode IDs and the application's authorized scope. Its
`retrieve_without_tracing` helper disables hosted tracing per invocation and
needs no model service. These retriever adapters are separate from the optional
vector-store bridge below.

Additional package APIs provide [query evidence](../src/scone_memory/retrieval/evidence_graph.py),
[bounded reranking](../src/scone_memory/retrieval/reranking.py),
[structural context](../src/scone_memory/retrieval/structural.py),
[recorded multi-hop retrieval](../src/scone_memory/retrieval/multihop.py), and
[encrypted workflow checkpoints](../src/scone_memory/agents/workflow.py).
The [retrieval workflow builder](../src/scone_memory/agents/retrieval.py) composes
both frameworks with retained-source checks.

`GET /v1/recall?graph_analysis=true` adds bounded community, hub and bridge
analysis to the scoped query result. Set `evidence_graph=true` as well to receive
the nodes and retained-source relationships behind those IDs; both options share
one graph build. The pure Python API is
`scone_memory.retrieval.graph_analysis.analyze_evidence_graph`.
Its versioned algorithm analyzes unique undirected recorded relationships and
reports directional hub counts separately. Coverage and omissions describe the
supplied graph, not the entire memory store; connectivity is not confidence.
Analysis failures leave ordinary recall available with an explicit unavailable
status. Neither option enables external services or model calls.

Activity graphs use `await memory.graph(space, fact_limit=400)` or
`GET /v1/graph?fact_limit=400`. The fact budget accepts 1–2,000 records and is
separate from the event window. Indexed reads filter by space and, for focused
graphs, source before applying a shared budget. Focused source groups are
visited in episode-ID order; selected facts are displayed in fact-ID order.
`facts_truncated` reports omitted or unvisited claims without claiming an exact
omission count. `fact_read_status` is `bounded`, `unavailable`, `failed`, or
`not_read`. Custom document stores can implement
[`GraphFactReader`](../src/scone_memory/core/graph_read.py); unsupported readers
produce an explicitly partial graph instead of scanning the full fact ledger.
This bounds fact selection, not all incident links or the total graph size.

The [self-hosted reranking evaluator](../src/scone_memory/testing/self_hosted_reranking.py)
and [Qdrant comparison](../src/scone_memory/testing/qdrant_comparison.py) are runnable
package modules. They use isolated fixtures and record failures as well as
successful retrieval; fixture scores do not establish general answer accuracy.
The root [.env.example](../../../.env.example) lists supported settings without
credentials. Keep private values in ignored `.env.local` with mode `0600`;
`../../scripts/serve-self-hosted.sh --check` validates its format before an explicit launch.

## Offline cross-encoder reranking

Install `scone-memory[offline-rerank]` to rank retained passages with a dedicated
CPU model instead of asking a chat model to assign relevance scores:

```python
from scone_memory.providers.offline_reranker import OfflineCrossEncoderReranker

reranker = OfflineCrossEncoderReranker(
    "/srv/models/ms-marco-MiniLM-L-6-v2",
    model_name="Xenova/ms-marco-MiniLM-L-6-v2",
)
memory = await MemoryEngine(
    documents, vectors, embedder, reranker=reranker,
    rerank_limit=16, rerank_timeout=2,
).open()
result = await memory.recall(
    "authorized-space", "Who approves external data exports?",
    limit=3, candidate_limit=32,
)
```

The directory must already contain the model's ONNX weights, configuration, and
tokenizer files. Provision those artifacts separately; the adapter makes no
download or inference network calls and never loads remote Python model code.
Supported plain ONNX models are `Xenova/ms-marco-MiniLM-L-6-v2`,
`Xenova/ms-marco-MiniLM-L-12-v2`, and `BAAI/bge-reranker-base`. Other models can
use the existing caller-supplied `Reranker` interface.
`model_identity` records artifact hashes for reproducible runs. Runtime settings
also accept `SCONE_RERANKER_CROSS_ENCODER_DIR` and
`SCONE_RERANKER_CROSS_ENCODER_MODEL`; these are mutually exclusive with the
existing trusted `SCONE_RERANKER_FACTORY` option.

Scores are raw model logits used for ordering, not confidence or a support
threshold. Negative scores can still identify the best available evidence.
Candidate depth, scope checks, result size, and reranking budgets remain owned by
Scone's retrieval pipeline. Every full query/passage pair must fit the configured
token limit and the model's declared limit. Oversized pairs fail reranking rather
than silently scoring a truncated prefix; the existing retrieval fallback and
failure trace remain visible. This does not expand a model's context window.

Inference runs off the event loop, with one active CPU job per adapter. Cancelling
the awaiting request does not stop an already running ONNX operation; its result
is discarded and its slot stays occupied until it finishes.

The isolated evaluator accepts `--cross-encoder-dir /srv/models/MODEL` together
with `--model MODEL_NAME`, `--embedding-cache EXISTING_BGE_CACHE`, and
`--output NEW_REPORT.json`. It compares ordinary fusion, expanded candidate
retrieval, and offline reranking on the same synthetic cases. It measures passage
retrieval, not generated-answer accuracy.

## Any LangChain VectorStore as the vector index

```python
from langchain_community.vectorstores import FAISS   # or another explicitly configured VectorStore
from scone_memory.backends import LangChainVectorIndex

from scone_memory.backends.langchain import LangChainVectorIndex as Bridge

def faiss_filter(space, as_of_ts, tags, where):          # FAISS hands a callable the metadata dict
    return lambda meta: Bridge._matches(meta, space, as_of_ts, tags, where)

index = LangChainVectorIndex(score="cosine_similarity", filter_builder=faiss_filter)
index.bind(FAISS(embedding_function=index.embeddings, index=faiss.IndexFlatIP(dim), docstore=InMemoryDocstore(),
                 index_to_docstore_id={}, distance_strategy=DistanceStrategy.MAX_INNER_PRODUCT, normalize_L2=True))
engine = MemoryEngine(documents, index, embedder)
```

The bridge makes the three things that differ between stores explicit
rather than guessing: vectors reach the store through `index.embeddings`
(or `add_embeddings` where the store has it); scope filters need a
`filter_builder(space, as_of_ts, tags, where)` that returns the store's
own filter, and without one the bridge over-fetches and filters on its
metadata, refusing (the lane reads as degraded, the lexical lane still
answers) whenever the window filled with out-of-scope candidates before
`limit` matches were found; and `score` names what the store's number
means (`cosine_similarity`, `cosine_distance`, `unit_l2_squared`, or the
default `unknown`, under which the order still drives fusion but no
similarity is shown and no confidence verdict is derived). The contract
runs through the bridge over `langchain_core`'s `InMemoryVectorStore`,
filtered and post-filtered, and over FAISS (inner product on unit
vectors, a callable filter over the metadata dict), which is the recipe
above.

## Sync facade and framework session adapters

Framework adapters live under `scone_memory.integrations`; each needs its
framework installed (`pip install 'scone-memory[langchain]'`,
`[llamaindex]`, `[openai-agents]`) and says so if it is missing.

```python
from scone_memory import SyncMemoryEngine
from scone_memory.integrations.langchain import SconeRetriever, SconeChatMessageHistory

memory = SyncMemoryEngine.from_env()
retriever = SconeRetriever(memory=memory, space="default", limit=5, where={"user_id": "mark"}, include_facts=True)
docs = retriever.invoke("where is the deploy runbook")   # Documents with episode_id, score, similarity, lanes in metadata
history = SconeChatMessageHistory(memory, "default", session_id="chat-1", extra={"user_id": "mark"})
history.add_messages([...])                              # one episode per message, in order, recallable like any memory
memory.close()                                          # finish owned loop work before exiting
```

Use `SyncMemoryEngine` as a context manager, or call `close()` when finished.
Closing first rejects new calls, then cancels and drains tasks on its dedicated
loop, finalizes async generators and waits for that loop's default executor.
Pending calls report cancellation after their async cleanup completes. A call
that already completed is not retroactively cancelled, and cancellation is not
a rollback of writes that reached storage.

`close(timeout=5.0)` bounds how long the calling thread waits. If cleanup or an
executor worker has not finished, it raises `TimeoutError` and leaves the loop
draining; new work remains rejected. Release any externally blocked work and call
`close()` again to wait for the same shutdown. Repeated/concurrent closes do not
cancel cleanup again. Blocking facade calls, including close, cannot be made
from its own event-loop thread; use the async engine there instead. Failed
construction also shuts down its worker loop.

This lifecycle owns the wrapper's loop, not arbitrary document/vector/backend
clients. Backend-resource ownership and explicit client closing remain the
caller's responsibility; successful facade shutdown alone does not certify that
every remote pool or external thread has been released.

`scone_memory.integrations.llamaindex.SconeRetriever(memory, space, ...)`
returns `NodeWithScore` nodes (`retrieve` needs a `SyncMemoryEngine`,
`aretrieve` takes either); `scone_memory.integrations.openai_agents.SconeSession(engine, space, session_id)`
is a `Session` for the Agents SDK runner (`get_items`, `add_items`,
`pop_item`, `clear_session`).

Conversation turns are stored one episode each with `session_id`, `role`
and `seq` metadata. A plain text message is stored as its text so recall
reads well over the transcript; anything else (tool calls, structured
content, extra fields) is stored verbatim as JSON. Either way what was
added is what comes back. Turns are deduplicated by position, not text
(`Record.dedup_key`), so the second "ok" in a conversation is a second
turn; a dump carries each episode's identity, so a re-imported transcript
keeps its repeats.
