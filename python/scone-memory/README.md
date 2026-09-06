# scone-memory

The memory layer for LLM applications, RAG pipelines and agents, in Python.
Every model call, retrieval step and agent in a system writes what happened
and what is true into one place, and reads it back scoped, dated and
explained.

Two kinds of memory live side by side:

- **Episodes**: what happened. Notes, chat turns, documents, tool output,
  kept verbatim and split into chunks that point back into the original
  bytes. Nothing stored is ever rewritten.
- **Facts**: what is true. Subject, predicate, object, and the interval
  over which it held. A new fact closes the one it supersedes and keeps the
  reason; asking about a past date returns what was true then.

The engine is written against three small protocols (documents, vectors,
embedder), so the same code runs in-process with no server, or against
MongoDB and Qdrant behind FastAPI. The in-process stores are the reference
implementation; the database adapters pass the same contract tests.
Vector indexes: in-memory, SQLite, Qdrant, Chroma (embedded or server),
LanceDB (embedded). Document stores: in-memory, SQLite, MongoDB.

## Install

```sh
pip install scone-memory                 # core, in-process stores
pip install 'scone-memory[mongo,qdrant]' # database adapters
pip install 'scone-memory[chroma]'       # Chroma vectors (SCONE_VECTORS=chroma; SCONE_CHROMA_PATH or SCONE_CHROMA_URL)
pip install 'scone-memory[lancedb]'      # LanceDB vectors (SCONE_VECTORS=lancedb, SCONE_LANCEDB_PATH)
pip install 'scone-memory[api]'          # FastAPI server
pip install 'scone-memory[local-embed]'  # bge-small ONNX, same model as the Rust core
```

## In-process

```python
import asyncio
from scone_memory import MemoryEngine, InMemoryDocumentStore, InMemoryVectorIndex, HashEmbedder

async def main():
    engine = await MemoryEngine(InMemoryDocumentStore(), InMemoryVectorIndex(), HashEmbedder()).open()
    await engine.remember("default", "Moved to Lisbon in March; the flat is on Rua Augusta", created_at="2024-03-02")
    await engine.assert_fact("default", "mark", "lives_in", "Lisbon", valid_from="2024-03-02")
    result = await engine.recall("default", "which street is my flat on")
    for item in result.items:
        print(item.score, item.created_at[:10], item.text)
    print([f.object for f in result.facts])

asyncio.run(main())
```

`HashEmbedder` is deterministic and dependency-free; it tracks word overlap,
not meaning. Use `LocalEmbedder()` (fastembed, bge-small) or
`RemoteEmbedder(base_url, model)` (any OpenAI-compatible `/embeddings`,
including Ollama) for real semantics.

## Server

```sh
export SCONE_API_KEY=change-me
export SCONE_DOCUMENTS=mongo  SCONE_MONGO_URL=mongodb://localhost:27017
export SCONE_VECTORS=qdrant   SCONE_QDRANT_URL=http://localhost:6333
export SCONE_EMBEDDER=local
scone-memory
```

The HTTP surface is the same as the Rust `scone serve`, so the
[`scone-client`](../scone-client) package (moving from `clients/python`) and
the MCP setup work against either. The bearer key decides the space; a key can never read
outside the space it was issued for.

| Route | What |
|---|---|
| `POST /v1/episodes` | remember; `{content, tags?, source?, created_at?, kind?}`; unknown fields are refused |
| `DELETE /v1/episodes/{id}` | forget |
| `GET /v1/recall?q&limit&as_of&tags&where&history` | hybrid recall plus the facts that held at `as_of`; `history=true` adds the closed facts that came before them |
| `GET /v1/facts?all&as_of` · `POST /v1/facts` · `POST /v1/facts/{id}/close` | the fact ledger |
| `GET /v1/profile` · `GET /v1/tags` · `GET /v1/status` · `GET /healthz` | overviews |

Errors are `{"error": "..."}` with 401, 404 or 422.

## What recall returns

Items carry `score` (rank within this query; the top item is always 1.0)
and `similarity` (cosine from the vector lane, when that lane saw the
chunk). Two lanes, vector and lexical, are fused by reciprocal rank with a
small recency term, capped at two chunks per episode. A lane that fails is
named in `degraded` and the other lane still answers. `context_reduction`
is the share of the space's bytes that were left behind.

`top_similarity` is the best cosine the vector lane saw for the query.
With `SCONE_SIMILARITY_FLOOR` set (a cosine, e.g. `0.45`), a recall whose
best hit falls below it, or that finds nothing, carries
`low_confidence: true` so the reader can decline to answer from weak
evidence; without a floor the field is `null` and nothing is judged. The
floor has no default because the right value depends on the embedder:
`scone-memory bench` prints, for each candidate floor, how many
no-evidence questions it would catch and how many answerable ones it
would wrongly withhold, and that sweep is where a floor comes from.

`history=true` (CLI `--history`) adds, for every matched fact, the closed
facts that held before it for the same subject and predicate, oldest
first, each with its interval and closing reason, bounded by `as_of`.

## Using it from a framework

Three adapters live under `scone_memory.integrations`; each needs its
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
```

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

## Tests

```sh
python -m venv .venv && .venv/bin/pip install -e '.[test]'
.venv/bin/pytest
SCONE_TEST_MONGO_URL=mongodb://localhost:27017 SCONE_TEST_QDRANT_URL=http://localhost:6333 .venv/bin/pytest
```

Behavioural tests are proven to fail before they are trusted:
`PROVE_RUNNER=".venv/bin/pytest -q" ../../scripts/prove-test.sh <file> <needle> <replacement> <test>`
breaks the code, requires red, restores, requires green.
