# Qwen embeddings with local Qdrant

Use `qwen/qwen3-embedding-8b` on OpenRouter with Scone's `RemoteEmbedder`
and the locally managed `QdrantVectorIndex`. This is the 8B embedding model,
not a chat model. Its native output has 4,096 dimensions. The user explicitly
approved this exception to the earlier restriction on Chinese embedding models.

```python
import os
from scone_memory import MemoryEngine
from scone_memory.backends import SqliteDocumentStore
from scone_memory.backends.qdrant import QdrantVectorIndex
from scone_memory.embedders.remote import RemoteEmbedder

embedder = RemoteEmbedder(
    "https://openrouter.ai/api/v1",
    "qwen/qwen3-embedding-8b",
    api_key=os.environ["OPENROUTER_API_KEY"],
    dim=4096,
    trust_env=False,
    query_prefix=(
        "Instruct: Find passages that provide evidence to answer the question.\n"
        "Query:"
    ),
)
engine = await MemoryEngine(
    SqliteDocumentStore("qwen-memory.db"),
    QdrantVectorIndex("http://127.0.0.1:64076", "scone_qwen8b_v1"),
    embedder,
).open()
try:
    await engine.remember("demo", "Morgan maintains the Cedar deployment.")
    result = await engine.recall("demo", "Who looks after Cedar?")
finally:
    await engine.close()
```

Use the environment-variable name containing your server-side key. Never put
the key in browser code. Only embedding inference leaves the machine; document
storage and Qdrant remain local. Jev can still be supplied as the engine's
reranker; Gemma remains the separate answer model.

Qwen's query format includes an instruction; document inputs remain unchanged.
Scone now supports the optional `embed_queries(texts)` method, falling back to
`embed(texts)` for existing embedders. Recall, tool selection, semantic
compression, summary traversal, and the comparison adapter use this distinction.
The comparison cache stores query and document vectors separately. A configured
query prefix changes the embedder identity. Rebuild into a fresh collection
when changing embedding configurations; do not mix these vectors with a prior
model's vectors, even if their dimensions match.

The live smoke test used the actual hosted Qwen model, 4,096-dimensional
vectors, Scone's ingestion and vector-only recall, and local Docker Qdrant.
It retrieved the paraphrased Cedar/Morgan passage ahead of an unrelated fruit
passage without degraded retrieval. This proves integration, not a quality
improvement over the existing 200-question experiment. The full neural-embedding
comparison remains to be run.

During setup, Docker's daemon and Qdrant port were unresponsive. VM logs showed
disk-write errors after the host filled up. After space was available, the
stalled backend/VM and supervisor were stopped and Docker restarted. The
original Qdrant container and bind-mounted storage were retained. Its existing
collection recovered green with 2,835 points; a separate Scone write/search/
tenant-isolation test passed and its temporary collection was removed. This
does not establish a checksum-level audit of every pre-existing point.

Sources: [OpenRouter model listing](https://openrouter.ai/qwen/qwen3-embedding-8b)
and [Qwen model card](https://huggingface.co/Qwen/Qwen3-Embedding-8B).
