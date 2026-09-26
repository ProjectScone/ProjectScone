"""Persistent local indices and shared API vectors for full-dataset runs."""
from __future__ import annotations

import asyncio
from dataclasses import dataclass
import json
from pathlib import Path
import time
from typing import cast
from urllib.parse import urlsplit
from uuid import UUID, uuid5

from llama_index.core import VectorStoreIndex
from llama_index.core.base.embeddings.base import BaseEmbedding
from llama_index.core.llms import MockLLM
from llama_index.core.retrievers import QueryFusionRetriever
from llama_index.core.retrievers.fusion_retriever import FUSION_MODES
from llama_index.core.schema import BaseNode, TextNode
from llama_index.retrievers.bm25 import BM25Retriever
from llama_index.vector_stores.qdrant import QdrantVectorStore
from qdrant_client import QdrantClient

from scone_memory import MemoryEngine
from scone_memory.backends import SqliteDocumentStore
from scone_memory.bench.comparative import CachedEmbedder, SconeEmbedding
from scone_memory.core.ports import Embedder
from scone_memory.ingestion.chunker import chunk_spans
from scone_memory.memory.engine import Record
from scone_memory.testing.public_qa import Document, Question

from .pipeline import Passage

SPACE = 'matched-public-qa'


def checkpoint_offset(path: Path, offset: int) -> None:
    pending = path.with_suffix('.pending')
    pending.write_text(json.dumps({'next': offset}))
    pending.replace(path)


class QueryOnlyEmbedder:
    """Reference adapters must use the precomputed QueryBundle embedding."""
    def __init__(self, inner: Embedder) -> None:
        self.id, self.dim = inner.id, inner.dim

    async def embed(self, texts: object) -> list[list[float]]:
        raise ValueError('reference tried to embed outside the shared cache')


def validate_url(url: str) -> None:
    parts = urlsplit(url)
    if (parts.scheme != 'http' or parts.hostname not in ('127.0.0.1', 'localhost', '::1')
            or parts.username or parts.password or parts.query or parts.fragment or parts.path not in ('', '/')):
        raise ValueError('benchmark requires a local Qdrant endpoint')


@dataclass
class Indices:
    engine: MemoryEngine
    reference: QueryFusionRetriever
    passages: dict[str, Passage]
    reference_keys: dict[str, str]
    client: QdrantClient


async def warm_vectors(cached: CachedEmbedder, documents: list[Document],
                       questions: list[Question], concurrency: int) -> None:
    texts = list(dict.fromkeys(doc.content[span.start:span.end]
        for doc in documents for span in chunk_spans(doc.content, 700)))
    print(f'Preparing {len(texts)} distinct chunk vectors and {len(questions)} query vectors', flush=True)
    semaphore = asyncio.Semaphore(concurrency)

    async def batch(offset: int, query: bool = False) -> None:
        async with semaphore:
            if query:
                await cached.embed_queries([q.question.strip() for q in questions[offset:offset + 32]])
            else:
                await cached.embed(texts[offset:offset + 32])
            if offset % 1024 == 0:
                print(f'Vectors {"query" if query else "corpus"} {offset}/{len(questions) if query else len(texts)}', flush=True)

    # Bounded task groups also bound in-flight response memory.
    for query, size in ((False, len(texts)), (True, len(questions))):
        for start in range(0, size, 32 * concurrency):
            async with asyncio.TaskGroup() as group:
                for offset in range(start, min(start + 32 * concurrency, size), 32):
                    group.create_task(batch(offset, query))


async def build(engine: MemoryEngine, cached: CachedEmbedder, documents: list[Document],
                output: Path, url: str) -> Indices:
    from scone_memory.backends.qdrant import QdrantVectorIndex

    validate_url(url)
    store = cast(SqliteDocumentStore, engine.documents)
    checkpoint = output / 'ingestion-offset.json'
    start = int(json.loads(checkpoint.read_text())['next']) if checkpoint.exists() else 0
    for offset in range(start, len(documents), 100):
        batch = documents[offset:offset + 100]
        await engine.remember_many(SPACE, [Record(doc.content, kind='file', source=doc.source_url,
            metadata={'document_id': doc.id}, created_at='2026-09-24') for doc in batch])
        checkpoint_offset(checkpoint, min(offset + 100, len(documents)))
        if offset % 1000 == 0:
            print(f'Indexed Scone {min(offset + 100, len(documents))}/{len(documents)}', flush=True)
    while True:
        count, remaining = await store.sync_text_index(SPACE)
        if not remaining:
            break
        if not count:
            raise ValueError('text index did not finish')
    if engine.vector_block is not None:
        raise ValueError('vector index unavailable')
    actual_documents = int(store.conn.execute('SELECT count(*) FROM episodes WHERE space=?', (SPACE,)).fetchone()[0])
    if actual_documents != len(documents):
        raise ValueError('incomplete document index')
    passages: dict[str, Passage] = {}
    for row in store.conn.execute('SELECT c.id, c.text, e.metadata FROM chunks c '
            'JOIN episodes e ON c.episode_id=e.id WHERE c.space=? ORDER BY c.id', (SPACE,)):
        key = str(row[0])
        passages[key] = Passage(key, str(json.loads(row[2])['document_id']), str(row[1]))
    native_vectors = cast(QdrantVectorIndex, engine.vectors)
    native_count = await native_vectors.client.count(native_vectors.collection, exact=True)
    if native_count.count != len(passages):
        raise ValueError('incomplete Scone vector index')
    client = QdrantClient(url=url, timeout=120)
    collection = native_vectors.collection + '_llamaindex'
    # qdrant-client counts its first upload attempt in max_retries.
    reference_store = QdrantVectorStore(collection, client=client, batch_size=64, max_retries=1)
    ordered = list(passages.values())
    namespace = UUID('59a5e121-b76b-4f70-8cb1-a548585f4407')
    reference_keys = {str(uuid5(namespace, p.key)): p.key for p in ordered}
    node_ids = {key: node_id for node_id, key in reference_keys.items()}
    checkpoint = output / 'reference-offset.json'
    start = int(json.loads(checkpoint.read_text())['next']) if checkpoint.exists() else 0
    for offset in range(start, len(ordered), 64):
        batch_passages = ordered[offset:offset + 64]
        vectors = await cached.embed([p.text for p in batch_passages])
        nodes: list[BaseNode] = [TextNode(id_=node_ids[p.key], text=p.text, embedding=v)
                 for p, v in zip(batch_passages, vectors, strict=True)]
        await asyncio.to_thread(reference_store.add, nodes)
        checkpoint_offset(checkpoint, min(offset + 64, len(ordered)))
        if offset % 1024 == 0:
            print(f'Indexed LlamaIndex {offset}/{len(ordered)}', flush=True)
    if client.count(collection, exact=True).count != len(ordered):
        raise ValueError('incomplete reference vector index')
    for name in (native_vectors.collection, collection):
        deadline = time.monotonic() + 1800
        while client.get_collection(name).status.value != 'green':
            if time.monotonic() >= deadline:
                raise TimeoutError('vector index did not become ready')
            print(f'Waiting for local index optimization: {name}', flush=True)
            await asyncio.sleep(10)
    adapter = cast(BaseEmbedding, SconeEmbedding(QueryOnlyEmbedder(cached)))
    index = VectorStoreIndex.from_vector_store(reference_store, embed_model=adapter)
    depth = min(64, len(ordered))
    bm25_path = output / 'bm25'
    if (bm25_path / 'retriever.json').exists():
        lexical = BM25Retriever.from_persist_dir(str(bm25_path))
    else:
        lexical = BM25Retriever.from_defaults(nodes=[TextNode(id_=node_ids[p.key], text=p.text) for p in ordered],
                                              similarity_top_k=depth)
        lexical.persist(str(bm25_path))
    reference = QueryFusionRetriever([index.as_retriever(similarity_top_k=depth), lexical],
        llm=MockLLM(), num_queries=1, similarity_top_k=32, use_async=False,
        mode=FUSION_MODES.RECIPROCAL_RANK, verbose=False)
    return Indices(engine, reference, passages, reference_keys, client)
