"""Bounded HNSW experiment against one owned UUID collection on local Qdrant."""
from __future__ import annotations

import asyncio
import importlib.metadata
import json
import statistics
import time
from uuid import uuid4

from _qdrant_profile import Options, RecallTrial, arguments, finish

async def main(options: Options) -> None:
    import numpy as np
    from qdrant_client import AsyncQdrantClient, models
    from scone_memory.backends.qdrant import QdrantVectorIndex
    from scone_memory.core.ports import VectorPoint

    collection = "scone_hnsw_audit_" + uuid4().hex
    client = AsyncQdrantClient(url=options.url)
    index = QdrantVectorIndex(collection=collection, client=client, metadata_indexes=("document_format", "entity_id"))
    rng = np.random.default_rng(413)
    vectors = rng.normal(size=(20000, 64)).astype(np.float32)
    vectors /= np.linalg.norm(vectors, axis=1, keepdims=True)
    queries = rng.normal(size=(24, 64)).astype(np.float32)
    queries /= np.linalg.norm(queries, axis=1, keepdims=True)
    scopes = {"unfiltered": {}, "format": {"document_format": "image"}, "entity": {"entity_id": "entity7"},
              "both": {"document_format": "image", "entity_id": "entity7"}}
    filters = {label: models.Filter(must=[models.FieldCondition(key="space", match=models.MatchValue(value="audit")),
        *[models.FieldCondition(key="meta." + key, match=models.MatchValue(value=value)) for key, value in where.items()]])
        for label, where in scopes.items()}
    results: list[dict[str, object]] = []
    report: dict[str, object] = {"collection": collection, "seed": 413, "points": 20000, "dimensions": 64,
              "numpy_version": np.__version__, "bit_generator": type(rng.bit_generator).__name__,
              "client_version": importlib.metadata.version("qdrant-client"),
              "method": "24 fixed queries, 1 warmup each per parameter, 3 repetitions in fixed parameter order; exact references before timings",
              "scopes": scopes, "results": results}
    creation_started = False
    try:
        report["server_info"] = (await client.info()).model_dump(mode="json")
        if await client.collection_exists(collection):
            raise RuntimeError("new UUID collection already exists")
        print("owned_collection", collection, flush=True)
        creation_started = True
        await client.create_collection(collection,
            vectors_config=models.VectorParams(size=64, distance=models.Distance.COSINE),
            hnsw_config=models.HnswConfigDiff(m=16, ef_construct=100, full_scan_threshold=16, max_indexing_threads=1),
            optimizers_config=models.OptimizersConfigDiff(indexing_threshold=0), shard_number=1)
        await index.ensure(64)
        started = time.perf_counter()
        for start in range(0, len(vectors), 500):
            await index.upsert([VectorPoint(chunk_id=i + 1, space="audit", episode_id=i + 1,
                created_at="2026-09-10T00:00:00Z", vector=vectors[i].tolist(),
                tags=("image",) if i % 10 == 0 else ("text",),
                metadata={"document_format": "image" if i % 10 == 0 else "text", "entity_id": "entity" + str((i // 10) % 100)})
                for i in range(start, min(start + 500, len(vectors)))])
        report["upload_seconds"] = time.perf_counter() - started
        started = time.perf_counter()
        await client.update_collection(collection, optimizers_config=models.OptimizersConfigDiff(indexing_threshold=64))
        while True:
            info = await client.get_collection(collection)
            elapsed = time.perf_counter() - started
            print("build", round(elapsed, 2), info.status, info.indexed_vectors_count, info.optimizer_status, flush=True)
            if (info.indexed_vectors_count or 0) >= 20000 and str(info.status) == "green" and str(info.optimizer_status) == "ok":
                break
            if elapsed > 180:
                raise TimeoutError("owned HNSW collection did not finish indexing within 180 seconds")
            await asyncio.sleep(2)
        report["build_seconds"] = time.perf_counter() - started
        report["collection_info"] = info.model_dump(mode="json")
        references: dict[tuple[str, int], list[tuple[int, float]]] = {}
        exact_trials = []
        for label, filter_value in filters.items():
            for number, query in enumerate(queries):
                started = time.perf_counter()
                response = await client.query_points(collection, query=query.tolist(), query_filter=filter_value,
                    search_params=models.SearchParams(exact=True), limit=10, with_payload=False)
                exact_trials.append({"scope": label, "query": number, "elapsed_ms": (time.perf_counter() - started) * 1000})
                references[label, number] = sorted([(int(point.id), float(point.score)) for point in response.points], key=lambda pair: (-pair[1], pair[0]))
        report["exact_trials"] = exact_trials
        for ef in (32, 64, 128, 256):
            for label, filter_value in filters.items():
                params = models.SearchParams(hnsw_ef=ef, exact=False)
                trials: list[RecallTrial] = []
                for query in queries:
                    await client.query_points(collection, query=query.tolist(), query_filter=filter_value,
                        search_params=params, limit=10, with_payload=False)
                for repeat in range(3):
                    for number, query in enumerate(queries):
                        started = time.perf_counter()
                        response = await client.query_points(collection, query=query.tolist(), query_filter=filter_value,
                            search_params=params, limit=10, with_payload=False)
                        elapsed = (time.perf_counter() - started) * 1000
                        actual = sorted([(int(point.id), float(point.score)) for point in response.points], key=lambda pair: (-pair[1], pair[0]))
                        reference = references[label, number]
                        recall = len({pair[0] for pair in actual} & {pair[0] for pair in reference}) / len(reference)
                        trials.append({"repeat": repeat, "query": number, "elapsed_ms": elapsed,
                            "actual": actual, "exact_reference": reference, "recall_at_10": recall})
                result: dict[str, object] = {"hnsw_ef": ef, "scope": label, "trials": trials,
                    "median_ms": statistics.median(trial["elapsed_ms"] for trial in trials),
                    "p95_ms": float(np.percentile([trial["elapsed_ms"] for trial in trials], 95)),
                    "recall_at_10": statistics.mean(trial["recall_at_10"] for trial in trials),
                    "queries_with_exact_top_10": sum(trial["recall_at_10"] == 1 for trial in trials)}
                results.append(result)
                print(json.dumps({key: value for key, value in result.items() if key != "trials"}), flush=True)
    except BaseException as error:
        report["error"] = f"{type(error).__name__}: {error}"
        raise
    finally:
        await finish(client, collection, creation_started, report, options.output)


if __name__ == "__main__":
    asyncio.run(main(arguments("Profile HNSW search effort against exact references: fixed 20,000 points, 64 dimensions, seed 413.")))
