"""Local, isolated payload-index benchmark; never runs against existing data."""
from __future__ import annotations

import asyncio
import importlib.metadata
import json
import statistics
import time
from uuid import uuid4

from _qdrant_profile import Options, Trial, arguments, finish

async def main(options: Options) -> None:
    import numpy as np
    from qdrant_client import models
    from scone_memory.backends.qdrant import QdrantVectorIndex
    from scone_memory.core.ports import VectorPoint

    collection = "scone_storage_audit_" + uuid4().hex
    index = QdrantVectorIndex(options.url, collection)
    rng = np.random.default_rng(413)
    vectors = rng.normal(size=(20000, 64)).astype(np.float32)
    vectors /= np.linalg.norm(vectors, axis=1, keepdims=True)
    queries = rng.normal(size=(8, 64)).astype(np.float32)
    queries /= np.linalg.norm(queries, axis=1, keepdims=True)
    scopes = {"format": {"document_format": "image"}, "entity": {"entity_id": "entity7"},
              "both": {"document_format": "image", "entity_id": "entity7"}}
    results: list[dict[str, object]] = []
    report: dict[str, object] = {"collection": collection, "seed": 413, "points": 20000, "dimensions": 64,
              "numpy_version": np.__version__, "bit_generator": type(rng.bit_generator).__name__,
              "client_version": importlib.metadata.version("qdrant-client"),
              "method": "8 fixed queries, 1 warmup each, then 5 repetitions; exact references; fixed stage order",
              "scopes": scopes, "results": results}
    creation_started = False
    try:
        report["server_info"] = (await index.client.info()).model_dump(mode="json")
        if await index.client.collection_exists(collection):
            raise RuntimeError("new UUID collection already exists")
        print("owned_collection", collection, flush=True)
        creation_started = True
        await index.ensure(64)
        for start in range(0, len(vectors), 500):
            await index.upsert([VectorPoint(chunk_id=i + 1, space="audit", episode_id=i + 1,
                created_at="2026-09-10T00:00:00Z", vector=vectors[i].tolist(),
                tags=("image",) if i % 10 == 0 else ("text",),
                metadata={"document_format": "image" if i % 10 == 0 else "text", "entity_id": "entity" + str((i // 10) % 100)})
                for i in range(start, min(start + 500, len(vectors)))])
        references: dict[tuple[str, int], list[tuple[int, float]]] = {}
        for label, where in scopes.items():
            for number, query in enumerate(queries):
                response = await index.client.query_points(collection, query=query.tolist(),
                    query_filter=models.Filter(must=[models.FieldCondition(key="space", match=models.MatchValue(value="audit")),
                        *[models.FieldCondition(key="meta." + key, match=models.MatchValue(value=value)) for key, value in where.items()]]),
                    search_params=models.SearchParams(exact=True), limit=10, with_payload=False)
                references[label, number] = sorted([(int(point.id), float(point.score)) for point in response.points], key=lambda pair: (-pair[1], pair[0]))
        for stage in ("without_metadata_indexes", "with_metadata_indexes"):
            if stage == "with_metadata_indexes":
                configured = QdrantVectorIndex(collection=collection, client=index.client, metadata_indexes=("document_format", "entity_id"))
                await configured.ensure(64)
            info = await index.client.get_collection(collection)
            for label, where in scopes.items():
                trials: list[Trial] = []
                for query in queries:
                    await index.search("audit", query.tolist(), 10, where=where)
                for repeat in range(5):
                    for number, query in enumerate(queries):
                        started = time.perf_counter()
                        actual = await index.search("audit", query.tolist(), 10, where=where)
                        elapsed = (time.perf_counter() - started) * 1000
                        trials.append({"repeat": repeat, "query": number, "elapsed_ms": elapsed,
                                       "actual": actual, "exact_reference": references[label, number]})
                mismatches = sum([pair[0] for pair in trial["actual"]] != [pair[0] for pair in trial["exact_reference"]] for trial in trials)
                score_error = max(abs(a[1] - b[1]) for trial in trials for a, b in zip(trial["actual"], trial["exact_reference"]))
                result: dict[str, object] = {"stage": stage, "scope": label, "indexed_vectors": info.indexed_vectors_count,
                          "payload_schema": list(info.payload_schema), "trials": trials,
                          "median_ms": statistics.median(trial["elapsed_ms"] for trial in trials),
                          "p95_ms": float(np.percentile([trial["elapsed_ms"] for trial in trials], 95)),
                          "ranked_id_mismatches": mismatches, "max_score_error": score_error}
                results.append(result)
                print(json.dumps({key: value for key, value in result.items() if key != "trials"}), flush=True)
    except BaseException as error:
        report["error"] = f"{type(error).__name__}: {error}"
        raise
    finally:
        await finish(index.client, collection, creation_started, report, options.output)


if __name__ == "__main__":
    asyncio.run(main(arguments("Profile metadata payload indexes: fixed 20,000 points, 64 dimensions, seed 413.")))
