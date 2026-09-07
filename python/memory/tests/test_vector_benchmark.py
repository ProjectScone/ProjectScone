"""Real vector-index measurements with literal independent nearest neighbors."""

import pytest

from scone_memory.core.ports import VectorPoint
from scone_memory.testing.vector_benchmark import SearchCase, measure_search


async def populate(engine):
    # Padding uses the declared width; expected neighbors come from literal axes.
    padding = [0.0] * (engine.embedder.dim - 2)
    a, b = [1.0, 0.0] + padding, [0.0, 1.0] + padding
    await engine.vectors.upsert([
        VectorPoint(1, "alpha", 1, "2024-01-01T00:00:00.000Z", a, ("red",), {"owner": "alice"}),
        VectorPoint(2, "alpha", 2, "2024-01-02T00:00:00.000Z", b, ("blue",), {"owner": "bob"}),
        VectorPoint(3, "beta", 3, "2024-01-01T00:00:00.000Z", a, ("red",), {"owner": "alice"}),
    ])
    return a


async def measured(index, cases, **kwargs):
    return await measure_search(index, cases, dataset_id="literal-axes-v1", backend_version="test-env",
                                deployment="local-test-only", **kwargs)


async def test_real_filtered_samples_retain_counts_and_ground_truth(engine):
    a = await populate(engine)
    cases = [SearchCase("nearest", "alpha", a, (1,), limit=1),
             SearchCase("metadata", "alpha", a, (2,), limit=1, where={"owner": "bob"}),
             SearchCase("space", "beta", a, (3,), limit=1, tags=("red",)),
             SearchCase("empty", "alpha", a, (), limit=1, as_of="2023-01-01T00:00:00.000Z")]
    observed = []

    async def observe(progress):
        observed.append(progress)

    report = await measured(engine.vectors, cases, repeats=2, on_progress=observe)
    assert report["sample_count"] == 8
    assert report["failed_samples"] == 0
    assert report["recall_at_k"] == {"value": 1.0, "n": 6, "denominator": "nonempty ground-truth samples, including failures"}
    assert [s["returned_ids"] for s in report["samples"]] == [[1], [2], [3], [], [1], [2], [3], []]
    assert report["samples"][3]["recall_at_k"] is None
    assert report["samples"][3]["empty_result_correct"] is True
    assert all(s["latency_ms"] >= 0 for s in report["samples"])
    assert report["latency_ms"]["n"] == 8
    assert observed[0] == {"status": "running", "done": 0, "total": 8}
    assert observed[-1] == {"status": "completed", "done": 8, "total": 8}
    assert report["product"] == "python"
    assert report["deployment"] == "local-test-only"


async def test_wrong_oracle_lowers_recall_instead_of_self_validation(engine):
    a = await populate(engine)
    report = await measured(engine.vectors, [SearchCase("wrong", "alpha", a, (2,), limit=1)], repeats=1)
    assert report["samples"][0]["returned_ids"] == [1]
    assert report["recall_at_k"]["value"] == 0.0
    assert report["failed_samples"] == 0  # retrieval quality differs from execution failure


async def test_search_failure_stays_in_recall_denominator(engine):
    await populate(engine)
    report = await measured(engine.vectors, [SearchCase("bad-dimension", "alpha", [1.0], (1,), limit=1)], repeats=1)
    assert report["failed_samples"] == 1
    assert report["recall_at_k"]["value"] == 0.0
    assert report["recall_at_k"]["n"] == 1
    assert report["samples"][0]["error_category"] == "ValueError"
    assert report["status"] == "failed"


async def test_observer_failure_is_visible_but_does_not_rewrite_search(engine):
    a = await populate(engine)

    async def unavailable(_):
        raise RuntimeError("secret endpoint must not appear in result")

    report = await measured(engine.vectors, [SearchCase("nearest", "alpha", a, (1,), limit=1)],
                            repeats=1, on_progress=unavailable)
    assert report["recall_at_k"]["value"] == 1.0
    assert report["observer_failures"] > 0
    assert report["status"] == "completed"
    assert "secret endpoint" not in str(report)


async def test_empty_workload_is_not_a_successful_benchmark(engine):
    with pytest.raises(ValueError, match="at least one"):
        await measured(engine.vectors, [], repeats=1)


async def test_only_empty_ground_truth_has_no_recall_percentage(engine):
    a = await populate(engine)
    report = await measured(engine.vectors, [SearchCase("none", "alpha", a, (), limit=1,
                            as_of="2023-01-01T00:00:00.000Z")], repeats=1)
    assert report["recall_at_k"]["value"] is None
    assert report["recall_at_k"]["n"] == 0
    assert report["samples"][0]["empty_result_correct"] is True


def test_latency_quantiles_use_documented_nearest_rank():
    from scone_memory.testing.vector_benchmark import _latencies

    result = _latencies([{"latency_ms": value} for value in (4, 1, 3, 2)])
    assert (result["n"], result["p50"], result["p95"]) == (4, 2, 4)


async def test_live_job_observer_records_actual_progress_in_its_space(engine):
    import httpx

    from scone_memory.api import create_app
    from scone_memory.observability.events import InMemoryEventLog

    a = await populate(engine)
    engine.events = InMemoryEventLog()
    app = create_app(engine, {"alpha-key": "alpha", "beta-key": "beta"})
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app), base_url="http://test") as client:
        async def observe(progress):
            response = await client.post("/v1/events", headers={"authorization": "Bearer alpha-key"}, json={
                "kind": "job", "payload": {"job_id": "literal-vector-smoke", "name": "Vector harness smoke (not ranking)",
                    "status": progress["status"], "progress": {"done": progress["done"], "total": progress["total"]},
                    "product": "python", "adapter": engine.vectors.name}})
            response.raise_for_status()

        report = await measured(engine.vectors, [SearchCase("nearest", "alpha", a, (1,), limit=1)],
                                repeats=2, on_progress=observe)
        assert report["observer_failures"] == 0
        visible = (await client.get("/v1/events", headers={"authorization": "Bearer alpha-key"})).json()["events"]
        assert visible[0]["payload"]["status"] == "completed"
        assert visible[0]["payload"]["progress"] == {"done": 2, "total": 2}
        assert visible[-1]["payload"]["progress"] == {"done": 0, "total": 2}
        other = await client.get("/v1/events", headers={"authorization": "Bearer beta-key"})
        assert other.json()["events"] == []
