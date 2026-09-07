"""Metrics are checked against values worked out by hand from a fixed
list of events, independent of the implementation."""

from __future__ import annotations

from fastapi.testclient import TestClient

from scone_memory import HashEmbedder, InMemoryDocumentStore, InMemoryEventLog, InMemoryVectorIndex, MemoryEngine
from scone_memory.api import create_app
from scone_memory.metrics import compute, nearest_rank
from scone_memory.ports import Event


def recall(i, ts, total, lanes, similarity=0.5, returned=10, space_bytes=100, where=None, error=None, embedder="hash-256",
           low_confidence=None, floor=None):
    payload = {
        "embedder": embedder, "latency_ms": {"total": total, "embed": 1.0, "vector": total / 2, "text": 1.0},
        "items": [{"chunk_id": 1, "episode_id": 1, "score": 1.0, "similarity": similarity, "lanes": lanes}] if lanes is not None else [],
        "returned_bytes": returned, "space_bytes": space_bytes, "where": where or {}, "degraded": [],
        "low_confidence": low_confidence, "similarity_floor": floor,
    }
    if error:
        payload["error"] = error
    return Event(i, ts, "default", "recall", payload)


EVENTS = [
    recall(1, "2025-01-01T10:00:00.000Z", 10.0, {"vector": 1, "text": 1}, similarity=0.9, returned=10, space_bytes=100),
    recall(2, "2025-01-01T11:00:00.000Z", 30.0, {"vector": 1}, similarity=0.3, returned=50, space_bytes=100),
    recall(3, "2025-01-02T10:00:00.000Z", 20.0, {"text": 1}, similarity=None, returned=20, space_bytes=100, where={"user_id": "ana"}),
    recall(4, "2025-01-02T11:00:00.000Z", 40.0, None, returned=0, space_bytes=0),
    recall(5, "2025-01-02T12:00:00.000Z", 5.0, None, error="both lanes failed"),
    Event(6, "2025-01-01T12:00:00.000Z", "default", "remember", {"fresh": 3, "deduplicated": 1, "chunks": 5, "bytes": 300, "embedder": "hash-256"}),
    Event(7, "2025-01-02T12:00:00.000Z", "default", "remember", {"records": 2, "error": "TimeoutError: x"}),
    Event(8, "2025-01-01T13:00:00.000Z", "default", "fact_assert", {"outcome": "new_active", "superseded": [1, 2]}),
    Event(9, "2025-01-01T13:01:00.000Z", "default", "fact_assert", {"outcome": "restated", "superseded": []}),
    Event(10, "2025-01-01T13:02:00.000Z", "default", "fact_assert", {"outcome": "new_closed", "superseded": []}),
    Event(11, "2025-01-01T13:03:00.000Z", "default", "fact_close", {"reason_kind": "manual"}),
    Event(12, "2025-01-01T14:00:00.000Z", "default", "feedback", {"recall_event_id": 1, "chunk_id": 1, "useful": False}),
    Event(13, "2025-01-01T14:01:00.000Z", "default", "feedback", {"recall_event_id": 1, "chunk_id": 1, "useful": True}),
    Event(14, "2025-01-01T14:02:00.000Z", "default", "feedback", {"recall_event_id": 2, "chunk_id": 1, "useful": False}),
    Event(15, "2025-01-03T09:00:00.000Z", "default", "recall", {"embedder": "hash-256", "latency_ms": {"total": 99.0}, "items": []}),
]


def metric(report, name):
    [m] = [m for m in report.metrics if m.name == name]
    return m


def test_nearest_rank_is_the_documented_quantile():
    assert nearest_rank([], 0.5) is None
    assert nearest_rank([10.0], 0.95) == 10.0
    assert nearest_rank([10.0, 20.0, 30.0, 40.0], 0.5) == 20.0  # ceil(0.5*4)=2nd
    assert nearest_rank([10.0, 20.0, 30.0, 40.0], 0.95) == 40.0  # ceil(3.8)=4th


def test_report_values_match_hand_computation():
    report = compute(EVENTS, since="2025-01-01T00:00:00.000Z", until="2025-01-03T00:00:00.000Z")
    # Window excludes event 15 (Jan 3). 14 events considered; 2 carry errors.
    assert report.coverage.events_considered == 14 and report.coverage.truncated is False
    assert report.coverage.earliest_retained == "2025-01-01T10:00:00.000Z"
    assert report.failures == {"recall": 1, "remember": 1}
    assert report.embedders == ["hash-256"]

    assert metric(report, "recall.count").value == 4  # events 1..4
    lat = metric(report, "recall.latency_ms.p50")
    assert (lat.value, lat.n) == (20.0, 4)  # sorted 10,20,30,40; ceil(0.5*4)=2nd -> 20
    assert metric(report, "recall.latency_ms.p95").value == 40.0
    assert metric(report, "recall.latency_ms.vector.p50").value == 10.0  # totals/2: 5,15,10,20 -> sorted 5,10,15,20 -> 2nd=10

    both = metric(report, "recall.top_item.both_lanes_share")
    assert (both.value, both.n, both.denominator) == (round(1 / 3, 4), 3, "successful recalls that returned at least one item")
    assert metric(report, "recall.top_item.vector_only_share").value == round(1 / 3, 4)
    assert metric(report, "recall.top_item.text_only_share").value == round(1 / 3, 4)

    sim = metric(report, "recall.top_similarity.hash-256")
    assert sim.n == 2 and sim.value["median"] == 0.3 and sim.value["p90"] == 0.9  # sims 0.3, 0.9
    assert "Uncalibrated" in sim.caveat

    red = metric(report, "recall.byte_context_reduction.median")
    # reductions: 0.9, 0.5, 0.8 (event 4 has space_bytes 0 -> excluded); sorted .5,.8,.9 -> 2nd = 0.8
    assert (red.value, red.n) == (0.8, 3)

    assert metric(report, "ingest.fresh_episodes").value == 3
    assert metric(report, "ingest.dedup_fraction").value == 0.25
    assert metric(report, "ingest.stored_bytes").value == 300
    assert metric(report, "ingest.calls").value == 1  # the failed call is a failure, not an ingest

    assert metric(report, "ledger.asserted").value == 2
    assert metric(report, "ledger.asserted_already_closed").value == 1
    assert metric(report, "ledger.restated").value == 1
    assert metric(report, "ledger.superseded").value == 2
    assert metric(report, "ledger.manual_closures").value == 1

    fb = metric(report, "feedback.useful_share")
    # (1,1): latest is useful=True; (2,1): False -> 1 useful of 2 judged
    assert (fb.value, fb.n, fb.denominator) == (0.5, 2, "judged items")
    assert metric(report, "feedback.judged_items").value == 2
    assert metric(report, "feedback.coverage").value == round(2 / 3, 4)  # 3 items returned by recalls 1..3

    assert metric(report, "recall.where_values.top5").value == {"user_id": {"ana": 1}}
    assert report.daily == {
        "2025-01-01": {"recall": 2, "remember": 1, "fact_assert": 3, "fact_close": 1, "feedback": 3},
        "2025-01-02": {"recall": 3, "remember": 1},
    }


def test_no_evidence_is_none_with_n_zero_never_a_default():
    report = compute([])
    assert report.coverage.events_considered == 0 and report.coverage.earliest_retained is None
    for m in report.metrics:
        if m.unit in ("ms", "share", "cosine", "share of bytes"):
            assert m.value is None and m.n == 0, m.name
    assert metric(report, "recall.count").value == 0


def test_truncation_is_reported_not_hidden():
    assert compute(EVENTS, truncated=True).coverage.truncated is True


def test_metrics_endpoint_reports_evidence_source_and_coverage():
    import asyncio

    engine = asyncio.run(MemoryEngine(InMemoryDocumentStore(), InMemoryVectorIndex(), HashEmbedder(), events=InMemoryEventLog()).open())
    with TestClient(create_app(engine, {"k": "default"})) as c:
        h = {"authorization": "Bearer k"}
        empty = c.get("/v1/metrics", headers=h).json()
        assert empty["evidence"] == "memory" and empty["coverage"]["events_considered"] == 0
        c.post("/v1/episodes", json={"content": "the harbour crane"}, headers=h)
        c.get("/v1/recall", params={"q": "harbour"}, headers=h)
        body = c.get("/v1/metrics", headers=h).json()
        assert body["coverage"]["events_considered"] == 2
        by_name = {m["name"]: m for m in body["metrics"]}
        assert by_name["recall.count"]["value"] == 1
        assert by_name["recall.latency_ms.p50"]["value"] > 0
        assert by_name["ingest.fresh_episodes"]["value"] == 1
        assert c.get("/v1/metrics", params={"limit": "1"}, headers=h).json()["coverage"]["truncated"] is True


def test_low_confidence_share_counts_only_judged_recalls():
    events = [
        recall(1, "2025-01-01T10:00:00.000Z", 10.0, {"vector": 1}, similarity=0.9, low_confidence=False, floor=0.5),
        recall(2, "2025-01-01T11:00:00.000Z", 10.0, {"vector": 1}, similarity=0.2, low_confidence=True, floor=0.5),
        recall(3, "2025-01-01T12:00:00.000Z", 10.0, None, low_confidence=True, floor=0.5),  # found nothing
        recall(4, "2025-01-01T13:00:00.000Z", 10.0, {"text": 1}, similarity=None),  # no floor: not judged
        recall(5, "2025-01-01T14:00:00.000Z", 5.0, None, error="both lanes failed", low_confidence=True, floor=0.5),
    ]
    m = metric(compute(events), "recall.low_confidence_share")
    assert (m.value, m.n) == (round(2 / 3, 4), 3), "two of the three judged recalls were flagged; the failed one and the unjudged one are outside"
    assert "0.5" in m.definition and "abstention accuracy" in m.caveat
    none = metric(compute([events[3]]), "recall.low_confidence_share")
    assert (none.value, none.n) == (None, 0), "no judged recall, no share"
