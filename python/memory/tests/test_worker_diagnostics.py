"""Consolidation diagnostics use explicit safe metadata, never error text."""

import logging

import pytest

from scone_memory import FakeChat, HashEmbedder, InMemoryDocumentStore, InMemoryVectorIndex, MemoryEngine
from scone_memory.ingestion.distill import Distiller
from scone_memory.ingestion.worker import ConsolidationWorker


@pytest.mark.parametrize("failed", [False, True])
async def test_consolidation_pass_logs_safe_counts_and_failure_class(caplog, failed):
    caplog.set_level(logging.INFO, logger="scone_memory.ingestion.worker")
    engine = await MemoryEngine(InMemoryDocumentStore(), InMemoryVectorIndex(), HashEmbedder()).open()
    await engine.remember("default", "private source text")
    worker = ConsolidationWorker(engine, Distiller(engine, FakeChat([
        "private malformed response" if failed else "[]",
    ])), ["default"])

    report = await worker.run_once("default")

    [started] = [record for record in caplog.records if getattr(record, "event", None) == "consolidation.started"]
    [finished] = [record for record in caplog.records if getattr(record, "event", None) == "consolidation.finished"]
    assert started.call_id == finished.call_id and started.mode == finished.mode == "consolidation"
    assert finished.outcome == ("failed" if failed else "completed")
    assert finished.exception_type == ("DistillError" if failed else None)
    assert finished.elapsed_ms >= 0
    assert finished.episodes == report.episodes == 1
    assert finished.proposed == finished.accepted == finished.parked == finished.rejected == finished.expired == 0
    assert "private" not in caplog.text
