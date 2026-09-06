"""The worker's pass report counts what the extraction gate withheld.
Separate from tests/test_worker.py, which another session is editing."""

from __future__ import annotations

from scone_memory import HashEmbedder, InMemoryDocumentStore, InMemoryEventLog, InMemoryVectorIndex, MemoryEngine
from scone_memory.distill import DistillOutcome, Extracted, RejectedExtraction
from scone_memory.worker import ConsolidationWorker


class GatedDistiller:
    """Stands in for the distiller: three candidates withheld, none stored."""

    async def distill_pending(self, space, limit):
        return [DistillOutcome(1, rejected=[
            RejectedExtraction(Extracted("a", "b", "c", 0.5), "no quote"),
            RejectedExtraction(Extracted("a", "b", "d", 0.5), "no quote"),
            RejectedExtraction(Extracted("x", "y", "z", 0.5), "hypothetical"),
        ])]


async def test_withheld_candidates_are_counted_by_reason_on_the_distill_event():
    engine = await MemoryEngine(InMemoryDocumentStore(), InMemoryVectorIndex(), HashEmbedder(), events=InMemoryEventLog()).open()
    worker = ConsolidationWorker(engine, GatedDistiller(), ["default"])
    report = await worker.run_once("default")
    assert (report.rejected, report.rejected_reasons) == (3, {"no quote": 2, "hypothetical": 1})
    assert (report.episodes, report.proposed) == (1, 0)
    [event] = await engine.events.query("default", kind="distill")
    assert event.payload["rejected"] == 3 and event.payload["rejected_reasons"] == {"no quote": 2, "hypothetical": 1}
