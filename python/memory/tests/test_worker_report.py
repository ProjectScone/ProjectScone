"""The worker's pass report counts what the extraction gate withheld.

Separate from tests/test_worker.py, which another session is editing. The
outcome here is built from the shape the worker reads (an object with
``rejected`` entries carrying ``reason``), not from the distiller's own
classes, so this test depends only on the worker: it must pass against
the committed tree even while the distiller is mid-change elsewhere."""

from __future__ import annotations

from dataclasses import dataclass, field

from scone_memory import HashEmbedder, InMemoryDocumentStore, InMemoryEventLog, InMemoryVectorIndex, MemoryEngine
from scone_memory.ingestion.worker import ConsolidationWorker
from scone_memory import FakeChat
from scone_memory.ingestion.distill import Distiller
from scone_memory.providers.llm import ChatError


@dataclass
class Withheld:
    reason: str


@dataclass
class Outcome:
    """The fields the worker reads from a DistillOutcome, and no others."""

    episode_id: int
    added: list = field(default_factory=list)
    closed: int = 0
    skipped: int = 0
    error: object = None
    rejected: list = field(default_factory=list)


class GatedDistiller:
    """Three candidates withheld, none stored."""

    async def distill_pending(self, space, limit):
        return [Outcome(1, rejected=[Withheld("no quote"), Withheld("no quote"), Withheld("hypothetical")])]


async def test_withheld_candidates_are_counted_by_reason_on_the_distill_event():
    engine = await MemoryEngine(InMemoryDocumentStore(), InMemoryVectorIndex(), HashEmbedder(), events=InMemoryEventLog()).open()
    worker = ConsolidationWorker(engine, GatedDistiller(), ["default"])
    report = await worker.run_once("default")
    assert (report.rejected, report.rejected_reasons) == (3, {"no quote": 2, "hypothetical": 1})
    assert (report.episodes, report.proposed) == (1, 0)
    [event] = await engine.events.query("default", kind="distill")
    assert event.payload["rejected"] == 3 and event.payload["rejected_reasons"] == {"no quote": 2, "hypothetical": 1}


async def test_an_outcome_without_a_rejected_field_reports_zero():
    class Plain:
        episode_id, added, closed, skipped, error = 1, [], 0, 0, None

    class OldDistiller:
        async def distill_pending(self, space, limit):
            return [Plain()]

    engine = await MemoryEngine(InMemoryDocumentStore(), InMemoryVectorIndex(), HashEmbedder(), events=InMemoryEventLog()).open()
    report = await ConsolidationWorker(engine, OldDistiller(), ["default"]).run_once("default")
    assert (report.rejected, report.rejected_reasons, report.error) == (0, {}, None)


async def test_failed_standalone_sources_keep_their_causes_in_persisted_events(engine):
    first = await engine.remember("default", "Ana moved to Lisbon.")
    second = await engine.remember("default", "Carol drinks coffee.")
    worker = ConsolidationWorker(
        engine,
        Distiller(engine, FakeChat([ChatError("chat server unreachable: ReadTimeout"), "not json"])),
        ["default"],
    )

    report = await worker.run_once("default")

    assert report.error == "DistillError: 2 episode(s) failed"
    [event] = await engine.events.query("default", kind="distill")
    errors = event.payload["episode_errors"]
    assert errors[str(first.episode_id)] == "ChatError: chat server unreachable: ReadTimeout"
    assert "JSON array" in errors[str(second.episode_id)]
    assert len(errors) == 2


async def test_parked_source_keeps_its_cause_without_another_model_attempt(engine):
    added = await engine.remember("default", "Ana moved to Lisbon.")
    worker = ConsolidationWorker(
        engine, Distiller(engine, FakeChat([ChatError("connection refused")]), max_attempts=1),
        ["default"],
    )
    await worker.run_once("default")

    report = await worker.run_once("default")

    assert report.episodes == 0 and report.parked == 1 and report.error is None
    [event, _] = await engine.events.query("default", kind="distill")
    assert event.payload["episode_errors"] == {
        str(added.episode_id): "parked: ChatError: connection refused",
    }
