"""The consolidation worker: turn pending episodes into proposed claims
on a timer, and leave a record of every pass.

Runs inside ``scone-memory serve`` when a chat model is configured, one
pass per configured space per interval. A pass either completes or
fails; either way it appends a ``distill`` event with what happened
(episodes read, proposals added, parked, error), so the Live view and
the metrics see consolidation as evidence rather than as a log line.
Nothing here decides truth: the distiller proposes and a person
approves, unless the deployer set accept_at.
"""

from __future__ import annotations

import asyncio
import logging
import time
import uuid
from dataclasses import dataclass, field
from typing import Callable, Mapping, Optional, Sequence

from .derive import Deriver
from .distill import DistillError, Distiller
from ..memory.engine import MemoryEngine
from ..core.errors import SconeError

logger = logging.getLogger(__name__)


@dataclass
class PassReport:
    space: str
    episodes: int = 0
    proposed: int = 0
    accepted: int = 0
    closed: int = 0
    skipped: int = 0
    parked: int = 0
    #: Per-source causes survive the pass and server restarts in the event
    #: log, including sources saved without an ingestion job receipt.
    episode_errors: dict[str, str] = field(default_factory=dict)
    #: Candidates the extraction gate withheld before they reached the
    #: store, and why, as counts per reason: the model's output that did
    #: not become a proposal is evidence too.
    rejected: int = 0
    rejected_reasons: dict[str, int] = field(default_factory=dict)
    #: Episodes the retention policy forgot in this pass.
    expired: int = 0
    #: The derivation pass, when the worker has a deriver: groups sent to
    #: the model, inferences proposed, restated, and rejected by the gate.
    derived_sent: int = 0
    derived_proposed: int = 0
    derived_restated: int = 0
    derived_rejected: int = 0
    error: Optional[str] = None
    latency_ms: float = 0.0

    def as_payload(self) -> dict:
        return {k: v for k, v in self.__dict__.items() if k != "space"}


class ConsolidationWorker:
    def __init__(
        self,
        engine: MemoryEngine,
        distiller: Optional[Distiller],
        spaces: Sequence[str],
        interval_s: float = 30.0,
        batch: int = 20,
        clock: Callable[[], float] = time.perf_counter,
        retention: Optional[Mapping[str, float]] = None,
        deriver: Optional["Deriver"] = None,
    ) -> None:
        """``distiller`` None means no model: the worker then only applies
        ``retention`` (episode kind -> days kept), which needs no model.
        ``deriver`` runs the derivation pass after extraction and retention."""
        if distiller is None and deriver is None and not retention:
            raise ValueError("a worker needs a distiller, a deriver, a retention policy, or some of them")
        self.engine = engine
        self.distiller = distiller
        self.deriver = deriver
        self.retention = dict(retention or {})
        self.spaces = list(spaces)
        self.interval_s = interval_s
        self.batch = batch
        self.clock = clock
        self.passes = 0
        self.last: dict[str, PassReport] = {}
        self._task: Optional[asyncio.Task] = None
        self._stop = asyncio.Event()

    async def run_once(self, space: str) -> PassReport:
        """One pass over one space; never raises, always records."""
        started, call_id = time.perf_counter(), uuid.uuid4().hex[:12]
        report: Optional[PassReport] = None
        outcome, exception_type = "cancelled", None
        logger.info("consolidation.started", extra={
            "event": "consolidation.started", "call_id": call_id, "mode": "consolidation",
        })
        try:
            report = await self._run_once(space)
            outcome = "failed" if report.error is not None else "completed"
            # _run_once constructs every error with its exception class first.
            # Keep only that class; provider/source error text never enters logs.
            exception_type = report.error.split(":", 1)[0] if report.error is not None else None
            return report
        except asyncio.CancelledError:
            exception_type = "CancelledError"
            raise
        except Exception as error:
            outcome, exception_type = "failed", type(error).__name__
            raise
        finally:
            counts = {name: getattr(report, name, 0) for name in (
                "episodes", "proposed", "accepted", "parked", "rejected", "expired",
            )}
            logger.log(logging.INFO if outcome == "completed" else logging.WARNING,
                       "consolidation.finished", extra={
                           "event": "consolidation.finished", "call_id": call_id,
                           "mode": "consolidation", "outcome": outcome,
                           "elapsed_ms": round((time.perf_counter() - started) * 1000, 3),
                           "exception_type": exception_type, **counts,
                       })

    async def _run_once(self, space: str) -> PassReport:
        started = self.clock()
        report = PassReport(space)
        try:
            outcomes = await self.distiller.distill_pending(space, limit=self.batch) if self.distiller is not None else []
        except DistillError as e:
            outcomes = e.outcomes
            report.error = f"DistillError: {e.failed} episode(s) failed"
        except SconeError as e:
            outcomes = []
            report.error = f"{type(e).__name__}: {e}"
        except Exception as e:  # noqa: BLE001 - a worker must not die on a transport hiccup
            outcomes = []
            report.error = type(e).__name__
        for o in outcomes:
            if o.error is not None and o.episode_id is not None:
                report.episode_errors[str(o.episode_id)] = o.error[:500]
            if o.error and o.error.startswith("parked"):
                report.parked += 1
                continue
            report.episodes += 1
            report.closed += o.closed
            report.skipped += o.skipped
            for r in getattr(o, "rejected", ()) or ():
                report.rejected += 1
                key = str(getattr(r, "reason", "unspecified"))[:80]
                report.rejected_reasons[key] = report.rejected_reasons.get(key, 0) + 1
            for fact in o.added:
                if fact.status == "proposed":
                    report.proposed += 1
                else:
                    report.accepted += 1
        if self.retention and report.error is None:
            try:
                report.expired = len((await self.engine.expire(space, self.retention, limit=self.batch)).forgotten)
            except SconeError as e:
                report.error = f"{type(e).__name__}: {e}"
        if self.deriver is not None and report.error is None:
            try:
                outcome = await self.deriver.derive(space, limit_groups=self.batch)
                report.derived_sent, report.derived_proposed = outcome.sent, len(outcome.proposed)
                report.derived_restated, report.derived_rejected = outcome.restated, len(outcome.rejected)
            except SconeError as e:
                report.error = f"{type(e).__name__}: {e}"
            except Exception as e:  # noqa: BLE001 - same rule as extraction: record, never die
                report.error = type(e).__name__
        report.latency_ms = round((self.clock() - started) * 1000, 3)
        self.last[space] = report
        self.passes += 1
        await self.engine._emit(space, "distill", report.as_payload())
        return report

    async def run_all(self) -> list[PassReport]:
        return [await self.run_once(space) for space in self.spaces]

    async def _loop(self) -> None:
        while not self._stop.is_set():
            await self.run_all()
            try:
                await asyncio.wait_for(self._stop.wait(), timeout=self.interval_s)
            except asyncio.TimeoutError:
                continue

    def start(self) -> None:
        if self._task is None:
            self._stop.clear()
            self._task = asyncio.create_task(self._loop(), name="scone-consolidation")

    async def stop(self) -> None:
        if self._task is None:
            return
        self._stop.set()
        try:
            await asyncio.wait_for(self._task, timeout=10)
        except (asyncio.TimeoutError, asyncio.CancelledError):
            self._task.cancel()
        self._task = None

    @property
    def running(self) -> bool:
        return self._task is not None and not self._task.done()
