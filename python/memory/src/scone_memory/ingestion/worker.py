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
import time
from dataclasses import dataclass, field
from typing import Callable, Mapping, Optional, Sequence

from .distill import DistillError, Distiller
from ..memory.engine import MemoryEngine
from ..core.errors import SconeError


@dataclass
class PassReport:
    space: str
    episodes: int = 0
    proposed: int = 0
    accepted: int = 0
    closed: int = 0
    skipped: int = 0
    parked: int = 0
    #: Candidates the extraction gate withheld before they reached the
    #: store, and why, as counts per reason: the model's output that did
    #: not become a proposal is evidence too.
    rejected: int = 0
    rejected_reasons: dict[str, int] = field(default_factory=dict)
    #: Episodes the retention policy forgot in this pass.
    expired: int = 0
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
    ) -> None:
        """``distiller`` None means no model: the worker then only applies
        ``retention`` (episode kind -> days kept), which needs no model."""
        if distiller is None and not retention:
            raise ValueError("a worker needs a distiller, a retention policy, or both")
        self.engine = engine
        self.distiller = distiller
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
