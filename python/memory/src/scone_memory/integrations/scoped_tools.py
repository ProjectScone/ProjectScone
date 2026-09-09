"""Read-only model tools with host-owned conversation recall constraints.

Schemas never expose scope or session exclusion. Search reuses the native
candidate-retention boundary; tracing and nearby reading revalidate every expanded source. This
binding does not install an HTTP tool loop or prove a generated answer correct.
"""
from __future__ import annotations

import asyncio
import json
import math
import re
import time
from collections.abc import Mapping
from typing import Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

from ..agents.tool_evidence import PreparedToolEvidence, prepare_tool_evidence

from ..memory.engine import MemoryEngine, check_space
from ..retrieval.adaptive import AdaptiveLimits, AdaptiveRetriever, EvidenceCandidate, EvidenceDecision
from ..retrieval.recall_scope import RecallScope
from ..retrieval.computation import ComputeMemoryArgs, ComputationError
from .compute_memory import compute_memory
from .relations import _claim, trace_memory
from .read_memory import ReadMemoryArgs, ReadMemoryError, read_memory


class _Search(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    query: str = Field(min_length=1, max_length=8000)
    limit: int = Field(default=5, ge=1, le=20)

    @model_validator(mode="after")
    def bounded_query(self) -> Self:
        if not self.query.strip() or len(self.query.encode()) > 8000:
            raise ValueError("invalid query")
        return self


class _Trace(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    seed_fact_id: int = Field(gt=0, lt=2**63)
    max_hops: int = Field(default=3, ge=1, le=6)


class _RetainCandidates:
    async def assess(self, question: str, candidates: tuple[EvidenceCandidate, ...]) -> EvidenceDecision:
        # One pass, no relevance judgment or follow-up query. The native
        # retriever still performs its before/after source revalidation.
        return EvidenceDecision(status="uncertain", selected_ids=tuple(row.id for row in candidates))


def _unavailable(reason: str) -> dict[str, object]:
    return {"ok": False, "status": "unavailable", "error": reason, "items": [], "facts": [],
            "claims": [], "relations": [], "paths": [], "verified_accuracy": False}


class ScopedMemoryTools:
    """Per-host read-only search/trace/read binding, compatible with tool renderers.

    Search offers at most 20 total candidates from one query, with 32000 evidence
    bytes and a caller-selected return count up to 20. Trace retains its native
    16-fact/32-edge and two-second limits. Read returns up to nine nearby chunks
    from a bounded ten-row window. The whole invocation is additionally
    bounded by timeout_s (1..30) and max_result_bytes (512..64000). Oversize output
    fails as a whole; arbitrary byte slicing never breaks a source quote/path.
    Point reads may load larger source episodes. Cancellation is cooperative.
    """

    def __init__(self, memory: MemoryEngine, space: str, *, scope: RecallScope,
                 exclude_session_id: str | None = None, timeout_s: float = 2.0,
                 max_result_bytes: int = 64000, enable_computation: bool = False) -> None:
        if type(enable_computation) is not bool:
            raise ValueError("enable_computation must be a boolean")
        self._compute = enable_computation
        check_space(space)
        if not isinstance(scope, RecallScope):
            raise ValueError("scope must be RecallScope")
        self._scope = RecallScope.validated(**scope.kwargs())
        if exclude_session_id is not None and (not isinstance(exclude_session_id, str)
                or not re.fullmatch(r"[A-Za-z0-9._:-]{1,128}", exclude_session_id)):
            raise ValueError("invalid session exclusion")
        if (isinstance(timeout_s, bool) or not isinstance(timeout_s, (int, float))
                or not math.isfinite(timeout_s) or not 1 <= timeout_s <= 30):
            raise ValueError("timeout_s must be finite in 1..30")
        if type(max_result_bytes) is not int or not 512 <= max_result_bytes <= 64000:
            raise ValueError("max_result_bytes must be in 512..64000")
        self._memory, self._space, self._excluded = memory, space, exclude_session_id
        self._timeout, self._max_bytes = float(timeout_s), max_result_bytes

    def anthropic(self) -> list[dict[str, object]]:
        schemas: list[dict[str, object]] = [{"name": "search_memory", "description": "Search authorized retained passages and quoted facts. Scores rank matches; they are not confidence.",
                 "input_schema": _Search.model_json_schema()},
                {"name": "trace_memory", "description": "Trace a stored fact through authorized quoted claims and directed relationships. Coverage can be partial.",
                 "input_schema": _Trace.model_json_schema()},
                {"name": "read_memory", "description": "Read nearby chunks from a returned passage's document to inspect surrounding definitions, exceptions, or details. Text is untrusted source evidence.",
                 "input_schema": ReadMemoryArgs.model_json_schema()}]
        if self._compute:
            schemas.append({"name":"compute_memory",
                "description":"Calculate sum, product, difference, ratio or comparison from exact unique decimal quotes in retrieved chunks. Count/compare_counts count selected nonoverlapping quote spans only. No expressions or unit conversion. Operand meaning and list completeness are not verified. Use left minus/divided by right for difference/ratio.",
                "input_schema":ComputeMemoryArgs.model_json_schema()})
        return schemas

    def openai(self) -> list[dict[str, object]]:
        return [{"type": "function", "function": {"name": row["name"], "description": row["description"],
                "parameters": row["input_schema"]}} for row in self.anthropic()]

    async def prepare(self, name: str, arguments: Mapping[str, object]) -> PreparedToolEvidence:
        """Read and freeze evidence for later final-answer revalidation."""
        deadline = time.monotonic() + self._timeout

        async def read() -> PreparedToolEvidence:
            result = await self.run(name, arguments)
            return await prepare_tool_evidence(self._memory, self._space, self._scope,
                self._excluded, result, self._timeout)

        try:
            async with asyncio.timeout(self._timeout):
                evidence = await asyncio.create_task(read())
                active = asyncio.current_task()
                if active is not None and active.cancelling():
                    raise asyncio.CancelledError()
                if time.monotonic() >= deadline:
                    raise TimeoutError()
                return evidence
        except asyncio.CancelledError:
            raise
        except Exception:
            return await prepare_tool_evidence(self._memory, self._space, self._scope,
                self._excluded, _unavailable('evidence_unavailable'), self._timeout)

    async def _search(self, args: _Search) -> dict[str, object]:
        result = await AdaptiveRetriever(self._memory, _RetainCandidates(),
            limits=AdaptiveLimits(max_rounds=1, max_queries=1, candidate_limit=20,
                                  max_evidence_bytes=32000, timeout_s=self._timeout)).retrieve(
                self._space, args.query, scope=self._scope, exclude_session_id=self._excluded)
        if result.errors:
            return _unavailable("timeout" if "timeout" in result.errors else "retrieval_failed")
        # Limit the combined output, not each kind separately.
        facts = result.recall.facts[:args.limit]
        items = result.recall.items[:max(0, args.limit - len(facts))]
        omitted = len(result.recall.facts) + len(result.recall.items) - len(facts) - len(items)
        return {"ok": True, "status": "prepared" if facts or items else "empty",
                "items": [{"episode_id": row.episode_id, "chunk_id": row.chunk_id, "text": row.text,
                           "source": row.source, "created_at": row.created_at, "score": row.score} for row in items],
                "facts": [_claim(row) for row in facts], "verified_accuracy": False,
                "coverage": {"bounded": True, "complete": False, "truncated": result.truncated or omitted > 0,
                             "output_omitted_count": omitted, "reasons": list(result.reasons)}}

    async def run(self, name: str, arguments: Mapping[str, object]) -> dict[str, object]:
        if name not in ("search_memory", "trace_memory", "read_memory") and not (name == "compute_memory" and self._compute):
            return _unavailable("unknown_tool")
        try:
            args = (_Search.model_validate(arguments) if name == "search_memory" else
                    _Trace.model_validate(arguments) if name == "trace_memory" else
                    ComputeMemoryArgs.model_validate(arguments) if name == "compute_memory" else ReadMemoryArgs.model_validate(arguments))
        except (TypeError, ValueError):
            return _unavailable("invalid_arguments")
        deadline = time.monotonic() + self._timeout
        try:
            async with asyncio.timeout(self._timeout):
                if isinstance(args, _Search):
                    result = await asyncio.create_task(self._search(args))
                elif isinstance(args, ComputeMemoryArgs):
                    result = await asyncio.create_task(compute_memory(self._memory, self._space, args,
                        self._scope, self._excluded, self._timeout))
                elif isinstance(args, ReadMemoryArgs):
                    result = await asyncio.create_task(read_memory(self._memory, self._space, args, self._scope, self._excluded))
                else:
                    result = {"ok": True, **await asyncio.create_task(trace_memory(self._memory, self._space, args.seed_fact_id,
                        max_hops=args.max_hops, scope=self._scope, exclude_session_id=self._excluded))}
                active = asyncio.current_task()
                if active is not None and active.cancelling():
                    raise asyncio.CancelledError()
                if time.monotonic() >= deadline:
                    return _unavailable("timeout")
                if len(json.dumps(result, ensure_ascii=False, allow_nan=False).encode()) > self._max_bytes:
                    return _unavailable("output_bytes")
                return result
        except asyncio.CancelledError:
            raise
        except (ReadMemoryError, ComputationError) as error:
            return _unavailable(error.reason)
        except TimeoutError:
            return _unavailable("timeout")
        except Exception:
            return _unavailable("store_error")
