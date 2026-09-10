"""Paired natural generation from native retained context, without label prompts."""
from __future__ import annotations

import argparse
import asyncio
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import re
import tempfile
import time
from typing import AsyncIterator, Callable, Literal, Protocol, Self, cast, runtime_checkable
from urllib.parse import urlparse

from pydantic import Field, model_validator


def _snapshot_code() -> dict[str, object]:
    package = Path(__file__).resolve().parent.parent
    paths = ("realtime/context.py", "retrieval/path_evidence.py", "retrieval/adaptive.py", "retrieval/search_history.py",
             "retrieval/evidence_groups.py", "retrieval/evidence_blend.py", "retrieval/adaptive_graph.py", "retrieval/multihop.py",
             "realtime/answer_review.py", "realtime/review_evidence.py", "realtime/evidence_answer.py", "realtime/text.py", "realtime/tool_answer.py",
             "providers/answer_reviewer.py", "providers/evidence_selector.py", "providers/llm.py", "providers/evidence_assessor.py", "testing/generation_ablation.py",
             "agents/evidence_loop.py", "agents/tool_evidence.py", "providers/tool_chat.py",
             "providers/structured_tool_chat.py", "providers/tool_synthesis.py",
             "integrations/scoped_tools.py", "integrations/read_memory.py")
    return {"kind": "disk_snapshot", "capture_stage": "evaluator_import_before_local_imports",
            "captured_at_utc": datetime.now(timezone.utc).isoformat(),
            "loaded_code_identity_verified": False,
            "files": {path: hashlib.sha256((package / path).read_bytes()).hexdigest() for path in paths},
            "limitation": "SHA-256 of disk bytes at evaluator import, before local imports and evaluation work. "
                "Files may change afterward or modules may already be cached; this is not proof of loaded-code identity. "
                "Only the listed implementation files are fingerprinted, not the full dependency environment."}


_CODE_PROVENANCE = _snapshot_code()

from ..backends.sqlite import SqliteDocumentStore, SqliteVectorIndex
from ..core.models import EpisodeKind
from ..core.ports import Embedder
from ..embedders.hash import HashEmbedder
from ..memory.engine import MemoryEngine, normalise_time
from ..providers.llm import OpenAICompatibleTextModel
from ..providers.tool_chat import SelfHostedToolChat
from ..providers.structured_tool_chat import SelfHostedStructuredToolChat
from ..agents.evidence_loop import EvidenceToolLoop, ToolLoopLimits, ToolLoopResult, ToolModel, ToolStep
from ..integrations.scoped_tools import ScopedMemoryTools
from ..providers.evidence_assessor import SelfHostedEvidenceAssessor
from ..providers.self_hosted import validate_self_hosted_identifier, validate_self_hosted_endpoint
from ..realtime.context import ContextReceipt, MemoryContext, _PREFIX
from ..realtime.answer_review import AnswerReviewLimits, review_answer
from ..realtime.tool_answer import review_tool_answer
from ..realtime.review_evidence import prepare_review_evidence
from ..providers.answer_reviewer import SelfHostedAnswerReviewer
from ..providers.evidence_selector import SelfHostedEvidenceSelector
from ..realtime.evidence_answer import EvidenceAnswerError, construct_evidence_answer
from ..retrieval.recall_scope import RecallScope
from ..realtime.events import ReplyCompleted, TextDelta, TextModel
from ..realtime.text import DEFAULT_SYSTEM_PROMPT
from ..retrieval.adaptive import AdaptiveLimits, AdaptiveRetriever, EmptySelectionPolicy, EvidencePolicy, FailurePolicy
from ..retrieval.multihop import MultiHopLimits
from .edge_retrieval_benchmark import EdgeFixture, FixtureCase, STAMP, _CachedBGE, _seed

MAX_BYTES = 16_000
MAX_EVIDENCE_CARDS = 24


class GenerationCase(FixtureCase):
    split: Literal["development", "held_out"]
    answer_checks: tuple[tuple[str, ...], ...] = Field(min_length=1, max_length=100)
    kind: EpisodeKind | None = None
    since: str | None = None
    until: str | None = None


class GenerationFixture(EdgeFixture):
    cases: tuple[GenerationCase, ...] = Field(min_length=1, max_length=100)

    @model_validator(mode="after")
    def validate_generation_cases(self) -> Self:
        documents = {document.id: document for document in self.documents}
        for case in self.cases:
            for alternatives in case.answer_checks:
                if not alternatives or any(not phrase.strip() or len(phrase.encode()) > 1000 for phrase in alternatives):
                    raise ValueError("answer checks require nonblank alternatives of at most 1000 UTF-8 bytes")
            for required in case.required:
                source = documents[required.document_id]
                created = normalise_time(source.created_at)
                if ((case.since and created < normalise_time(case.since)) or (case.until and created > normalise_time(case.until))):
                    raise ValueError("required source is outside the case time scope")
                if case.kind not in (None, "file"):
                    raise ValueError("fixture documents are seeded as files")
        return self


def load_generation_fixture(path: Path) -> GenerationFixture:
    raw = path.read_bytes()
    if len(raw) > 8_000_000:
        raise ValueError("fixture exceeds 8 MB")
    return GenerationFixture.model_validate_json(raw)


def score_answer(answer: str, checks: tuple[tuple[str, ...], ...]) -> dict[str, object]:
    verdicts: list[dict[str, object]] = []
    for alternatives in checks:
        match = next((found for phrase in alternatives
                      if (found := re.search(r"(?<!\w)" + re.escape(phrase) + r"(?!\w)", answer, re.IGNORECASE)) is not None), None)
        verdicts.append({"alternatives": list(alternatives), "matched": match is not None,
                         "matched_quote": match.group() if match else None})
    return {"key_fact_coverage": sum(bool(verdict["matched"]) for verdict in verdicts) / len(verdicts) if verdicts else 0.0,
            "checks": verdicts, "semantic_entailment": "unmeasured", "unsupported_claim_rate": None}


@runtime_checkable
class _AsyncClose(Protocol):
    async def aclose(self) -> None: ...


async def capture_public_reply(provider: TextModel, messages: list[dict[str, str]], *, timeout: float) -> dict[str, object]:
    """Keep bounded public deltas and require adapter completion; always close."""
    started = time.perf_counter()
    first_token_ms: float | None = None
    parts: list[str] = []
    size, completed, status = 0, False, "incomplete"
    error_type: str | None = None
    cleanup_error: str | None = None
    events: AsyncIterator[TextDelta | ReplyCompleted] | None = None
    try:
        events = provider.respond(messages)
        async with asyncio.timeout(timeout):
            async for event in events:
                if isinstance(event, ReplyCompleted):
                    completed = bool("".join(parts).strip())
                    status = "completed" if completed else "empty"
                    break
                if not isinstance(event, TextDelta) or not isinstance(event.text, str):
                    status, error_type = "invalid_event", "UnexpectedPublicEvent"
                    break
                if event.text and first_token_ms is None:
                    first_token_ms = (time.perf_counter() - started) * 1000
                raw = event.text.encode()
                remaining = MAX_BYTES - size
                if len(raw) > remaining:
                    clipped = raw[:remaining].decode(errors="ignore")
                    parts.append(clipped)
                    size += len(clipped.encode())
                    status = "output_limit"
                    break
                parts.append(event.text)
                size += len(raw)
    except TimeoutError:
        status, error_type = "timeout", "TimeoutError"
    except Exception as exc:
        error_type = type(exc).__name__
        cause = type(exc.__cause__).__name__
        status = "timeout" if "Timeout" in cause else "failed"
        if "length" in str(exc).lower() or "truncat" in str(exc).lower():
            status = "truncated"
    finally:
        try:
            if isinstance(events, _AsyncClose):
                await asyncio.wait_for(events.aclose(), timeout=2.0)
        except Exception as exc:
            cleanup_error = type(exc).__name__
        finally:
            try:
                await asyncio.wait_for(provider.aclose(), timeout=2.0)
            except Exception as exc:
                cleanup_error = type(exc).__name__
    return {"status": status, "completed": completed, "answer_text": "".join(parts), "output_bytes": size,
            "truncated": status in ("truncated", "output_limit"), "error_type": error_type,
            "cleanup_error_type": cleanup_error, "first_token_ms": round(first_token_ms, 3) if first_token_ms is not None else None,
            "total_ms": round((time.perf_counter() - started) * 1000, 3)}


def _mapping(value: object) -> dict[str, object]:
    return cast(dict[str, object], value) if isinstance(value, dict) and all(isinstance(key, str) for key in value) else {}


def _records(value: object) -> list[dict[str, object]]:
    return [_mapping(item) for item in value] if isinstance(value, list) else []


async def _context_coverage(engine: MemoryEngine, case: GenerationCase, request: list[dict[str, str]],
                            labels: dict[int, str], *, selected_ids: frozenset[str] | None = None) -> dict[str, object]:
    pieces: list[tuple[int, str]] = []
    for message in request:
        content = message.get("content", "")
        prefix, separator, serialized = content.partition("\n")
        if not separator or not prefix.startswith(_PREFIX.rstrip("\n")):
            continue
        payload = _mapping(json.loads(serialized))
        record_kinds = [("sources", "episode_id", "text", "chunk", "chunk_id"),
                        ("claims", "source_episode_id", "quote", "fact", "fact_id"),
                        ("relations", "source_episode_id", "quote", "link", "link_id")]
        for key, id_key, text_key, kind, evidence_key in record_kinds:
            for record in _records(payload.get(key)):
                if selected_ids is not None and f"{kind}:{record.get(evidence_key)}" not in selected_ids:
                    continue
                episode_id, text = record.get(id_key), record.get(text_key)
                if type(episode_id) is int and isinstance(text, str):
                    pieces.append((episode_id, text))
    retained: list[tuple[str, str]] = []
    included_episode_ids: set[int] = set()
    invalid = leaks = 0
    for episode_id, text in dict.fromkeys(pieces):
        episode = await engine.documents.get_episode(case.space, episode_id)
        if episode is None:
            invalid += 1
            continue
        if (episode.space != case.space or any(episode.metadata.get(key) != value for key, value in case.where.items())
                or (case.kind is not None and episode.kind != case.kind)
                or (case.source_prefix is not None and not (episode.source or "").startswith(case.source_prefix))
                or (case.since is not None and episode.created_at < normalise_time(case.since))
                or (case.until is not None and episode.created_at > normalise_time(case.until))):
            leaks += 1
        elif text not in episode.content:
            invalid += 1
        else:
            retained.append((labels.get(episode_id, f"unlabeled:{episode_id}"), text))
            included_episode_ids.add(episode_id)
    matches = [any(document_id == required.document_id and required.quote in text for document_id, text in retained)
               for required in case.required]
    return {"evidence_coverage": sum(matches) / len(matches), "included_source_ids": sorted({document_id for document_id, _ in retained}),
            "included_episode_ids": sorted(included_episode_ids),
            "required_evidence": [{"document_id": required.document_id, "quote": required.quote, "matched": matched}
                                  for required, matched in zip(case.required, matches)],
            "scope_leaks": leaks, "invalid_provenance": invalid}


async def _review_reply(engine: MemoryEngine, case: GenerationCase, request: list[dict[str, object]],
                        receipt: ContextReceipt, row: dict[str, object], reviewer: SelfHostedAnswerReviewer,
                        timeout: float, policy: str) -> None:
    row["draft_answer_text"] = row["answer_text"]
    row["draft_first_token_ms"], row["draft_total_ms"] = row["first_token_ms"], row["total_ms"]
    row["draft_completed"], row["draft_status"] = row["completed"], row["status"]
    if not row["completed"] or receipt["status"] != "prepared":
        row["answer_review"] = {"status": "skipped", "reason": "incomplete_generation" if not row["completed"] else "no_memory_evidence",
                                "verified_accuracy": False}
        row["review_ms"] = 0.0
        row["first_token_ms"] = row["total_ms"] if row["completed"] and row["answer_text"] else None
        return
    started = time.perf_counter()
    review_deadline = time.monotonic() + timeout
    try:
        scope = RecallScope.validated(where=case.where, kind=case.kind, source_prefix=case.source_prefix,
                                      since=case.since, until=case.until)
        async with asyncio.timeout_at(review_deadline):
            material = await prepare_review_evidence(engine, case.space, scope, receipt["session_id"], request, receipt)
        reviewed = await review_answer(reviewer, case.query, cast(str, row["answer_text"]),
            material.evidence, material.evidence_ids,
            limits=AnswerReviewLimits(timeout_s=timeout, max_answer_bytes=MAX_BYTES, max_evidence_bytes=MAX_BYTES),
            validate_evidence=material.validate, deadline=review_deadline)
        row["answer_review"] = reviewed.receipt.model_dump(mode="json")
        if (reviewed.receipt.source_status in ("stale", "unavailable")
                or (policy == "require_supported" and reviewed.receipt.status != "supported")):
            row.update(status="review_rejected", completed=False, answer_text="", error_type="AnswerReviewRejected")
        else:
            row["answer_text"] = reviewed.answer
    except asyncio.CancelledError:
        raise
    except Exception:
        row["answer_review"] = {"status": "unavailable", "source_status": "unavailable", "verified_accuracy": False}
        row.update(status="review_unavailable", completed=False, answer_text="", error_type="AnswerReviewUnavailable")
    row["output_bytes"] = len(cast(str, row["answer_text"]).encode())
    review_ms = (time.perf_counter() - started) * 1000
    row["review_ms"] = round(review_ms, 3)
    row["total_ms"] = round(cast(float, row["draft_total_ms"]) + review_ms, 3)
    # Reviewed mode buffers the draft; the final answer is delivered together.
    row["first_token_ms"] = row["total_ms"] if row["answer_text"] else None


async def _extractive_reply(engine: MemoryEngine, case: GenerationCase, request: list[dict[str, object]],
                            receipt: ContextReceipt, selector: SelfHostedEvidenceSelector,
                            timeout: float) -> dict[str, object]:
    started = time.perf_counter()
    deadline = time.monotonic() + timeout
    answer = ""
    error_type: str | None = None
    status = "completed"
    answer_receipt: dict[str, object]
    try:
        scope = RecallScope.validated(where=case.where, kind=case.kind, source_prefix=case.source_prefix,
                                      since=case.since, until=case.until)
        async with asyncio.timeout_at(deadline):
            material = await prepare_review_evidence(engine, case.space, scope, receipt["session_id"], request, receipt)
        result = await construct_evidence_answer(selector, case.query, material, timeout_s=timeout,
            max_cards=MAX_EVIDENCE_CARDS, max_evidence_bytes=MAX_BYTES, max_answer_bytes=MAX_BYTES, deadline=deadline)
        answer, answer_receipt = result.answer, result.receipt
        if not answer.strip():
            status = "empty"
    except asyncio.CancelledError:
        raise
    except Exception as error:
        reason = error.reason if isinstance(error, EvidenceAnswerError) else (
            "source_validation_timeout" if isinstance(error, TimeoutError) else "source_validation_failed")
        status = "timeout" if reason.endswith("timeout") else "evidence_answer_unavailable"
        error_type = "EvidenceAnswerUnavailable"
        answer_receipt = {"status": "unavailable", "reason": reason, "mode": "extractive", "verified_accuracy": False,
                          "source_status": "stale" if reason == "stale_evidence" else "unavailable"}
    total_ms = round((time.perf_counter() - started) * 1000, 3)
    return {"status": status, "completed": status == "completed", "answer_text": answer, "output_bytes": len(answer.encode()),
            "truncated": False, "error_type": error_type, "cleanup_error_type": None,
            "first_token_ms": total_ms if answer else None, "total_ms": total_ms, "evidence_answer": answer_receipt}


def _validate_endpoint(endpoint: str) -> str:
    parsed = urlparse(endpoint)
    if parsed.scheme not in ("http", "https") or not parsed.hostname or parsed.username or parsed.password or parsed.fragment:
        raise ValueError("provide an explicit HTTP(S) self-hosted endpoint without URL credentials")
    return endpoint.rstrip("/")


def _tool_limits(timeout: float) -> ToolLoopLimits:
    return ToolLoopLimits(timeout_s=float(timeout), max_transcript_bytes=MAX_BYTES,
                          max_tool_bytes=MAX_BYTES, max_reply_bytes=MAX_BYTES)


async def _tool_reply(engine: MemoryEngine, case: GenerationCase, labels: dict[int, str],
                      provider: ToolModel, timeout: float, initial_search: bool, *,
                      reviewer: SelfHostedAnswerReviewer | None = None, review_timeout: float = 20.0,
                      review_policy: str = 'report') -> dict[str, object]:
    requests: list[dict[str, object]] = []
    observed_packets: dict[str, str] = {}

    class ObservedModel:
        async def complete(self, messages: list[dict[str, object]], tools: list[dict[str, object]]) -> ToolStep:
            raw = json.dumps({'messages':messages, 'tools':tools}, ensure_ascii=False, allow_nan=False).encode()
            requests.append({'bytes':len(raw), 'sha256':hashlib.sha256(raw).hexdigest()})
            for message in messages:
                call_id, content = message.get('tool_call_id'), message.get('content')
                if message.get('role') == 'tool' and isinstance(call_id, str) and isinstance(content, str):
                    observed_packets[call_id] = content
            return await provider.complete(messages, tools)

    row: dict[str, object] = {'answer_mode':'tool_generation', 'structured_paths':False,
        'adaptive_retrieval':False, 'ordered_quotes':False, 'answer_text':'', 'completed':False,
        'status':'failed', 'context_status':'unavailable', 'context_bytes':0, 'output_bytes':0,
        'error_type':None, 'first_token_ms':None, 'tool_retrieval':None}
    coverage_request: list[dict[str, str]] = []
    started = time.perf_counter()
    try:
        scope = RecallScope.validated(where=case.where, kind=case.kind, source_prefix=case.source_prefix,
                                      since=case.since, until=case.until)
        tools = ScopedMemoryTools(engine, case.space, scope=scope)
        result = await EvidenceToolLoop(ObservedModel(), tools, limits=_tool_limits(timeout), initial_search=initial_search).run(
            [{'role':'system','content':DEFAULT_SYSTEM_PROMPT}, {'role':'user','content':case.query}])
        row.update(answer_text=result.text, completed=True, status='completed',
            context_status='prepared' if result.evidence_ids else 'empty',
            context_bytes=sum(len(packet.encode()) for packet in result.evidence_packets),
            output_bytes=len(result.text.encode()), tool_retrieval={
                'model_calls':result.model_calls, 'tool_calls':result.tool_calls,
                'outcomes':[outcome.model_dump(mode='json') for outcome in result.tool_outcomes],
                'source_status':result.source_status, 'evidence_ids':list(result.evidence_ids),
                'verified_accuracy':False})
        if reviewer is not None:
            row['draft_total_ms'] = round((time.perf_counter() - started) * 1000, 3)
            await _review_tool_result(case, result, row, reviewer, review_timeout, review_policy)
    except asyncio.CancelledError:
        raise
    except Exception as error:
        row.update(status='timeout' if isinstance(error, TimeoutError) else 'failed', error_type=type(error).__name__)
    row.update(total_ms=round((time.perf_counter()-started)*1000,3),
               generation_provider_calls=len(requests), model_requests=requests,
               observed_tool_bytes=sum(len(packet.encode()) for packet in observed_packets.values()))
    if reviewer is not None:
        row['first_token_ms'] = row['total_ms'] if row['completed'] and row['answer_text'] else None
    # Audit evidence that reached the provider even when its answer failed.
    # This conversion is never sent to the model and contains no answer labels.
    for raw in observed_packets.values():
        packet = _mapping(json.loads(raw))
        payload = {'sources':packet.get('items', []),
                   'claims':[*_records(packet.get('facts')), *_records(packet.get('claims'))],
                   'relations':packet.get('relations', [])}
        coverage_request.append({'role':'user', 'content':_PREFIX + json.dumps(payload)})
    row.update(await _context_coverage(engine, case, coverage_request, labels))
    return row


async def _review_tool_result(case: GenerationCase, result: ToolLoopResult, row: dict[str, object],
                              reviewer: SelfHostedAnswerReviewer, timeout: float, policy: str) -> None:
    row['draft_answer_text'], row['draft_completed'], row['draft_status'] = result.text, True, 'completed'
    started = time.perf_counter()
    try:
        reviewed = await review_tool_answer(reviewer, case.query, result,
            AnswerReviewLimits(timeout_s=timeout, max_answer_bytes=MAX_BYTES, max_evidence_bytes=MAX_BYTES))
        row['answer_review'] = reviewed.receipt.model_dump(mode='json')
        if (reviewed.receipt.source_status != 'retained'
                or (policy == 'require_supported' and reviewed.receipt.status != 'supported')):
            row.update(status='review_rejected', completed=False, answer_text='', error_type='AnswerReviewRejected')
        elif not await result.validate():
            row['answer_review'] = {**reviewed.receipt.model_dump(mode='json'), 'source_status':'stale'}
            row.update(status='review_rejected', completed=False, answer_text='', error_type='StaleToolEvidence')
        else:
            row['answer_text'] = reviewed.answer
    except asyncio.CancelledError:
        raise
    except Exception as error:
        row['answer_review'] = {**_mapping(row.get('answer_review')), 'status':'unavailable',
                               'source_status':'unavailable', 'verified_accuracy':False}
        row.update(status='review_unavailable', completed=False, answer_text='', error_type=type(error).__name__)
    row['output_bytes'] = len(cast(str, row['answer_text']).encode())
    source_status = _mapping(row['answer_review'])['source_status']
    row['tool_retrieval'] = {**_mapping(row['tool_retrieval']),
        'source_status':result.source_status if source_status == 'retained' else source_status}
    row['review_ms'] = round((time.perf_counter() - started) * 1000, 3)


async def run_ablation(fixture: Path, *, output: Path, model: str | None = None, endpoint: str | None = None,
                       embedding_cache: Path | None = None, repeats: int = 1, timeout: float = 30.0,
                       ordered_quotes: bool = False,
                       adaptive_model: str | None = None, adaptive_timeout: float = 30.0,
                       adaptive_rounds: int = 3, baseline_paths: bool = False, group_relations: bool = False,
                       adaptive_failure_policy: FailurePolicy = "retain_verified", expand_relations: bool = False,
                       adaptive_empty_selection_policy: EmptySelectionPolicy = "retain_verified",
                       adaptive_evidence_policy: EvidencePolicy = "model_selected",
                       review_model: str | None = None, review_timeout: float = 20.0,
                       review_policy: Literal["report", "require_supported"] = "report",
                       review_quote_mode: Literal['text', 'spans'] = 'text',
                       evidence_selector_model: str | None = None, evidence_answer_timeout: float = 20.0,
                       tool_mode: Literal['off', 'native', 'structured'] = 'off', tool_initial_search: bool = True,
                       tool_think: bool | None = None,
                       model_factory: Callable[[], TextModel] | None = None) -> dict[str, object]:
    if tool_mode not in ('off', 'native', 'structured') or type(tool_initial_search) is not bool:
        raise ValueError('invalid tool evaluation configuration')
    if tool_think is not None and type(tool_think) is not bool:
        raise ValueError('tool_think must be a boolean or None')
    if tool_mode != 'off':
        if not model or not endpoint:
            raise ValueError('tool evaluation requires an explicit model and endpoint')
        validate_self_hosted_identifier(model)
        validate_self_hosted_endpoint(endpoint)
        if any((adaptive_model, evidence_selector_model, ordered_quotes, baseline_paths,
                group_relations, expand_relations)):
            raise ValueError('tool evaluation cannot combine independent candidate pipelines')
    if evidence_selector_model is not None and review_model is not None:
        raise ValueError("evidence_selector_model and review_model are mutually exclusive")
    if (isinstance(evidence_answer_timeout, bool) or not isinstance(evidence_answer_timeout, (int, float))
            or not math.isfinite(evidence_answer_timeout) or not 1 <= evidence_answer_timeout <= 180):
        raise ValueError("evidence_answer_timeout must be finite in 1..180")
    if evidence_selector_model is not None:
        try:
            validate_self_hosted_identifier(evidence_selector_model)
        except ValueError:
            raise ValueError("evidence_selector_model must be bounded nonblank text") from None
        if not endpoint:
            raise ValueError("evidence_selector_model requires an explicit self-hosted endpoint")
    if type(review_policy) is not str or review_policy not in ("report", "require_supported"):
        raise ValueError("review_policy must be report or require_supported")
    if type(review_quote_mode) is not str or review_quote_mode not in ('text', 'spans'):
        raise ValueError('review_quote_mode must be text or spans')
    if review_quote_mode != 'text' and review_model is None:
        raise ValueError('review_quote_mode requires review_model')
    if (isinstance(review_timeout, bool) or not isinstance(review_timeout, (int, float))
            or not math.isfinite(review_timeout) or not 1 <= review_timeout <= 180):
        raise ValueError("review_timeout must be finite in 1..180")
    if review_model is not None:
        try:
            validate_self_hosted_identifier(review_model)
        except ValueError:
            raise ValueError("review_model must be bounded nonblank text") from None
        if not endpoint:
            raise ValueError("review_model requires an explicit self-hosted endpoint")
    if type(expand_relations) is not bool:
        raise ValueError("expand_relations must be a boolean")
    if expand_relations and adaptive_model is None:
        raise ValueError("expand_relations requires adaptive_model")
    if type(adaptive_failure_policy) is not str or adaptive_failure_policy not in ("empty", "retain_verified"):
        raise ValueError("adaptive_failure_policy must be empty or retain_verified")
    if type(adaptive_empty_selection_policy) is not str or adaptive_empty_selection_policy not in ("retain_verified", "empty"):
        raise ValueError("adaptive_empty_selection_policy must be retain_verified or empty")
    if type(adaptive_evidence_policy) is not str or adaptive_evidence_policy not in ("model_selected", "original_and_selected"):
        raise ValueError("adaptive_evidence_policy must be model_selected or original_and_selected")
    if type(ordered_quotes) is not bool:
        raise ValueError("ordered_quotes must be a boolean")
    if type(group_relations) is not bool:
        raise ValueError("group_relations must be a boolean")
    if group_relations and adaptive_model is None:
        raise ValueError("group_relations requires adaptive_model")
    if type(baseline_paths) is not bool:
        raise ValueError("baseline_paths must be a boolean")
    if (isinstance(adaptive_timeout, bool) or not isinstance(adaptive_timeout, (int, float))
            or not math.isfinite(adaptive_timeout) or not 1 <= adaptive_timeout <= 180):
        raise ValueError("adaptive_timeout must be finite in [1,180]")
    if type(adaptive_rounds) is not int or not 1 <= adaptive_rounds <= 4:
        raise ValueError("adaptive_rounds must be an integer in 1..4")
    if adaptive_model is not None:
        try:
            validate_self_hosted_identifier(adaptive_model)
        except ValueError:
            raise ValueError("adaptive_model must be bounded nonblank text") from None
        if not endpoint:
            raise ValueError("adaptive_model requires an explicit self-hosted endpoint")
    elif baseline_paths:
        raise ValueError("baseline_paths requires adaptive_model")
    if type(repeats) is not int or not 1 <= repeats <= 10 or isinstance(timeout, bool) or not math.isfinite(timeout) or timeout <= 0 or timeout > 180:
        raise ValueError("repeats must be 1..10 and timeout must be finite in (0,180]")
    corpus = load_generation_fixture(fixture)
    if not corpus.cases[0].where:
        raise ValueError("fixture seeding requires a nonempty metadata scope")
    if model_factory is None:
        if not model or not endpoint:
            raise ValueError("an installed model and explicit self-hosted endpoint are required")
        base_url = _validate_endpoint(endpoint)
        def factory() -> TextModel:
            return OpenAICompatibleTextModel(base_url, model, timeout=timeout, max_output_tokens=512, trust_env=False)
        model_factory = factory
    elif endpoint is not None:
        _validate_endpoint(endpoint)
    assessor = (SelfHostedEvidenceAssessor(endpoint, adaptive_model, timeout=adaptive_timeout,
        group_relations=group_relations, max_evidence_bytes=16_000)
                if adaptive_model is not None and endpoint is not None else None)
    reviewer = (SelfHostedAnswerReviewer(endpoint, review_model, timeout=review_timeout,
        max_answer_bytes=MAX_BYTES, max_evidence_bytes=MAX_BYTES, quote_mode=review_quote_mode)
        if review_model is not None and endpoint is not None else None)
    selector = (SelfHostedEvidenceSelector(endpoint, evidence_selector_model, timeout=evidence_answer_timeout)
                if evidence_selector_model is not None and endpoint is not None else None)
    graph_limits = MultiHopLimits() if expand_relations else None
    results: list[dict[str, object]] = []
    report: dict[str, object] = {"schema_version": 1, "state": "running", "fixture_sha256": hashlib.sha256(fixture.read_bytes()).hexdigest(),
        "code_provenance": _CODE_PROVENANCE,
        "ordered_quotes": ordered_quotes,
        "candidate_answer_mode": 'tool_generation' if tool_mode != 'off' else "extractive" if selector is not None else "model_generation",
        "tool_mode": tool_mode, "tool_initial_search": tool_initial_search if tool_mode != 'off' else None,
        "tool_think": tool_think,
        "tool_limits": _tool_limits(timeout).model_dump(mode='json') if tool_mode != 'off' else None,
        "evidence_selector_model": evidence_selector_model, "evidence_answer_timeout_seconds": evidence_answer_timeout,
        "evidence_answer_max_cards": MAX_EVIDENCE_CARDS if selector is not None else None,
        "evidence_answer_max_evidence_bytes": MAX_BYTES if selector is not None else None,
        "evidence_answer_max_answer_bytes": MAX_BYTES if selector is not None else None,
        "review_model": review_model, "review_timeout_seconds": review_timeout, "review_policy": review_policy,
        "review_quote_mode": review_quote_mode if reviewer is not None else None,
        "review_max_rounds": 2 if reviewer is not None else None,
        "review_max_output_tokens": 2048 if reviewer is not None else None,
        "review_max_answer_bytes": MAX_BYTES if reviewer is not None else None,
        "review_max_evidence_bytes": MAX_BYTES if reviewer is not None else None,
        "adaptive_model": adaptive_model, "adaptive_timeout_seconds": adaptive_timeout,
        "adaptive_failure_policy": adaptive_failure_policy if assessor is not None else None,
        "adaptive_empty_selection_policy": adaptive_empty_selection_policy if assessor is not None else None,
        "adaptive_evidence_policy": adaptive_evidence_policy if assessor is not None else None,
        "assessment_timeout_seconds": adaptive_timeout if assessor is not None else None,
        "adaptive_rounds": adaptive_rounds, "baseline_paths": baseline_paths, "group_relations": group_relations,
        "expand_relations": expand_relations, "graph_limits": graph_limits.model_dump(mode="json") if graph_limits is not None else None,
        "assessment_max_evidence_bytes": 16_000 if assessor is not None else None,
        "model": model or "injected-test-provider", "endpoint": endpoint, "repeats": repeats, "timeout_seconds": timeout,
        "system_prompt_sha256": hashlib.sha256(DEFAULT_SYSTEM_PROMPT.encode()).hexdigest(), "max_prompt_bytes": MAX_BYTES,
        "max_output_bytes": MAX_BYTES, "max_output_tokens": 512, "results": results,
        "limitations": ["Only public generated text or host-rendered source excerpts are recorded; no hidden reasoning is requested or captured.",
            "Host-quoted excerpts can inflate lexical coverage; extractive scores do not measure generative answer accuracy.",
            "Exact phrase alternatives are mechanical key-fact checks, not semantic entailment or unsupported-claim measurement.",
            "Development and held-out splits are declared in the fixture; scores do not guarantee 98–100% general performance.",
            "Generation failures and partial answers remain visible; evidence coverage is separate from answer checks.",
            "Tool candidates may make multiple model requests. Their byte counts describe the model-neutral transcript and schemas, not provider wire encoding.",
            "All candidate evidence is evaluated after the turn; requirement labels never guide its retrieval or generation."]}
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8") as destination:
        destination.write(json.dumps(report, indent=2) + "\n")
    def checkpoint() -> None:
        output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    try:
        embedder: Embedder
        if embedding_cache is None:
            embedder = HashEmbedder()
        else:
            embedder = _CachedBGE(embedding_cache)
        report["embedder"] = embedder.id
        with tempfile.TemporaryDirectory(prefix="scone-generation-ablation-") as directory:
            database = str(Path(directory) / "memory.sqlite")
            engine = await MemoryEngine(SqliteDocumentStore(database), SqliteVectorIndex(database), embedder, clock=lambda: STAMP).open()
            try:
                episodes, _ = await _seed(engine, corpus, 0)
                labels = {episode_id: document_id for document_id, episode_id in episodes.items()}
                adaptive = (AdaptiveRetriever(engine, assessor,
                    limits=AdaptiveLimits(timeout_s=float(adaptive_timeout), max_rounds=adaptive_rounds),
                    failure_policy=adaptive_failure_policy, graph_limits=graph_limits,
                    empty_selection_policy=adaptive_empty_selection_policy, evidence_policy=adaptive_evidence_policy)
                    if assessor is not None else None)
                for case in corpus.cases:
                    for trial in range(repeats):
                        for candidate in ((False, True) if trial % 2 == 0 else (True, False)):
                            if candidate and tool_mode != 'off':
                                assert model is not None and endpoint is not None
                                provider = (SelfHostedStructuredToolChat if tool_mode == 'structured' else SelfHostedToolChat)(
                                    endpoint, model, timeout_s=timeout, max_tokens=512, think=tool_think)
                                tool_row = await _tool_reply(engine, case, labels, provider, timeout, tool_initial_search,
                                    reviewer=reviewer, review_timeout=review_timeout, review_policy=review_policy)
                                tool_row.update(case_id=case.id, split=case.split, query=case.query, trial=trial, variant='candidate')
                                tool_row.update(score_answer(cast(str, tool_row['answer_text']), case.answer_checks))
                                tool_row['successful_key_fact_coverage'] = tool_row['key_fact_coverage'] if tool_row['completed'] else 0.0
                                results.append(tool_row)
                                checkpoint()
                                continue
                            structured = candidate or baseline_paths
                            adaptive_retriever = adaptive if candidate else None
                            context = MemoryContext(engine, case.space, f"ablation-{case.id}-{trial}", where=case.where,
                                kind=case.kind, source_prefix=case.source_prefix, since=case.since, until=case.until,
                                structured_paths=structured, path_quotes=ordered_quotes and candidate,
                                adaptive_retriever=adaptive_retriever,
                                recall_timeout=adaptive_timeout if adaptive_retriever is not None else 2.0)
                            started = time.perf_counter()
                            request, receipt = await context.prepare([{"role": "system", "content": DEFAULT_SYSTEM_PROMPT},
                                                                     {"role": "user", "content": case.query}])
                            context_ms = (time.perf_counter() - started) * 1000
                            messages = cast(list[dict[str, str]], request)
                            prompt = json.dumps(messages, ensure_ascii=False).encode()
                            row: dict[str, object] = {"case_id": case.id, "split": case.split, "query": case.query,
                                "trial": trial, "variant": "candidate" if candidate else "baseline",
                                "answer_mode": "extractive" if selector is not None and candidate and receipt["status"] == "prepared" else "model_generation",
                                "generation_provider_calls": 0,
                                "adaptive_retrieval": adaptive_retriever is not None,
                                "adaptive_failure_policy": adaptive_failure_policy if adaptive_retriever is not None else None,
                                "adaptive_empty_selection_policy": adaptive_empty_selection_policy if adaptive_retriever is not None else None,
                                "adaptive_evidence_policy": adaptive_evidence_policy if adaptive_retriever is not None else None,
                                "group_relations": group_relations and adaptive_retriever is not None,
                                "expand_relations": expand_relations and adaptive_retriever is not None,
                                "structured_paths": structured, "ordered_quotes": ordered_quotes and candidate,
                                "context_ms": round(context_ms, 3),
                                "prompt_bytes": len(prompt), "prompt_sha256": hashlib.sha256(prompt).hexdigest(),
                                "context_bytes": receipt["context_bytes"], "context_status": receipt["status"],
                                "path_count": receipt.get("path_count", 0), "multihop_status": receipt.get("multihop_status", "absent"),
                                "context_receipt": receipt}
                            row.update(await _context_coverage(engine, case, messages, labels))
                            if selector is not None and candidate and receipt["status"] == "prepared":
                                row.update(await _extractive_reply(engine, case, request, receipt, selector, evidence_answer_timeout))
                            elif len(prompt) > MAX_BYTES:
                                row.update(status="prompt_limit", completed=False, answer_text="", output_bytes=0,
                                           error_type="PromptByteLimit", first_token_ms=None, total_ms=0.0)
                            else:
                                row["generation_provider_calls"] = 1
                                row.update(await capture_public_reply(model_factory(), messages, timeout=timeout))
                            if selector is not None and candidate and receipt["status"] != "prepared":
                                row["evidence_answer"] = {"status": "skipped", "reason": "no_memory_evidence", "verified_accuracy": False}
                            if reviewer is not None and candidate:
                                await _review_reply(engine, case, request, receipt, row, reviewer, review_timeout, review_policy)
                            if row["answer_mode"] == "extractive":
                                raw_ids = _mapping(row.get("evidence_answer")).get("evidence_ids")
                                selected_ids = frozenset(value for value in raw_ids if isinstance(value, str)) if isinstance(raw_ids, (list, tuple)) else frozenset()
                                selected_coverage = await _context_coverage(engine, case, messages, labels, selected_ids=selected_ids)
                                row["selected_evidence_coverage"] = selected_coverage["evidence_coverage"]
                                row["selected_evidence_scope_leaks"] = selected_coverage["scope_leaks"]
                                row["selected_evidence_invalid_provenance"] = selected_coverage["invalid_provenance"]
                            row.update(score_answer(cast(str, row["answer_text"]), case.answer_checks))
                            row["successful_key_fact_coverage"] = row["key_fact_coverage"] if row["completed"] else 0.0
                            results.append(row)
                            checkpoint()
            finally:
                await engine.close()
        report["state"] = "completed"
    except BaseException as exc:
        report["state"], report["error_type"] = "failed", type(exc).__name__
        checkpoint()
        raise
    checkpoint()
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fixture", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--endpoint", required=True)
    parser.add_argument("--embedding-cache", type=Path)
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--timeout", type=float, default=30)
    parser.add_argument('--tool-mode', choices=('off','native','structured'), default='off')
    parser.add_argument('--tool-initial-search', action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument('--tool-think', action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--ordered-quotes", action="store_true", help="Include ordered verbatim quotes only in the structured-path candidate")
    parser.add_argument("--adaptive-model", help="Explicit installed self-hosted evidence assessor model for the candidate")
    parser.add_argument("--adaptive-timeout", type=float, default=30, help="Candidate adaptive retrieval budget in seconds (1..180)")
    parser.add_argument("--adaptive-rounds", type=int, default=3, help="Maximum adaptive assessment rounds (1..4)")
    parser.add_argument("--baseline-paths", action="store_true", help="Use structured paths for the baseline; requires --adaptive-model")
    parser.add_argument("--group-relations", action="store_true", help="Select exact fact components atomically; requires --adaptive-model")
    parser.add_argument("--adaptive-failure-policy", choices=("empty", "retain_verified"), default="retain_verified",
                        help="On assessor failure, reverify candidates (default) or return empty evidence")
    parser.add_argument("--adaptive-empty-selection-policy", choices=("retain_verified", "empty"), default="retain_verified",
                        help="On a valid empty selection, retain verified candidates (default) or return empty evidence")
    parser.add_argument("--adaptive-evidence-policy", choices=("model_selected", "original_and_selected"), default="model_selected",
                        help="Use model-selected evidence (default) or also preserve evidence from the original query")
    parser.add_argument("--expand-relations", action="store_true",
                        help="Gather bounded graph evidence before assessment; requires --adaptive-model")
    parser.add_argument("--review-model", help="Explicit installed answer reviewer model for the candidate")
    parser.add_argument("--review-timeout", type=float, default=20, help="Whole answer review budget including source checks")
    parser.add_argument("--review-policy", choices=("report", "require_supported"), default="report")
    parser.add_argument('--review-quote-mode', choices=('text', 'spans'), default='text')
    parser.add_argument("--evidence-selector-model", help="Select source cards for a host-rendered candidate answer; mutually exclusive with --review-model")
    parser.add_argument("--evidence-answer-timeout", type=float, default=20, help="Whole extractive answer budget including source checks")
    args = parser.parse_args()
    report = asyncio.run(run_ablation(args.fixture, output=args.output, model=args.model, endpoint=args.endpoint,
        embedding_cache=args.embedding_cache, repeats=args.repeats, timeout=args.timeout, ordered_quotes=args.ordered_quotes,
        tool_mode=args.tool_mode, tool_initial_search=args.tool_initial_search, tool_think=args.tool_think,
        adaptive_model=args.adaptive_model, adaptive_timeout=args.adaptive_timeout,
        adaptive_rounds=args.adaptive_rounds, baseline_paths=args.baseline_paths, group_relations=args.group_relations,
        adaptive_failure_policy=args.adaptive_failure_policy, expand_relations=args.expand_relations,
        adaptive_empty_selection_policy=args.adaptive_empty_selection_policy,
        adaptive_evidence_policy=args.adaptive_evidence_policy,
        review_model=args.review_model, review_timeout=args.review_timeout, review_policy=args.review_policy,
        review_quote_mode=args.review_quote_mode,
        evidence_selector_model=args.evidence_selector_model, evidence_answer_timeout=args.evidence_answer_timeout))
    print(json.dumps({"output": str(args.output), "state": report["state"]}))


if __name__ == "__main__":
    main()
