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
    paths = ("realtime/context.py", "retrieval/path_evidence.py", "retrieval/adaptive.py",
             "retrieval/evidence_groups.py", "providers/llm.py", "providers/evidence_assessor.py", "testing/generation_ablation.py")
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
from ..providers.evidence_assessor import SelfHostedEvidenceAssessor
from ..providers.self_hosted import validate_self_hosted_identifier
from ..realtime.context import MemoryContext, _PREFIX
from ..realtime.events import ReplyCompleted, TextDelta, TextModel
from ..realtime.text import DEFAULT_SYSTEM_PROMPT
from ..retrieval.adaptive import AdaptiveLimits, AdaptiveRetriever
from .edge_retrieval_benchmark import EdgeFixture, FixtureCase, STAMP, _CachedBGE, _seed

MAX_BYTES = 16_000


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
                            labels: dict[int, str]) -> dict[str, object]:
    pieces: list[tuple[int, str]] = []
    for message in request:
        content = message.get("content", "")
        prefix, separator, serialized = content.partition("\n")
        if not separator or not prefix.startswith(_PREFIX.rstrip("\n")):
            continue
        payload = _mapping(json.loads(serialized))
        for key, id_key, text_key in (("sources", "episode_id", "text"), ("claims", "source_episode_id", "quote")):
            for record in _records(payload.get(key)):
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


def _validate_endpoint(endpoint: str) -> str:
    parsed = urlparse(endpoint)
    if parsed.scheme not in ("http", "https") or not parsed.hostname or parsed.username or parsed.password or parsed.fragment:
        raise ValueError("provide an explicit HTTP(S) self-hosted endpoint without URL credentials")
    return endpoint.rstrip("/")


async def run_ablation(fixture: Path, *, output: Path, model: str | None = None, endpoint: str | None = None,
                       embedding_cache: Path | None = None, repeats: int = 1, timeout: float = 30.0,
                       ordered_quotes: bool = False,
                       adaptive_model: str | None = None, adaptive_timeout: float = 30.0,
                       adaptive_rounds: int = 3, baseline_paths: bool = False, group_relations: bool = False,
                       model_factory: Callable[[], TextModel] | None = None) -> dict[str, object]:
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
    results: list[dict[str, object]] = []
    report: dict[str, object] = {"schema_version": 1, "state": "running", "fixture_sha256": hashlib.sha256(fixture.read_bytes()).hexdigest(),
        "code_provenance": _CODE_PROVENANCE,
        "ordered_quotes": ordered_quotes,
        "adaptive_model": adaptive_model, "adaptive_timeout_seconds": adaptive_timeout,
        "assessment_timeout_seconds": adaptive_timeout if assessor is not None else None,
        "adaptive_rounds": adaptive_rounds, "baseline_paths": baseline_paths, "group_relations": group_relations,
        "assessment_max_evidence_bytes": 16_000 if assessor is not None else None,
        "model": model or "injected-test-provider", "endpoint": endpoint, "repeats": repeats, "timeout_seconds": timeout,
        "system_prompt_sha256": hashlib.sha256(DEFAULT_SYSTEM_PROMPT.encode()).hexdigest(), "max_prompt_bytes": MAX_BYTES,
        "max_output_bytes": MAX_BYTES, "max_output_tokens": 512, "results": results,
        "limitations": ["Only public TextDelta output is recorded; no hidden reasoning is requested or captured.",
            "Exact phrase alternatives are mechanical key-fact checks, not semantic entailment or unsupported-claim measurement.",
            "Development and held-out splits are declared in the fixture; scores do not guarantee 98–100% general performance.",
            "Generation failures and partial answers remain visible; evidence coverage is separate from answer checks."]}
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
                    limits=AdaptiveLimits(timeout_s=float(adaptive_timeout), max_rounds=adaptive_rounds))
                    if assessor is not None else None)
                for case in corpus.cases:
                    for trial in range(repeats):
                        for candidate in ((False, True) if trial % 2 == 0 else (True, False)):
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
                                "adaptive_retrieval": adaptive_retriever is not None,
                                "group_relations": group_relations and adaptive_retriever is not None,
                                "structured_paths": structured, "ordered_quotes": ordered_quotes and candidate,
                                "context_ms": round(context_ms, 3),
                                "prompt_bytes": len(prompt), "prompt_sha256": hashlib.sha256(prompt).hexdigest(),
                                "context_bytes": receipt["context_bytes"], "context_status": receipt["status"],
                                "path_count": receipt.get("path_count", 0), "multihop_status": receipt.get("multihop_status", "absent"),
                                "context_receipt": receipt}
                            row.update(await _context_coverage(engine, case, messages, labels))
                            if len(prompt) > MAX_BYTES:
                                row.update(status="prompt_limit", completed=False, answer_text="", output_bytes=0,
                                           error_type="PromptByteLimit", first_token_ms=None, total_ms=0.0)
                            else:
                                row.update(await capture_public_reply(model_factory(), messages, timeout=timeout))
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
    parser.add_argument("--ordered-quotes", action="store_true", help="Include ordered verbatim quotes only in the structured-path candidate")
    parser.add_argument("--adaptive-model", help="Explicit installed self-hosted evidence assessor model for the candidate")
    parser.add_argument("--adaptive-timeout", type=float, default=30, help="Candidate adaptive retrieval budget in seconds (1..180)")
    parser.add_argument("--adaptive-rounds", type=int, default=3, help="Maximum adaptive assessment rounds (1..4)")
    parser.add_argument("--baseline-paths", action="store_true", help="Use structured paths for the baseline; requires --adaptive-model")
    parser.add_argument("--group-relations", action="store_true", help="Select exact fact components atomically; requires --adaptive-model")
    args = parser.parse_args()
    report = asyncio.run(run_ablation(args.fixture, output=args.output, model=args.model, endpoint=args.endpoint,
        embedding_cache=args.embedding_cache, repeats=args.repeats, timeout=args.timeout, ordered_quotes=args.ordered_quotes,
        adaptive_model=args.adaptive_model, adaptive_timeout=args.adaptive_timeout,
        adaptive_rounds=args.adaptive_rounds, baseline_paths=args.baseline_paths, group_relations=args.group_relations))
    print(json.dumps({"output": str(args.output), "state": report["state"]}))


if __name__ == "__main__":
    main()
