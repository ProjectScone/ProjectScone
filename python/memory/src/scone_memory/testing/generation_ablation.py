"""Paired natural generation from native retained context, without label prompts."""
from __future__ import annotations

import argparse
import asyncio
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

from ..backends.sqlite import SqliteDocumentStore, SqliteVectorIndex
from ..core.models import EpisodeKind
from ..core.ports import Embedder
from ..embedders.hash import HashEmbedder
from ..memory.engine import MemoryEngine, normalise_time
from ..providers.llm import OpenAICompatibleTextModel
from ..realtime.context import MemoryContext, _PREFIX
from ..realtime.events import ReplyCompleted, TextDelta, TextModel
from ..realtime.text import DEFAULT_SYSTEM_PROMPT
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
                       model_factory: Callable[[], TextModel] | None = None) -> dict[str, object]:
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
    results: list[dict[str, object]] = []
    report: dict[str, object] = {"schema_version": 1, "state": "running", "fixture_sha256": hashlib.sha256(fixture.read_bytes()).hexdigest(),
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
                for case in corpus.cases:
                    for trial in range(repeats):
                        for structured in ((False, True) if trial % 2 == 0 else (True, False)):
                            context = MemoryContext(engine, case.space, f"ablation-{case.id}-{trial}", where=case.where,
                                kind=case.kind, source_prefix=case.source_prefix, since=case.since, until=case.until,
                                structured_paths=structured)
                            started = time.perf_counter()
                            request, receipt = await context.prepare([{"role": "system", "content": DEFAULT_SYSTEM_PROMPT},
                                                                     {"role": "user", "content": case.query}])
                            context_ms = (time.perf_counter() - started) * 1000
                            messages = cast(list[dict[str, str]], request)
                            prompt = json.dumps(messages, ensure_ascii=False).encode()
                            row: dict[str, object] = {"case_id": case.id, "split": case.split, "query": case.query,
                                "trial": trial, "structured_paths": structured, "context_ms": round(context_ms, 3),
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
    args = parser.parse_args()
    report = asyncio.run(run_ablation(args.fixture, output=args.output, model=args.model, endpoint=args.endpoint,
        embedding_cache=args.embedding_cache, repeats=args.repeats, timeout=args.timeout))
    print(json.dumps({"output": str(args.output), "state": report["state"]}))


if __name__ == "__main__":
    main()
