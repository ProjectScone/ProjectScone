"""Three predeclared retrieval cases using an explicit LLM or offline cross-encoder.

Requires an existing BGE model cache. No live memory or runtime configuration
is read. Cross-encoder mode requires existing local model files and makes no
chat calls. Run with python -m scone_memory.testing.self_hosted_reranking.
"""
from __future__ import annotations

import asyncio
import argparse
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import tempfile
import time

from scone_memory import MemoryEngine, Record
from scone_memory.backends.sqlite import SqliteDocumentStore, SqliteVectorIndex
from scone_memory.embedders.local import LocalEmbedder
from scone_memory.providers.self_hosted import validate_self_hosted_endpoint, validate_self_hosted_identifier

from scone_memory.providers.self_hosted_reranker import SelfHostedLLMReranker
from scone_memory.providers.offline_reranker import OfflineCrossEncoderReranker


@dataclass(frozen=True)
class Case:
    name: str
    question: str
    target: str
    distraction: str


CASES = (
    Case("credential-response", "What should the team do when the primary secret is exposed?",
         "If an access credential leaks, revoke it, issue a replacement, and invalidate active sessions.",
         "The dashboard label is 'primary secret exposed'. The team reviews how that label is displayed; this note specifies no response procedure."),
    Case("event-retention", "How long do we keep unsuccessful checkout attempts?",
         "Abandoned purchase events expire after thirty days.",
         "The checkout attempts report marks unsuccessful attempts. It tracks how often the keep button is clicked; this note specifies no retention period."),
    Case("export-approval", "Who must sign off before a customer export leaves the company?",
         "Data extracts require approval from the privacy lead prior to external transfer.",
         "The customer export screen has a sign off label before the company logo. The design note describes its alignment and identifies no approver."),
)


@dataclass(frozen=True)
class Options:
    endpoint: str
    model: str
    embedding_cache: Path
    output: Path
    cross_encoder_dir: Path | None = None


@dataclass(frozen=True)
class Variant:
    name: str
    candidate_limit: int | None
    rerank: bool


VARIANTS = (
    Variant("default_fusion", None, False),
    Variant("expanded_fusion", 16, False),
    Variant("expanded_self_hosted_rerank", 16, True),
)


def parse_options(argv: Sequence[str] | None = None) -> Options:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--endpoint", default="http://127.0.0.1:11434/v1",
                        help="Self-hosted OpenAI-compatible chat base URL (default: %(default)s).")
    parser.add_argument("--model", required=True, help="Served model identifier, or vendor-supported model name with --cross-encoder-dir.")
    parser.add_argument("--cross-encoder-dir", type=Path,
                        help="Existing offline cross-encoder model directory; use no chat provider.")
    parser.add_argument("--embedding-cache", required=True, type=Path,
                        help="Existing FastEmbed cache containing BAAI/bge-small-en-v1.5; downloads are disabled.")
    parser.add_argument("--output", required=True, type=Path, help="JSON diagnostic report path (must not exist).")
    args = parser.parse_args(argv)
    try:
        endpoint = validate_self_hosted_endpoint(str(args.endpoint))
        model = validate_self_hosted_identifier(str(args.model))
    except ValueError as error:
        parser.error(str(error))
    cache = Path(args.embedding_cache).expanduser().resolve()
    output = Path(args.output).expanduser().resolve()
    cross_encoder_dir = Path(args.cross_encoder_dir).expanduser().resolve() if args.cross_encoder_dir is not None else None
    if not cache.is_dir():
        parser.error("--embedding-cache must be an existing directory with cached BGE model files")
    if output.exists():
        parser.error("--output already exists; choose a new report path")
    if cross_encoder_dir is not None and not cross_encoder_dir.is_dir():
        parser.error("--cross-encoder-dir must be an existing directory with offline model files")
    return Options(endpoint, model, cache, output, cross_encoder_dir)


def _code_provenance(offline: bool) -> dict[str, object]:
    package = Path(__file__).resolve().parent.parent
    paths = ["testing/self_hosted_reranking.py", "memory/engine.py", "retrieval/reranking.py",
             "core/models.py", "backends/sqlite.py", "embedders/local.py"]
    paths.extend(["providers/offline_reranker.py"] if offline else ["providers/self_hosted_reranker.py", "providers/llm.py"])
    return {"kind": "disk_snapshot", "capture_stage": "before_model_initialization",
            "captured_at_utc": datetime.now(timezone.utc).isoformat(), "loaded_code_identity_verified": False,
            "files": {path: hashlib.sha256((package / path).read_bytes()).hexdigest() for path in paths},
            "limitation": "Disk bytes at evaluation start, not proof of already imported code or the full dependency environment."}


async def evaluate(options: Options) -> None:
    validate_self_hosted_endpoint(options.endpoint)
    validate_self_hosted_identifier(options.model)
    if not options.embedding_cache.is_dir():
        raise ValueError("embedding_cache must be an existing directory with cached BGE model files")
    if options.cross_encoder_dir is not None and not options.cross_encoder_dir.is_dir():
        raise ValueError("cross_encoder_dir must be an existing directory with offline model files")
    # Reserve the destination before initializing either model. Exclusive open
    # also prevents a concurrent run from replacing an earlier result.
    options.output.parent.mkdir(parents=True, exist_ok=True)
    with options.output.open("x", encoding="utf-8") as destination:
        destination.write(json.dumps({"state": "running", "diagnostic_only": True}) + "\n")
        destination.flush()
        try:
            report = await _evaluate(options)
        except BaseException as error:
            destination.seek(0)
            destination.truncate()
            destination.write(json.dumps({"state": "failed", "diagnostic_only": True, "error_type": type(error).__name__}) + "\n")
            raise
        destination.seek(0)
        destination.truncate()
        destination.write(json.dumps(report, indent=2) + "\n")


async def _evaluate(options: Options) -> dict[str, object]:
    # Set before FastEmbed/Hugging Face is imported by LocalEmbedder. FastEmbed
    # honors HF_HUB_OFFLINE for both Hugging Face and fallback model downloads.
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    offline = options.cross_encoder_dir is not None
    provenance = _code_provenance(offline)
    reranker: SelfHostedLLMReranker | OfflineCrossEncoderReranker
    model_identity: dict[str, object] | None = None
    if options.cross_encoder_dir is not None:
        reranker = OfflineCrossEncoderReranker(options.cross_encoder_dir, model_name=options.model)
        model_identity = reranker.model_identity
    else:
        reranker = SelfHostedLLMReranker(options.endpoint, options.model)
    embedder = LocalEmbedder(cache_dir=str(options.embedding_cache))
    rows: list[dict[str, object]] = []
    with tempfile.TemporaryDirectory(prefix="scone-self-hosted-rerank-") as directory:
        for case in CASES:
            path = Path(directory) / f"{case.name}.db"
            memory = await MemoryEngine(SqliteDocumentStore(path), SqliteVectorIndex(path), embedder,
                                        reranker=reranker, rerank_limit=16, rerank_timeout=10).open()
            try:
                source = await memory.remember_many("fixture", [
                    Record(f"Design variation {number}. {case.distraction}", kind="file",
                           source=f"design/{number}.txt", metadata={"project": "fixture"},
                           created_at="2026-01-01") for number in range(8)
                ] + [Record(case.target, kind="file", source="policy/answer.txt",
                            metadata={"project": "fixture"}, created_at="2026-01-01")])
                target = source[-1].episode_id
                # Warm the same local embedding execution path; excluded from query timings.
                await embedder.embed([case.question])
                for variant in VARIANTS:
                    started = time.perf_counter()
                    result = await memory.recall("fixture", case.question, limit=1,
                                                 where={"project": "fixture"},
                                                 candidate_limit=variant.candidate_limit, rerank=variant.rerank)
                    variant_name = "expanded_offline_cross_encoder" if offline and variant.rerank else variant.name
                    row: dict[str, object] = {"case": case.name, "question": case.question, "variant": variant_name,
                           "trial": 1,
                           "target_source_id": target, "selected_source_ids": [item.episode_id for item in result.items],
                           "selected_text": [item.text for item in result.items],
                           "selected_rerank_scores": [item.rerank_score for item in result.items],
                           "target_found": any(item.episode_id == target for item in result.items),
                           "elapsed_ms": round((time.perf_counter() - started) * 1000, 3),
                           "returned_bytes": result.returned_bytes,
                           "trace": result.rerank.model_dump() if result.rerank else None,
                           "degraded": result.degraded}
                    rows.append(row)
                    print(json.dumps(row), flush=True)
            finally:
                await memory.close()
    return {
        "state": "completed", "diagnostic_only": True, "trials_per_variant": 1,
        "endpoint": None if offline else options.endpoint, "model": options.model, "embedder": embedder.id,
        "reranker_backend": "offline_cross_encoder" if offline else "self_hosted_llm",
        "cross_encoder_dir": str(options.cross_encoder_dir) if offline else None,
        "model_identity": model_identity, "code_provenance": provenance,
        "embedding_cache": str(options.embedding_cache), "offline_embeddings": True,
        "backend": "isolated SQLite FTS5 plus SQLite vector index",
        "rerank_limit": 16, "rerank_timeout_seconds": 10,
        "cases_predeclared": True, "model_calls": reranker.calls, "reranker_calls": reranker.calls,
        "chat_model_calls": 0 if offline else reranker.calls, "rows": rows,
        "limitations": ["Three synthetic cases, not a representative benchmark.",
                        "Reranking scores are raw ordering signals, not calibrated confidence or answer accuracy.",
                        "The offline cross-encoder uses only explicitly supplied local model files; embedding downloads are disabled.",
                        "Single run per variant; timings mix cold and warm model state.",
                        "No live memory writes. No universal recall or latency guarantee."]
    }


def main(argv: Sequence[str] | None = None) -> None:
    asyncio.run(evaluate(parse_options(argv)))


if __name__ == "__main__":
    main()
