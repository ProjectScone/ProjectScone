"""One diagnostic trial of three predeclared cases against an explicit self-hosted model.

Requires an existing BGE model cache. No live memory or runtime configuration
is read. Run explicitly with python -m scone_memory.testing.self_hosted_reranking.
"""
from __future__ import annotations

import asyncio
import argparse
from collections.abc import Sequence
from dataclasses import dataclass
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
    parser.add_argument("--model", required=True, help="Already-served self-hosted model identifier.")
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
    if not cache.is_dir():
        parser.error("--embedding-cache must be an existing directory with cached BGE model files")
    if output.exists():
        parser.error("--output already exists; choose a new report path")
    return Options(endpoint, model, cache, output)


async def evaluate(options: Options) -> None:
    # Set before FastEmbed/Hugging Face is imported by LocalEmbedder. FastEmbed
    # honors HF_HUB_OFFLINE for both Hugging Face and fallback model downloads.
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
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
                    row: dict[str, object] = {"case": case.name, "question": case.question, "variant": variant.name,
                           "trial": 1,
                           "target_source_id": target, "selected_source_ids": [item.episode_id for item in result.items],
                           "selected_text": [item.text for item in result.items],
                           "target_found": any(item.episode_id == target for item in result.items),
                           "elapsed_ms": round((time.perf_counter() - started) * 1000, 3),
                           "returned_bytes": result.returned_bytes,
                           "trace": result.rerank.model_dump() if result.rerank else None,
                           "degraded": result.degraded}
                    rows.append(row)
                    print(json.dumps(row), flush=True)
            finally:
                await memory.close()
    report = {
        "diagnostic_only": True, "trials_per_variant": 1,
        "endpoint": options.endpoint, "model": options.model, "embedder": embedder.id,
        "embedding_cache": str(options.embedding_cache), "offline_embeddings": True,
        "backend": "isolated SQLite FTS5 plus SQLite vector index",
        "rerank_limit": 16, "rerank_timeout_seconds": 10,
        "cases_predeclared": True, "model_calls": reranker.calls, "rows": rows,
        "limitations": ["Three synthetic cases, not a representative benchmark.",
                        "Single run per variant; timings mix cold and warm model state.",
                        "No live memory writes. No universal recall or latency guarantee."]
    }
    options.output.parent.mkdir(parents=True, exist_ok=True)
    with options.output.open("x", encoding="utf-8") as destination:
        destination.write(json.dumps(report, indent=2) + "\n")


def main(argv: Sequence[str] | None = None) -> None:
    asyncio.run(evaluate(parse_options(argv)))


if __name__ == "__main__":
    main()
