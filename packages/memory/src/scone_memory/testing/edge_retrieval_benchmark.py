"""Offline evidence coverage evaluation on versioned original edge fixtures."""
from __future__ import annotations

import argparse
import asyncio
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import statistics
import tempfile
import time
from typing import Iterable, Literal, Self, Sequence, cast

from pydantic import BaseModel, ConfigDict, Field, JsonValue, model_validator

from ..backends.sqlite import SqliteDocumentStore, SqliteVectorIndex
from ..core.models import RecallResult
from ..core.ports import Embedder, TextFilter
from ..embedders.hash import HashEmbedder
from ..memory.engine import MemoryEngine, Record, check_space, normalise_time
from ..retrieval.multihop import MultiHopLimits, expand_multihop
from ..retrieval.structural import StructuralLimits, expand_structural_context

STAMP = "2026-09-07T00:00:00.000Z"


class _FixtureModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)


class FixtureDocument(_FixtureModel):
    id: str = Field(min_length=1)
    content: str = Field(min_length=1, max_length=2_000_000)
    source: str = Field(min_length=1)
    space: str = Field(min_length=1)
    tags: tuple[str, ...]
    where: dict[str, str]
    created_at: str


class FixtureFact(_FixtureModel):
    id: str = Field(min_length=1)
    document_id: str
    subject: str = Field(min_length=1)
    predicate: str = Field(min_length=1)
    object: str = Field(min_length=1)
    quote: str = Field(min_length=1)
    valid_from: str


class RequiredEvidence(_FixtureModel):
    document_id: str
    quote: str = Field(min_length=1)


class FixtureCase(_FixtureModel):
    id: str = Field(min_length=1)
    mode: Literal["structural", "multihop"]
    query: str = Field(min_length=1)
    space: str
    where: dict[str, str]
    source_prefix: str | None
    seed_fact_ids: tuple[str, ...]
    required: tuple[RequiredEvidence, ...] = Field(min_length=1)


def _inside(space: str, source: str | None, metadata: dict[str, str], case: FixtureCase) -> bool:
    return (space == case.space and all(metadata.get(key) == value for key, value in case.where.items())
            and (case.source_prefix is None or (source is not None and source.startswith(case.source_prefix))))


class EdgeFixture(_FixtureModel):
    schema_version: Literal[1]
    description: str
    documents: tuple[FixtureDocument, ...] = Field(min_length=1, max_length=10_000)
    facts: tuple[FixtureFact, ...] = Field(max_length=10_000)
    cases: tuple[FixtureCase, ...] = Field(min_length=1, max_length=100)

    @model_validator(mode="after")
    def validate_references(self) -> Self:
        for records in (self.documents, self.facts, self.cases):
            ids = [record.id for record in records]
            if len(ids) != len(set(ids)):
                raise ValueError("fixture IDs must be unique within each collection")
        documents = {document.id: document for document in self.documents}
        facts = {fact.id: fact for fact in self.facts}
        identities = {(document.space, document.content.strip()) for document in self.documents}
        if len(identities) != len(self.documents):
            raise ValueError("different document labels share an engine deduplication identity")
        for document in self.documents:
            check_space(document.space)
            normalise_time(document.created_at)
            if len(document.content.encode()) > 2_000_000:
                raise ValueError("fixture source exceeds UTF-8 byte limit")
        for fact in self.facts:
            normalise_time(fact.valid_from)
            fact_document = documents.get(fact.document_id)
            if fact_document is None or fact.quote not in fact_document.content:
                raise ValueError("fact must cite a literal quote from a known source")
        for case in self.cases:
            check_space(case.space)
            if len(set(case.seed_fact_ids)) != len(case.seed_fact_ids):
                raise ValueError("duplicate seed fact ID")
            for required in case.required:
                required_document = documents.get(required.document_id)
                if required_document is None or required.quote not in required_document.content:
                    raise ValueError("required evidence must be a literal source quote")
                if not _inside(required_document.space, required_document.source, required_document.where, case):
                    raise ValueError("required evidence is outside the case scope")
            for seed_id in case.seed_fact_ids:
                seed = facts.get(seed_id)
                if seed is None:
                    raise ValueError("seed fact reference is unknown")
                document = documents[seed.document_id]
                if not _inside(document.space, document.source, document.where, case):
                    raise ValueError("seed fact is outside the case scope")
        return self


def load_fixture(path: Path) -> EdgeFixture:
    raw = path.read_bytes()
    if len(raw) > 8_000_000:
        raise ValueError("fixture exceeds 8 MB")
    return EdgeFixture.model_validate_json(raw)


class _CachedBGE:
    id = "bge-small-en-v1.5"
    dim = 384

    def __init__(self, cache: Path) -> None:
        if not cache.is_dir():
            raise FileNotFoundError("embedding cache must be an existing local directory")
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"
        from ..providers._onnx import prepare_onnx_runtime
        prepare_onnx_runtime()
        from fastembed import TextEmbedding
        self._model = TextEmbedding(model_name="BAAI/bge-small-en-v1.5", cache_dir=str(cache),
                                    local_files_only=True, threads=1)

    async def embed(self, texts: Sequence[str]) -> list[list[float]]:
        vectors = cast(Iterable[Iterable[float]], self._model.embed(list(texts)))
        return [[float(value) for value in vector] for vector in vectors]


@dataclass(frozen=True)
class _Piece:
    episode_id: int
    text: str


def _recalled(result: RecallResult) -> list[_Piece]:
    return [_Piece(item.episode_id, item.text) for item in result.items] + [
        _Piece(fact.source_episode_id, fact.quote) for fact in result.facts
        if fact.source_episode_id is not None and fact.quote is not None]


async def _coverage(engine: MemoryEngine, case: FixtureCase, required_ids: dict[str, int],
                    pieces: list[_Piece]) -> dict[str, object]:
    unique = list(dict.fromkeys(pieces))
    accepted: list[_Piece] = []
    scope_leaks = invalid = 0
    for piece in unique:
        episode = await engine.documents.get_episode(case.space, piece.episode_id)
        if episode is None or not _inside(episode.space, episode.source, episode.metadata, case):
            scope_leaks += 1
        elif piece.text not in episode.content:
            invalid += 1
        else:
            accepted.append(piece)
    matched = [any(piece.episode_id == required_ids[required.document_id] and required.quote in piece.text
                   for piece in accepted) for required in case.required]
    required_sources = {required_ids[required.document_id] for required in case.required}
    matched_sources = required_sources & {piece.episode_id for piece in accepted}
    return {"quote_coverage": sum(matched) / len(matched), "source_coverage": len(matched_sources) / len(required_sources),
            "required_quotes": len(matched), "matched_quotes": sum(matched),
            "required_sources": len(required_sources), "matched_sources": len(matched_sources),
            "matches": [{"document_id": required.document_id, "quote": required.quote, "matched": found}
                        for required, found in zip(case.required, matched)],
            "text_bytes": sum(len(piece.text.encode()) for piece in unique),
            "scope_leaks": scope_leaks, "invalid_provenance": invalid}


async def _seed(engine: MemoryEngine, fixture: EdgeFixture, size: int) -> tuple[dict[str, int], dict[str, int]]:
    episodes: dict[str, int] = {}
    for document in fixture.documents:
        added = await engine.remember(document.space, document.content, kind="file", source=document.source,
                                      tags=document.tags, metadata=document.where, created_at=document.created_at)
        episodes[document.id] = added.episode_id
    facts: dict[str, int] = {}
    by_id = {document.id: document for document in fixture.documents}
    for fact in fixture.facts:
        source = by_id[fact.document_id]
        stored = await engine.assert_fact(source.space, fact.subject, fact.predicate, fact.object,
                                          source_episode_id=episodes[source.id], quote=fact.quote,
                                          valid_from=fact.valid_from)
        facts[fact.id] = stored.fact_id
    case = fixture.cases[0]
    other_metadata = dict(case.where)
    other_metadata[next(iter(other_metadata))] += "-benchmark-other"
    used_spaces = {document.space for document in fixture.documents}
    foreign_space = "benchmark-outsider"
    while foreign_space in used_spaces:
        foreign_space += "-x"
    query_terms = " and ".join(item.query for item in fixture.cases)
    for group, space, metadata in (("same-project", case.space, case.where),
                                    ("other-project", case.space, other_metadata),
                                    ("foreign-space", foreign_space, case.where)):
        for offset in range(0, size, 100):
            await engine.remember_many(space, [Record(
                f"{query_terms} planning memo {group} {number}. No operational decision is recorded.",
                kind="file", source=f"distractors/{group}/{number}.md", tags=("approved",),
                metadata=metadata, created_at="2026-01-01")
                for number in range(offset, min(size, offset + 100))])
    return episodes, facts


def _latency(values: list[float]) -> dict[str, float]:
    return {"latency_ms_median": round(statistics.median(values), 3), "latency_ms_max": round(max(values), 3)}


async def _case(engine: MemoryEngine, case: FixtureCase, episodes: dict[str, int], facts: dict[str, int],
                size: int, repeats: int) -> dict[str, object]:
    scope = TextFilter(where=case.where, source_prefix=case.source_prefix)
    baseline_ms: list[float] = []
    expanded_ms: list[float] = []
    baseline: dict[str, object] = {}
    enriched: dict[str, object] = {}
    work: dict[str, JsonValue] = {}
    caps: dict[str, JsonValue] = {}
    diagnostic: dict[str, object] | None = None
    baseline_trials: list[float] = []
    enriched_trials: list[float] = []
    recalled_seed_ids: list[int] = []
    for _ in range(repeats):
        started = time.perf_counter()
        result = await engine.recall(case.space, case.query, limit=1, where=case.where, source_prefix=case.source_prefix)
        recall_ms = (time.perf_counter() - started) * 1000
        baseline_ms.append(recall_ms)
        pieces = _recalled(result)
        baseline = await _coverage(engine, case, episodes, pieces)
        started = time.perf_counter()
        if case.mode == "structural":
            structural_limits = StructuralLimits()
            structural = await expand_structural_context(engine.documents, case.space, result, scope=scope, limits=structural_limits)
            added = [_Piece(section.episode_id, section.text) for section in structural.sections]
            work = {key: value for key, value in structural.model_dump(mode="json").items() if key != "sections"}
            work["output_bytes"] = len(structural.model_dump_json().encode())
            caps = structural_limits.model_dump(mode="json")
        else:
            multihop_limits = MultiHopLimits()
            multihop = await expand_multihop(engine.documents, case.space, seeds=result, scope=scope, limits=multihop_limits)
            added = [_Piece(fact.source_episode_id, fact.quote) for fact in multihop.facts
                     if fact.source_episode_id is not None and fact.quote is not None]
            work = multihop.counts.model_dump(mode="json")
            work["coverage"] = multihop.coverage.model_dump(mode="json")
            caps = multihop_limits.model_dump(mode="json")
            recalled_seed_ids = [fact.fact_id for fact in result.facts]
        expanded_ms.append(recall_ms + (time.perf_counter() - started) * 1000)
        enriched = await _coverage(engine, case, episodes, pieces + added)
        baseline_trials.append(cast(float, baseline["quote_coverage"]))
        enriched_trials.append(cast(float, enriched["quote_coverage"]))
    if case.mode == "multihop" and case.seed_fact_ids:
        explicit = await expand_multihop(engine.documents, case.space,
                                         seed_fact_ids=[facts[seed] for seed in case.seed_fact_ids], scope=scope)
        diagnostic = await _coverage(engine, case, episodes, [_Piece(fact.source_episode_id, fact.quote)
            for fact in explicit.facts if fact.source_episode_id is not None and fact.quote is not None])
        diagnostic["label"] = "predeclared explicit fact IDs; traversal diagnostic, excluded from primary coverage"
    baseline.update(_latency(baseline_ms))
    enriched.update(_latency(expanded_ms))
    return {"case_id": case.id, "mode": case.mode, "query": case.query, "distractors_per_group": size,
            "distractor_total": size * 3, "repeats": repeats, "baseline_limit": 1,
            "baseline": baseline, "enriched": enriched, "baseline_quote_coverage_trials": baseline_trials,
            "enriched_quote_coverage_trials": enriched_trials, "work": work, "caps": caps,
            "seed_mode": "actual_recall_facts" if case.mode == "multihop" else "actual_recall_chunks",
            "recalled_fact_ids": recalled_seed_ids, "explicit_seed_diagnostic": diagnostic}


async def run(fixture: Path, sizes: list[int], repeats: int, embedding_cache: Path | None = None) -> dict[str, object]:
    if not 1 <= repeats <= 100 or not sizes or any(type(size) is not int or not 0 <= size <= 100_000 for size in sizes):
        raise ValueError("repeats must be 1..100; sizes must be nonempty integers in 0..100000")
    corpus = load_fixture(fixture)
    if not corpus.cases[0].where or any((case.space, case.where) != (corpus.cases[0].space, corpus.cases[0].where)
                                       for case in corpus.cases):
        raise ValueError("distractor workload requires one shared nonempty metadata scope and space")
    embedder: Embedder
    if embedding_cache is None:
        embedder = HashEmbedder()
    else:
        embedder = _CachedBGE(embedding_cache)
    results: list[dict[str, object]] = []
    for size in sizes:
        with tempfile.TemporaryDirectory(prefix="scone-edge-fixture-") as directory:
            path = str(Path(directory) / "memory.sqlite")
            engine = await MemoryEngine(SqliteDocumentStore(path), SqliteVectorIndex(path), embedder, clock=lambda: STAMP).open()
            try:
                episodes, facts = await _seed(engine, corpus, size)
                for case in corpus.cases:
                    results.append(await _case(engine, case, episodes, facts, size, repeats))
            finally:
                await engine.close()
    return {"schema_version": 1, "fixture": str(fixture), "fixture_sha256": hashlib.sha256(fixture.read_bytes()).hexdigest(),
            "embedder": embedder.id, "backend": "isolated temporary SQLite", "network_required": False,
            "workload": "two predeclared source-and-quote evidence cases", "results": results,
            "limitations": ["Measures retained source and literal quote coverage, not answer accuracy or general semantic retrieval.",
                            "Hash embeddings measure lexical overlap; cached BGE is an optional separate run.",
                            "SQLite vector retrieval scans the selected space; no large-scale latency claim.",
                            "Median/max timings combine first and warm queries; enrichment latency includes baseline recall.",
                            "Text bytes deduplicate identical source/text pairs but retain overlapping spans; expansion output bytes include JSON.",
                            "Work counters cover expansion only, excluding ingestion, baseline recall and evaluation source checks.",
                            "Explicit fact seed diagnostics are reported separately and never improve primary coverage."]}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fixture", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--sizes", type=int, nargs="+", default=[0, 100, 1000])
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--embedding-cache", type=Path)
    args = parser.parse_args()
    report = asyncio.run(run(args.fixture, args.sizes, args.repeats, args.embedding_cache))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"output": str(args.output), "embedder": report["embedder"]}))


if __name__ == "__main__":
    main()
