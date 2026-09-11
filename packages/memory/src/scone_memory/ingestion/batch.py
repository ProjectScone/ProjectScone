"""Batch ingestion and crash recovery over explicit storage ports.

The public engine checks space access and reports remember events. This component
owns validation, chunking, deduplication, embedding, write ordering and recovery.
"""
from __future__ import annotations

from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass
import math
from typing import cast

from ..core.errors import InvalidInput
from ..core.models import Added, MAX_CONTENT_BYTES
from ..core.ports import DocumentStore, Embedder, Event, NewChunk, NewEpisode, VectorIndex, VectorPoint
from ..core.validation import KINDS, normalise_metadata, normalise_tags, normalise_time
from .chunker import byte_spans, chunk_spans
from .records import Record, RecoveryReport, _DupOf, _Pending, content_hash

EMBED_BATCH = 64


async def _embed_chunks(embedder: Embedder, texts: Sequence[str]) -> list[list[float]]:
    """Require complete, valid provider responses before accepting any chunk vectors."""
    vectors: list[list[float]] = []
    for offset in range(0, len(texts), EMBED_BATCH):
        batch = texts[offset:offset + EMBED_BATCH]
        response = await embedder.embed(batch)
        if not isinstance(response, list) or len(response) != len(batch):
            raise ValueError('embedding response must contain one vector per input text')
        for vector in response:
            if not isinstance(vector, list) or len(vector) != embedder.dim:
                raise ValueError('embedding vector does not match the configured dimension')
            try:
                valid = all(isinstance(value, (int, float)) and not isinstance(value, bool)
                            and math.isfinite(value) for value in vector)
            except OverflowError:
                valid = False
            if not valid:
                raise ValueError('embedding vector must contain finite numeric values')
            # Providers may reuse their response buffers on the next call.
            vectors.append(list(vector))
    return vectors


@dataclass(frozen=True)
class IngestionRuntime:
    """Storage references captured at dispatch; supplied callbacks remain live."""

    documents: DocumentStore
    vectors: VectorIndex
    embedder: Embedder
    clock: Callable[[], str]
    chunk_target: int
    embed_text: Callable[[NewEpisode, str], str]
    emit: Callable[[str, str, dict[str, object]], Awaitable[Event | None]]


async def remember_many(runtime: IngestionRuntime, space: str, records: Sequence[Record]) -> list[Added]:
    when = runtime.clock()
    results: list[Added | _DupOf | None] = []
    fresh: list[_Pending] = []
    seen: dict[str, int] = {}  # content hash -> index into results
    for record in records:
        content = record.content
        if not isinstance(content, str) or not content.strip():
            raise InvalidInput("content must not be empty")
        if len(content.encode()) > MAX_CONTENT_BYTES:
            raise InvalidInput(f"content exceeds {MAX_CONTENT_BYTES} bytes")
        kind = record.kind
        if kind not in KINDS:
            raise InvalidInput(f"kind must be one of {KINDS}, got {record.kind!r}")
        clean_tags = normalise_tags(record.tags)
        clean_meta = normalise_metadata(record.metadata or {})
        happened = normalise_time(record.created_at) if record.created_at else when
        digest = record.content_hash or content_hash(space, content, record.dedup_key)
        if not digest or len(digest) > 128:
            raise InvalidInput("content_hash must be 1..=128 chars")
        if digest in seen:
            results.append(Added(episode_id=-1, deduplicated=True, chunks=0))
            fresh_or_dup = seen[digest]
            results[-1] = _DupOf(fresh_or_dup)  # resolved after inserts
            continue
        existing = await runtime.documents.episode_by_hash(space, digest)
        if existing is not None:
            seen[digest] = len(results)
            results.append(Added(episode_id=existing.episode_id, deduplicated=True, chunks=0, outcome="duplicate"))
            continue
        seen[digest] = len(results)
        results.append(None)
        spans = chunk_spans(content, runtime.chunk_target)
        fresh.append(
            _Pending(
                slot=len(results) - 1,
                texts=[content[sp.start : sp.end] for sp in spans],
                new=NewEpisode(
                    space=space,
                    kind=kind,
                    content=content,
                    content_hash=digest,
                    created_at=happened,
                    ingested_at=when,
                    source=record.source,
                    tags=clean_tags,
                    metadata=clean_meta,
                ),
                spans=[(sp.start, sp.end) for sp in byte_spans(content, spans)],
            )
        )

    if fresh:
        texts = [runtime.embed_text(p.new, t) for p in fresh for t in p.texts]
        vectors = await _embed_chunks(runtime.embedder, texts)
        await write_batch(runtime, space, fresh, vectors, results)
        await runtime.documents.bump_revision(space)

    resolved: list[Added] = []
    for r in results:
        if isinstance(r, _DupOf):
            target = cast(Added, results[r.slot])
            resolved.append(Added(episode_id=target.episode_id, deduplicated=True, chunks=0, outcome="duplicate"))
        else:
            resolved.append(cast(Added, r))
    return resolved


async def write_batch(
    runtime: IngestionRuntime, space: str, fresh: list["_Pending"], vectors: list[list[float]], results: list[Added | _DupOf | None]
) -> None:
    written: list[int] = []
    try:
        offset = 0
        for pending in fresh:
            # The mark outlives a crash; clear_inflight below is the
            # last thing this episode's write does (see recover()).
            await runtime.documents.mark_inflight(space, pending.new.content_hash)
            episode = await runtime.documents.insert_episode(pending.new)
            written.append(episode.episode_id)
            chunks = await runtime.documents.insert_chunks(
                [
                    NewChunk(
                        episode_id=episode.episode_id,
                        space=space,
                        ordinal=i,
                        start=a,
                        end=b,
                        text=text,
                        created_at=episode.created_at,
                    )
                    for i, ((a, b), text) in enumerate(zip(pending.spans, pending.texts))
                ]
            )
            await runtime.vectors.upsert(
                [
                    VectorPoint(
                        chunk_id=c.chunk_id,
                        space=space,
                        episode_id=episode.episode_id,
                        created_at=episode.created_at,
                        vector=v,
                        tags=pending.new.tags,
                        metadata=pending.new.metadata,
                    )
                    for c, v in zip(chunks, vectors[offset : offset + len(chunks)])
                ]
            )
            offset += len(chunks)
            await runtime.documents.clear_inflight(space, pending.new.content_hash)
            results[pending.slot] = Added(episode_id=episode.episode_id, deduplicated=False, chunks=len(chunks))
    except Exception:
        # Undo the partial batch so a retry starts clean.
        for episode_id in written:
            removed = await runtime.documents.delete_episode(space, episode_id)
            await runtime.vectors.delete(removed)
        for pending in fresh:
            await runtime.documents.clear_inflight(space, pending.new.content_hash)
        raise


async def recover(runtime: IngestionRuntime) -> RecoveryReport:
    """Finish or forget what a crash interrupted. A remember marks its
    episode identity in the document store before writing and clears
    the mark once the rows and the vectors are all durable. Anything
    still marked here was cut off somewhere in between: an episode row
    without chunks, or chunks without vectors, or no row at all. Each
    is brought to the complete state (chunks and vectors rebuilt from
    the stored content, which never changed) or, if nothing landed,
    the mark is dropped. Recorded as one "recover" event per open when
    there was anything to do, so the evidence shows it happened."""
    report = RecoveryReport()
    marks = await runtime.documents.inflight()
    for space, digest in marks:
        episode = await runtime.documents.episode_by_hash(space, digest)
        if episode is None:
            report.forgotten += 1
        else:
            chunks = await runtime.documents.chunks_of(space, episode.episode_id)
            if not chunks:
                spans = chunk_spans(episode.content, runtime.chunk_target)
                chunks = await runtime.documents.insert_chunks(
                    [
                        NewChunk(
                            episode_id=episode.episode_id,
                            space=space,
                            ordinal=i,
                            start=bs.start,
                            end=bs.end,
                            text=episode.content[sp.start : sp.end],
                            created_at=episode.created_at,
                        )
                        for i, (sp, bs) in enumerate(zip(spans, byte_spans(episode.content, spans)))
                    ]
                )
                report.rechunked += 1
            as_new = NewEpisode(
                space=space, kind=episode.kind, content=episode.content, content_hash=episode.content_hash,
                created_at=episode.created_at, ingested_at=episode.ingested_at, source=episode.source,
                tags=episode.tags, metadata=episode.metadata,
            )
            texts = [runtime.embed_text(as_new, c.text) for c in chunks]
            vectors = await _embed_chunks(runtime.embedder, texts)
            await runtime.vectors.upsert(
                [
                    VectorPoint(
                        chunk_id=c.chunk_id, space=space, episode_id=episode.episode_id, created_at=episode.created_at,
                        vector=v, tags=episode.tags, metadata=episode.metadata,
                    )
                    for c, v in zip(chunks, vectors)
                ]
            )
            report.completed += 1
        await runtime.documents.clear_inflight(space, digest)
    if marks:
        for space in sorted({s for s, _ in marks}):
            await runtime.emit(space, "recover", {
                "marks": sum(1 for s, _ in marks if s == space),
                "completed": report.completed, "rechunked": report.rechunked, "forgotten": report.forgotten,
            })
    return report
