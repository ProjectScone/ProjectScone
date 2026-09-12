"""Batch ingestion and crash recovery over explicit storage ports.

The public engine checks space access and reports remember events. This component
owns validation, chunking, deduplication, embedding, write ordering and recovery.
"""
from __future__ import annotations

from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass
from typing import cast

from ..core.errors import InvalidInput, SconeError
from ..core.models import Added, MAX_CONTENT_BYTES
from ..core.ports import DocumentStore, Embedder, Event, NewChunk, NewEpisode, VectorIndex, VectorPoint
from ..core.validation import KINDS, normalise_metadata, normalise_tags, normalise_time
from .chunker import Span, byte_spans, chunk_spans
from .code import code_language, code_spans
from .records import Record, RecoveryReport, _DupOf, _Pending, content_hash

EMBED_BATCH = 64


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
    #: Whether a source stored under a name that says it is code is cut at
    #: its declarations rather than every chunk_target characters.
    code_aware: bool = True


async def _each(runtime: IngestionRuntime, space: str, records: Sequence[Record]) -> list[Added]:
    """A batch where every record stands or falls on its own.

    The records that pass are stored together, so they still deduplicate
    against each other the way a batch does; the ones that do not are
    answered where they were asked, in order, so a caller can line the
    answers up against what they sent."""
    kept: list[Record] = []
    where: list[int] = []
    answers: list[Added | None] = []
    for record in records:
        try:
            _check(space, record)
        except SconeError as wrong:
            answers.append(Added(episode_id=-1, outcome="failed", reason=str(wrong)))
            continue
        where.append(len(answers))
        answers.append(None)
        kept.append(record)
    if kept:
        for slot, outcome in zip(where, await remember_many(runtime, space, kept)):
            answers[slot] = outcome
    return [answer if answer is not None else Added(episode_id=-1, outcome="failed",
                                                    reason="nothing was stored for it")
            for answer in answers]


def _check(space: str, record: Record) -> None:
    """Everything a record must be before anything is written for it. The
    batch checks these as it goes; a partial batch asks first, so that a
    record that cannot be stored is answered rather than raised."""
    content = record.content
    if not isinstance(content, str) or not content.strip():
        raise InvalidInput("content must not be empty")
    if len(content.encode()) > MAX_CONTENT_BYTES:
        raise InvalidInput(f"content exceeds {MAX_CONTENT_BYTES} bytes")
    if record.kind not in KINDS:
        raise InvalidInput(f"kind must be one of {KINDS}, got {record.kind!r}")
    normalise_tags(record.tags)
    normalise_metadata(record.metadata or {})
    if record.created_at:
        normalise_time(record.created_at)
    digest = record.content_hash or content_hash(space, content, record.dedup_key)
    if not digest or len(digest) > 128:
        raise InvalidInput("content_hash must be 1..=128 chars")


def spans_for(runtime: IngestionRuntime, content: str, source: str | None) -> list[Span]:
    """Where to cut: at declarations when the source is code and the name
    it was stored under says which language, and by length otherwise."""
    language = code_language(source) if runtime.code_aware else None
    if language is None:
        return chunk_spans(content, runtime.chunk_target)
    return list(code_spans(content, runtime.chunk_target, language=language))


async def remember_many(runtime: IngestionRuntime, space: str, records: Sequence[Record], *,
                        partial: bool = False) -> list[Added]:
    """Store a batch. One bad record refuses the whole batch, because a
    caller who sent one usually wants to fix it and send the lot again,
    and a half-stored batch nobody asked for is worse than a refusal.

    ``partial`` is for the caller who is importing from somewhere messy
    and wants what can be stored: each record is judged on its own, a bad
    one comes back as failed with the reason, and the rest are stored."""
    if partial:
        return await _each(runtime, space, records)
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
        spans = spans_for(runtime, content, record.source)
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
        vectors: list[list[float]] = []
        for i in range(0, len(texts), EMBED_BATCH):
            vectors.extend(await runtime.embedder.embed(texts[i : i + EMBED_BATCH]))
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
                spans = spans_for(runtime, episode.content, episode.source)
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
            vectors: list[list[float]] = []
            for i in range(0, len(texts), EMBED_BATCH):
                vectors.extend(await runtime.embedder.embed(texts[i : i + EMBED_BATCH]))
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
