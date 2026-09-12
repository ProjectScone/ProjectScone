"""Batch ingestion and crash recovery over explicit storage ports.

The public engine checks space access and reports remember events. This component
owns validation, chunking, deduplication, embedding, write ordering and recovery.
"""
from __future__ import annotations

from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass
import hashlib
import json
import math
from typing import cast

from ..core.errors import InvalidInput, SconeError
from ..core.models import Added, MAX_CONTENT_BYTES
from ..core.ports import DocumentStore, Embedder, EmbeddingCheckpoint, Event, NewChunk, NewEpisode, VectorIndex, VectorPoint
from ..core.validation import KINDS, normalise_metadata, normalise_tags, normalise_time
from .chunker import Span, byte_spans, chunk_spans
from .structure_chunks import structured_spans
from .code import code_language, code_spans
from .records import Record, RecoveryReport, _DupOf, _Pending, content_hash

EMBED_BATCH = 64


def _validated_vectors(response: object, count: int, dimension: int) -> list[list[float]]:
    """Vectors from one batch, checked. ``dimension`` is the width already
    settled for this operation, or 0 when the embedder declares none and
    nothing has settled it yet."""
    if not isinstance(response, list) or len(response) != count:
        raise ValueError('embedding response must contain one vector per input text')
    # Not every embedder declares a width -- a remote model behind an
    # endpoint that does not advertise one reports 0, and the abstention
    # floor reads the width off the vectors themselves for exactly that
    # case. Checking against 0 would refuse every vector such an embedder
    # ever returned. With no declared width the width of the first vector
    # settles it, and the caller carries that forward across every batch of
    # one operation: settling it per batch would let 64 vectors be eight
    # wide and the next sixteen, and an index built from those is silently
    # incoherent.
    expected = dimension if dimension else None
    vectors: list[list[float]] = []
    for vector in response:
        if not isinstance(vector, list) or not vector:
            raise ValueError('embedding vector does not match the configured dimension')
        if expected is None:
            expected = len(vector)
        if len(vector) != expected:
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


async def _embed_chunks(embedder: Embedder, texts: Sequence[str], *,
                        checkpoint: EmbeddingCheckpoint | None = None,
                        binding: str = '') -> list[list[float]]:
    """Persist only complete validated batches, and validate receipts on reuse."""
    identifier, dimension = embedder.id, embedder.dim
    prefix = ''
    if checkpoint is not None:
        digest = hashlib.sha256(json.dumps(['embedding-batch-v1', identifier, dimension, EMBED_BATCH, binding]).encode())
        for text in texts:
            encoded = text.encode('utf-8')
            digest.update(len(encoded).to_bytes(8, 'big'))
            digest.update(encoded)
        prefix = digest.hexdigest()
    vectors: list[list[float]] = []
    settled = dimension
    for offset in range(0, len(texts), EMBED_BATCH):
        batch = texts[offset:offset + EMBED_BATCH]
        key = f'{prefix}:{offset}'
        saved = checkpoint.get(key) if checkpoint is not None else None
        if saved is None:
            response: object = await embedder.embed(batch)
        else:
            try:
                response = json.loads(saved)
            except (ValueError, UnicodeError, RecursionError):
                raise ValueError('embedding checkpoint is invalid') from None
        if embedder.id != identifier or embedder.dim != dimension:
            raise ValueError('embedding identity changed during indexing')
        validated = _validated_vectors(response, len(batch), settled)
        # An undeclared width is settled by the first vector of the
        # operation and holds for every batch after it, checkpointed ones
        # included -- one operation produces one width or it fails.
        if not settled and validated:
            settled = len(validated[0])
        if saved is None and checkpoint is not None:
            checkpoint.put(key, json.dumps(validated, allow_nan=False, separators=(',', ':')).encode())
        vectors.extend(validated)
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
    embedding_checkpoint: EmbeddingCheckpoint | None = None
    #: Whether a source stored under a name that says it is code is cut at
    #: its declarations rather than every chunk_target characters.
    code_aware: bool = True
    #: Whether prose is cut at the structure it carries -- headings and
    #: numbered clauses -- rather than at the target alone. Off by
    #: default: cut positions decide what chunks exist, and stored
    #: offsets are part of the shared specification, so this is a
    #: caller's choice and not ours to make for an existing space.
    structure_aware: bool = False


def validated_record(space: str, record: Record, when: str) -> NewEpisode:
    """Validate and snapshot caller input before any source can be removed."""
    content = record.content
    if not isinstance(content, str) or not content.strip():
        raise InvalidInput("content must not be empty")
    if len(content.encode()) > MAX_CONTENT_BYTES:
        raise InvalidInput(f"content exceeds {MAX_CONTENT_BYTES} bytes")
    kind = record.kind
    if kind not in KINDS:
        raise InvalidInput(f"kind must be one of {KINDS}, got {record.kind!r}")
    if record.source is not None and not isinstance(record.source, str):
        raise InvalidInput("source must be a string or None")
    clean_tags = normalise_tags(record.tags)
    clean_meta = normalise_metadata(record.metadata or {})
    try:
        for value in (record.source or '', *clean_tags, *clean_meta.values()):
            value.encode('utf-8')
    except UnicodeError as error:
        raise InvalidInput("record source, tags and metadata must be valid UTF-8") from error
    happened = normalise_time(record.created_at) if record.created_at else when
    digest = record.content_hash or content_hash(space, content, record.dedup_key)
    if not digest or len(digest) > 128:
        raise InvalidInput("content_hash must be 1..=128 chars")
    return NewEpisode(space=space, kind=kind, content=content, content_hash=digest,
                      created_at=happened, ingested_at=when, source=record.source,
                      tags=clean_tags, metadata=clean_meta)


def chunk_record(runtime: IngestionRuntime, new: NewEpisode, *, slot: int = 0) -> _Pending:
    spans = spans_for(runtime, new.content, new.source)
    return _Pending(slot=slot, new=new, texts=[new.content[sp.start:sp.end] for sp in spans],
                    spans=[(sp.start, sp.end) for sp in byte_spans(new.content, spans)])


async def embed_pending(runtime: IngestionRuntime, space: str, fresh: Sequence[_Pending]) -> list[list[float]]:
    texts = [runtime.embed_text(p.new, text) for p in fresh for text in p.texts]
    binding = '' if runtime.embedding_checkpoint is None else json.dumps(
        [space, [(p.new.content_hash, p.spans) for p in fresh]], separators=(',', ':'))
    return await _embed_chunks(runtime.embedder, texts, checkpoint=runtime.embedding_checkpoint, binding=binding)


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
            validated_record(space, record, runtime.clock())
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



def spans_for(runtime: IngestionRuntime, content: str, source: str | None) -> list[Span]:
    """Where to cut: at declarations when the source is code and the name
    it was stored under says which language, at the document's own
    headings and clauses when asked, and by length otherwise."""
    language = code_language(source) if runtime.code_aware else None
    if language is not None:
        return list(code_spans(content, runtime.chunk_target, language=language))
    if runtime.structure_aware:
        # Prose only. Code has a better boundary than a heading, and the
        # branch above already took it.
        return list(structured_spans(content, runtime.chunk_target).spans)
    return chunk_spans(content, runtime.chunk_target)


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
        new = validated_record(space, record, when)
        digest = new.content_hash
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
        fresh.append(chunk_record(runtime, new, slot=len(results) - 1))

    if fresh:
        vectors = await embed_pending(runtime, space, fresh)
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
