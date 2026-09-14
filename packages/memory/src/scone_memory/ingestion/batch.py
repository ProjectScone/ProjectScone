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
from typing import Sequence, cast

from ..core.errors import InvalidInput, SconeError
from ..core.models import Added, MAX_CONTENT_BYTES
from ..core.ports import DocumentStore, Embedder, EmbeddingCheckpoint, Event, NewChunk, NewEpisode, VectorIndex, VectorPoint
from ..core.validation import KINDS, normalise_metadata, normalise_tags, normalise_time
from .chunker import Span, byte_spans, chunk_spans
from .semantic_chunks import semantic_spans
from .structure_chunks import structured_spans
from .embedding_cache import EmbeddingCache, cache_key
from .code import code_context, code_language, code_spans
from .structure import Heading, parse_structure
from .records import Chunking, Record, RetainedVideoRecord, RecoveryReport, _DupOf, _Pending, content_hash

from .vectors import validated_vectors

#: Kept as a private name here because the tests and the abstention floor
#: reach for it at this address; the implementation moved to `vectors`.
_validated_vectors = validated_vectors

EMBED_BATCH = 64




async def _embed_chunks(embedder: Embedder, texts: Sequence[str], *,
                        checkpoint: EmbeddingCheckpoint | None = None,
                        binding: str = '', cache: EmbeddingCache | None = None,
                        reused: list[bool] | None = None) -> list[list[float]]:
    """Persist only complete validated batches, and validate receipts on reuse.

    With a ``cache``, a text the cache holds a vector for is not sent to
    the embedder, and ``reused`` (when given) receives one flag per text
    saying whether its vector came from the cache. An embedder of
    undeclared width bypasses the cache: a vector cannot be checked
    against a width nobody has stated."""
    identifier, dimension = embedder.id, embedder.dim
    if cache is not None and dimension:
        keys = [cache_key(identifier, dimension, text) for text in texts]
        # A cache is not evidence and may not refuse a write: one that
        # fails is a miss, told so, and the embedder answers instead.
        try:
            found = cache.take(keys, dimension)
        except Exception as error:  # noqa: BLE001 - whatever the store raised, the record is still stored
            cache.failed("taking", error)
            found = {}
        if reused is not None:
            reused.extend(key in found for key in keys)
        # Each missing text once, however many chunks share it.
        wanted = list(dict.fromkeys(key for key in keys if key not in found))
        first_text: dict[str, str] = {}
        for key, text in zip(keys, texts):
            first_text.setdefault(key, text)
        fresh = (await _embed_chunks(embedder, [first_text[key] for key in wanted], checkpoint=checkpoint, binding=binding)
                 if wanted else [])
        # Kept once the whole batch is validated: a partial batch fails
        # before this line and leaves nothing behind.
        answered = dict(zip(wanted, fresh))
        if answered:
            try:
                cache.keep(answered, dimension)
            except Exception as error:  # noqa: BLE001
                cache.failed("keeping", error)
        return [list(answered[key]) if key in answered else found[key] for key in keys]
    if reused is not None:
        reused.extend(False for _ in texts)
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
    #: Vectors kept by the text they embed, so a chunk unchanged since it
    #: was last stored is not embedded again. None embeds everything.
    embedding_cache: EmbeddingCache | None = None
    #: Whether a source stored under a name that says it is code is cut at
    #: its declarations rather than every chunk_target characters.
    code_aware: bool = True
    #: Whether prose is cut at the structure it carries -- headings and
    #: numbered clauses -- rather than at the target alone. Off by
    #: default: cut positions decide what chunks exist, and stored
    #: offsets are part of the shared specification, so this is a
    #: caller's choice and not ours to make for an existing space.
    structure_aware: bool = False
    #: Whether prose is cut where its subject changes rather than where
    #: the target lands. Off by default for the same reason as
    #: `structure_aware`, and costs one extra embedding call per episode
    #: -- its sentences are embedded to find the boundaries.
    semantic_aware: bool = False
    #: Whether each chunk is embedded with the headings it sits under (for
    #: code, its file and declarations). Changes vectors, never stored text.
    heading_context: bool = False
    context_inputs: Callable[[NewEpisode, Sequence[tuple[int, int]]], Awaitable[list[str]]] | None = None
    #: Whether each chunk's context -- what it is under and does not say --
    #: is indexed beside its text for the context lane. Stored text is
    #: never changed by it; a store without the index is left alone.
    context_lane: bool = False
    # With an episode id, verification also repairs its original/manifest links.
    verify_visual: Callable[[str, Record, int | None], Awaitable[None]] | None = None
    #: The headings an imported file marked itself, read from its retained
    #: manifest; None for an episode that is not an imported file.
    document_headings: Callable[[NewEpisode], Awaitable[tuple[Heading, ...] | None]] | None = None


def validated_record(space: str, record: Record, when: str, *, verified_visual: bool = False) -> NewEpisode:
    """Validate and snapshot caller input before any source can be removed."""
    content = record.content
    if not isinstance(content, str) or (not content.strip() and not (verified_visual and content == "")):
        raise InvalidInput("content must not be empty")
    if len(content.encode()) > MAX_CONTENT_BYTES:
        raise InvalidInput(f"content exceeds {MAX_CONTENT_BYTES} bytes")
    kind = record.kind
    if kind not in KINDS:
        raise InvalidInput(f"kind must be one of {KINDS}, got {record.kind!r}")
    if record.source is not None and not isinstance(record.source, str):
        raise InvalidInput("source must be a string or None")
    clean_tags = normalise_tags(record.tags)
    clean_meta = dict(normalise_metadata(record.metadata or {}))
    asked, held = record.chunking, clean_meta.get("chunking")
    if asked is not None and held is not None and asked != held:
        raise InvalidInput(f"metadata says chunking={held!r} but the record asks for {asked!r}")
    mode = asked if asked is not None else held
    if mode is not None:
        # Whether the mode exists and fits the source is `cut_for`'s to say,
        # and it says so before anything is stored.
        clean_meta["chunking"] = mode
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


async def _validated_record(runtime: IngestionRuntime, space: str, record: Record, when: str) -> NewEpisode:
    visual = isinstance(record, RetainedVideoRecord) and record.content == ''
    if visual:
        if runtime.verify_visual is None:
            raise InvalidInput('visual-only documents require retained attachment verification')
        await runtime.verify_visual(space, record, None)
    return validated_record(space, record, when, verified_visual=visual)


#: Bytes of heading path put in front of one chunk's embedding input. A
#: path longer than this keeps its innermost headings, and the receipt
#: counts the chunks it was cut for.
MAX_HEADING_CONTEXT_BYTES = 256
#: Names the rule that builds a heading path into an embedding input. It is
#: part of the vector writer's identity, so change it whenever that rule, or
#: the bound above, changes what a chunk is embedded with.
HEADING_CONTEXT_VERSION = "heading-path-v2"


async def headings_of(runtime: IngestionRuntime, episode: NewEpisode | None) -> tuple[Heading, ...] | None:
    """The headings a file episode's own manifest marks, or None when there is none to read."""
    if episode is None or runtime.document_headings is None:
        return None
    return await runtime.document_headings(episode)


def context_lines(content: str, source: str | None, spans: Sequence[tuple[int, int]],
                  target: int, headings: Sequence[Heading] | None = None) -> tuple[list[str], int]:
    """The line put in front of each chunk before it is embedded, and how
    many were cut to the bound.

    For code whose name says its language, the file and the declarations
    the chunk sits in (``code_context``). For prose, the titles of the
    headings above the chunk's first byte, outermost first: those its text
    carries, and those an imported file marked itself (``headings``), each
    of which runs to the next of the same or a higher level. An empty line
    means nothing to add. Nothing here reads or changes stored text."""
    language = code_language(source)
    if language is not None:
        found = list(code_spans(content, target, language=language))
        declared = {(cut_at.start, cut_at.end): span.names
                    for cut_at, span in zip(byte_spans(content, cast(list[Span], found)), found)}
        return [code_context(source, declared.get(span, ())) for span in spans], 0
    try:
        sections = [(section.start, section.end, section.level, section.title)
                    for section in parse_structure(content).sections if section.level > 0 and section.title]
    except ValueError:
        sections = []
    marks = sorted(headings or (), key=lambda heading: heading.start)
    size = len(content.encode())
    for index, heading in enumerate(marks):
        end = next((later.start for later in marks[index + 1:] if later.level <= heading.level), size)
        sections.append((heading.start, end, heading.level, heading.title))
    lines: list[str] = []
    cut = 0
    for start, _ in spans:
        titles = [title for _, _, _, title in sorted(
            (section for section in sections if section[0] <= start < section[1]), key=lambda section: section[2])]
        line = " > ".join(titles)
        if len(line.encode()) > MAX_HEADING_CONTEXT_BYTES:
            cut += 1
            while len(titles) > 1 and len(" > ".join(titles).encode()) > MAX_HEADING_CONTEXT_BYTES:
                titles.pop(0)
            line = " > ".join(titles).encode()[:MAX_HEADING_CONTEXT_BYTES].decode("utf-8", errors="ignore")
        lines.append(line)
    return lines, cut


async def chunk_record(runtime: IngestionRuntime, new: NewEpisode, *, slot: int = 0) -> _Pending:
    cut = await cut_for(runtime, new.content, new.source, new.metadata.get("chunking"), episode=new)
    byte_offsets = [(sp.start, sp.end) for sp in byte_spans(new.content, cut.spans)]
    receipt = None
    if runtime.heading_context:
        lines, lines_cut = context_lines(new.content, new.source, byte_offsets, runtime.chunk_target,
                                         await headings_of(runtime, new))
        receipt = {"mode": "headings", "chunks_with_context": sum(1 for line in lines if line),
                   "context_bytes": sum(len(line.encode()) for line in lines), "context_cut": lines_cut}
    return _Pending(slot=slot, new=new, texts=[new.content[sp.start:sp.end] for sp in cut.spans],
                    spans=byte_offsets, chunking=cut.chunking, structure=cut.structure, embedding_context=receipt)


async def embedding_inputs(runtime: IngestionRuntime, new: NewEpisode, spans: Sequence[tuple[int, int]],
                           texts: Sequence[str]) -> list[str]:
    if not texts:
        return []
    if runtime.context_inputs is not None:
        content = new.content.encode()
        previous = 0
        if len(spans) != len(texts):
            raise InvalidInput('document embedding chunk does not match its source span')
        for (start, end), text in zip(spans, texts):
            if (type(start) is not int or type(end) is not int
                    or not previous <= start < end <= len(content)
                    or content[start:end] != text.encode()):
                raise InvalidInput('document embedding chunk does not match its source span')
            previous = end
    contextual = await runtime.context_inputs(new, spans) if runtime.context_inputs is not None else texts
    if len(contextual) != len(texts):
        raise InvalidInput('embedding context must preserve the chunk count')
    if runtime.heading_context:
        # One place for both ingestion and recovery, so a recovered chunk is
        # embedded exactly as an uninterrupted one would have been.
        lines, _ = context_lines(new.content, new.source, spans, runtime.chunk_target, await headings_of(runtime, new))
        contextual = [f"{line}\n{text}" if line else text for line, text in zip(lines, contextual)]
    return [runtime.embed_text(new, text) for text in contextual]


async def embed_pending_counted(runtime: IngestionRuntime, space: str,
                                fresh: Sequence[_Pending]) -> tuple[list[list[float]], list[int]]:
    """Every pending record's vectors in order, and for each record how
    many of them the embedding cache answered rather than the embedder."""
    texts: list[str] = []
    counts: list[int] = []
    for pending in fresh:
        inputs = await embedding_inputs(runtime, pending.new, pending.spans, pending.texts)
        counts.append(len(inputs))
        texts.extend(inputs)
    binding = '' if runtime.embedding_checkpoint is None else json.dumps(
        [space, [(p.new.content_hash, p.spans) for p in fresh]], separators=(',', ':'))
    flags: list[bool] = []
    vectors = await _embed_chunks(runtime.embedder, texts, checkpoint=runtime.embedding_checkpoint, binding=binding,
                                  cache=runtime.embedding_cache, reused=flags)
    reused: list[int] = []
    at = 0
    for count in counts:
        reused.append(sum(flags[at:at + count]))
        at += count
    return vectors, reused


async def embed_pending(runtime: IngestionRuntime, space: str, fresh: Sequence[_Pending]) -> list[list[float]]:
    vectors, _ = await embed_pending_counted(runtime, space, fresh)
    return vectors


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
            await _validated_record(runtime, space, record, runtime.clock())
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



#: The ways one record can ask to be cut.
CHUNKINGS: tuple[str, ...] = ("length", "code", "structure", "semantic")


@dataclass(frozen=True)
class Cut:
    """Where a record was cut, which way, and the structure chunker's own
    counts when that was the way."""

    spans: list[Span]
    chunking: Chunking
    structure: dict[str, object] | None = None


async def cut_for(runtime: IngestionRuntime, content: str, source: str | None,
                  chunking: str | None = None, *, episode: NewEpisode | None = None) -> Cut:
    """Where to cut, and which way it was: at declarations when the source
    is code and the name it was stored under says which language, at the
    document's own headings and clauses when asked, where its subject
    changes when asked, and by length otherwise. An explicit ``chunking``
    is the record's own choice and wins over the engine's rule; None is
    the rule as it was.

    The one dispatch point, and awaitable even though only one branch
    needs it. A synchronous twin would let a caller reach chunking
    without an embedder and receive length-cut chunks believing they
    were something else.

    Given the ``episode``, a structure cut also cuts at the headings an
    imported file marked itself, which its text alone does not show."""
    if chunking is not None and chunking not in CHUNKINGS:
        raise InvalidInput(f"chunking must be one of {', '.join(CHUNKINGS)}, not {chunking!r}")
    language = code_language(source) if (runtime.code_aware or chunking == "code") else None
    if chunking == "code" and language is None:
        raise InvalidInput("chunking=code needs a source whose name says its language")
    if language is not None and chunking in (None, "code"):
        return Cut(list(code_spans(content, runtime.chunk_target, language=language)), "code")
    mode = cast(Chunking, chunking or ("structure" if runtime.structure_aware else "semantic" if runtime.semantic_aware else "length"))
    if mode == "structure":
        # Prose only. Code has a better boundary than a heading, and the
        # branch above already took it unless the record said otherwise.
        found = structured_spans(content, runtime.chunk_target, headings=await headings_of(runtime, episode))
        receipt = {key: value for key, value in found.record().items() if key not in ("spans", "chunks")}
        return Cut(list(found.spans), "structure", receipt)
    if mode == "semantic":
        return Cut(list(await semantic_spans(content, runtime.embedder, runtime.chunk_target)), "semantic")
    return Cut(chunk_spans(content, runtime.chunk_target), "length")


async def spans_for(runtime: IngestionRuntime, content: str, source: str | None,
                    chunking: str | None = None, *, episode: NewEpisode | None = None) -> list[Span]:
    """The spans of ``cut_for``, for callers that need only where."""
    return (await cut_for(runtime, content, source, chunking, episode=episode)).spans


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
    if any(isinstance(record, RetainedVideoRecord) for record in records):
        if len(records) != 1:
            raise InvalidInput('visual-only retention requires a single document operation')
        return [await _retain_visual(runtime, space, records[0])]
    when = runtime.clock()
    results: list[Added | _DupOf | None] = []
    fresh: list[_Pending] = []
    seen: dict[str, int] = {}  # content hash -> index into results
    for record in records:
        new = await _validated_record(runtime, space, record, when)
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
        fresh.append(await chunk_record(runtime, new, slot=len(results) - 1))

    if fresh:
        vectors, reused = await embed_pending_counted(runtime, space, fresh)
        await write_batch(runtime, space, fresh, vectors, results, reused=reused)
        await runtime.documents.bump_revision(space)

    resolved: list[Added] = []
    for r in results:
        if isinstance(r, _DupOf):
            target = cast(Added, results[r.slot])
            resolved.append(Added(episode_id=target.episode_id, deduplicated=True, chunks=0, outcome="duplicate"))
        else:
            resolved.append(cast(Added, r))
    return resolved


async def _retain_visual(runtime: IngestionRuntime, space: str, record: Record) -> Added:
    """An interrupted link operation leaves its source marked for verified repair.

    Unlike a text batch, this operation has no embeddings to prepare or roll back.
    Deleting its episode on a link failure would strand already-linked evidence.
    """
    new = await _validated_record(runtime, space, record, runtime.clock())
    if new.content != '' or runtime.verify_visual is None:
        raise InvalidInput('visual-only retention requires empty content and verified evidence')
    episode = await runtime.documents.episode_by_hash(space, new.content_hash)
    duplicate = episode is not None
    await runtime.documents.mark_inflight(space, new.content_hash)
    if episode is None:
        episode = await runtime.documents.insert_episode(new)
    await runtime.verify_visual(space, record, episode.episode_id)
    await runtime.documents.bump_revision(space)
    await runtime.documents.clear_inflight(space, new.content_hash)
    return Added(episode_id=episode.episode_id, deduplicated=duplicate, chunks=0,
                 outcome='duplicate' if duplicate else 'accepted')


async def write_batch(
    runtime: IngestionRuntime, space: str, fresh: list["_Pending"], vectors: list[list[float]], results: list[Added | _DupOf | None],
    reused: Sequence[int] | None = None,
) -> None:
    """``reused`` says, per pending record, how many of its vectors came
    from the embedding cache; it is written into the record's receipt."""
    written: list[int] = []
    try:
        offset = 0
        for index, pending in enumerate(fresh):
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
            if chunks and runtime.context_lane:
                from .context_terms import index_episode_context

                await index_episode_context(runtime.documents, space, pending.new, chunks)
            if chunks:
                await runtime.vectors.upsert([
                    VectorPoint(
                        chunk_id=c.chunk_id, space=space, episode_id=episode.episode_id,
                        created_at=episode.created_at, vector=v, tags=pending.new.tags,
                        metadata=pending.new.metadata,
                    )
                    for c, v in zip(chunks, vectors[offset:offset + len(chunks)])
                ])
            offset += len(chunks)
            await runtime.documents.clear_inflight(space, pending.new.content_hash)
            results[pending.slot] = Added(episode_id=episode.episode_id, deduplicated=False, chunks=len(chunks),
                                          embeddings_reused=reused[index] if reused is not None else 0,
                                          chunking=pending.chunking, structure=pending.structure,
                                          embedding_context=pending.embedding_context)
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
            if episode.content == '':
                if runtime.verify_visual is None or chunks:
                    raise InvalidInput('visual-only recovery requires verified evidence and zero chunks')
                await runtime.verify_visual(space, RetainedVideoRecord(content='', kind=episode.kind,
                    source=episode.source, metadata=episode.metadata, content_hash=episode.content_hash), episode.episode_id)
                await runtime.documents.clear_inflight(space, digest)
                report.completed += 1
                continue
            as_new = NewEpisode(
                space=space, kind=episode.kind, content=episode.content, content_hash=episode.content_hash,
                created_at=episode.created_at, ingested_at=episode.ingested_at, source=episode.source,
                tags=episode.tags, metadata=episode.metadata,
            )
            if not chunks:
                spans = await spans_for(runtime, episode.content, episode.source, episode.metadata.get("chunking"),
                                        episode=as_new)
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
            texts = await embedding_inputs(runtime, as_new, [(c.start, c.end) for c in chunks],
                                           [c.text for c in chunks])
            vectors = await _embed_chunks(runtime.embedder, texts, cache=runtime.embedding_cache)
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
