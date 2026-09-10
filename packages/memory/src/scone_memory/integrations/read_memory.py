"""Read an authorized passage's neighboring stored chunks without re-chunking."""
from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

from ..core.chunk_window import ChunkWindowLookup
from ..core.models import Chunk, Episode
from ..core.ports import TextFilter
from ..core.timeutil import parse_rfc3339
from ..memory.engine import MemoryEngine
from ..retrieval.multihop import _source_matches
from ..retrieval.recall_scope import RecallScope


class ReadMemoryArgs(BaseModel):
    model_config = ConfigDict(extra='forbid', strict=True)
    chunk_id: int = Field(gt=0, lt=2**63)
    before: int = Field(default=1, ge=0, le=4)
    after: int = Field(default=1, ge=0, le=4)


class ReadMemoryError(ValueError):
    def __init__(self, reason: str = 'evidence_unavailable') -> None:
        self.reason = reason if reason in ('evidence_unavailable', 'unsupported_chunk_window') else 'evidence_unavailable'
        super().__init__('memory read unavailable')


async def read_memory(memory: MemoryEngine, space: str, args: ReadMemoryArgs,
                      scope: RecallScope, excluded_session: str | None) -> dict[str, object]:
    """At most ten chunk rows and one episode, plus retained-source checks.

    The wrapper owns the invocation deadline and output byte limit. Source point
    reads can load a whole episode. Returned text remains exact stored chunks;
    final-answer source validation is performed by PreparedToolEvidence.
    """
    args = ReadMemoryArgs.model_validate(args.model_dump())
    documents = memory.documents
    if not isinstance(documents, ChunkWindowLookup):
        raise ReadMemoryError('unsupported_chunk_window')
    revision = await documents.revision(space)
    seeds = await documents.get_chunks(space, [args.chunk_id])
    if len(seeds) != 1:
        raise ReadMemoryError()
    seed = Chunk.model_validate(seeds[0].model_dump(), strict=True).model_copy(deep=True)
    if (seed.space != space or seed.chunk_id != args.chunk_id or not 0 <= seed.ordinal < 2**31
            or not 0 < seed.episode_id < 2**63):
        raise ReadMemoryError()
    episode = await documents.get_episode(space, seed.episode_id)
    if episode is None:
        raise ReadMemoryError()
    episode = Episode.model_validate(episode.model_dump(warnings=False), strict=True).model_copy(deep=True)
    if (episode.space != space or episode.episode_id != seed.episode_id
            or not _source_matches(episode, TextFilter(**scope.kwargs()), excluded_session)
            or parse_rfc3339(episode.created_at) > parse_rfc3339(memory.clock())):
        raise ReadMemoryError()
    raw = episode.content.encode()
    start, end = max(0, seed.ordinal - args.before), seed.ordinal + args.after
    limit = seed.ordinal - start + args.after + 2  # One look-ahead row, at most ten.
    rows = await documents.page_chunks(space, seed.episode_id, start_ordinal=start, limit=limit)
    if len(rows) > limit:
        raise ReadMemoryError()
    chunks = tuple(Chunk.model_validate(row.model_dump(), strict=True).model_copy(deep=True) for row in rows)
    if (len({row.chunk_id for row in chunks}) != len(chunks) or len({row.ordinal for row in chunks}) != len(chunks)
            or list(chunks) != sorted(chunks, key=lambda row: (row.ordinal, row.chunk_id))):
        raise ReadMemoryError()
    for row in chunks:
        if (not 0 < row.chunk_id < 2**63 or row.space != space or row.episode_id != episode.episode_id
                or row.created_at != episode.created_at or row.ordinal < start
                or not 0 <= row.start < row.end <= len(raw) or raw[row.start:row.end] != row.text.encode()):
            raise ReadMemoryError()
    selected = tuple(row for row in chunks if row.ordinal <= end)
    if seed not in selected:
        raise ReadMemoryError()
    current = await documents.get_chunks(space, [row.chunk_id for row in selected])
    if (len(current) != len(selected) or {row.chunk_id:row for row in current} != {row.chunk_id:row for row in selected}
            or await documents.get_episode(space, episode.episode_id) != episode
            or await documents.revision(space) != revision):
        raise ReadMemoryError()
    return {'ok':True, 'status':'prepared', 'facts':[], 'verified_accuracy':False,
        'items':[{'episode_id':row.episode_id, 'chunk_id':row.chunk_id, 'text':row.text,
                  'source':episode.source, 'created_at':row.created_at, 'score':0.0} for row in selected],
        'coverage':{'bounded':True, 'complete':False, 'mode':'chunk_window',
                    'requested_start_ordinal':start, 'requested_end_ordinal':end,
                    'returned_ordinals':[row.ordinal for row in selected],
                    'has_more_after':any(row.ordinal > end for row in chunks),
                    'truncated':selected[0].start > 0 or selected[-1].end < len(raw)}}
