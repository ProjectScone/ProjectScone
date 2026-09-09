"""Bind exact calculations to authorized retained document snapshots."""
from __future__ import annotations

import json
from typing import cast

from ..agents.tool_evidence import prepare_tool_evidence
from ..core.models import Chunk, Episode
from ..memory.engine import MemoryEngine
from ..retrieval.computation import ComputeMemoryArgs, ComputationError, MatchedInput, evaluate_computation, _number
from ..retrieval.recall_scope import RecallScope


def _check_source_spans(result: dict[str, object], chunks: dict[int, Chunk], sources: dict[int, Episode]) -> None:
    # Different overlapping chunks can contain the same physical source span.
    # Count/aggregate it once per group even when the model uses different IDs.
    for side in ('left', 'right'):
        seen: list[tuple[int, int, int]] = []
        for span in cast(list[MatchedInput], result[side]):
            chunk = chunks[span['chunk_id']]
            start = chunk.start + len(chunk.text[:span['start_char']].encode())
            end = chunk.start + len(chunk.text[:span['end_char']].encode())
            if any(episode == chunk.episode_id and start < other_end and other_start < end
                   for episode, other_start, other_end in seen):
                raise ComputationError()
            seen.append((chunk.episode_id, start, end))
            if result['operation'] not in ('count', 'compare_counts'):
                content = sources[chunk.episode_id].content
                source_start = len(content.encode()[:start].decode())
                source_span: MatchedInput = {**span, 'start_char':source_start,
                                            'end_char':source_start + len(span['quote'])}
                _number(source_span, content)


async def compute_memory(memory: MemoryEngine, space: str, args: ComputeMemoryArgs,
                         scope: RecallScope, excluded_session: str | None,
                         timeout_s: float) -> dict[str, object]:
    """Point-read at most 16 chunks; wrapper owns total deadline/output bounds.

    Original passages, source metadata and revision are validated using the same
    retained-source contract as other tools. No full scan/window capability is
    needed. The caller's prepare() binds a final-publication validator as well.
    """
    args = ComputeMemoryArgs.model_validate(args.model_dump())
    ids = sorted({row.chunk_id for row in args.left + args.right})
    documents = memory.documents
    revision = await documents.revision(space)
    rows = await documents.get_chunks(space, ids)
    if len(rows) != len(ids):
        raise ComputationError('evidence_unavailable')
    try:
        snapshots = [Chunk.model_validate(row.model_dump(), strict=True).model_copy(deep=True) for row in rows]
        chunks = {row.chunk_id:row for row in snapshots}
        if (sorted(chunks) != ids or any(row.space != space for row in snapshots)):
            raise ValueError('invalid chunks')
        sources: dict[int, Episode] = {}
        for chunk in snapshots:
            if chunk.episode_id in sources:
                continue
            source = await documents.get_episode(space, chunk.episode_id)
            if source is None:
                raise ValueError('missing source')
            sources[chunk.episode_id] = Episode.model_validate(source.model_dump(warnings=False), strict=True).model_copy(deep=True)
        packet: dict[str, object] = {'ok':True, 'status':'prepared', 'facts':[], 'verified_accuracy':False,
            'items':[{'chunk_id':row.chunk_id, 'episode_id':row.episode_id, 'text':row.text,
                      'source':sources[row.episode_id].source, 'created_at':row.created_at, 'score':0.0}
                     for row in snapshots],
            'coverage':{'bounded':True, 'complete':False, 'mode':'selected_spans'}}
        prepared = await prepare_tool_evidence(memory, space, scope, excluded_session, packet, timeout_s)
        current = await documents.get_chunks(space, ids)
        if len(current) != len(snapshots) or {row.chunk_id:row for row in current} != chunks:
            raise ValueError('changed chunks')
        for episode_id, source in sources.items():
            if await documents.get_episode(space, episode_id) != source:
                raise ValueError('changed source')
        if await documents.revision(space) != revision or not await prepared.validate():
            raise ValueError('changed source')
    except ValueError:
        raise ComputationError('evidence_unavailable') from None
    result = evaluate_computation(args, {row.chunk_id:row.text for row in snapshots})
    _check_source_spans(result, chunks, sources)
    detached: dict[str, object] = json.loads(prepared.payload)
    detached['computation'] = result
    return detached
