"""Opt-in resumable retrieval with fixed authorization and retained-source checks.

The caller supplies its engine, fixed scope, private journal path and key.
Framework integrations are optional; no browser or model service is required.
"""
from __future__ import annotations

from pathlib import Path
import hashlib
from typing import cast

from ..memory.engine import MemoryEngine
from .workflow import JSONValue, StepContext, WorkflowRunner, WorkflowStep

from ..integrations.composition import build_retrieval_workflow, retrieve_without_tracing


def _records(value: JSONValue) -> list[dict[str, JSONValue]]:
    if not isinstance(value, list) or any(not isinstance(item, dict) for item in value):
        raise ValueError('invalid evidence records')
    return cast(list[dict[str, JSONValue]], value)


def build_edge_retrieval_runner(
    memory: MemoryEngine, journal: Path, *, key: bytes, space: str,
    where: dict[str, str], limit: int = 5,
) -> WorkflowRunner:
    """Resume a verified evidence snapshot; use a new run ID for a fresh search.

    Core workflow callbacks are trusted application code. This builder fixes
    space/project authorization outside model control and validates every source
    before producing or replaying a cached citation packet.
    """
    fixed_where = dict(where)
    workflow = build_retrieval_workflow(memory, space, where=fixed_where, limit=limit)

    async def verify(context: StepContext) -> bool:
        if context.space != space or context.scope != {'where': fixed_where}:
            return False
        for stage in ('retrieve', 'evidence'):
            if stage not in context.completed:
                continue
            for record in _records(context.completed[stage]):
                episode_id, chunk_id = record.get('episode_id'), record.get('chunk_id')
                if type(episode_id) is not int or type(chunk_id) is not int:
                    return False
                episode = await memory.documents.get_episode(space, episode_id)
                chunks = await memory.documents.get_chunks(space, [chunk_id])
                if episode is None or len(chunks) != 1:
                    return False
                chunk = chunks[0]
                if (episode.space != space or chunk.space != space or chunk.episode_id != episode_id
                        or any(episode.metadata.get(name) != value for name, value in fixed_where.items())
                        or episode.content_hash != record.get('content_hash')
                        or hashlib.sha256(episode.content.encode()).hexdigest() != record.get('source_sha256')
                        or chunk.text != record.get('text') or episode.source != record.get('source')
                        or episode.content.encode()[chunk.start:chunk.end] != chunk.text.encode()):
                    return False
        return True

    async def retrieve(context: StepContext) -> JSONValue:
        if not isinstance(context.inputs, dict):
            raise ValueError('query required')
        query = context.inputs.get('query')
        if not isinstance(query, str):
            raise ValueError('query required')
        documents = await retrieve_without_tracing(workflow, query)
        evidence: list[JSONValue] = []
        for document in documents:
            episode_id = document.metadata.get('episode_id')
            chunk_id = document.metadata.get('chunk_id')
            if type(episode_id) is not int or type(chunk_id) is not int:
                raise ValueError('retained source IDs required')
            episode = await memory.documents.get_episode(space, episode_id)
            if episode is None:
                raise ValueError('source unavailable')
            evidence.append({'episode_id': episode_id, 'chunk_id': chunk_id,
                             'text': document.page_content, 'source': episode.source,
                             'content_hash': episode.content_hash,
                             'source_sha256': hashlib.sha256(episode.content.encode()).hexdigest()})
        return evidence

    async def evidence(context: StepContext) -> JSONValue:
        if not await verify(context):
            raise ValueError('source unavailable')
        return context.completed['retrieve']

    return WorkflowRunner(journal, key=key, steps=[
        WorkflowStep('retrieve', '1', retrieve, idempotent=True, retryable=True),
        WorkflowStep('evidence', '1', evidence, idempotent=True, retryable=True),
    ], source_verifier=verify)


__all__ = ["build_edge_retrieval_runner"]
