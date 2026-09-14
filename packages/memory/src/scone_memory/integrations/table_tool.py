"""A document's tables, listed and asked inside the agent loop; the arithmetic stays exact.

An agent that can search can find a spreadsheet. What it could not do
was add a column up without doing arithmetic in prose. These two tools
give it the table's own cells: ``list_tables`` names what a document
declares, ``query_table`` answers a structured question over one table
by ``retrieval.table_query`` -- exact fractions, every cell quoted --
and ties each quoted cell to the stored chunk that holds it, so the
final answer's sources are checked like any other tool evidence and a
source that changes underneath is caught before publication.

The model's part is turning words into the arguments; nothing it says
reaches the numbers.
"""

from __future__ import annotations

import json
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from ..agents.tool_evidence import prepare_tool_evidence
from ..core.errors import InvalidInput, NotFound
from ..core.models import Episode
from ..core.ports import TextFilter
from ..memory.engine import MemoryEngine
from ..retrieval.multihop import _source_matches
from ..retrieval.recall_scope import RecallScope
from ..retrieval.table_query import TableQueryArgs, answer_from, episode_tables

#: Chunks one answer's evidence may name; the tool evidence contract reads no more.
MAX_ITEMS = 20


class ListTablesArgs(BaseModel):
    model_config = ConfigDict(extra='forbid', frozen=True, strict=True)
    episode_id: int = Field(gt=0, lt=2**63)


class QueryTableToolArgs(TableQueryArgs):
    episode_id: int = Field(gt=0, lt=2**63)


class TableToolError(ValueError):
    def __init__(self, reason: Literal['evidence_unavailable'] = 'evidence_unavailable') -> None:
        self.reason = reason
        super().__init__('table evidence unavailable')


async def _scoped_episode(memory: MemoryEngine, space: str, episode_id: int, scope: RecallScope,
                          excluded_session: str | None) -> Episode:
    try:
        episode = await memory.episode(space, episode_id)
    except (NotFound, InvalidInput):
        raise TableToolError() from None
    if not _source_matches(episode, TextFilter(**scope.kwargs()), excluded_session):
        raise TableToolError()
    return episode


async def list_tables(memory: MemoryEngine, space: str, args: ListTablesArgs, scope: RecallScope,
                      excluded_session: str | None) -> dict[str, object]:
    """The tables an in-scope document declares; a document that is not one is unavailable."""
    await _scoped_episode(memory, space, args.episode_id, scope, excluded_session)
    try:
        tables = await episode_tables(memory, space, args.episode_id)
    except InvalidInput:
        raise TableToolError() from None
    return {'ok': True, 'status': 'prepared' if tables else 'empty', 'episode_id': args.episode_id,
            'tables': [table.summary() for table in tables], 'items': [], 'facts': [], 'verified_accuracy': False,
            'coverage': {'bounded': True, 'complete': True, 'mode': 'declared_tables'}}


async def query_table_tool(memory: MemoryEngine, space: str, args: QueryTableToolArgs, scope: RecallScope,
                           excluded_session: str | None, timeout_s: float) -> dict[str, object]:
    """The exact answer, its cells each tied to the chunk holding it, as prepared tool evidence."""
    episode = await _scoped_episode(memory, space, args.episode_id, scope, excluded_session)
    try:
        tables = await episode_tables(memory, space, args.episode_id)
    except InvalidInput:
        raise TableToolError() from None
    answer = answer_from(tables, TableQueryArgs(operation=args.operation, column=args.column, table=args.table,
                                                where=args.where))
    chunks = await memory.documents.chunks_of(space, args.episode_id)
    raw = episode.content.encode('utf-8')
    holders: dict[int, object] = {}
    cells: list[dict[str, object]] = []
    unanchored = 0
    for cell in answer.cells:
        holder = next((c for c in chunks if c.start <= cell.start and cell.end <= c.end), None)
        if holder is None or raw[cell.start:cell.end] != cell.text.encode('utf-8'):
            unanchored += 1
            cells.append({**cell.record(), 'chunk_id': None})
            continue
        holders.setdefault(holder.chunk_id, holder)
        cells.append({**cell.record(), 'chunk_id': holder.chunk_id})
    named = list(holders.values())[:MAX_ITEMS]
    # A cell is anchored only to a chunk the evidence packet carries and
    # checks for freshness; a cell in a chunk past that bound is returned
    # with no anchor and counted, never with an anchor nobody rechecked.
    kept_ids = {c.chunk_id for c in named}  # type: ignore[attr-defined]
    beyond = 0
    for quoted in cells:
        if quoted['chunk_id'] is not None and quoted['chunk_id'] not in kept_ids:
            quoted['chunk_id'] = None
            beyond += 1
    packet: dict[str, object] = {
        'ok': True, 'status': 'prepared', 'facts': [], 'verified_accuracy': False,
        'items': [{'chunk_id': c.chunk_id, 'episode_id': c.episode_id, 'text': c.text,  # type: ignore[attr-defined]
                   'source': episode.source, 'created_at': c.created_at, 'score': 0.0} for c in named],  # type: ignore[attr-defined]
        'coverage': {'bounded': True, 'complete': False, 'mode': 'table_cells',
                     'truncated': len(holders) > MAX_ITEMS or answer.quotes_truncated,
                     'cells_unanchored': unanchored, 'cells_beyond_evidence': beyond}}
    prepared = await prepare_tool_evidence(memory, space, scope, excluded_session, packet, timeout_s)
    detached: dict[str, object] = json.loads(prepared.payload)
    detached['table'] = {**answer.record(), 'cells': cells}
    return detached
