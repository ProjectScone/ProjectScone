"""Validate and remap imported fact replacement edges without overwriting history."""
from __future__ import annotations

from collections.abc import Callable, Hashable, Mapping, Sequence
from typing import TypeVar

from ..core.errors import InvalidInput
from ..core.graph_read import LedgerPager, checked_ledger_page
from ..core.models import Fact
from ..core.ports import DocumentStore, NewFact


Node = TypeVar('Node', bound=Hashable)
MAX_DESTINATION_FACTS = 100_000

def _identity(value: object) -> int:
    if type(value) is not int or not 0 < value < 2**63:
        raise InvalidInput('supersession requires positive 64-bit fact identities')
    return value


def _acyclic(edges: Mapping[Node, Node], starts: Sequence[Node]) -> None:
    finished: set[Node] = set()
    for start in starts:
        path: set[Node] = set()
        node = start
        while node in edges and node not in finished:
            if node in path:
                raise InvalidInput('supersession history contains a cycle or a collapsed self-reference')
            path.add(node)
            node = edges[node]
        finished.update(path)


def parse(records: Sequence[Mapping]) -> dict[int, int]:
    if not any(record.get('superseded_by') is not None for record in records):
        return {}
    known: set[int] = set()
    edges: dict[int, int] = {}
    for record in records:
        if record.get('fact_id') is None and record.get('superseded_by') is None:
            continue
        fact_id = _identity(record.get('fact_id'))
        if fact_id in known:
            raise InvalidInput('supersession history repeats a source fact identity')
        known.add(fact_id)
        if record.get('superseded_by') is not None:
            edges[fact_id] = _identity(record['superseded_by'])
    if any(target not in known for target in edges.values()):
        raise InvalidInput('supersession history references a fact missing from the archive')
    _acyclic(edges, list(edges))
    return edges


async def inventory(documents: DocumentStore, space: str) -> list[Fact]:
    if not isinstance(documents, LedgerPager) or not callable(documents.page_facts):
        raise InvalidInput('supersession import requires a complete paged destination ledger')
    prepare = getattr(documents, 'prepare_ledger_read', None)
    if callable(prepare):
        await prepare(space)
    facts: list[Fact] = []
    before = None
    while True:
        limit = min(1000, MAX_DESTINATION_FACTS - len(facts) + 1)
        page = await documents.page_facts(space, before, limit)
        problem = checked_ledger_page(page, space=space, before_id=before, limit=limit)
        if problem:
            raise InvalidInput(f'supersession destination inventory is invalid: {problem}')
        if not page:
            return facts
        if len(facts) + len(page) > MAX_DESTINATION_FACTS:
            raise InvalidInput(f'supersession destination inventory exceeds {MAX_DESTINATION_FACTS} facts')
        for raw in page:
            fact = Fact.model_validate(raw.model_dump(), strict=True)
            _identity(fact.fact_id)
            if fact.superseded_by is not None:
                _identity(fact.superseded_by)
            facts.append(fact)
        before = facts[-1].fact_id


def preflight(
    edges: Mapping[int, int], prepared: Sequence[tuple[Mapping, NewFact]],
    existing: Sequence[Fact], identity: Callable[[Fact | NewFact], Hashable],
) -> None:
    if not edges:
        return
    matched: dict[Hashable, Fact] = {}
    ambiguous: set[Hashable] = set()
    for fact in existing:
        key = identity(fact)
        if key in matched:
            ambiguous.add(key)
        matched[key] = fact
    source_keys = {int(record['fact_id']): identity(new) for record, new in prepared if record.get('fact_id') is not None}

    def node(key: Hashable) -> Hashable:
        return ('stored', matched[key].fact_id) if key in matched else ('new', key)

    graph: dict[Hashable, Hashable] = {
        ('stored', fact.fact_id): ('stored', fact.superseded_by)
        for fact in existing if fact.superseded_by is not None
    }
    starts: list[Hashable] = []
    for source, target in edges.items():
        if source_keys[source] in ambiguous or source_keys[target] in ambiguous:
            raise InvalidInput('supersession matches an ambiguous destination fact identity')
        start, successor = node(source_keys[source]), node(source_keys[target])
        if start in graph and graph[start] != successor:
            raise InvalidInput('supersession conflicts with imported or destination replacement history')
        graph[start] = successor
        starts.append(start)
    _acyclic(graph, starts)


async def restore(documents: DocumentStore, space: str, edges: Mapping[int, int], fact_map: Mapping[int, int]) -> int:
    if not edges:
        return 0
    desired: dict[int, int] = {}
    for source, target in edges.items():
        start, successor = fact_map[source], fact_map[target]
        if start in desired and desired[start] != successor:
            raise InvalidInput('supersession identities collapsed to conflicting successors')
        desired[start] = successor
    updates: list[Fact] = []
    for fact_id, successor in desired.items():
        fact = await documents.get_fact(space, fact_id)
        target_fact = await documents.get_fact(space, successor)
        if fact is None or target_fact is None or fact.space != space or target_fact.space != space or fact.fact_id != fact_id or target_fact.fact_id != successor:
            raise InvalidInput('supersession destination identity changed during import')
        if fact.superseded_by == successor:
            continue
        if fact.superseded_by is not None:
            raise InvalidInput('supersession conflicts with destination replacement history')
        updates.append(fact.model_copy(update={'superseded_by': successor}))
    if updates:
        # Invalidate before writes: an update can commit and lose its reply.
        # Retry then finds the edge already present, but the old revision is gone.
        await documents.bump_revision(space)
    try:
        for fact in updates:
            await documents.update_fact(fact)
    finally:
        # Readers may have cached between pre-invalidation and a write. Settle
        # after every attempt, including a lost acknowledgment. A no-op retry
        # also settles an earlier process interruption that skipped this block.
        await documents.bump_revision(space)
    return len(updates)
