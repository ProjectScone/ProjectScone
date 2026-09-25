"""Opt-in routing over an already-authorized immutable document snapshot.

This module never loads documents or grants access. The caller supplies the
current authorized snapshot on every call; a route is not evidence or authority.
"""
from __future__ import annotations

import asyncio
from collections import OrderedDict
from dataclasses import dataclass, replace
import hashlib
import math
import time
from typing import Protocol

from ..ingestion.structure import parse_structure


@dataclass(frozen=True)
class SectionNode:
    id: str
    parent_id: str | None
    title: str
    text: str
    start: int
    end: int
    body_start: int


@dataclass(frozen=True)
class SectionSnapshot:
    scope: str
    document_id: str
    version: str
    nodes: tuple[SectionNode, ...]
    content: str

    @classmethod
    def from_markdown(cls, scope: str, document_id: str, content: str) -> SectionSnapshot:
        if not scope.strip() or not document_id.strip() or max(len(scope), len(document_id)) > 512:
            raise ValueError('snapshot requires bounded scope and document identity')
        structure = parse_structure(content)
        raw = content.encode()
        bodies: dict[str, list[str]] = {s.section_id: [] for s in structure.sections}
        starts: dict[str, int] = {}
        for block in structure.blocks:
            if block.kind != 'heading':
                starts.setdefault(block.section_id, block.start)
                bodies[block.section_id].append(raw[block.start:block.end].decode())
        nodes = tuple(SectionNode(s.section_id, s.parent_section_id, s.title,
                                  ''.join(bodies[s.section_id]), s.start, s.end, starts.get(s.section_id, s.start)) for s in structure.sections)
        return cls(scope, document_id, structure.content_hash, nodes, content)

    def original(self, identifier: str) -> str:
        node = next(n for n in self.nodes if n.id == identifier)
        return self.content.encode()[node.start:node.end].decode()

    def path(self, identifier: str) -> tuple[str, ...]:
        nodes = {node.id: node for node in self.nodes}
        result: list[str] = []
        current: str | None = identifier
        while current is not None:
            node = nodes[current]
            if node.title:
                result.append(node.title)
            current = node.parent_id
        return tuple(reversed(result))


@dataclass(frozen=True)
class RouteOption:
    section_id: str
    title: str
    terminal: bool
    outline: tuple[str, ...] = ()


@dataclass(frozen=True)
class RouteMenu:
    path: tuple[str, ...]
    options: tuple[RouteOption, ...]


@dataclass(frozen=True)
class ChoiceBatch:
    # Each distribution has one probability per option followed by no-match.
    distributions: tuple[tuple[float, ...], ...]
    model: str
    input_tokens: int = 0
    output_tokens: int = 0


class SectionChooser(Protocol):
    @property
    def definition(self) -> str: ...
    async def choose(self, query: str, menus: tuple[RouteMenu, ...]) -> ChoiceBatch: ...


@dataclass(frozen=True)
class RouteResult:
    version: str
    section_ids: tuple[str, ...]
    scores: tuple[float, ...]
    reason: str
    cache_hit: bool = False
    requests: int = 0
    elapsed_ms: float = 0
    model: str = ''
    input_tokens: int = 0
    output_tokens: int = 0


@dataclass(frozen=True)
class _Path:
    section_id: str
    terminal: bool = False
    log_score: float = 0
    decisions: int = 0

    @property
    def score(self) -> float:
        return math.exp(self.log_score / self.decisions) if self.decisions else 1.0


def _menu(snapshot: SectionSnapshot, path: _Path) -> RouteMenu:
    node = next(node for node in snapshot.nodes if node.id == path.section_id)
    children = tuple(n for n in snapshot.nodes if n.parent_id == node.id)
    parents = {n.parent_id for n in snapshot.nodes}
    options = tuple(RouteOption(n.id, n.title or '(untitled)', n.id not in parents,
        tuple(desc.title for desc in snapshot.nodes if n.start < desc.start < n.end and desc.title))
        for n in children)
    if node.text.strip() or children:
        options += (RouteOption(node.id, 'Read this entire section, including its subsections', True),)
    return RouteMenu(snapshot.path(node.id), options)


def _validate(batch: ChoiceBatch, menus: tuple[RouteMenu, ...]) -> None:
    if (not batch.model.strip() or any(type(t) is not int or t < 0
            for t in (batch.input_tokens, batch.output_tokens))):
        raise ValueError('invalid route model or usage')
    if len(batch.distributions) != len(menus):
        raise ValueError('mismatched menu count')
    for probabilities, menu in zip(batch.distributions, menus, strict=True):
        if (len(probabilities) != len(menu.options) + 1
                or any(type(p) not in (int, float) or not math.isfinite(p) or not 0 <= p <= 1
                       for p in probabilities)
                or not math.isclose(sum(probabilities), 1.0, abs_tol=.005)):
            raise ValueError('invalid route distribution')


class SectionRouter:
    def __init__(self, chooser: SectionChooser, *, beam_width: int = 3, max_rounds: int = 8,
                 timeout: float = 10.0, cache_size: int = 256) -> None:
        if (type(beam_width) is not int or not 1 <= beam_width <= 8
                or type(max_rounds) is not int or not 1 <= max_rounds <= 16
                or type(cache_size) is not int or not 0 <= cache_size <= 4096
                or type(timeout) not in (int, float) or not math.isfinite(timeout) or not 0 < timeout <= 60):
            raise ValueError('invalid routing bounds')
        self.chooser = chooser
        self.beam_width, self.max_rounds, self.timeout = beam_width, max_rounds, timeout
        self.cache_size = cache_size
        self._cache: OrderedDict[str, RouteResult] = OrderedDict()

    async def route(self, query: str, snapshot: SectionSnapshot) -> RouteResult:
        if not query.strip() or len(query.encode()) > 8000:
            raise ValueError('invalid routing query')
        if SectionSnapshot.from_markdown(snapshot.scope, snapshot.document_id, snapshot.content) != snapshot:
            raise ValueError('snapshot does not match source structure')
        if not snapshot.content.strip():
            return RouteResult(snapshot.version, (), (), 'empty_document')
        started = time.perf_counter()
        identity = (snapshot.scope, snapshot.document_id, snapshot.version, snapshot.nodes, snapshot.content,
                    query, self.chooser.definition, self.beam_width, self.max_rounds)
        key = hashlib.sha256(repr(identity).encode()).hexdigest()
        if key in self._cache:
            self._cache.move_to_end(key)
            return replace(self._cache[key], cache_hit=True, requests=0, input_tokens=0,
                           output_tokens=0, elapsed_ms=(time.perf_counter() - started) * 1000)
        # All cache keys bind to the full immutable snapshot, including headings and body.
        result = RouteResult(snapshot.version, (), (), 'no_match')
        beam = [_Path(snapshot.nodes[0].id)]
        requests = input_tokens = output_tokens = 0
        model = ''
        try:
            async with asyncio.timeout(self.timeout):
                for _ in range(self.max_rounds):
                    pending = [p for p in beam if not p.terminal]
                    if not pending:
                        break
                    menus = tuple(_menu(snapshot, p) for p in pending)
                    if any(len(menu.options) > 254 for menu in menus):
                        result = replace(result, reason='menu_too_wide')
                        break
                    requests += 1
                    batch = await self.chooser.choose(query, menus)
                    _validate(batch, menus)
                    model = batch.model
                    input_tokens += batch.input_tokens
                    output_tokens += batch.output_tokens
                    expanded = [p for p in beam if p.terminal]
                    for path, menu, probabilities in zip(pending, menus, batch.distributions, strict=True):
                        for option, probability in zip(menu.options, probabilities[:-1], strict=True):
                            if probability <= 0 or probability <= probabilities[-1]:
                                continue
                            expanded.append(_Path(option.section_id, option.terminal,
                                path.log_score + math.log(probability), path.decisions + 1))
                    beam = sorted(expanded, key=lambda p: (-p.score, p.section_id))[:self.beam_width]
                    if not beam:
                        break
                else:
                    if any(not p.terminal for p in beam):
                        result = replace(result, reason='budget_exhausted')
                if result.reason == 'no_match' and beam and all(p.terminal for p in beam):
                    result = replace(result, section_ids=tuple(p.section_id for p in beam),
                                     scores=tuple(p.score for p in beam), reason='routed')
        except TimeoutError:
            result = replace(result, reason='timeout')
        except ValueError:
            result = replace(result, reason='invalid_response')
        except Exception:
            result = replace(result, reason='provider_failed')
        result = replace(result, requests=requests, model=model, input_tokens=input_tokens,
            output_tokens=output_tokens, elapsed_ms=(time.perf_counter() - started) * 1000)
        if result.reason in ('routed', 'no_match') and self.cache_size:
            self._cache[key] = result
            while len(self._cache) > self.cache_size:
                self._cache.popitem(last=False)
        return result


@dataclass(frozen=True)
class FetchDecision:
    mode: str
    model: str
    probabilities: tuple[float, ...]
    input_tokens: int = 0
    output_tokens: int = 0


class FetchChooser(Protocol):
    async def choose_fetch(self, query: str, sections: tuple[tuple[str, int], ...],
                           max_bytes: int) -> FetchDecision: ...
