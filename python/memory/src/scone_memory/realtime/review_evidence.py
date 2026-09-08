"""Revalidate exactly the retained evidence delivered to an answer model.

Only bounded point reads occur here: no recall, graph expansion, or new links.
The receipt revision rejects native writes since context preparation. Full
immutable snapshots then detect direct-store changes during answer review.
Individual source reads may load large episodes through the existing store API.
"""
from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
import hashlib
import json
from typing import cast

from ..core.models import Chunk, Episode, Fact, FactLink
from ..core.ports import TextFilter
from ..core.timeutil import parse_rfc3339
from ..memory.engine import MemoryEngine, check_space
from ..retrieval.evidence_graph import MAX_CHUNKS, MAX_FACTS, MAX_LINKS, MAX_SOURCES
from ..retrieval.multihop import PointFactLinks, _source_matches
from ..retrieval.recall_scope import RecallScope
from .context import ContextReceipt, _PATH_GUIDANCE, _PREFIX

_ERROR = "prepared evidence unavailable"
_MAX_BYTES = 128000


@dataclass(frozen=True)
class PreparedReviewEvidence:
    evidence: str
    evidence_ids: tuple[str, ...]
    _validator: Callable[[], Awaitable[bool]] = field(repr=False, compare=False)

    async def validate(self) -> bool:
        """False means evidence changed; sanitized errors mean unavailable."""
        return await self._validator()


class _Changed(ValueError):
    pass


def _json(value: object) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _mapping(value: object) -> dict[str, object]:
    if not isinstance(value, dict) or not all(isinstance(key, str) for key in value):
        raise _Changed()
    return cast(dict[str, object], value)


def _records(value: object, cap: int) -> list[dict[str, object]]:
    if not isinstance(value, list) or len(value) > cap:
        raise _Changed()
    return [_mapping(record) for record in value]


def _id(value: object) -> int:
    if type(value) is not int or not 0 < value < 2**63:
        raise _Changed()
    return value


def _unique_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise _Changed()
        result[key] = value
    return result


def _claim(fact: Fact) -> dict[str, object]:
    return {key: getattr(fact, key) for key in ("fact_id", "subject", "predicate", "object", "origin", "status",
        "valid_from", "valid_until", "confidence", "source_episode_id", "quote")}


def _relation(link: FactLink) -> dict[str, object]:
    return {key: getattr(link, key) for key in ("link_id", "from_fact", "to_fact", "kind", "source_episode_id", "quote")}


def _fingerprints(records: list[dict[str, object]], id_key: str) -> dict[str, str]:
    return {str(_id(record.get(id_key))): hashlib.sha256(_json(record).encode()).hexdigest() for record in records}


def _paths(paths: list[dict[str, object]], claims: dict[int, dict[str, object]],
           relations: dict[int, dict[str, object]]) -> None:
    for path in paths:
        if set(path) - {"fact_ids", "steps", "ordered_evidence"}:
            raise _Changed()
        raw_ids = path.get("fact_ids")
        if not isinstance(raw_ids, list) or not 2 <= len(raw_ids) <= 7:
            raise _Changed()
        ids = [_id(value) for value in raw_ids]
        if len(set(ids)) != len(ids) or not set(ids).issubset(claims):
            raise _Changed()
        steps = _records(path.get("steps"), 6)
        if len(steps) != len(ids) - 1:
            raise _Changed()
        for left, right, step in zip(ids, ids[1:], steps):
            source, target = _id(step.get("from_fact")), _id(step.get("to_fact"))
            direction = step.get("direction")
            if direction not in ("forward", "reverse") or (source, target) != ((left, right) if direction == "forward" else (right, left)):
                raise _Changed()
            if step.get("kind") == "subject_object":
                if (set(step) != {"from_fact", "to_fact", "kind", "direction"} or direction != "forward"
                        or not isinstance(claims[source]["object"], str)
                        or claims[source]["object"] != claims[target]["subject"]):
                    raise _Changed()
            else:
                relation = relations.get(_id(step.get("link_id")))
                if (set(step) != {"from_fact", "to_fact", "kind", "direction", "link_id"}
                        or relation is None or step.get("kind") == "contradicts"
                        or any(step.get(key) != relation[key] for key in ("from_fact", "to_fact", "kind"))):
                    raise _Changed()
        if "ordered_evidence" in path:
            expected = [{"fact_id": value, "source_episode_id": claims[value]["source_episode_id"],
                         "quote": claims[value]["quote"]} for value in ids]
            if _json(path["ordered_evidence"]) != _json(expected):
                raise _Changed()


@dataclass(frozen=True)
class _Packet:
    block: str
    revision: int
    sources: tuple[dict[str, object], ...]
    claims: tuple[dict[str, object], ...]
    relations: tuple[dict[str, object], ...]
    graph_sources: dict[int, dict[str, object]]
    graph_chunks: dict[int, dict[str, object]]
    evidence_ids: tuple[str, ...]


def _packet(request: list[dict[str, object]], receipt: ContextReceipt, session_id: str) -> _Packet:
    revision = receipt.get("evidence_revision")
    if (receipt.get("status") != "prepared" or receipt.get("session_id") != session_id
            or receipt.get("evidence_graph_status") != "prepared" or receipt.get("evidence_graph_stale")
            or type(revision) is not int or revision < 0):
        raise _Changed()
    blocks = [message.get("content") for message in request if isinstance(message.get("content"), str)
              and cast(str, message["content"]).startswith("Scone retrieved source material:")]
    if len(blocks) != 1 or not isinstance(blocks[0], str):
        raise _Changed()
    block = blocks[0]
    raw = block.encode()
    if (len(raw) > _MAX_BYTES or receipt.get("context_bytes") != len(raw)
            or hashlib.sha256(raw).hexdigest() != receipt.get("context_sha256")):
        raise _Changed()
    prefix, separator, serialized = block.partition("\n")
    allowed_prefixes = {_PREFIX.rstrip("\n"), _PREFIX.rstrip("\n") + " " + _PATH_GUIDANCE.rstrip("\n")}
    if not separator or prefix not in allowed_prefixes:
        raise _Changed()
    packet = _mapping(json.loads(serialized, object_pairs_hook=_unique_object))
    if (set(packet) - {"schema_version", "coverage", "sources", "claims", "relations", "paths"}
            or type(packet.get("schema_version")) is not int or packet["schema_version"] != 1):
        raise _Changed()
    _mapping(packet.get("coverage"))
    sources = _records(packet.get("sources"), MAX_CHUNKS)
    claims = _records(packet.get("claims", []), MAX_FACTS)
    relations = _records(packet.get("relations", []), MAX_LINKS)
    paths = _records(packet.get("paths", []), 16)
    ids = tuple(f"{kind}:{_id(record.get(key))}" for records, kind, key in (
        (sources, "chunk", "chunk_id"), (claims, "fact", "fact_id"), (relations, "link", "link_id")) for record in records)
    if not ids or len(set(ids)) != len(ids):
        raise _Changed()
    if (_fingerprints(claims, "fact_id") != receipt.get("claim_fingerprints")
            or _fingerprints(relations, "link_id") != receipt.get("relation_fingerprints")):
        raise _Changed()
    source_hashes = {str(_id(record.get("chunk_id"))): hashlib.sha256(cast(str, record["text"]).encode()).hexdigest()
                     for record in sources if isinstance(record.get("text"), str)}
    if len(source_hashes) != len(sources) or source_hashes != receipt.get("evidence_fingerprints"):
        raise _Changed()
    _paths(paths, {_id(record.get("fact_id")): record for record in claims},
           {_id(record.get("link_id")): record for record in relations})
    serialized_graph = _json(_mapping(receipt.get("evidence_graph")))
    if len(serialized_graph.encode()) > 512000:
        raise _Changed()
    graph = _mapping(json.loads(serialized_graph))
    nodes = _records(graph.get("nodes"), 128)
    _records(graph.get("edges"), 512)
    graph_sources, graph_chunks = {}, {}
    for node in nodes:
        data = _mapping(node.get("data"))
        if node.get("kind") == "episode":
            graph_sources[_id(data.get("episode_id"))] = node
        elif node.get("kind") == "chunk":
            graph_chunks[_id(data.get("chunk_id"))] = node
    if len(graph_sources) > MAX_SOURCES or len(graph_chunks) > MAX_CHUNKS:
        raise _Changed()
    return _Packet(block, revision, tuple(sources), tuple(claims), tuple(relations), graph_sources, graph_chunks, ids)


async def _capture(memory: MemoryEngine, space: str, scope: RecallScope, session_id: str,
                   packet: _Packet) -> tuple[str, ...]:
    documents = memory.documents
    if await documents.revision(space) != packet.revision:
        raise _Changed()
    boundary = memory.clock()
    parse_rfc3339(boundary)
    scope_filter = TextFilter(**scope.kwargs())
    sources: dict[int, Episode] = {}
    snapshots: list[str] = []

    async def source(episode_id: int) -> Episode:
        if episode_id in sources:
            return sources[episode_id]
        if len(sources) >= MAX_SOURCES:
            raise _Changed()
        episode = await documents.get_episode(space, episode_id)
        if (episode is None or episode.episode_id != episode_id or episode.space != space
                or not _source_matches(episode, scope_filter, session_id)
                or parse_rfc3339(episode.created_at) > parse_rfc3339(boundary)):
            raise _Changed()
        proof = packet.graph_sources.get(episode_id)
        if proof is None:
            raise _Changed()
        data = _mapping(proof.get("data"))
        if (proof.get("id") != f"episode:{episode_id}" or proof.get("ts") != episode.created_at
                or data.get("source") != episode.source or data.get("kind") != episode.kind
                or data.get("preview") != episode.content[:480]):
            raise _Changed()
        # JSON strings are immutable even when an adapter reuses mutable models.
        snapshots.append(_json(episode.model_dump(mode="json")))
        sources[episode_id] = episode.model_copy(deep=True)
        return sources[episode_id]

    for record in packet.sources:
        chunk_id, episode_id = _id(record.get("chunk_id")), _id(record.get("episode_id"))
        chunks = await documents.get_chunks(space, [chunk_id])
        if len(chunks) != 1:
            raise _Changed()
        chunk: Chunk = chunks[0]
        episode = await source(episode_id)
        expected: dict[str, object] = {"chunk_id": chunk_id, "episode_id": episode_id, "text": chunk.text,
                                      "source": episode.source, "created_at": chunk.created_at}
        expected.update({key: episode.metadata[key] for key in ("project", "role") if key in episode.metadata})
        proof = packet.graph_chunks.get(chunk_id)
        data = _mapping(proof.get("data")) if proof is not None else {}
        if (chunk.chunk_id != chunk_id or chunk.space != space or chunk.episode_id != episode_id
                or chunk.created_at != episode.created_at or chunk.start < 0 or chunk.end < chunk.start
                or chunk.end > len(episode.content.encode()) or episode.content.encode()[chunk.start:chunk.end] != chunk.text.encode()
                or _json(record) != _json(expected) or proof is None or proof.get("id") != f"chunk:{chunk_id}"
                or proof.get("ts") != chunk.created_at or data.get("episode_id") != episode_id
                or data.get("start") != chunk.start or data.get("end") != chunk.end or data.get("text") != chunk.text[:480]):
            raise _Changed()
        snapshots.append(_json(chunk.model_dump(mode="json")))
    facts: dict[int, Fact] = {}
    for record in packet.claims:
        fact_id = _id(record.get("fact_id"))
        fact = await documents.get_fact(space, fact_id)
        if (fact is None or fact.fact_id != fact_id or fact.space != space or fact.excluded
                or fact.status != "active" or not fact.holds_at(boundary) or _json(record) != _json(_claim(fact))):
            raise _Changed()
        episode = await source(_id(fact.source_episode_id))
        if not fact.quote or fact.quote not in episode.content:
            raise _Changed()
        snapshots.append(_json(fact.model_dump(mode="json")))
        facts[fact_id] = fact.model_copy(deep=True)
    if packet.relations and not isinstance(documents, PointFactLinks):
        raise _Changed()
    for record in packet.relations:
        if not isinstance(documents, PointFactLinks):
            raise _Changed()
        link_id = _id(record.get("link_id"))
        link = await documents.get_fact_link(space, link_id)
        if (link is None or link.link_id != link_id or link.space != space or link.from_fact not in facts
                or link.to_fact not in facts or parse_rfc3339(link.created_at) > parse_rfc3339(boundary)
                or _json(record) != _json(_relation(link))):
            raise _Changed()
        episode = await source(_id(link.source_episode_id))
        if not link.quote or link.quote not in episode.content:
            raise _Changed()
        snapshots.append(_json(link.model_dump(mode="json")))
    if await documents.revision(space) != packet.revision:
        raise _Changed()
    return tuple(snapshots)


async def prepare_review_evidence(memory: MemoryEngine, space: str, scope: RecallScope, session_id: str,
                                  request: list[dict[str, object]], receipt: ContextReceipt) -> PreparedReviewEvidence:
    """Freeze delivered evidence, rejecting unavailable or changed preparation.

    Each read pass has a one-second cooperative timeout. Revision changes or
    record changes make ``validate`` false; store errors are sanitized errors.
    The surrounding review controller owns the entire review deadline.
    """
    try:
        check_space(space)
        if not isinstance(session_id, str) or not 1 <= len(session_id) <= 128:
            raise _Changed()
        fixed_scope = RecallScope.validated(**scope.kwargs())
        packet = _packet(request, receipt, session_id)
        async with asyncio.timeout(1.0):
            original = await _capture(memory, space, fixed_scope, session_id, packet)
    except asyncio.CancelledError:
        raise
    except TimeoutError:
        raise TimeoutError(_ERROR) from None
    except Exception:
        raise ValueError(_ERROR) from None

    async def validate() -> bool:
        try:
            async with asyncio.timeout(1.0):
                return await _capture(memory, space, fixed_scope, session_id, packet) == original
        except asyncio.CancelledError:
            raise
        except _Changed:
            return False
        except TimeoutError:
            raise TimeoutError(_ERROR) from None
        except Exception:
            raise ValueError(_ERROR) from None

    return PreparedReviewEvidence(packet.block, packet.evidence_ids, validate)
