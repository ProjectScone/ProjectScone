"""Opt-in parent-section context, using only bounded retained recall sources."""
from __future__ import annotations

from typing import Protocol, Sequence

from pydantic import BaseModel, ConfigDict, Field

from ..core.models import MAX_CONTENT_BYTES, Chunk, Episode, RecallResult
from ..core.ports import TextFilter
from ..core.timeutil import parse_rfc3339
from ..ingestion.structure import DocumentStructure, StructureBlock, StructureSection, parse_structure
from ..memory.engine import check_space


class StructuralDocuments(Protocol):
    async def get_chunks(self, space: str, chunk_ids: Sequence[int]) -> list[Chunk]: ...
    async def get_episode(self, space: str, episode_id: int) -> Episode | None: ...


class StructuralLimits(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid", strict=True)
    max_bytes: int = Field(default=16_000, ge=1, le=256_000)
    max_sections: int = Field(default=8, ge=1, le=64)
    max_sources: int = Field(default=4, ge=1, le=32)
    max_chunks: int = Field(default=24, ge=1, le=128)
    max_source_bytes: int = Field(default=MAX_CONTENT_BYTES, ge=1, le=MAX_CONTENT_BYTES)


class StructuralSection(StructureSection):
    episode_id: int
    source: str | None
    content_hash: str
    source_sha256: str
    section_start: int
    section_end: int
    text: str
    matched_chunk_ids: tuple[int, ...]
    blocks: tuple[StructureBlock, ...]
    truncated: bool = False


class StructuralContext(BaseModel):
    model_config = ConfigDict(frozen=True)
    sections: tuple[StructuralSection, ...] = ()
    returned_bytes: int = 0
    sources_considered: int = 0
    chunks_considered: int = 0
    revalidation_source_reads: int = 0
    revalidation_chunks: int = 0
    omitted_chunks: int = 0
    omitted_sections: int = 0
    omitted_sources: int = 0
    provenance_missing: int = 0
    out_of_scope: int = 0
    invalid_provenance: int = 0
    truncated: bool = False


def _matches(episode: Episode, scope: TextFilter, exclude_session_id: str | None) -> bool:
    created = parse_rfc3339(episode.created_at)
    return (all(tag in episode.tags for tag in scope.tags)
            and all(episode.metadata.get(key) == value for key, value in scope.where.items())
            and (scope.conditions is None or bool(scope.conditions.matches(episode.metadata)))
            and (scope.kind is None or episode.kind == scope.kind)
            and (scope.source_prefix is None or (episode.source is not None and episode.source.startswith(scope.source_prefix)))
            and (scope.since is None or created >= parse_rfc3339(scope.since))
            and (scope.until is None or created <= parse_rfc3339(scope.until))
            and (scope.as_of is None or created <= parse_rfc3339(scope.as_of))
            and (exclude_session_id is None or (episode.metadata.get("session_id") != exclude_session_id
                                               and episode.source != exclude_session_id)))


async def expand_structural_context(documents: StructuralDocuments, space: str, result: RecallResult,
                                    *, scope: TextFilter, limits: StructuralLimits | None = None,
                                    exclude_session_id: str | None = None) -> StructuralContext:
    """Return the smallest containing section for each revalidated recall chunk.

    Reads chunk IDs from the bounded recall prefix and at most max_sources
    distinct episodes; never scans an index, calls a model or changes recall.
    Rechecks space, scope and exact chunk byte spans; fingerprints source bytes
    with independent SHA-256. Episode content_hash is an opaque identity. Oversized
    sections are omitted whole so code fences/table headers never get clipped.
    Limits count UTF-8 source text bytes, not JSON envelope/metadata bytes.
    Retention is observed at the store read; no cross-store snapshot is claimed.
    """
    check_space(space)
    budget = limits or StructuralLimits()
    items = result.items[:budget.max_chunks]
    chunks = {chunk.chunk_id: chunk for chunk in await documents.get_chunks(space, [item.chunk_id for item in items])} if items else {}
    sources: dict[int, tuple[Episode, DocumentStructure] | None] = {}
    omitted_sources: set[int] = set()
    selected: dict[tuple[int, str], tuple[Episode, DocumentStructure, StructureSection, list[int]]] = {}
    missing = invalid = outside = 0
    for item in items:
        chunk = chunks.get(item.chunk_id)
        if chunk is None:
            missing += 1
            continue
        if chunk.space != space or chunk.episode_id != item.episode_id or chunk.text != item.text:
            invalid += 1
            continue
        if chunk.episode_id not in sources:
            if len(sources) >= budget.max_sources:
                omitted_sources.add(chunk.episode_id)
                continue
            sources[chunk.episode_id] = None
            episode = await documents.get_episode(space, chunk.episode_id)
            if episode is None or episode.space != space or episode.episode_id != chunk.episode_id:
                missing += 1
                continue
            if not _matches(episode, scope, exclude_session_id):
                outside += 1
                continue
            raw = episode.content.encode("utf-8")
            if len(raw) > budget.max_source_bytes:
                omitted_sources.add(chunk.episode_id)
                continue
            try:
                structure = parse_structure(episode.content)
            except ValueError:
                omitted_sources.add(chunk.episode_id)
                continue
            sources[chunk.episode_id] = (episode.model_copy(deep=True), structure)
        retained = sources[chunk.episode_id]
        if retained is None:
            continue
        episode, structure = retained
        raw = episode.content.encode("utf-8")
        if not (0 <= chunk.start < chunk.end <= len(raw)) or raw[chunk.start:chunk.end] != chunk.text.encode("utf-8"):
            invalid += 1
            continue
        section = max((section for section in structure.sections
                       if section.start <= chunk.start and chunk.end <= section.end), key=lambda section: section.level)
        key = (episode.episode_id, section.section_id)
        if key not in selected:
            selected[key] = (episode, structure, section, [])
        if chunk.chunk_id not in selected[key][3]:
            selected[key][3].append(chunk.chunk_id)
    expanded: list[StructuralSection] = []
    returned_bytes = omitted_sections = 0
    for episode, structure, section, chunk_ids in selected.values():
        size = section.end - section.start
        if len(expanded) >= budget.max_sections or returned_bytes + size > budget.max_bytes:
            omitted_sections += 1
            continue
        expanded.append(StructuralSection(
            **section.model_dump(), episode_id=episode.episode_id, source=episode.source,
            content_hash=episode.content_hash, section_start=section.start, section_end=section.end,
            source_sha256=structure.content_hash,
            text=episode.content.encode("utf-8")[section.start:section.end].decode("utf-8"),
            matched_chunk_ids=tuple(chunk_ids),
            blocks=tuple(block for block in structure.blocks if section.start <= block.start and block.end <= section.end)))
        returned_bytes += size
    # Recheck after all expansion reads. This catches changes during traversal;
    # the store still does not offer an atomic multi-record snapshot.
    valid_sources: set[int] = set()
    source_ids = dict.fromkeys(section.episode_id for section in expanded)
    for episode_id in source_ids:
        retained_source = sources[episode_id]
        assert retained_source is not None
        current = await documents.get_episode(space, episode_id)
        if current is None:
            missing += 1
        elif current.space != space or current.episode_id != episode_id:
            invalid += 1
        elif not _matches(current, scope, exclude_session_id):
            outside += 1
        elif current != retained_source[0]:
            invalid += 1
        else:
            valid_sources.add(episode_id)
    final_ids = list(dict.fromkeys(chunk_id for section in expanded if section.episode_id in valid_sources
                                  for chunk_id in section.matched_chunk_ids))
    final_chunks = {chunk.chunk_id: chunk for chunk in await documents.get_chunks(space, final_ids)} if final_ids else {}
    valid_chunks: set[int] = set()
    for chunk_id in final_ids:
        current_chunk = final_chunks.get(chunk_id)
        if current_chunk is None:
            missing += 1
        elif current_chunk != chunks[chunk_id]:
            invalid += 1
        else:
            valid_chunks.add(chunk_id)
    verified: list[StructuralSection] = []
    for expanded_section in expanded:
        matched_ids = tuple(chunk_id for chunk_id in expanded_section.matched_chunk_ids if chunk_id in valid_chunks)
        if expanded_section.episode_id in valid_sources and matched_ids:
            verified.append(expanded_section.model_copy(update={"matched_chunk_ids": matched_ids}))
    expanded = verified
    returned_bytes = sum(section.end - section.start for section in expanded)
    omitted_chunks = len(result.items) - len(items)
    return StructuralContext(sections=tuple(expanded), returned_bytes=returned_bytes,
                             sources_considered=len(sources), chunks_considered=len(items),
                             revalidation_source_reads=len(source_ids), revalidation_chunks=len(final_ids),
                             omitted_chunks=omitted_chunks, omitted_sections=omitted_sections,
                             omitted_sources=len(omitted_sources), provenance_missing=missing,
                             invalid_provenance=invalid, out_of_scope=outside,
                             truncated=bool(omitted_chunks or omitted_sections or omitted_sources))
