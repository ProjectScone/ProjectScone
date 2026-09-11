"""One source, followed through: sections, chunks, the claims quoting it and
the entities they name.

An episode's content is the unchanged source. Its Markdown headings give
sections, its chunks are stored byte spans, and each claim that cites it
carries a quote located here by exact UTF-8 byte offsets, the first
occurrence and how many there are. The claims name entities in the
projection. Mentions are something else: names of known entities found in
the text, reported apart, because a name appearing is not the source
asserting anything about it. A lowercase single word is never taken as a
name. Chunks and claims are capped, and the caps are reported.
"""

from __future__ import annotations

import hashlib
import re
from typing import TYPE_CHECKING

from ..core.graph_read import GraphFactReader
from ..ingestion.structure import parse_structure
from .context import name_index
from .read import load_projection

if TYPE_CHECKING:
    from ..memory.engine import MemoryEngine

_WORD = re.compile(r"\w+")
_LONGEST = 6


def _word_spans(content: str) -> list[tuple[str, int, int]]:
    """Each word with its UTF-8 byte span, computed in one pass."""
    spans: list[tuple[str, int, int]] = []
    offset, position = 0, 0
    for match in _WORD.finditer(content):
        offset += len(content[position:match.start()].encode("utf-8"))
        width = len(match.group().encode("utf-8"))
        spans.append((match.group(), offset, offset + width))
        offset += width
        position = match.end()
    return spans


_ATTEMPTS = 3


async def sources_view(engine: "MemoryEngine", space: str, episode_id: int, *, max_chunks: int = 64,
                       max_claims: int = 200) -> dict[str, object]:
    """The view, read between two matching revisions when the space holds
    still long enough; ``consistent`` says whether it did."""
    view: dict[str, object] = {}
    for _attempt in range(_ATTEMPTS):
        before = await engine.revision(space)
        view = await _once(engine, space, episode_id, max_chunks=max_chunks, max_claims=max_claims)
        if await engine.revision(space) == before:
            view["consistent"] = True
            return view
    view["consistent"] = False
    coverage = view["coverage"]
    assert isinstance(coverage, dict)
    coverage["reasons"] = [*coverage["reasons"], "ledger_changed_during_read"]
    coverage["truncated"] = True
    return view


async def _once(engine: "MemoryEngine", space: str, episode_id: int, *, max_chunks: int,
                max_claims: int) -> dict[str, object]:
    episode = await engine.episode(space, episode_id)
    content = episode.content
    body = content.encode("utf-8")
    reasons: list[str] = []
    structure = parse_structure(content)
    headed = [section for section in structure.sections if section.level > 0 and section.title]
    sections = [{"id": section.section_id, "title": section.title, "level": section.level,
                 "parent": section.parent_section_id, "start": section.start, "end": section.end}
                for section in headed]

    def section_at(offset: int) -> str | None:
        inside = [section for section in headed if section.start <= offset < section.end]
        return max(inside, key=lambda section: (section.level, section.start)).title if inside else None

    returned = await engine.documents.chunks_of(space, episode_id)
    chunks = sorted((chunk for chunk in returned if chunk.space == space and chunk.episode_id == episode_id
                     and 0 <= chunk.start <= chunk.end <= len(body)), key=lambda chunk: chunk.ordinal)
    if len(chunks) != len(returned):
        reasons.append("foreign_chunks")
    if len(chunks) > max_chunks:
        reasons.append("chunk_limit")
    shown_chunks = [{"chunk_id": chunk.chunk_id, "ordinal": chunk.ordinal, "start": chunk.start, "end": chunk.end,
                     "section": section_at(chunk.start)} for chunk in chunks[:max_chunks]]

    documents = engine.documents
    if isinstance(documents, GraphFactReader):
        rows = [fact for fact in await documents.facts_for_graph(space, episode_id, max_claims + 1)
                if fact.space == space and fact.source_episode_id == episode_id]
    else:
        rows = []
        reasons.append("claims_unavailable")
    if len(rows) > max_claims:
        reasons.append("claim_limit")
    rows = rows[:max_claims]
    projection, read = await load_projection(engine, space, mode="all")
    found_reasons = read.get("reasons")
    reasons += [str(reason) for reason in found_reasons] if isinstance(found_reasons, list) else []
    roles = {role.fact_id: role for role in projection.roles}
    entities = {entity.entity_id: entity for entity in projection.entities}
    named: dict[str, list[int]] = {}
    claims: list[dict[str, object]] = []
    for fact in rows:
        span: dict[str, int] | None = None
        occurrences = 0
        if fact.quote:
            quote = fact.quote.encode("utf-8")
            start = body.find(quote)
            if start >= 0:
                span = {"start": start, "end": start + len(quote)}
                occurrences = body.count(quote)
        role = roles.get(fact.fact_id)
        ids = [entity_id for entity_id in (role.subject_id, role.object_id) if entity_id] if role else []
        for entity_id in ids:
            named.setdefault(entity_id, []).append(fact.fact_id)
        claims.append({
            "fact_id": fact.fact_id, "subject": fact.subject, "predicate": fact.predicate, "object": fact.object,
            "status": fact.status, "excluded": fact.excluded, "quote": fact.quote, "span": span,
            "occurrences": occurrences,
            "grounding": "source_unquoted" if not fact.quote else "quote_verified" if span else "quote_not_found",
            "section": section_at(span["start"]) if span else None,
            "chunks": [chunk.chunk_id for chunk in chunks if span and chunk.start < span["end"]
                       and span["start"] < chunk.end],
            "entities": ids})

    index = name_index(projection)
    words = _word_spans(content)
    mentions: dict[str, dict[str, object]] = {}
    position = 0
    while position < len(words):
        for size in range(min(_LONGEST, len(words) - position), 0, -1):
            run = words[position:position + size]
            matches = index.get(" ".join(word.casefold() for word, _, _ in run))
            if not matches or (size == 1 and not any(character.isupper() for character in run[0][0])):
                continue
            start, end = run[0][1], run[-1][2]
            for entity in matches:
                if entity.entity_id in named:
                    continue
                found = mentions.setdefault(entity.entity_id, {
                    "id": entity.entity_id, "key": entity.key, "label": entity.label,
                    "span": {"start": start, "end": end}, "occurrences": 0})
                found["occurrences"] = int(str(found["occurrences"])) + 1
            position += size
            break
        else:
            position += 1
    return {
        "schema_version": 1, "space": space,
        "episode": {"id": episode.episode_id, "kind": episode.kind, "source": episode.source,
                    "created_at": episode.created_at, "bytes": len(body),
                    "content_sha256": hashlib.sha256(body).hexdigest()},
        "sections": sections, "chunks": shown_chunks, "claims": claims,
        "entities": [{"id": entity_id, "key": entities[entity_id].key, "label": entities[entity_id].label,
                      "kind": entities[entity_id].kind, "claims": fact_ids}
                     for entity_id, fact_ids in sorted(named.items()) if entity_id in entities],
        "mentions": sorted(mentions.values(), key=lambda mention: str(mention["id"])),
        "coverage": {"chunks_total": len(chunks), "chunks_shown": len(shown_chunks), "claims_shown": len(claims),
                     "truncated": bool(reasons), "reasons": reasons, "read": read},
    }
