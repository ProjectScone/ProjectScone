"""Entities a question resembles, found by vector when it names none.

Graph context finds its seeds by name: the runs of a question's words that
are an entity's name. A question can ask about a thing without naming it
("which manufacturing firm do we know?"), so each entity is also described
in one line (its label, its kind, its strongest relations both ways and its
values) and embedded with the engine's own embedder; the entities nearest
the question's vector become seeds too.

Lines are embedded once each: vectors are kept by the embedder's id and the
line, so an entity whose line has not changed is never embedded again, in
any projection. Only the most connected ``MAX_SIMILAR_ENTITIES`` are
considered, and the rest are counted, because with a remote embedder every
new line is a call. No floor is guessed here: each seed carries its score,
and a caller who has measured one passes ``min_similarity``.
"""

from __future__ import annotations

from collections import OrderedDict, defaultdict
import hashlib
import math
from typing import TYPE_CHECKING, Iterable, Sequence

from .project import Entity, EntityProjection

if TYPE_CHECKING:
    from ..memory.engine import MemoryEngine

#: Entities, most connected first, whose lines a question is compared with.
MAX_SIMILAR_ENTITIES = 5_000
#: Relations each way and values that describe an entity.
_PER_SIDE = 8
_BATCH = 256
#: Vectors kept across calls, least recently used dropped first.
_KEPT = 50_000
_VECTORS: OrderedDict[str, tuple[float, ...]] = OrderedDict()


def _words(predicate: str) -> str:
    return " ".join(predicate.replace("_", " ").split())


def entity_lines(projection: EntityProjection) -> dict[str, str]:
    """One line per entity, for embedding rather than reading: its label and
    kind, then what it points at, its values and what points at it, each
    side strongest first. Nothing here is cited; a seed found by it is
    cited by the packet like any other."""
    label = {entity.entity_id: entity.label for entity in projection.entities}
    outgoing: dict[str, list[tuple[int, str]]] = defaultdict(list)
    incoming: dict[str, list[tuple[int, str]]] = defaultdict(list)
    values: dict[str, list[tuple[int, str]]] = defaultdict(list)
    for relation in projection.relations:
        weight = -len(relation.fact_ids)
        outgoing[relation.subject_id].append((weight, f"{_words(relation.predicate)} {label[relation.object_id]}"))
        incoming[relation.object_id].append((weight, f"{label[relation.subject_id]} {_words(relation.predicate)} it"))
    for attribute in projection.attributes:
        values[attribute.entity_id].append((-len(attribute.fact_ids), f"{_words(attribute.predicate)} {attribute.value}"))

    def strongest(items: list[tuple[int, str]]) -> list[str]:
        return [text for _, text in sorted(items)[:_PER_SIDE]]

    lines: dict[str, str] = {}
    for entity in projection.entities:
        head = f"{entity.label}, {entity.kind}" if entity.kind else entity.label
        parts = [*strongest(outgoing[entity.entity_id]), *strongest(values[entity.entity_id]),
                 *strongest(incoming[entity.entity_id])]
        lines[entity.entity_id] = f"{head}: {'; '.join(parts)}" if parts else head
    return lines


def _key(embedder_id: str, text: str) -> str:
    return hashlib.sha256(f"{embedder_id}\0{text}".encode("utf-8", "surrogatepass")).hexdigest()


async def _vectors(engine: "MemoryEngine", texts: Sequence[str]) -> list[tuple[float, ...]]:
    """Vectors for the texts, embedding only those not kept already."""
    embedder = engine.embedder
    keys = [_key(embedder.id, text) for text in texts]
    missing = list(dict.fromkeys((key, text) for key, text in zip(keys, texts) if key not in _VECTORS))
    for start in range(0, len(missing), _BATCH):
        batch = missing[start:start + _BATCH]
        for (key, _), vector in zip(batch, await embedder.embed([text for _, text in batch])):
            _VECTORS[key] = tuple(vector)
    found = []
    for key in keys:
        _VECTORS.move_to_end(key)
        found.append(_VECTORS[key])
    while len(_VECTORS) > _KEPT:
        _VECTORS.popitem(last=False)
    return found


def _cosine(left: Sequence[float], right: Sequence[float]) -> float:
    norm = math.sqrt(sum(x * x for x in left)) * math.sqrt(sum(y * y for y in right))
    return sum(x * y for x, y in zip(left, right)) / norm if norm else 0.0


async def similar_entities(engine: "MemoryEngine", projection: EntityProjection, question: str, *, limit: int,
                           min_similarity: float | None = None,
                           exclude: Iterable[str] = ()) -> tuple[list[tuple[Entity, float]], int]:
    """The ``limit`` entities whose lines are nearest the question, closest
    first, above zero and at least ``min_similarity`` when given, skipping
    ``exclude``; and how many entities were left uncompared by the cap."""
    if limit < 1:
        return [], 0
    ranked = sorted(projection.entities, key=lambda entity: (-(entity.as_subject + entity.as_object), entity.entity_id))
    considered, cut = ranked[:MAX_SIMILAR_ENTITIES], max(0, len(ranked) - MAX_SIMILAR_ENTITIES)
    skipped = set(exclude)
    candidates = [entity for entity in considered if entity.entity_id not in skipped]
    if not candidates:
        return [], cut
    lines = entity_lines(projection)
    vectors = await _vectors(engine, [question, *(lines[entity.entity_id] for entity in candidates)])
    scored = sorted(((_cosine(vectors[0], vector), entity) for entity, vector in zip(candidates, vectors[1:])),
                    key=lambda item: (-item[0], item[1].entity_id))
    return [(entity, round(score, 3)) for score, entity in scored
            if score > 0 and (min_similarity is None or score >= min_similarity)][:limit], cut
