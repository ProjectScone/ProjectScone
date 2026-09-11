"""An entity lane for recall: passages naming what a question is about, or
the things one relation away from it.

A question often names one thing while its answer sits in a passage about
a neighbour: "Which city is Alice Chen's employer in?" is answered by
"Acme Robotics is headquartered in Lisbon", which shares no word with it.
The ledger records that Alice Chen works at Acme Robotics. The lane:

1. finds the entities the question names (the longest name at each
   position, common words never counting);
2. adds up to three neighbours of each, those with the most facts behind
   the relation first, at most twelve entities in all;
3. makes one lexical search for passages naming any of them, under the
   same filter as the other lanes;
4. keeps a passage only when its text really names one of them: a run of
   its words, read as names and questions are, equals a spelling of one.
   Passages naming a neighbour but no entity of the question rank first:
   the second hop the other lanes cannot reach, while passages naming the
   question's own entities are found by them anyway.

The lane is fused by reciprocal rank at ``ENTITY_WEIGHT``, so its best
second-hop passage can stand beside results two lanes agree on. It is
off unless a recall asks for it, and it reads only a projection the
caller supplies.
"""

from __future__ import annotations

from collections import defaultdict

from ..core.models import QueryEntity
from ..core.ports import DocumentStore, TextFilter
from ..core.validation import MAX_QUERY
from ..entities.context import mentioned
from ..entities.project import Entity, EntityProjection
from ..entities.query import name_words

MAX_SEEDS = 4
NEIGHBOURS_PER_SEED = 3
MAX_ENTITIES = 12
#: The lane's weight in reciprocal-rank fusion beside the text and vector
#: lanes' 1. Judged on the bench; the lane is off unless asked for.
ENTITY_WEIGHT = 2.0


def lane_entities(projection: EntityProjection, query: str) -> list[QueryEntity]:
    """The question's entities and their strongest neighbours."""
    entities = {entity.entity_id: entity for entity in projection.entities}
    seeds = mentioned(projection, query, limit=MAX_SEEDS)
    chosen: list[QueryEntity] = [QueryEntity(entity_id=seed.entity_id, key=seed.key, label=seed.label, role="seed",
                                             matched=seed.label) for seed in seeds]
    taken = {seed.entity_id for seed in seeds}
    touching: dict[str, list[tuple[int, str, str]]] = defaultdict(list)
    for relation in projection.relations:
        if relation.subject_id != relation.object_id:
            touching[relation.subject_id].append((len(relation.fact_ids), relation.object_id, relation.predicate))
            touching[relation.object_id].append((len(relation.fact_ids), relation.subject_id, relation.predicate))
    for seed in seeds:
        added = 0
        for _support, far, predicate in sorted(touching[seed.entity_id], key=lambda item: (-item[0], item[1])):
            if len(chosen) >= MAX_ENTITIES or added >= NEIGHBOURS_PER_SEED:
                break
            if far in taken:
                continue
            taken.add(far)
            added += 1
            entity = entities[far]
            chosen.append(QueryEntity(entity_id=far, key=entity.key, label=entity.label, role="neighbour",
                                      matched=predicate))
    return chosen


_LONGEST_NAME = 6  # words


def name_phrases(entity: Entity) -> set[str]:
    """Every spelling of an entity as its words, the way names are matched."""
    return {phrase for phrase in (" ".join(name_words(spelling)) for spelling in
                                  (entity.key, entity.label, *(form.text for form in entity.surface_forms))) if phrase}


def text_phrases(text: str) -> set[str]:
    """Every run of up to six words in a passage, read with the same words
    as names and questions: a combining mark stays with its letter and a
    symbol inside a word (C++, R&D) stays part of it."""
    words = name_words(text)
    return {" ".join(words[start:start + size]) for start in range(len(words))
            for size in range(1, min(_LONGEST_NAME, len(words) - start) + 1)}


async def entity_lane(documents: DocumentStore, projection: EntityProjection, space: str, query: str, depth: int,
                      scope: TextFilter) -> tuple[list[tuple[int, float]], list[QueryEntity]]:
    """Ranked chunks naming the question's entities or their neighbours, and
    the entities searched for. Exactly one lexical search."""
    chosen = lane_entities(projection, query)
    if not chosen:
        return [], []
    by_id = {entity.entity_id: entity for entity in projection.entities}
    seeds = {phrase for item in chosen if item.role == "seed" for phrase in name_phrases(by_id[item.entity_id])}
    neighbours = {phrase for item in chosen if item.role == "neighbour"
                  for phrase in name_phrases(by_id[item.entity_id])}

    lane_query = " ".join(dict.fromkeys(item.label for item in chosen))[:MAX_QUERY]
    hits = await documents.search_text(space, lane_query, depth, scope)
    if not hits:
        return [], chosen
    texts = {chunk.chunk_id: text_phrases(chunk.text)
             for chunk in await documents.get_chunks(space, [cid for cid, _ in hits])}
    second_hop, first_hop = [], []
    for chunk_id, score in hits:
        phrases = texts.get(chunk_id)
        if phrases is None:
            continue
        names_seed = not seeds.isdisjoint(phrases)
        names_neighbour = not neighbours.isdisjoint(phrases)
        if names_neighbour and not names_seed:
            second_hop.append((chunk_id, score))
        elif names_seed or names_neighbour:
            first_hop.append((chunk_id, score))
    return second_hop + first_hop, chosen
