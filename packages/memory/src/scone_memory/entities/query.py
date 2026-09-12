"""Asking the entity graph: which entity a name means, what surrounds it,
and how two entities connect.

All answers come from recorded relations and cite the facts behind them.
A name that could mean several entities is reported as ambiguous with every
candidate, never resolved by guessing. A path keeps each relation's
direction, and heavily linked hubs are never used as shortcuts between
other entities.
"""

from __future__ import annotations

from collections import OrderedDict, defaultdict, deque
from dataclasses import dataclass
import re
from typing import Literal
import unicodedata

from ..core.validation import entity_key
from .project import Attribute, Entity, EntityProjection, Implied, Relation

ResolveTier = Literal["id", "key", "variant", "prefix", "tokens"]


@dataclass(frozen=True)
class Candidate:
    entity_id: str
    key: str
    label: str


@dataclass(frozen=True)
class Resolution:
    status: Literal["resolved", "ambiguous", "not_found"]
    tier: ResolveTier | None
    #: The first ``limit`` candidates by key; ``total`` counts them all.
    candidates: tuple[Candidate, ...]
    total: int = 0


def _candidate(entity: Entity) -> Candidate:
    return Candidate(entity.entity_id, entity.key, entity.label)


#: Words a spelling may lead with that do not name the thing itself. Not
#: "st": St. Louis is a city, not Louis.
_LEADING = frozenset({"the", "a", "an", "dr", "mr", "mrs", "ms", "mx", "miss", "prof", "sir", "dame"})
#: Punctuation that opens or closes a phrase in running text. Only these
#: leave a word, and only from the side they sit on: a leading dot or a
#: path prefix (.env, ../config) and symbols inside a name (C#, C++,
#: node.js) are part of it and kept.
_OPENING = "\"'([{\u2018\u201c\u00ab"
_CLOSING = ".,;:!?\"')]}\u2019\u201d\u00bb"
_POSSESSIVE = re.compile(r"['\u2019]s$")


def name_words(text: str) -> list[str]:
    """The words of a name or a question as names are matched: compatibility
    forms unified (NFKC), case folded, opening and closing punctuation and a
    possessive stripped from their sides of each word. Symbols inside a
    word (C#, C++, node.js, ../config) are part of it and kept."""
    words = []
    for word in unicodedata.normalize("NFKC", text).casefold().split():
        word = _POSSESSIVE.sub("", word.lstrip(_OPENING).rstrip(_CLOSING)).rstrip(_CLOSING)
        if word:
            words.append(word)
    return words


def variant_fold(name: str) -> str:
    """A spelling reduced for lookup: its ``name_words`` with a leading
    article or title dropped, so "Dr. Alice Chen" and "ACME, Inc." meet
    "alice chen" and "acme inc". Lookup only: identity stays with
    ``entity_key``."""
    words = name_words(name)
    while len(words) > 1 and words[0] in _LEADING:
        words = words[1:]
    return " ".join(words)


_VARIANTS: OrderedDict[str, dict[str, tuple[Entity, ...]]] = OrderedDict()


def _variants(projection: EntityProjection) -> dict[str, tuple[Entity, ...]]:
    found = _VARIANTS.get(projection.digest)
    if found is None:
        grouped: dict[str, list[Entity]] = defaultdict(list)
        for entity in projection.entities:
            grouped[variant_fold(entity.key)].append(entity)
        found = {variant: tuple(items) for variant, items in grouped.items()}
        _VARIANTS[projection.digest] = found
        while len(_VARIANTS) > 4:
            _VARIANTS.popitem(last=False)
    else:
        _VARIANTS.move_to_end(projection.digest)
    return found


def resolve(projection: EntityProjection, name: str, *, limit: int = 20) -> Resolution:
    """The entity a name or id means, by the first tier that matches:
    the id itself; the same key (case and spacing folded); the same
    variant (titles, possessives and punctuation aside); a key the name
    begins at a word boundary; a key holding every word of the name."""
    wanted = entity_key(name)
    entities = sorted(projection.entities, key=lambda entity: entity.key)
    variant = variant_fold(name)
    tiers: list[tuple[ResolveTier, list[Entity]]] = [
        ("id", [entity for entity in entities if entity.entity_id == name.strip()]),
        ("key", [entity for entity in entities if entity.key == wanted]),
        ("variant", sorted(_variants(projection).get(variant, ()), key=lambda entity: entity.key) if variant else []),
        ("prefix", [entity for entity in entities if wanted and entity.key.startswith(wanted + " ")]),
        ("tokens", [entity for entity in entities
                    if wanted and set(wanted.split()) <= set(entity.key.split())]),
    ]
    for tier, matches in tiers:
        if matches:
            status: Literal["resolved", "ambiguous"] = "resolved" if len(matches) == 1 else "ambiguous"
            return Resolution(status, tier, tuple(_candidate(entity) for entity in matches[:limit]), len(matches))
    return Resolution("not_found", None, (), 0)


@dataclass(frozen=True)
class Hop:
    relation_id: str
    subject_id: str
    predicate: str
    object_id: str
    #: "forward" when the path follows the relation from subject to object.
    direction: Literal["forward", "reverse"]
    fact_ids: tuple[int, ...]


@dataclass(frozen=True)
class Path:
    entity_ids: tuple[str, ...]
    hops: tuple[Hop, ...]


@dataclass(frozen=True)
class PathResult:
    status: Literal["found", "none_within_limit", "disconnected", "unknown_entity"]
    paths: tuple[Path, ...]
    hubs_skipped: tuple[str, ...]
    truncated: bool


def _links(projection: EntityProjection) -> dict[str, dict[str, Relation]]:
    """For each entity, its neighbours and the strongest relation to each."""
    links: dict[str, dict[str, Relation]] = defaultdict(dict)
    for relation in sorted(projection.relations, key=lambda relation: (-len(relation.fact_ids), relation.relation_id)):
        if relation.subject_id == relation.object_id:
            continue
        links[relation.subject_id].setdefault(relation.object_id, relation)
        links[relation.object_id].setdefault(relation.subject_id, relation)
    return links


def paths_between(projection: EntityProjection, source: str, target: str, *, max_hops: int = 4,
                  limit: int = 3, hub_degree: int = 200) -> PathResult:
    """Up to ``limit`` distinct shortest routes from source to target.

    Relations are walked in either direction and each hop says which. An
    entity with more than ``hub_degree`` neighbours is never passed through
    (only an endpoint may be a hub), so a person connected to everything does
    not make every pair of entities look related.
    """
    known = {entity.entity_id for entity in projection.entities}
    if source not in known or target not in known:
        return PathResult("unknown_entity", (), (), False)
    links = _links(projection)
    depth, parents = {source: 0}, defaultdict(list)
    skipped: set[str] = set()
    queue = deque([source])
    while queue:
        node = queue.popleft()
        if node not in (source, target) and len(links[node]) > hub_degree:
            skipped.add(node)
            continue
        # Explore the whole component, so a target beyond max_hops is told
        # apart from one that cannot be reached at all.
        for neighbour in sorted(links[node]):
            if neighbour not in depth:
                depth[neighbour] = depth[node] + 1
                queue.append(neighbour)
            if depth[neighbour] == depth[node] + 1:
                parents[neighbour].append(node)
    if target not in depth:
        return PathResult("disconnected", (), tuple(sorted(skipped)), False)
    if depth[target] > max_hops:
        return PathResult("none_within_limit", (), tuple(sorted(skipped)), False)

    routes: list[list[str]] = []
    truncated = False

    def back(node: str, trail: list[str]) -> None:
        nonlocal truncated
        if len(routes) >= limit:
            truncated = True
            return
        if node == source:
            routes.append([source, *reversed(trail)])
            return
        for parent in sorted(parents[node]):
            back(parent, [*trail, node])

    back(target, [])
    paths = []
    for route in routes:
        hops = []
        for left, right in zip(route, route[1:]):
            relation = links[left][right]
            hops.append(Hop(relation.relation_id, relation.subject_id, relation.predicate, relation.object_id,
                            "forward" if relation.subject_id == left else "reverse", relation.fact_ids))
        paths.append(Path(tuple(route), tuple(hops)))
    return PathResult("found", tuple(paths), tuple(sorted(skipped)), truncated)


@dataclass(frozen=True)
class Neighbourhood:
    entity: Entity
    outgoing: tuple[Relation, ...]
    incoming: tuple[Relation, ...]
    attributes: tuple[Attribute, ...]
    #: What follows from claims about this entity under the space's
    #: vocabulary, either way round. Kept apart from what was said.
    follows: tuple["Implied", ...] = ()
    #: Counted claims the entity takes part in, as subject or object.
    claims: int = 0
    relations_total: int = 0
    #: How many follow from claims about it, before any limit.
    follows_total: int = 0
    truncated: bool = False


def neighbourhood(projection: EntityProjection, entity_id: str, *, limit: int = 100) -> Neighbourhood | None:
    """An entity's relations in both directions and its values, strongest first."""
    entity = next((item for item in projection.entities if item.entity_id == entity_id), None)
    if entity is None:
        return None
    strongest = lambda relation: (-len(relation.fact_ids), relation.predicate, relation.relation_id)  # noqa: E731
    outgoing = sorted((r for r in projection.relations if r.subject_id == entity_id), key=strongest)
    incoming = sorted((r for r in projection.relations if r.object_id == entity_id and r.subject_id != entity_id),
                      key=strongest)
    attributes = sorted((a for a in projection.attributes if a.entity_id == entity_id),
                        key=lambda attribute: (attribute.predicate, attribute.value))
    follows = sorted((item for item in projection.implied
                      if entity_id in (item.subject_id, item.object_id)), key=strongest)
    total = len(outgoing) + len(incoming)
    claims = sum(1 for role in projection.roles if entity_id in (role.subject_id, role.object_id))
    return Neighbourhood(entity, tuple(outgoing[:limit]), tuple(incoming[:limit]), tuple(attributes[:limit]),
                         tuple(follows[:limit]), claims, total, len(follows),
                         len(outgoing) > limit or len(incoming) > limit or len(attributes) > limit
                         or len(follows) > limit)
