"""Project a space's fact ledger into entities, relations and attributes.

A pure function of fact rows: no store, clock, model or randomness is read,
and the same rows in any order give byte-identical ids and digest. Subjects
are entities; objects are entities or values as ``classify`` decides.
Entity-to-entity claims become relations and value claims become
attributes, grouped with every fact that supports them, so each item in the
graph can be traced back to its claims. Declined facts never held and are
left out; proposed, closed and excluded facts are counted apart, and views
decide what to show.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
from functools import lru_cache
import hashlib
import json
import re
from typing import Iterable, Literal, Sequence, cast

from ..core.models import Fact
from ..core.timeutil import format_rfc3339, parse_rfc3339
from ..core.validation import entity_key
from .classify import CLASSIFIER_VERSION, ClassificationContext, ObjectClassification, classify_object, reference_flag
from .ids import attribute_id, implied_id, key_id, relation_id
from .meanings import MAX_IMPLIED, MAX_STEPS, MAX_WALKED, RelationMeanings
from .kinds import KIND_HINTS_VERSION, EntityKind, KindStatus, hint, infer_kind

PROJECTION_VERSION = "scone.entities/1"
_MAX_FORMS = 8

Grounding = Literal["quoted", "unquoted", "unsourced"]


@dataclass(frozen=True, slots=True)
class Support:
    """How many facts back an item, by status, grounding and origin."""

    facts: int = 0
    active: int = 0
    closed: int = 0
    proposed: int = 0
    excluded: int = 0
    quoted: int = 0
    unquoted: int = 0
    unsourced: int = 0
    stated: int = 0
    extracted: int = 0
    inferred: int = 0


@dataclass(frozen=True, slots=True)
class SurfaceForm:
    text: str
    count: int


@dataclass(frozen=True, slots=True)
class Entity:
    entity_id: str
    key: str
    label: str
    surface_forms: tuple[SurfaceForm, ...]
    kind: EntityKind | None
    kind_status: KindStatus
    kind_basis: tuple[int, ...]
    flags: tuple[str, ...]
    as_subject: int
    as_object: int


@dataclass(frozen=True, slots=True)
class FactRole:
    fact_id: int
    subject_id: str
    object_id: str | None
    classification: ObjectClassification
    predicate: str
    status: str
    excluded: bool
    origin: str
    grounding: Grounding
    valid_from: str
    valid_until: str | None
    source_episode_id: int | None


@dataclass(frozen=True, slots=True)
class Relation:
    relation_id: str
    subject_id: str
    predicate: str
    object_id: str
    fact_ids: tuple[int, ...]
    support: Support
    first_valid_from: str
    #: None while any in-ledger member still holds.
    last_valid_until: str | None


@dataclass(frozen=True, slots=True)
class Implied:
    """A relation nobody stated, which follows from ones they did.

    Its own type, so that no view, count or export can mistake it for a
    claim. It rests on the same facts as the relations it follows from,
    holds only while all of them do, and is no better supported than the
    weakest of them."""

    relation_id: str
    subject_id: str
    predicate: str
    object_id: str
    fact_ids: tuple[int, ...]
    support: Support
    first_valid_from: str
    last_valid_until: str | None
    #: "inverse", "symmetric" or "transitive".
    follows: str
    #: The stated relations it was worked out from, in order.
    follows_from: tuple[str, ...]
    #: Every stretch of valid time the claims under it actually shared,
    #: half-open and in order. This is the truth of when it held; the two
    #: fields above are the first beginning and the last ending of these,
    #: and a chain that held twice with a gap between says so here.
    periods: tuple[tuple[str, str | None], ...] = ()


@dataclass(frozen=True, slots=True)
class Attribute:
    attribute_id: str
    entity_id: str
    predicate: str
    value: str
    literal_kind: str | None
    fact_ids: tuple[int, ...]
    support: Support


@dataclass(frozen=True)
class EntityProjection:
    space: str
    revision: int
    entities: tuple[Entity, ...]
    relations: tuple[Relation, ...]
    attributes: tuple[Attribute, ...]
    roles: tuple[FactRole, ...]
    digest: str
    version: str = PROJECTION_VERSION
    #: What follows from the stated relations under the space's vocabulary,
    #: kept apart from them so that what was said is never confused with
    #: what was worked out. Empty when no meanings are configured.
    implied: tuple[Implied, ...] = ()
    #: Whether more followed than a projection will hold.
    implied_capped: bool = False
    #: The vocabulary the implications were worked out under, so a view can
    #: say what it applied and where it stopped. None when none was given.
    meanings: RelationMeanings | None = None
    #: Where those meanings came from: ``space`` when the space holds its
    #: own vocabulary, ``process`` when they are the reading process's
    #: configuration, ``none`` when there are none. Carried on the
    #: projection so every view that serialises one can say it, rather
    #: than each route having to ask separately -- and so the claim that
    #: an answer says where its vocabulary came from is true of every
    #: answer built from a projection.
    vocabulary_source: str = "none"
    vocabulary_why: str = ""

    def components(self) -> list[frozenset[str]]:
        """Groups of entities connected by relations in either direction."""
        parent = {entity.entity_id: entity.entity_id for entity in self.entities}

        def root(item: str) -> str:
            while parent[item] != item:
                parent[item] = parent[parent[item]]
                item = parent[item]
            return item

        for relation in self.relations:
            a, b = root(relation.subject_id), root(relation.object_id)
            if a != b:
                parent[max(a, b)] = min(a, b)
        groups: dict[str, set[str]] = defaultdict(set)
        for item in parent:
            groups[root(item)].add(item)
        return sorted((frozenset(group) for group in groups.values()), key=lambda group: (-len(group), min(group)))


@lru_cache(maxsize=4096)
def _moment(text: str) -> str:
    # A space's facts share few distinct instants; each is parsed once.
    return format_rfc3339(parse_rfc3339(text))


def _grounding(fact: Fact) -> Grounding:
    if fact.source_episode_id is None:
        return "unsourced"
    return "quoted" if fact.quote else "unquoted"


def _support(facts: Iterable[Fact]) -> Support:
    counts: Counter[str] = Counter()
    for fact in facts:
        counts["facts"] += 1
        counts[fact.status] += 1
        counts["excluded"] += fact.excluded
        counts[_grounding(fact)] += 1
        counts[fact.origin] += 1
    return Support(**{name: counts[name] for name in Support.__slots__})


def classification_context(claims: Iterable[tuple[str, str]]) -> ClassificationContext:
    """What the classifier needs from a set of (subject, object) claims:
    every subject's key anchors, and an object two or more subjects share
    counts as shared. Every view classifies through this, so the same
    claims come out the same everywhere."""
    anchors: set[str] = set()
    sharers: dict[str, set[str]] = defaultdict(set)
    for subject, obj in claims:
        anchors.add(entity_key(subject))
        sharers[entity_key(obj)].add(entity_key(subject))
    return ClassificationContext(anchors=frozenset(anchors),
                                 shared_objects=frozenset(key for key, subjects in sharers.items() if len(subjects) > 1))


def quoted_form(key: str, quote: str | None) -> str | None:
    """The quote's own spelling of a key: case and spacing as written."""
    if not quote:
        return None
    pattern = r"(?<!\w)" + r"\s+".join(re.escape(token) for token in key.split()) + r"(?!\w)"
    found = re.search(pattern, quote, re.IGNORECASE)
    return found.group(0) if found else None


def _label(forms: Counter[str], key: str) -> str:
    return min(forms, key=lambda form: (-forms[form], not any(c.isupper() for c in form), form)) if forms else key


def _canonical(value: object) -> bytes:
    return json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"),
                      default=list).encode("utf-8", "surrogatepass")


def project_entities(space: str, facts: Iterable[Fact], *, revision: int,
                     meanings: RelationMeanings | None = None,
                     vocabulary_source: str = "none",
                     vocabulary_why: str = "") -> EntityProjection:
    rows = sorted(facts, key=lambda fact: fact.fact_id)
    for fact in rows:
        if fact.space != space:
            raise ValueError(f"fact {fact.fact_id} belongs to space {fact.space!r}, not {space!r}")
    held = [fact for fact in rows if fact.status != "declined"]
    context = classification_context((fact.subject, fact.object) for fact in held)

    forms: dict[str, Counter[str]] = defaultdict(Counter)
    hints: dict[str, list[tuple[EntityKind, int]]] = defaultdict(list)
    roles_count: dict[str, Counter[str]] = defaultdict(Counter)
    roles: list[FactRole] = []
    relation_facts: dict[tuple[str, str, str], list[Fact]] = defaultdict(list)
    attribute_facts: dict[tuple[str, str, str], list[Fact]] = defaultdict(list)
    literal_kinds: dict[tuple[str, str, str], str | None] = {}
    for fact in held:
        subject_key = entity_key(fact.subject)
        subject_id = key_id(space, subject_key)
        forms[subject_key][quoted_form(subject_key, fact.quote) or fact.subject.strip()] += 1
        roles_count[subject_key]["subject"] += 1
        if (kind := hint(fact.predicate, "subject")) is not None:
            hints[subject_key].append((kind, fact.fact_id))
        classification = classify_object(fact.object, fact.predicate, context)
        object_id = None
        if classification.object_class == "entity":
            object_key = entity_key(fact.object)
            object_id = key_id(space, object_key)
            forms[object_key][fact.object.strip()] += 1
            roles_count[object_key]["object"] += 1
            if (kind := hint(fact.predicate, "object")) is not None:
                hints[object_key].append((kind, fact.fact_id))
            relation_facts[(subject_id, fact.predicate, object_id)].append(fact)
        else:
            group = (subject_id, fact.predicate, fact.object.strip())
            attribute_facts[group].append(fact)
            literal_kinds[group] = classification.literal_kind
        roles.append(FactRole(
            fact.fact_id, subject_id, object_id, classification, fact.predicate, fact.status, fact.excluded,
            fact.origin, _grounding(fact), _moment(fact.valid_from),
            None if fact.valid_until is None else _moment(fact.valid_until), fact.source_episode_id))

    entities = []
    for key in sorted(forms):
        kind, status, basis = infer_kind(hints[key])
        flag = reference_flag(key)
        ordered = sorted(forms[key].items(), key=lambda item: (-item[1], item[0]))[:_MAX_FORMS]
        entities.append(Entity(
            key_id(space, key), key, _label(forms[key], key), tuple(SurfaceForm(text, count) for text, count in ordered),
            kind, status, basis, () if flag is None else (flag,),
            roles_count[key]["subject"], roles_count[key]["object"]))

    relations = []
    spans: dict[str, tuple[tuple[str, str | None], ...]] = {}
    for (subject_id, predicate, object_id), members in relation_facts.items():
        in_ledger = [fact for fact in members if fact.in_ledger]
        ends = [fact.valid_until for fact in in_ledger]
        spans[relation_id(space, subject_id, predicate, object_id)] = merged_periods(
            [(_moment(fact.valid_from), _moment(fact.valid_until) if fact.valid_until else None)
             for fact in in_ledger])
        relations.append(Relation(
            relation_id(space, subject_id, predicate, object_id), subject_id, predicate, object_id,
            tuple(fact.fact_id for fact in members), _support(members),
            min(_moment(fact.valid_from) for fact in members),
            None if not in_ledger or any(end is None for end in ends) else max(_moment(e) for e in ends if e)))
    attributes = [
        Attribute(attribute_id(space, entity_id, predicate, value), entity_id, predicate, value,
                  literal_kinds[(entity_id, predicate, value)], tuple(fact.fact_id for fact in members),
                  _support(members))
        for (entity_id, predicate, value), members in attribute_facts.items()]
    entities.sort(key=lambda entity: entity.entity_id)
    relations.sort(key=lambda relation: relation.relation_id)
    attributes.sort(key=lambda attribute: attribute.attribute_id)
    implied, capped = _implied(space, relations, spans, meanings)
    digest = hashlib.sha256(_canonical({
        "version": [PROJECTION_VERSION, CLASSIFIER_VERSION, KIND_HINTS_VERSION], "space": space,
        "entities": [_plain(item) for item in entities], "relations": [_plain(item) for item in relations],
        "attributes": [_plain(item) for item in attributes], "roles": [_plain(item) for item in roles],
        **({"meanings": meanings.record(), "implied": [_plain(item) for item in implied]}
           if meanings else {}),
    })).hexdigest()
    return EntityProjection(space, revision, tuple(entities), tuple(relations), tuple(attributes), tuple(roles),
                            digest, implied=tuple(implied), implied_capped=capped, meanings=meanings,
                            vocabulary_source=vocabulary_source, vocabulary_why=vocabulary_why)


def _implied(space: str, relations: list[Relation], spans: dict[str, tuple[tuple[str, str | None], ...]],
             meanings: RelationMeanings | None) -> tuple[list[Implied], bool]:
    """What follows from the stated relations under the space's vocabulary.

    Nothing already said is implied, nothing is implied twice, and nothing
    is implied about a thing and itself. A chain holds only over the
    stretches of time its claims actually shared, worked out from the
    claims themselves rather than from the first and last moment of each
    relation, because a relation made of two spells with a gap between
    them does not hold during the gap. A chain that shares no moment with
    itself is not a chain at all and is never recorded.

    The walk is bounded twice over: a thing is reached by the shortest
    route to it and never expanded again, and the whole walk stops after
    MAX_WALKED claims are examined. Either bound, and the cap on how many
    implications a projection holds, is reported rather than assumed."""
    if not meanings or not relations:
        return [], False
    said = {(item.subject_id, item.predicate, item.object_id) for item in relations}
    found: dict[tuple[str, str, str], Implied] = {}
    capped = False
    walked = 0

    def keep(subject: str, predicate: str, other: str, follows: str,
             path: tuple[Relation, ...], shared: tuple[tuple[str, str | None], ...]) -> bool:
        """Record one implied relation, when its claims ever held together.
        False when there is no room left for another."""
        nonlocal capped
        key = (subject, predicate, other)
        if subject == other or key in said or key in found or not shared:
            return True
        if len(found) >= MAX_IMPLIED:
            capped = True
            return False
        # It is no better grounded than its least grounded claim: a chain
        # with one unsourced link is an unsourced chain, whatever the rest
        # of it was quoted from.
        weakest = min(path, key=lambda item: (item.support.quoted, item.support.unquoted,
                                              item.support.facts, item.relation_id))
        found[key] = Implied(
            implied_id(space, subject, predicate, other), subject, predicate, other,
            tuple(sorted({fact for item in path for fact in item.fact_ids})), weakest.support,
            shared[0][0], shared[-1][1], follows, tuple(item.relation_id for item in path), shared)
        return True

    for item in relations:
        other_side = meanings.opposite(item.predicate)
        if other_side is not None and not keep(item.object_id, other_side, item.subject_id,
                                               "symmetric" if meanings.reads_both_ways(item.predicate)
                                               else "inverse", (item,),
                                               spans.get(item.relation_id, ())):
            break
    onward: dict[tuple[str, str], list[Relation]] = defaultdict(list)
    for item in relations:
        if meanings.carries_through(item.predicate):
            onward[(item.predicate, item.subject_id)].append(item)
    for item in relations:
        if not meanings.carries_through(item.predicate) or capped:
            continue
        # Each thing is reached by the shortest route from this claim and
        # never expanded twice: a dense graph has more paths than anyone
        # can walk, and walking them all finds nothing a shorter route did
        # not already find.
        reached = {item.subject_id, item.object_id}
        frontier: list[tuple[str, tuple[Relation, ...]]] = [(item.object_id, (item,))]
        for _ in range(MAX_STEPS - 1):
            beyond: list[tuple[str, tuple[Relation, ...]]] = []
            for at, path in frontier:
                for edge in onward.get((item.predicate, at), ()):
                    walked += 1
                    if walked > MAX_WALKED:
                        capped = True
                        return sorted(found.values(), key=lambda r: r.relation_id), capped
                    if edge.object_id in reached:
                        continue
                    reached.add(edge.object_id)
                    step = (*path, edge)
                    shared = _shared(spans, step)
                    if not keep(item.subject_id, item.predicate, edge.object_id, "transitive", step, shared):
                        return sorted(found.values(), key=lambda r: r.relation_id), capped
                    # A stretch shared by every claim so far only ever
                    # shrinks, so a chain that already shares nothing can
                    # never share anything further along. Walking on would
                    # spend claims looking for an answer that cannot exist.
                    if shared:
                        beyond.append((edge.object_id, step))
            frontier = beyond
            if not frontier:
                break
    return sorted(found.values(), key=lambda r: r.relation_id), capped


def merged_periods(periods: Iterable[tuple[str, str | None]]) -> tuple[tuple[str, str | None], ...]:
    """One stretch of time per spell, overlapping and touching spells
    joined, in order. An open end swallows everything after it."""
    ordered = sorted(periods, key=lambda period: (period[0], period[1] is None, period[1] or ""))
    joined: list[tuple[str, str | None]] = []
    for start, end in ordered:
        if end is not None and end <= start:
            continue
        if joined and (joined[-1][1] is None or joined[-1][1] >= start):
            was = joined[-1]
            joined[-1] = (was[0], None if was[1] is None or end is None else max(was[1], end))
        else:
            joined.append((start, end))
    return tuple(joined)


def _shared(spans: dict[str, tuple[tuple[str, str | None], ...]],
            path: Sequence[Relation]) -> tuple[tuple[str, str | None], ...]:
    """The stretches of time every claim along a path held at once."""
    common = spans.get(path[0].relation_id, ())
    for step in path[1:]:
        if not common:
            return ()
        common = _overlap(common, spans.get(step.relation_id, ()))
    return common


def _overlap(left: Sequence[tuple[str, str | None]],
             right: Sequence[tuple[str, str | None]]) -> tuple[tuple[str, str | None], ...]:
    """Where two ordered sets of stretches meet."""
    shared: list[tuple[str, str | None]] = []
    one = other = 0
    while one < len(left) and other < len(right):
        stops, ends = left[one][1], right[other][1]
        start = max(left[one][0], right[other][0])
        finish = None if stops is None or ends is None else min(stops, ends)
        if finish is None:
            finish = stops if ends is None else ends
        if finish is None or start < finish:
            shared.append((start, finish))
        # Whichever stretch ends first cannot meet anything later.
        if stops is None:
            other += 1
        elif ends is None or stops <= ends:
            one += 1
        else:
            other += 1
    return tuple(shared)


_SCALARS = frozenset({str, int, float, bool, type(None)})


def _plain(item: object) -> object:
    # Most values are scalars; test the exact type first, the costly
    # attribute lookup only for the rest. The output is unchanged.
    kind = type(item)
    if kind in _SCALARS:
        return item
    if kind is tuple:
        return [_plain(value) for value in cast("tuple[object, ...]", item)]
    slots = getattr(kind, "__slots__", None)
    if slots is not None:
        return {name: _plain(getattr(item, name)) for name in slots}
    return item
