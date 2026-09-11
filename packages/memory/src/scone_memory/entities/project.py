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
import hashlib
import json
import re
from typing import Iterable, Literal

from ..core.models import Fact
from ..core.timeutil import format_rfc3339, parse_rfc3339
from ..core.validation import entity_key
from .classify import CLASSIFIER_VERSION, ClassificationContext, ObjectClassification, classify_object, reference_flag
from .ids import attribute_id, key_id, relation_id
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


def _moment(text: str) -> str:
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


def _quoted_form(key: str, quote: str | None) -> str | None:
    """The quote's own spelling of a key: case and spacing as written."""
    if not quote:
        return None
    pattern = r"(?<!\w)" + r"\s+".join(re.escape(token) for token in key.split()) + r"(?!\w)"
    found = re.search(pattern, quote, re.IGNORECASE)
    return found.group(0) if found else None


def _label(forms: Counter[str], key: str) -> str:
    return min(forms, key=lambda form: (-forms[form], not any(c.isupper() for c in form), form)) if forms else key


def _canonical(value: object) -> bytes:
    return json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"), default=list).encode("utf-8")


def project_entities(space: str, facts: Iterable[Fact], *, revision: int) -> EntityProjection:
    rows = sorted(facts, key=lambda fact: fact.fact_id)
    for fact in rows:
        if fact.space != space:
            raise ValueError(f"fact {fact.fact_id} belongs to space {fact.space!r}, not {space!r}")
    held = [fact for fact in rows if fact.status != "declined"]
    anchors = frozenset(entity_key(fact.subject) for fact in held)
    sharers: dict[str, set[str]] = defaultdict(set)
    for fact in held:
        sharers[entity_key(fact.object)].add(entity_key(fact.subject))
    context = ClassificationContext(anchors=anchors,
                                    shared_objects=frozenset(key for key, subjects in sharers.items() if len(subjects) > 1))

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
        forms[subject_key][_quoted_form(subject_key, fact.quote) or fact.subject.strip()] += 1
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
    for (subject_id, predicate, object_id), members in relation_facts.items():
        in_ledger = [fact for fact in members if fact.in_ledger]
        ends = [fact.valid_until for fact in in_ledger]
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
    digest = hashlib.sha256(_canonical({
        "version": [PROJECTION_VERSION, CLASSIFIER_VERSION, KIND_HINTS_VERSION], "space": space,
        "entities": [_plain(item) for item in entities], "relations": [_plain(item) for item in relations],
        "attributes": [_plain(item) for item in attributes], "roles": [_plain(item) for item in roles],
    })).hexdigest()
    return EntityProjection(space, revision, tuple(entities), tuple(relations), tuple(attributes), tuple(roles), digest)


def _plain(item: object) -> object:
    if hasattr(item, "__slots__"):
        return {name: _plain(getattr(item, name)) for name in item.__slots__}
    if isinstance(item, tuple):
        return [_plain(value) for value in item]
    return item
