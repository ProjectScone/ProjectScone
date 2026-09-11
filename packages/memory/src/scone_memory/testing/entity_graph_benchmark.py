"""Graph quality on a versioned synthetic fixture.

A fixture is JSON lines of four kinds, all synthetic:

- ``fact``: a claim to load (subject, predicate, object, optional
  valid_from);
- ``same_entity``: names that all mean one thing, the gold clusters;
- ``literal``: whether a fact's object should be a value (``true``) or a
  thing (``false``);
- ``path``: two names that should connect within ``hops``, or with
  ``"expected": false``, that should not.

Each row must be exactly what its kind allows: its required fields,
nothing unknown, text where text is due, ``true`` or ``false`` for
``value`` and ``expected``, whole hops from 1 to 4 and RFC 3339 times.
Gold must be about what the view holds at the fixture's ``as_of``: a
``same_entity`` name or a path end that no fact due by then names, or a
``literal`` on a claim that is not a fact or does not hold then, could
only be vacuously right. Such a fixture is refused with ``FixtureError``,
naming each line, instead of scored.

The report scores what a person would check by hand, from the evidence in
the projection. Something missing is never counted as right:

- ``claims_missing``: facts the view should hold at ``as_of`` that no
  relation or attribute carries. A fact is carried only by an item with
  its own subject, predicate and object (or value), not by any item
  citing its id. Which facts the view should hold is the view's own rule
  for the stored fact; a fact the store lost should be held once it has
  begun;
- ``claims_out_of_view``: facts that do not hold at ``as_of``, because
  they begin later or the ledger closed them, so the view rightly leaves
  them out;
- ``gold_names_missing`` and ``path_ends_missing``: gold names and path
  ends with no entity;
- ``cluster_fragments`` and ``fragmentation``: entities each gold cluster
  became, 1 being whole (a missing name is a fragment of its own);
- ``alias_bcubed_f1``: B-cubed F1 of the projection's entities against the
  gold clusters, over the gold names; a missing name adds nothing to
  precision or recall;
- ``literal_error_rate``: labelled objects not carried the labelled way,
  as an attribute for a value and a relation for a thing;
- ``connected_claim_share``: labelled claims between things that a
  relation carries;
- ``path_recall``: expected paths found within their hops, by hop count,
  and ``path_false_positives``, paths found where none should be;
- ``view_bytes``: the knowledge view, report and context packet sizes;
- ``build_ms_per_10k``: projection time scaled to 10,000 facts.

Everything but the timing is deterministic and hashed into
``artefact_sha256``; two runs of one fixture give the same hash.
``failures`` compares a report with thresholds that start at the recorded
baseline and only tighten. A score the fixture has no gold for is None,
unmeasured, and a threshold on it fails rather than passing on nothing.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import asdict, dataclass, field
from datetime import datetime
import hashlib
import json
from pathlib import Path
import time
from typing import Mapping

from ..core.errors import SconeError
from ..core.models import Fact
from ..core.timeutil import parse_rfc3339
from ..core.validation import check_space, entity_key
from ..entities.analysis import analyze_projection
from ..entities.context import graph_context
from ..entities.project import Attribute, Relation, project_entities
from ..entities.query import paths_between
from ..entities.read import load_projection
from ..entities.report import build_report
from ..entities.view import counts, knowledge_view

#: Recorded on fixtures-v1; lower is better for the *_max rows.
THRESHOLDS_V1: Mapping[str, float] = {
    "claims_missing_max": 0,
    "gold_names_missing_max": 0,
    "path_ends_missing_max": 0,
    "literal_error_rate_max": 0.0,
    "connected_claim_share_min": 1.0,
    "path_recall_min": 1.0,
    "path_false_positives_max": 0.0,
    "fragmentation_max": 1.2,
    "alias_bcubed_f1_min": 0.9,
}


class FixtureError(ValueError):
    """Gold that names something the fixture never loads."""


@dataclass(frozen=True)
class EntityGraphReport:
    fixture: str
    facts: int
    entities: int
    relations: int
    attributes: int
    claims_missing: int
    claims_out_of_view: int
    gold_names_missing: int
    path_ends_missing: int
    cluster_fragments: dict[str, int]
    fragmentation: float | None
    alias_bcubed_f1: float | None
    literal_error_rate: float | None
    connected_claim_share: float | None
    path_recall: dict[int, float]
    path_false_positives: int | None
    view_bytes: dict[str, int]
    build_ms_per_10k: float
    artefact_sha256: str = field(default="")

    def record(self) -> dict[str, object]:
        return asdict(self)


def failures(report: EntityGraphReport, thresholds: Mapping[str, float]) -> list[str]:
    """Each threshold the report breaches, as a readable line."""
    found: list[str] = []

    def check(name: str, value: float | None, limit: float, above: bool) -> None:
        if value is None:
            found.append(f"{name}: unmeasured, the fixture has no gold for it")
        elif (value > limit) if above else (value < limit):
            found.append(f"{name}: {value} {'>' if above else '<'} {limit}")

    check("claims_missing", report.claims_missing, thresholds["claims_missing_max"], True)
    check("gold_names_missing", report.gold_names_missing, thresholds["gold_names_missing_max"], True)
    check("path_ends_missing", report.path_ends_missing, thresholds["path_ends_missing_max"], True)
    check("literal_error_rate", report.literal_error_rate, thresholds["literal_error_rate_max"], True)
    check("connected_claim_share", report.connected_claim_share, thresholds["connected_claim_share_min"], False)
    if not report.path_recall:
        check("path_recall", None, thresholds["path_recall_min"], False)
    for hops, recall in sorted(report.path_recall.items()):
        check(f"path_recall[{hops}]", recall, thresholds["path_recall_min"], False)
    check("path_false_positives", report.path_false_positives, thresholds["path_false_positives_max"], True)
    check("fragmentation", report.fragmentation, thresholds["fragmentation_max"], True)
    check("alias_bcubed_f1", report.alias_bcubed_f1, thresholds["alias_bcubed_f1_min"], False)
    return found


#: Per kind: the fields a row must have, and the ones it may have.
_FIELDS: Mapping[str, tuple[frozenset[str], frozenset[str]]] = {
    "meta": (frozenset(), frozenset({"name", "space", "as_of"})),
    "fact": (frozenset({"subject", "predicate", "object"}), frozenset({"valid_from"})),
    "same_entity": (frozenset({"names"}), frozenset()),
    "literal": (frozenset({"subject", "predicate", "object", "value"}), frozenset()),
    "path": (frozenset({"from", "to", "hops"}), frozenset({"expected"})),
}
_TEXT = frozenset({"name", "space", "subject", "predicate", "object", "from", "to", "as_of", "valid_from"})
_TIMES = frozenset({"as_of", "valid_from"})
#: The graph surfaces walk at most four hops.
_MAX_HOPS = 4
_AS_OF = "2025-06-01T00:00:00Z"
_VALID_FROM = "2024-01-01T00:00:00Z"


def _field_problem(key: str, value: object) -> str | None:
    if key in _TEXT:
        if not isinstance(value, str) or not value.strip():
            return f"{key} must be non-empty text"
        if key in _TIMES:
            try:
                parse_rfc3339(value)
            except ValueError:
                return f"{key} must be an RFC 3339 time"
        if key == "space":
            try:
                check_space(value)
            except SconeError:
                return "space is not a valid space name"
    elif key in ("value", "expected") and not isinstance(value, bool):
        return f"{key} must be true or false"
    elif key == "hops" and (isinstance(value, bool) or not isinstance(value, int) or not 1 <= value <= _MAX_HOPS):
        return f"hops must be a whole number from 1 to {_MAX_HOPS}"
    elif key == "names" and not (isinstance(value, list) and value
                                 and all(isinstance(name, str) and name.strip() for name in value)):
        return "names must be a non-empty list of non-empty text"
    return None


Row = dict[str, object]


def _load(path: Path) -> tuple[Row, list[tuple[int, Row]]]:
    """The meta row, and every other row with its line number. Refused
    whole if any row is not exactly what its kind allows."""
    meta: Row = {}
    rows: list[tuple[int, Row]] = []
    problems: list[str] = []
    for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as error:
            problems.append(f"line {number}: not JSON ({error.msg})")
            continue
        if not isinstance(row, dict):
            problems.append(f"line {number}: a row must be an object")
            continue
        kind = row.get("kind")
        if not isinstance(kind, str) or kind not in _FIELDS:
            problems.append(f"line {number}: unknown kind {kind!r}")
            continue
        required, allowed = _FIELDS[kind]
        fields = set(row) - {"kind"}
        problems += [f"line {number}: {kind} needs {key}" for key in sorted(required - fields)]
        problems += [f"line {number}: {kind} has no field {key}" for key in sorted(fields - required - allowed)]
        problems += [f"line {number}: {problem}" for key in sorted(fields & (required | allowed))
                     if (problem := _field_problem(key, row[key])) is not None]
        if kind != "meta":
            rows.append((number, row))
        elif meta:
            problems.append(f"line {number}: a second meta row")
        else:
            meta = row
    if problems:
        raise FixtureError("; ".join(problems))
    return meta, rows


def _claim(row: Row) -> tuple[str, str, str]:
    """A fact row's claim as the store keeps it: the subject by its key."""
    return entity_key(str(row["subject"])), str(row["predicate"]), str(row["object"])


def _begins(row: Row) -> datetime:
    return parse_rfc3339(str(row.get("valid_from", _VALID_FROM)))


def _said(row: Row) -> str:
    return f"{row['subject']!r} {row['predicate']} {row['object']!r}"


def _checked(rows: list[tuple[int, Row]], when: datetime) -> None:
    """Refuse gold about a name or claim that no fact due by ``when``
    states: it could only ever be vacuously right."""
    stated = {_claim(row) for _, row in rows if row["kind"] == "fact"}
    due = {_claim(row) for _, row in rows if row["kind"] == "fact" and _begins(row) <= when}
    named = {key for subject, _, value in due for key in (subject, entity_key(value))}
    problems: list[str] = []
    for number, row in rows:
        if row["kind"] == "same_entity":
            problems += [f"line {number}: same_entity name {name!r} is in no fact due by as_of"
                         for name in _names(row) if entity_key(str(name)) not in named]
        elif row["kind"] == "literal" and _claim(row) not in stated:
            problems.append(f"line {number}: literal {_said(row)} is not a fact")
        elif row["kind"] == "path":
            problems += [f"line {number}: path end {row[side]!r} is in no fact due by as_of" for side in ("from", "to")
                         if entity_key(str(row[side])) not in named]
    if problems:
        raise FixtureError("; ".join(problems))


def _names(row: Row) -> list[object]:
    names = row.get("names")
    return names if isinstance(names, list) else []


def _bcubed(gold: list[list[str]], entity_of: Mapping[str, str | None]) -> float | None:
    """B-cubed F1 over the gold names: a name's predicted cluster is every
    gold name on the same entity. A name with no entity was not clustered
    at all, so it adds nothing to either sum; None when there is no gold."""
    cluster_of = {name: frozenset(cluster) for cluster in gold for name in cluster}
    names = sorted(cluster_of)
    if not names:
        return None
    precision = recall = 0.0
    for name in names:
        entity = entity_of.get(name)
        if entity is None:
            continue
        predicted = frozenset(other for other in names if entity_of.get(other) == entity)
        precision += len(predicted & cluster_of[name]) / len(predicted)
        recall += len(predicted & cluster_of[name]) / len(cluster_of[name])
    precision, recall = precision / len(names), recall / len(names)
    return round(2 * precision * recall / (precision + recall), 6) if precision + recall else 0.0


async def run_entity_graph_benchmark(path: Path) -> EntityGraphReport:
    from ..embedders.hash import HashEmbedder
    from ..backends.memory import InMemoryDocumentStore, InMemoryVectorIndex
    from ..memory.engine import MemoryEngine

    meta, numbered = _load(path)
    space = str(meta.get("space", "bench"))
    moment = str(meta.get("as_of", _AS_OF))
    when = parse_rfc3339(moment)
    _checked(numbered, when)
    rows = [row for _, row in numbered]
    engine = await MemoryEngine(InMemoryDocumentStore(), InMemoryVectorIndex(), HashEmbedder(),
                                clock=lambda: moment).open()
    try:
        for row in rows:
            if row["kind"] == "fact":
                await engine.assert_fact(space, str(row["subject"]), str(row["predicate"]), str(row["object"]),
                                         valid_from=str(row.get("valid_from", _VALID_FROM)))
        projection, coverage = await load_projection(engine, space, mode="current", as_of=moment)
        by_key = {entity.key: entity.entity_id for entity in projection.entities}

        gold = [[str(name) for name in _names(row)] for row in rows if row["kind"] == "same_entity"]
        entity_of = {name: by_key.get(entity_key(name)) for cluster in gold for name in cluster}
        fragments = {cluster[0]: len({entity_of[name] or f"missing:{name}" for name in cluster}) for cluster in gold}

        # A claim restated in another interval is another fact: 34, then 35,
        # then 34 again is three facts, told apart by when each began.
        facts = await engine.documents.list_facts(space, include_closed=True)
        by_claim: dict[tuple[str, str, str], list[Fact]] = defaultdict(list)
        for fact in facts:
            by_claim[(fact.subject, fact.predicate, fact.object)].append(fact)

        def stored(row: Row) -> Fact | None:
            """The stored fact a fact row made: its claim, from when it began."""
            begins = _begins(row)
            return next((fact for fact in by_claim[_claim(row)] if parse_rfc3339(fact.valid_from) == begins), None)

        relations_of: dict[int, list[Relation]] = defaultdict(list)
        attributes_of: dict[int, list[Attribute]] = defaultdict(list)
        for relation in projection.relations:
            for fact_id in relation.fact_ids:
                relations_of[fact_id].append(relation)
        for attribute in projection.attributes:
            for fact_id in attribute.fact_ids:
                attributes_of[fact_id].append(attribute)

        def holds(row: Row) -> bool:
            """Whether the view should hold the claim at as_of, by its own
            rule; a claim the store lost should, once it has begun."""
            fact = stored(row)
            if fact is None:
                return _begins(row) <= when
            return counts(fact.status, fact.excluded, fact.valid_from, fact.valid_until, "current", when)

        def carried(row: Row) -> tuple[bool, bool]:
            """Whether a relation, and whether an attribute, carries this very
            claim: its fact, on its own subject, predicate and object."""
            fact = stored(row)
            if fact is None:
                return False, False
            subject, predicate = by_key.get(entity_key(str(row["subject"]))), str(row["predicate"])
            thing = by_key.get(entity_key(str(row["object"])))
            return (any(relation.subject_id == subject and relation.predicate == predicate
                        and relation.object_id == thing for relation in relations_of[fact.fact_id]),
                    any(attribute.entity_id == subject and attribute.predicate == predicate
                        and attribute.value == str(row["object"]).strip() for attribute in attributes_of[fact.fact_id]))

        stated = [row for row in rows if row["kind"] == "fact"]
        in_view = [row for row in stated if holds(row)]
        claims_missing = sum(not any(carried(row)) for row in in_view)
        rows_of: dict[tuple[str, str, str], list[Row]] = defaultdict(list)
        for row in stated:
            rows_of[_claim(row)].append(row)

        def held(label: Row) -> Row | None:
            """The fact row a label is about: of its claim's, the one the
            view should hold at as_of."""
            return next((row for row in rows_of[_claim(label)] if holds(row)), None)

        unheld = [f"line {number}: literal {_said(row)} does not hold at as_of; "
                  + ("its fact begins later" if all(_begins(fact_row) > when for fact_row in rows_of[_claim(row)])
                     else "the ledger closed it")
                  for number, row in numbered if row["kind"] == "literal" and held(row) is None]
        if unheld:
            raise FixtureError("; ".join(unheld))
        labels = [(label, held(label)) for label in rows if label["kind"] == "literal"]
        wrong, things, connected = 0, 0, 0
        for label, about in labels:
            thing, value = carried(about) if about is not None else (False, False)
            wrong += (thing, value) != ((False, True) if label["value"] else (True, False))
            if not label["value"]:
                things += 1
                connected += thing

        recall: dict[int, list[bool]] = {}
        false_positives, negatives, ends_missing = 0, 0, 0
        for row in (row for row in rows if row["kind"] == "path"):
            ends = [by_key.get(entity_key(str(row[side]))) for side in ("from", "to")]
            ends_missing += sum(end is None for end in ends)
            found = bool(ends[0] and ends[1] and paths_between(projection, ends[0], ends[1],
                                                               max_hops=int(str(row["hops"])), limit=1).paths)
            if row.get("expected", True):
                recall.setdefault(int(str(row["hops"])), []).append(found)
            else:
                negatives += 1
                false_positives += found

        view = knowledge_view(projection, mode="current", as_of=moment, limit=150, attribute_limit=300,
                              coverage=coverage)
        report = build_report(projection, analyze_projection(projection), meta=view["projection"],  # type: ignore[arg-type]
                              filters={"status": "current", "as_of": moment}, coverage=coverage)
        first_path = next((row for row in rows if row["kind"] == "path"), None)
        context_bytes = 0
        if first_path is not None:
            packet = await graph_context(engine, space, names=[str(first_path["from"]), str(first_path["to"])],
                                         as_of=moment)
            context_bytes = len(packet.text.encode())
        view_bytes = {"knowledge": len(json.dumps(view, sort_keys=True, default=str).encode()),
                      "report": len(json.dumps(report, sort_keys=True, default=str).encode()),
                      "context": context_bytes}

        counted = list(facts)
        started = time.perf_counter()
        project_entities(space, counted, revision=1)
        elapsed = (time.perf_counter() - started) * 1000
    finally:
        await engine.close()

    result = EntityGraphReport(
        fixture=str(meta.get("name", path.name)), facts=len(facts), entities=len(projection.entities),
        relations=len(projection.relations), attributes=len(projection.attributes), claims_missing=claims_missing,
        claims_out_of_view=len(stated) - len(in_view),
        gold_names_missing=sum(entity is None for entity in entity_of.values()), path_ends_missing=ends_missing,
        cluster_fragments=fragments,
        fragmentation=round(sum(fragments.values()) / len(fragments), 6) if fragments else None,
        alias_bcubed_f1=_bcubed(gold, entity_of),
        literal_error_rate=round(wrong / len(labels), 6) if labels else None,
        connected_claim_share=round(connected / things, 6) if things else None,
        path_recall={hops: round(sum(hits) / len(hits), 6) for hops, hits in sorted(recall.items())},
        path_false_positives=false_positives if negatives else None, view_bytes=view_bytes,
        build_ms_per_10k=round(elapsed * 10_000 / max(1, len(counted)), 3))
    stable = {key: value for key, value in result.record().items() if key not in ("build_ms_per_10k", "artefact_sha256")}
    digest = hashlib.sha256(json.dumps(stable, sort_keys=True).encode()).hexdigest()
    return EntityGraphReport(**{**result.record(), "artefact_sha256": digest})  # type: ignore[arg-type]
