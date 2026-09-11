"""Graph quality on a versioned synthetic fixture.

A fixture is JSON lines of four kinds, all synthetic:

- ``fact``: a claim to load (subject, predicate, object, optional
  valid_from);
- ``same_entity``: names that all mean one thing, the gold clusters;
- ``literal``: whether a fact's object should be a value (``true``) or a
  thing (``false``);
- ``path``: two names that should connect within ``hops``, or with
  ``"expected": false``, that should not.

Gold must be about what the fixture loads: a ``same_entity`` name or a
path end that no fact names, or a ``literal`` that is not one of the
facts, could only ever be vacuously right, so the fixture is refused with
``FixtureError`` instead of scored.

The report scores what a person would check by hand, from the evidence in
the projection. Something missing is never counted as right:

- ``claims_missing``: loaded facts that became neither a relation nor an
  attribute;
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

from dataclasses import asdict, dataclass, field
import hashlib
import json
from pathlib import Path
import time
from typing import Mapping

from ..core.validation import entity_key
from ..entities.analysis import analyze_projection
from ..entities.context import graph_context
from ..entities.project import project_entities
from ..entities.query import paths_between
from ..entities.read import load_projection
from ..entities.report import build_report
from ..entities.view import knowledge_view

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


def _load(path: Path) -> tuple[dict[str, object], list[dict[str, object]]]:
    meta: dict[str, object] = {}
    rows: list[dict[str, object]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            row = json.loads(line)
            if row.get("kind") == "meta":
                meta = row
            else:
                rows.append(row)
    return meta, rows


def _names(row: Mapping[str, object]) -> list[object]:
    names = row.get("names")
    return names if isinstance(names, list) else []


def _claim(row: Mapping[str, object]) -> tuple[str, str, str]:
    """A fact row's claim as the store keeps it: the subject by its key."""
    return entity_key(str(row["subject"])), str(row["predicate"]), str(row["object"])


_KINDS = frozenset({"fact", "same_entity", "literal", "path"})


def _checked(rows: list[dict[str, object]]) -> None:
    """Refuse gold about a name or claim that no fact in the fixture states."""
    claims = {_claim(row) for row in rows if row.get("kind") == "fact"}
    named = {key for subject, _, value in claims for key in (subject, entity_key(value))}
    problems: list[str] = []
    for row in rows:
        kind = row.get("kind")
        if kind not in _KINDS:
            problems.append(f"unknown kind {kind!r}")
        elif kind == "same_entity":
            if not _names(row):
                problems.append("a same_entity row with no names")
            problems += [f"same_entity name {name!r} is in no fact" for name in _names(row)
                         if entity_key(str(name)) not in named]
        elif kind == "literal" and _claim(row) not in claims:
            problems.append(f"literal {row['subject']!r} {row['predicate']} {row['object']!r} is not a fact")
        elif kind == "path":
            problems += [f"path end {row[side]!r} is in no fact" for side in ("from", "to")
                         if entity_key(str(row[side])) not in named]
    if problems:
        raise FixtureError("; ".join(problems))


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

    meta, rows = _load(path)
    _checked(rows)
    space = str(meta.get("space", "bench"))
    moment = str(meta.get("as_of", "2025-06-01T00:00:00Z"))
    engine = await MemoryEngine(InMemoryDocumentStore(), InMemoryVectorIndex(), HashEmbedder(),
                                clock=lambda: moment).open()
    try:
        for row in rows:
            if row["kind"] == "fact":
                await engine.assert_fact(space, str(row["subject"]), str(row["predicate"]), str(row["object"]),
                                         valid_from=str(row.get("valid_from", "2024-01-01T00:00:00Z")))
        projection, coverage = await load_projection(engine, space, mode="current", as_of=moment)
        by_key = {entity.key: entity.entity_id for entity in projection.entities}

        gold = [[str(name) for name in _names(row)] for row in rows if row["kind"] == "same_entity"]
        entity_of = {name: by_key.get(entity_key(name)) for cluster in gold for name in cluster}
        fragments = {cluster[0]: len({entity_of[name] or f"missing:{name}" for name in cluster}) for cluster in gold}

        facts = {(fact.subject, fact.predicate, fact.object): fact
                 for fact in await engine.documents.list_facts(space, include_closed=True)}
        as_thing = {fact_id for relation in projection.relations for fact_id in relation.fact_ids}
        as_value = {fact_id for attribute in projection.attributes for fact_id in attribute.fact_ids}

        def carried(row: Mapping[str, object]) -> tuple[bool, bool]:
            """Whether a relation, and whether an attribute, carries the claim."""
            fact = facts.get(_claim(row))
            return (fact is not None and fact.fact_id in as_thing), (fact is not None and fact.fact_id in as_value)

        claims_missing = sum(not any(carried(row)) for row in rows if row["kind"] == "fact")
        labels = [row for row in rows if row["kind"] == "literal"]
        wrong, things, connected = 0, 0, 0
        for label in labels:
            thing, value = carried(label)
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

        counted = [fact for fact in facts.values()]
        started = time.perf_counter()
        project_entities(space, counted, revision=1)
        elapsed = (time.perf_counter() - started) * 1000
    finally:
        await engine.close()

    result = EntityGraphReport(
        fixture=str(meta.get("name", path.name)), facts=len(facts), entities=len(projection.entities),
        relations=len(projection.relations), attributes=len(projection.attributes), claims_missing=claims_missing,
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
