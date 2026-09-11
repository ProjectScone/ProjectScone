"""What a space's graph is made of: its vocabulary, not its contents.

The kinds its entities have, the predicates its facts use and, for each
predicate, which kinds it joins and which kinds of value it takes:
(person) -works_at-> (organisation) twice, (person) -age-> a quantity
twice. An agent reads this before asking the graph anything, to know what
it could ask. It is the schema the ledger has, counted from the projection
a view reads, so it honours the same status and moment; nothing here is
declared ahead of the facts, and a kind is only as known as the entity's
``kind_status`` says.
"""

from __future__ import annotations

from collections import Counter
import hashlib
import json
from typing import TYPE_CHECKING, Iterable, cast

from .context import one_line
from .project import EntityProjection
from .read import load_projection, read_record
from .view import projection_meta

if TYPE_CHECKING:
    from ..memory.engine import MemoryEngine
    from .view import StatusMode

#: Predicates listed at most; ``predicates_total`` counts them all.
MAX_PREDICATES = 1000
#: Characters of a predicate shown; a longer one is clipped and marked.
MAX_TERM = 200
#: The encoded size of the listed predicates, by default and at most.
MAX_BYTES = 64_000
MAX_BYTES_LIMIT = 1_000_000

#: A subject's kind with the other end's kind (or the value's), and how many
#: facts the item stands on.
_Pair = tuple[str | None, str | None]


def _kind_order(kind: str | None) -> tuple[bool, str]:
    return kind is None, kind or ""


def _pairs(items: Iterable[tuple[str | None, str | None, int]], second: str, unit: str) -> list[dict[str, object]]:
    counted: Counter[_Pair] = Counter()
    facts: Counter[_Pair] = Counter()
    for subject, other, supported in items:
        counted[(subject, other)] += 1
        facts[(subject, other)] += supported
    ordered = sorted(counted.items(), key=lambda item: (-item[1], _kind_order(item[0][0]), _kind_order(item[0][1])))
    return [{"subject": subject, second: other, unit: count, "facts": facts[(subject, other)]}
            for (subject, other), count in ordered]


def _facts(entry: dict[str, object]) -> int:
    return cast(int, entry["facts"])


def _term(predicate: str) -> dict[str, object]:
    """A predicate as shown. One past ``MAX_TERM`` characters is clipped and
    says so, with its full length and a digest, so two long predicates that
    share a start are still told apart."""
    if len(predicate) <= MAX_TERM:
        return {"predicate": predicate}
    return {"predicate": predicate[:MAX_TERM - 1] + "…", "clipped": True, "length": len(predicate),
            "term_sha256": hashlib.sha256(predicate.encode("utf-8", "surrogatepass")).hexdigest()}


def _size(entry: dict[str, object]) -> int:
    """Bytes the entry takes as JSON in a response, and one for its comma."""
    return len(json.dumps(entry, ensure_ascii=False).encode("utf-8", "backslashreplace")) + 1


def graph_schema(projection: EntityProjection, *, limit: int = 200, max_bytes: int = MAX_BYTES) -> dict[str, object]:
    """The projection's kinds, predicates and the kinds each predicate joins,
    most used first. Predicates are listed while there are fewer than
    ``limit`` and their JSON fits ``max_bytes``; the rest are counted, and
    ``truncated_by`` says which bound cut the list."""
    if not 1 <= limit <= MAX_PREDICATES:
        raise ValueError(f"limit must be from 1 to {MAX_PREDICATES}")
    if not 1_024 <= max_bytes <= MAX_BYTES_LIMIT:
        raise ValueError(f"max_bytes must be from 1024 to {MAX_BYTES_LIMIT}")
    # An entity whose hints disagree has no kind; at a join it is shown as
    # contested, not filed with the entities nothing hinted at.
    kind_of: dict[str, str | None] = {
        entity.entity_id: entity.kind or ("contested" if entity.kind_status == "conflict" else None)
        for entity in projection.entities}
    joins: dict[str, list[tuple[str | None, str | None, int]]] = {}
    values: dict[str, list[tuple[str | None, str | None, int]]] = {}
    for relation in projection.relations:
        joins.setdefault(relation.predicate, []).append(
            (kind_of.get(relation.subject_id), kind_of.get(relation.object_id), len(relation.fact_ids)))
    for attribute in projection.attributes:
        values.setdefault(attribute.predicate, []).append(
            (kind_of.get(attribute.entity_id), attribute.literal_kind, len(attribute.fact_ids)))
    entries: list[dict[str, object]] = []
    for predicate in joins.keys() | values.keys():
        joined, valued = joins.get(predicate, []), values.get(predicate, [])
        entries.append({**_term(predicate), "facts": sum(item[2] for item in joined + valued),
                        "relations": len(joined), "attributes": len(valued),
                        "joins": _pairs(joined, "object", "relations"),
                        "values": _pairs(valued, "kind", "attributes")})
    entries.sort(key=lambda entry: (-_facts(entry), str(entry["predicate"]), str(entry.get("term_sha256", ""))))
    listed: list[dict[str, object]] = []
    used = 0
    for entry in entries[:limit]:
        if used + _size(entry) > max_bytes:
            break
        listed.append(entry)
        used += _size(entry)
    cut_by = [] if len(listed) == len(entries) else ["limit"] if len(listed) == limit else ["max_bytes"]

    known = Counter((entity.kind, str(entity.kind_status)) for entity in projection.entities)
    kinds = [{"kind": kind, "status": status, "entities": count} for (kind, status), count in sorted(
        known.items(), key=lambda item: (-item[1], _kind_order(item[0][0]), item[0][1]))]
    return {
        "totals": {"entities": len(projection.entities), "relations": len(projection.relations),
                   "attributes": len(projection.attributes), "facts": sum(map(_facts, entries)),
                   "predicates": len(entries), "kinds": len(kinds)},
        "kinds": kinds,
        "predicates": listed,
        "predicates_total": len(entries),
        "truncated": bool(cut_by),
        "truncated_by": cut_by,
    }


def _plural(count: int, word: str, words: str | None = None) -> str:
    return f"{count} {word}" if count == 1 else f"{count} {words or word + 's'}"


def _end(kind: object, thing: bool) -> str:
    name = one_line(kind if kind is not None else "unknown")
    return f"({name})" if thing else name


def schema_lines(schema: dict[str, object], *, header: str, reasons: Iterable[str]) -> str:
    """The schema as one line per item, for a model: coverage first, then
    totals, kinds and each predicate's joins. Stored text is kept to one
    line, so no predicate can start a line of its own."""
    reasons = list(reasons)
    totals = cast(dict[str, int], schema["totals"])
    lines = [header, f"coverage: {'limited: ' + ', '.join(reasons) if reasons else 'complete'}",
             f"totals: {_plural(totals['entities'], 'entity', 'entities')}, "
             f"{_plural(totals['relations'], 'relation')}, {_plural(totals['attributes'], 'value')}, "
             f"{_plural(totals['facts'], 'fact')}, {_plural(totals['predicates'], 'predicate')}"]
    for kind in cast(list[dict[str, object]], schema["kinds"]):
        status, count = str(kind["status"]), cast(int, kind["entities"])
        name = kind["kind"] if kind["kind"] is not None else "contested" if status == "conflict" else None
        lines.append(f"kind: {_end(name, False)}, {_plural(count, 'entity', 'entities')}"
                     + (f", {status}" if kind["kind"] is not None else ""))
    for entry in cast(list[dict[str, object]], schema["predicates"]):
        shapes = [f"{_end(join['subject'], True)} -> {_end(join['object'], True)} x{join['relations']}"
                  for join in cast(list[dict[str, object]], entry["joins"])]
        shapes += [f"{_end(value['subject'], True)} -> {_end(value['kind'], False)} x{value['attributes']}"
                   for value in cast(list[dict[str, object]], entry["values"])]
        lines.append(f"predicate: {one_line(entry['predicate'])}, {_plural(_facts(entry), 'fact')}: "
                     + "; ".join(shapes))
    left = cast(int, schema["predicates_total"]) - len(cast(list[object], schema["predicates"]))
    if left:
        lines.append(f"omitted: {_plural(left, 'predicate')}")
    return "\n".join(lines)


async def schema_record(engine: "MemoryEngine", space: str, *, status: "StatusMode" = "current",
                        as_of: str | None = None, limit: int = 200, max_bytes: int = MAX_BYTES) -> dict[str, object]:
    """The schema of one view, as every surface answers it: which projection
    it counts, at what moment, and whether the read was whole."""
    when = as_of if as_of is not None else engine.clock()
    projection, coverage = await load_projection(engine, space, mode=status, as_of=when)
    complete, read = read_record(coverage)
    schema = graph_schema(projection, limit=limit, max_bytes=max_bytes)
    # A capped read counts only the predicates it saw: the list is partial.
    cuts = (["read"] if not complete else []) + cast(list[str], schema["truncated_by"])
    return {"schema_version": 1, "space": space, "projection": projection_meta(projection),
            "filters": {"status": status, "as_of": when}, **schema, "truncated": bool(cuts), "truncated_by": cuts,
            "complete": complete, "coverage": read}


def schema_text(record: dict[str, object]) -> str:
    """A ``schema_record`` as lines for a model."""
    projection = cast(dict[str, object], record["projection"])
    filters = cast(dict[str, str], record["filters"])
    reasons = cast(dict[str, object], record["coverage"]).get("reasons")
    header = (f"schema: space {one_line(record['space'])}, {filters['status']} facts as of {filters['as_of']}, "
              f"projection {str(projection['digest'])[:12]} at revision {projection['revision']}")
    return schema_lines(record, header=header, reasons=[str(r) for r in reasons] if isinstance(reasons, list) else [])
