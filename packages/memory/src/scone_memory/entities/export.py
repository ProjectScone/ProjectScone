"""Export the entity graph to formats other tools read.

- ``json``: node-link JSON (nodes, links, graph metadata) as graph libraries
  and d3 read it, with each node's values and degree for layout.
- ``graphml``: GraphML XML for Gephi, yEd and NetworkX.
- ``cypher``: MERGE statements to load the graph into Neo4j or Memgraph,
  one per line. Relations are ``RELATES`` edges carrying their predicate as
  a property, so no predicate becomes query syntax.
- ``csv``: a zip of entities.csv, relations.csv and attributes.csv. Cells
  that a spreadsheet would read as formulas are prefixed with a quote.
- ``jsonld``: JSON-LD linked data, one node per entity.
- ``obsidian``: a zip of Markdown notes, one per entity, wiki-linked through
  its relations, with an index.

Every format carries the fact ids behind each relation and value, the
projection digest it came from and, when given, ``about``: the view's
filters and what its read left out, so a partial export says so itself. Output is byte-for-byte deterministic: sorted
items, fixed zip timestamps. Text is escaped for its format, never trusted.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
import csv
import io
import json
import re
from typing import Callable, Literal, Mapping
import xml.etree.ElementTree as ElementTree
import zipfile

from .markdown import literal
from .project import Entity, EntityProjection

ExportFormat = Literal["json", "graphml", "cypher", "csv", "jsonld", "obsidian"]
EXPORT_FORMATS: tuple[ExportFormat, ...] = ("json", "graphml", "cypher", "csv", "jsonld", "obsidian")
_ZIP_TIME = (1980, 1, 1, 0, 0, 0)


@dataclass(frozen=True)
class Export:
    body: bytes
    media_type: str
    filename: str


def _degrees(projection: EntityProjection) -> dict[str, int]:
    degree: dict[str, int] = defaultdict(int)
    for relation in projection.relations:
        degree[relation.subject_id] += 1
        degree[relation.object_id] += 1
    return degree


def _attributes(projection: EntityProjection) -> dict[str, list[dict[str, object]]]:
    found: dict[str, list[dict[str, object]]] = defaultdict(list)
    for attribute in projection.attributes:
        found[attribute.entity_id].append({"predicate": attribute.predicate, "value": attribute.value,
                                           "literal_kind": attribute.literal_kind,
                                           "fact_ids": list(attribute.fact_ids)})
    return found


def _meta(projection: EntityProjection) -> dict[str, object]:
    return {"space": projection.space, "digest": projection.digest, "version": projection.version,
            "revision": projection.revision}


def _about_text(about: Mapping[str, object]) -> str:
    return json.dumps(dict(about), ensure_ascii=False, sort_keys=True)


def _about_lines(about: Mapping[str, object]) -> list[str]:
    coverage = about.get("coverage")
    if not isinstance(coverage, Mapping):
        return []
    lines = [f"{literal(about.get('status', ''))} facts as of {literal(about.get('as_of', ''))}: "
             f"{coverage.get('facts_counted', 0)} counted of {coverage.get('facts_read', 0)} read."]
    reasons = coverage.get("reasons")
    if isinstance(reasons, list) and reasons:
        lines.append(f"Limited by {literal(', '.join(map(str, reasons)))}: this is part of the space, not all of it.")
    return ["", *lines]


def _node_link(projection: EntityProjection, about: Mapping[str, object]) -> Export:
    degree, values = _degrees(projection), _attributes(projection)
    data = {
        "directed": True, "multigraph": True, "graph": {**_meta(projection), "about": dict(about)},
        "nodes": [{"id": entity.entity_id, "key": entity.key, "label": entity.label, "kind": entity.kind,
                   "kind_status": entity.kind_status, "names": [form.text for form in entity.surface_forms],
                   "flags": list(entity.flags), "degree": degree[entity.entity_id],
                   "attributes": values[entity.entity_id]} for entity in projection.entities],
        "links": [{"id": relation.relation_id, "source": relation.subject_id, "target": relation.object_id,
                   "predicate": relation.predicate, "fact_ids": list(relation.fact_ids),
                   "facts": relation.support.facts} for relation in projection.relations],
    }
    return Export(json.dumps(data, ensure_ascii=False, indent=2, sort_keys=True).encode("utf-8"),
                  "application/json", "graph.json")


def _graphml(projection: EntityProjection, about: Mapping[str, object]) -> Export:
    ns = "http://graphml.graphdrawing.org/xmlns"
    ElementTree.register_namespace("", ns)
    root = ElementTree.Element(f"{{{ns}}}graphml")
    for key, target, name in (("label", "node", "label"), ("key", "node", "key"), ("kind", "node", "kind"),
                              ("predicate", "edge", "predicate"), ("fact_ids", "edge", "fact_ids"),
                              ("digest", "graph", "digest"), ("about", "graph", "about")):
        ElementTree.SubElement(root, f"{{{ns}}}key", {"id": key, "for": target, "attr.name": name, "attr.type": "string"})
    graph = ElementTree.SubElement(root, f"{{{ns}}}graph", {"id": projection.space, "edgedefault": "directed"})
    ElementTree.SubElement(graph, f"{{{ns}}}data", {"key": "digest"}).text = projection.digest
    ElementTree.SubElement(graph, f"{{{ns}}}data", {"key": "about"}).text = _about_text(about)
    for entity in projection.entities:
        node = ElementTree.SubElement(graph, f"{{{ns}}}node", {"id": entity.entity_id})
        for key, value in (("label", entity.label), ("key", entity.key), ("kind", entity.kind or "")):
            ElementTree.SubElement(node, f"{{{ns}}}data", {"key": key}).text = value
    for relation in projection.relations:
        edge = ElementTree.SubElement(graph, f"{{{ns}}}edge", {"id": relation.relation_id,
                                      "source": relation.subject_id, "target": relation.object_id})
        ElementTree.SubElement(edge, f"{{{ns}}}data", {"key": "predicate"}).text = relation.predicate
        ElementTree.SubElement(edge, f"{{{ns}}}data", {"key": "fact_ids"}).text = " ".join(map(str, relation.fact_ids))
    return Export(ElementTree.tostring(root, encoding="utf-8", xml_declaration=True), "application/graphml+xml",
                  "graph.graphml")


def _cypher_text(value: object) -> str:
    text = str(value).replace("\\", "\\\\").replace("'", "\\'").replace('"', '\\"')
    return "'" + text.replace("\n", "\\n").replace("\r", "\\r") + "'"


def _cypher(projection: EntityProjection, about: Mapping[str, object]) -> Export:
    lines = [f"// Scone knowledge graph for space {projection.space}, projection {projection.digest}",
             "// Load with cypher-shell; statements are idempotent MERGEs.",
             "// about: " + _about_text(about)]
    for entity in projection.entities:
        lines.append(f"MERGE (e:Entity {{id: {_cypher_text(entity.entity_id)}}}) SET e.key = {_cypher_text(entity.key)}, "
                     f"e.label = {_cypher_text(entity.label)}, e.kind = {_cypher_text(entity.kind or '')};")
    for relation in projection.relations:
        lines.append(f"MATCH (a:Entity {{id: {_cypher_text(relation.subject_id)}}}), "
                     f"(b:Entity {{id: {_cypher_text(relation.object_id)}}}) "
                     f"MERGE (a)-[r:RELATES {{id: {_cypher_text(relation.relation_id)}}}]->(b) "
                     f"SET r.predicate = {_cypher_text(relation.predicate)}, r.fact_ids = {list(relation.fact_ids)};")
    return Export(("\n".join(lines) + "\n").encode("utf-8"), "text/plain; charset=utf-8", "graph.cypher")


def _cell(value: object) -> str:
    text = "" if value is None else str(value)
    return "'" + text if text[:1] in ("=", "+", "-", "@", "\t", "\r") else text


def _zip(files: dict[str, str]) -> bytes:
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", compression=zipfile.ZIP_DEFLATED) as bundle:
        for name in sorted(files):
            info = zipfile.ZipInfo(name, date_time=_ZIP_TIME)
            info.compress_type = zipfile.ZIP_DEFLATED
            bundle.writestr(info, files[name].encode("utf-8"))
    return buffer.getvalue()


def _table(header: list[str], rows: list[list[object]]) -> str:
    buffer = io.StringIO()
    writer = csv.writer(buffer, lineterminator="\n")
    writer.writerow(header)
    writer.writerows([[_cell(value) for value in row] for row in rows])
    return buffer.getvalue()


def _csv(projection: EntityProjection, about: Mapping[str, object]) -> Export:
    files = {
        "about.json": json.dumps({**_meta(projection), "about": dict(about)}, ensure_ascii=False, indent=2,
                                 sort_keys=True) + "\n",
        "entities.csv": _table(["id", "key", "label", "kind", "kind_status"],
                               [[e.entity_id, e.key, e.label, e.kind, e.kind_status] for e in projection.entities]),
        "relations.csv": _table(["id", "subject_id", "predicate", "object_id", "fact_ids"],
                                [[r.relation_id, r.subject_id, r.predicate, r.object_id, " ".join(map(str, r.fact_ids))]
                                 for r in projection.relations]),
        "attributes.csv": _table(["id", "entity_id", "predicate", "value", "literal_kind", "fact_ids"],
                                 [[a.attribute_id, a.entity_id, a.predicate, a.value, a.literal_kind,
                                   " ".join(map(str, a.fact_ids))] for a in projection.attributes]),
    }
    return Export(_zip(files), "application/zip", "graph-csv.zip")


def _json_ld(projection: EntityProjection, about: Mapping[str, object]) -> Export:
    def urn(entity_id: str) -> str:
        return f"urn:scone:{projection.space}:{entity_id}"

    nodes: dict[str, dict[str, object]] = {
        entity.entity_id: {"@id": urn(entity.entity_id), "@type": (entity.kind or "entity").capitalize(),
                           "label": entity.label, "key": entity.key}
        for entity in projection.entities}
    for relation in projection.relations:
        links = nodes[relation.subject_id].setdefault(relation.predicate, [])
        assert isinstance(links, list)
        links.append({"@id": urn(relation.object_id), "facts": list(relation.fact_ids)})
    for attribute in projection.attributes:
        values = nodes[attribute.entity_id].setdefault(attribute.predicate, [])
        assert isinstance(values, list)
        values.append({"@value": attribute.value, "facts": list(attribute.fact_ids)})
    data = {"@context": {"@vocab": "https://projectscone.dev/vocab#",
                         "label": "http://www.w3.org/2000/01/rdf-schema#label"},
            "digest": projection.digest, "about": dict(about), "@graph": [nodes[entity_id] for entity_id in sorted(nodes)]}
    return Export(json.dumps(data, ensure_ascii=False, indent=2, sort_keys=True).encode("utf-8"),
                  "application/ld+json", "graph.jsonld")


_UNSAFE = re.compile(r'[\\/:*?"<>|#^\[\]\x00-\x1f]+')


def _note_names(entities: tuple[Entity, ...]) -> dict[str, str]:
    """A distinct, safe note name for every entity: characters file systems
    or wiki links would misread become '-', and clashes get the id's tail."""
    names: dict[str, str] = {}
    taken: set[str] = set()
    for entity in sorted(entities, key=lambda entity: entity.entity_id):
        base = " ".join(_UNSAFE.sub("-", entity.label).split()).strip(". ")[:100] or "entity"
        name = base if base.casefold() not in taken else f"{base} ({entity.entity_id[-6:]})"
        taken.add(name.casefold())
        names[entity.entity_id] = name
    return names


def _obsidian(projection: EntityProjection, about: Mapping[str, object]) -> Export:
    names = _note_names(projection.entities)
    outgoing: dict[str, list[str]] = defaultdict(list)
    incoming: dict[str, list[str]] = defaultdict(list)
    values: dict[str, list[str]] = defaultdict(list)

    def cited(fact_ids: tuple[int, ...]) -> str:
        return "fact " + str(fact_ids[0]) if len(fact_ids) == 1 else "facts " + ", ".join(map(str, fact_ids))

    for relation in projection.relations:
        outgoing[relation.subject_id].append(
            f"- {literal(relation.predicate)} [[{names[relation.object_id]}]] ({cited(relation.fact_ids)})")
        incoming[relation.object_id].append(
            f"- [[{names[relation.subject_id]}]] {literal(relation.predicate)} ({cited(relation.fact_ids)})")
    for attribute in projection.attributes:
        values[attribute.entity_id].append(
            f"- {literal(attribute.predicate)}: {literal(attribute.value)} ({cited(attribute.fact_ids)})")
    files: dict[str, str] = {}
    for entity in projection.entities:
        body = ["---", f"id: {entity.entity_id}", f"key: {json.dumps(entity.key, ensure_ascii=False)}",
                f"kind: {entity.kind or 'unknown'}", "---", "", f"# {literal(entity.label)}", ""]
        for title, lines in (("Relations", outgoing[entity.entity_id]), ("Referenced by", incoming[entity.entity_id]),
                             ("Values", values[entity.entity_id])):
            if lines:
                body += [f"## {title}", "", *sorted(lines), ""]
        files[f"entities/{names[entity.entity_id]}.md"] = "\n".join(body)
    files["index.md"] = "\n".join([f"# Knowledge graph: {literal(projection.space)}", "",
                                   f"Projection `{projection.digest[:12]}`, {len(projection.entities)} entities.",
                                   *_about_lines(about), "",
                                   *sorted(f"- [[{name}]]" for name in names.values())]) + "\n"
    return Export(_zip(files), "application/zip", "graph-obsidian.zip")


_WRITERS: dict[str, Callable[[EntityProjection, Mapping[str, object]], Export]] = {
    "json": _node_link, "graphml": _graphml, "cypher": _cypher, "csv": _csv, "jsonld": _json_ld, "obsidian": _obsidian,
}


def export_graph(projection: EntityProjection, format: ExportFormat, *,
                 about: Mapping[str, object] | None = None) -> Export:
    return _WRITERS[format](projection, about or {})
