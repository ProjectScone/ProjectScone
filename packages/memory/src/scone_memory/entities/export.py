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
import unicodedata
from urllib.parse import quote
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


def _visible(character: str) -> str:
    """A visible stand-in for a character a format cannot carry: a C0
    control becomes its Control Pictures symbol (U+0001 is shown as
    U+2401), and anything else, such as a lone surrogate, becomes U+FFFD."""
    code = ord(character)
    return chr(0x2400 + code) if code < 0x20 else "\u2421" if code == 0x7F else "\ufffd"


# Characters XML 1.0 has no way to write, even as a character reference.
_NOT_XML = re.compile("[\x00-\x08\x0b\x0c\x0e-\x1f\ud800-\udfff\ufffe\uffff]")
# Characters a Cypher string writes as a \\u escape rather than raw.
_CYPHER_ESCAPED = re.compile("[\x00-\x1f\x7f\ud800-\udfff\u2028\u2029]")
# Characters a CSV cell cannot carry to the tools that read it.
_NOT_CSV = re.compile("[\x00\ud800-\udfff]")


def _encoded(text: str) -> bytes:
    """UTF-8, with any lone surrogate written as a backslash escape. Used
    only where a surrogate can appear solely inside a JSON or YAML string,
    whose own escape that is."""
    return text.encode("utf-8", "backslashreplace")


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
    return Export(_encoded(json.dumps(data, ensure_ascii=False, indent=2, sort_keys=True)), "application/json",
                  "graph.json")


def _graphml(projection: EntityProjection, about: Mapping[str, object]) -> Export:
    """Entities and values are both nodes (``type`` says which), so a value
    keeps its own edge (``link`` says relation or value) carrying the
    predicate and the facts behind it.

    XML 1.0 cannot hold some characters the ledger can (U+0001, U+FFFE, a
    lone surrogate). Text shows each as a visible stand-in, and the element
    gains an ``exact`` field: its original values as JSON, which can."""
    ns = "http://graphml.graphdrawing.org/xmlns"
    ElementTree.register_namespace("", ns)
    root = ElementTree.Element(f"{{{ns}}}graphml")
    for key, target in (("type", "node"), ("label", "node"), ("key", "node"), ("kind", "node"),
                        ("literal_kind", "node"), ("link", "edge"), ("predicate", "edge"), ("fact_ids", "edge"),
                        ("digest", "graph"), ("about", "graph"), ("exact", "all")):
        ElementTree.SubElement(root, f"{{{ns}}}key", {"id": key, "for": target, "attr.name": key,
                                                      "attr.type": "string"})
    graph = ElementTree.SubElement(root, f"{{{ns}}}graph", {"id": projection.space, "edgedefault": "directed"})

    def fields(parent: ElementTree.Element, values: tuple[tuple[str, str], ...]) -> None:
        exact = {}
        for key, value in values:
            shown = _NOT_XML.sub(lambda match: _visible(match.group()), value)
            if shown != value:
                exact[key] = value
            ElementTree.SubElement(parent, f"{{{ns}}}data", {"key": key}).text = shown
        if exact:
            ElementTree.SubElement(parent, f"{{{ns}}}data", {"key": "exact"}).text = json.dumps(exact, sort_keys=True)

    def edge(edge_id: str, source: str, target: str, link: str, predicate: str, fact_ids: tuple[int, ...]) -> None:
        element = ElementTree.SubElement(graph, f"{{{ns}}}edge", {"id": edge_id, "source": source, "target": target})
        fields(element, (("link", link), ("predicate", predicate), ("fact_ids", " ".join(map(str, fact_ids)))))

    fields(graph, (("digest", projection.digest), ("about", _about_text(about))))
    for entity in projection.entities:
        node = ElementTree.SubElement(graph, f"{{{ns}}}node", {"id": entity.entity_id})
        fields(node, (("type", "entity"), ("label", entity.label), ("key", entity.key), ("kind", entity.kind or "")))
    for attribute in projection.attributes:
        node = ElementTree.SubElement(graph, f"{{{ns}}}node", {"id": attribute.attribute_id})
        fields(node, (("type", "value"), ("label", attribute.value), ("literal_kind", attribute.literal_kind or "")))
    for relation in projection.relations:
        edge(relation.relation_id, relation.subject_id, relation.object_id, "relation", relation.predicate,
             relation.fact_ids)
    for attribute in projection.attributes:
        edge(f"{attribute.attribute_id}:has", attribute.entity_id, attribute.attribute_id, "value", attribute.predicate,
             attribute.fact_ids)
    # A parser folds a raw CR (and CR LF) into LF; a character reference
    # survives. The serializer writes CR only inside text, never in markup.
    body = ElementTree.tostring(root, encoding="utf-8", xml_declaration=True).replace(b"\r", b"&#13;")
    return Export(body, "application/graphml+xml", "graph.graphml")


def _cypher_text(value: object) -> str:
    """A Cypher string literal: quotes and backslashes escaped, and controls,
    line separators and lone surrogates as \\u escapes Cypher decodes."""
    text = str(value).replace("\\", "\\\\").replace("'", "\\'").replace('"', '\\"')
    return "'" + _CYPHER_ESCAPED.sub(lambda match: f"\\u{ord(match.group()):04x}", text) + "'"


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
    for attribute in projection.attributes:
        lines.append(f"MATCH (e:Entity {{id: {_cypher_text(attribute.entity_id)}}}) "
                     f"MERGE (v:Value {{id: {_cypher_text(attribute.attribute_id)}}}) "
                     f"SET v.value = {_cypher_text(attribute.value)}, v.literal_kind = {_cypher_text(attribute.literal_kind or '')} "
                     f"MERGE (e)-[r:HAS_VALUE]->(v) "
                     f"SET r.predicate = {_cypher_text(attribute.predicate)}, r.fact_ids = {list(attribute.fact_ids)};")
    return Export(("\n".join(lines) + "\n").encode("utf-8"), "text/plain; charset=utf-8", "graph.cypher")


def _cell(value: object) -> str:
    text = _NOT_CSV.sub(lambda match: _visible(match.group()), "" if value is None else str(value))
    return "'" + text if text[:1] in ("=", "+", "-", "@", "\t", "\r") else text


def _zip(files: dict[str, str]) -> bytes:
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", compression=zipfile.ZIP_DEFLATED) as bundle:
        for name in sorted(files):
            info = zipfile.ZipInfo(name, date_time=_ZIP_TIME)
            info.compress_type = zipfile.ZIP_DEFLATED
            bundle.writestr(info, _encoded(files[name]))
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


_PREDICATES = "https://projectscone.dev/predicate/"


def _json_ld(projection: EntityProjection, about: Mapping[str, object]) -> Export:
    """Linked data in two layers. Each entity node states its relations and
    values plainly, as any RDF tool reads them. Each relation and value is
    also a ``Claim`` node naming its subject, predicate, object or value,
    and the facts behind it, so provenance belongs to the statement, not to
    the thing it points at. Predicates are percent-encoded ``p:`` terms, so
    none can take the place of a node's own ``@id``, ``label`` or ``key``."""
    def urn(item_id: str) -> str:
        return f"urn:scone:{projection.space}:{item_id}"

    def term(predicate: str) -> str:
        return "p:" + quote(predicate.encode("utf-8", "surrogatepass"), safe="-_.~")

    nodes: dict[str, dict[str, object]] = {
        entity.entity_id: {"@id": urn(entity.entity_id), "@type": (entity.kind or "entity").capitalize(),
                           "label": entity.label, "key": entity.key}
        for entity in projection.entities}
    stated: dict[str, dict[str, list[object]]] = defaultdict(lambda: defaultdict(list))
    claims: list[dict[str, object]] = []
    for relation in projection.relations:
        stated[relation.subject_id][term(relation.predicate)].append({"@id": urn(relation.object_id)})
        claims.append({"@id": urn(relation.relation_id), "@type": "Claim",
                       "subject": {"@id": urn(relation.subject_id)}, "predicate": {"@id": term(relation.predicate)},
                       "object": {"@id": urn(relation.object_id)}, "facts": list(relation.fact_ids)})
    for attribute in projection.attributes:
        stated[attribute.entity_id][term(attribute.predicate)].append(attribute.value)
        claims.append({"@id": urn(attribute.attribute_id), "@type": "Claim",
                       "subject": {"@id": urn(attribute.entity_id)}, "predicate": {"@id": term(attribute.predicate)},
                       "value": attribute.value, "literal_kind": attribute.literal_kind or "",
                       "facts": list(attribute.fact_ids)})
    record = {"@id": urn(f"export:{projection.digest}"), "@type": "Export", "digest": projection.digest,
              "about": dict(about)}
    graph = [{**nodes[entity_id], **stated[entity_id]} for entity_id in sorted(nodes)]
    # Only "@context" and "@graph" at the top: anything beside them would make
    # the document a node and put every statement in a named graph.
    data = {"@context": {"@vocab": "https://projectscone.dev/vocab#", "p": _PREDICATES,
                         "label": "http://www.w3.org/2000/01/rdf-schema#label"},
            "@graph": [record, *graph, *sorted(claims, key=lambda claim: str(claim["@id"]))]}
    return Export(_encoded(json.dumps(data, ensure_ascii=False, indent=2, sort_keys=True)), "application/ld+json",
                  "graph.jsonld")


_UNSAFE = re.compile('[\\\\/:*?"<>|#^\\[\\]\x00-\x1f\x7f\ud800-\udfff]+')


# Names Windows keeps for devices, whatever extension follows them.
_DEVICES = frozenset({"con", "prn", "aux", "nul", *(f"com{n}" for n in range(1, 10)),
                      *(f"lpt{n}" for n in range(1, 10))})


def _folded(name: str) -> str:
    """How a disk that ignores case and Unicode normalisation sees a name."""
    return unicodedata.normalize("NFC", name).casefold()


def _free(candidate: str, taken: set[str]) -> bool:
    return _folded(candidate) not in taken


def _note_names(entities: tuple[Entity, ...]) -> dict[str, str]:
    """A distinct, safe note name for every entity.

    Characters that file systems or wiki links misread become '-', and a
    name Windows keeps for a device gains a leading '_'. Every name is
    checked against every name already given, folded the way a case- and
    normalisation-insensitive disk folds it; a clash takes more of the
    entity's id, then a number, until it is free, so no note can overwrite
    another."""
    names: dict[str, str] = {}
    taken = {"index"}  # the vault's own index note
    for entity in sorted(entities, key=lambda entity: entity.entity_id):
        base = " ".join(_UNSAFE.sub("-", entity.label).split()).strip(". ")[:100]
        base = base or "entity"
        if _folded(base).split(".")[0].rstrip() in _DEVICES:
            base = "_" + base
        digits = entity.entity_id.split(":", 1)[-1]
        tried = [base, *(f"{base} ({digits[:size]})" for size in (6, 12)), f"{base} ({digits})"]
        name = next((candidate for candidate in tried if _free(candidate, taken)), None)
        number = 2
        while name is None:
            candidate = f"{base} ({digits} {number})"
            name, number = (candidate if _free(candidate, taken) else None), number + 1
        taken.add(_folded(name))
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
