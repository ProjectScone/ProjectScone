"""Exporting the entity graph to the formats other tools read.

Every format carries the same entities and relations with their fact ids,
round-trips through a standard parser, and escapes what its syntax needs:
names with quotes, angle brackets, commas, slashes and leading '=' must not
break a file or smuggle in code.
"""
from __future__ import annotations

import csv
import io
import json
import math
import re
import unicodedata
import xml.etree.ElementTree as ElementTree
import zipfile

import pytest

from scone_memory.core.models import Fact
from scone_memory.core.validation import entity_key
from scone_memory.entities.export import EXPORT_FORMATS, export_graph
from scone_memory.entities.ids import key_id
from scone_memory.entities.project import project_entities

HOSTILE = 'O\'Brien "Bob" <script>'


def fact(number: int, subject: str, predicate: str, object_: str) -> Fact:
    return Fact(fact_id=number, space="alpha", subject=subject.casefold(), predicate=predicate, object=object_,
                valid_from="2025-01-01T00:00:00Z")


LEDGER = [fact(1, "alice chen", "works_at", "Acme, Inc."), fact(2, "acme, inc.", "based_in", "Lisbon"),
          fact(3, "alice chen", "knows", HOSTILE), fact(4, "alice chen", "joined_on", "May 2021"),
          fact(5, "=cmd|' /c calc'!A0", "knows", "Alice Chen")]


@pytest.fixture
def projection():
    return project_entities("alpha", LEDGER, revision=3)


def test_every_format_is_offered():
    assert set(EXPORT_FORMATS) == {"json", "graphml", "gexf", "cypher", "csv", "jsonld", "obsidian", "wiki", "mermaid",
                                   "svg", "canvas", "html"}


def test_node_link_json_carries_entities_relations_and_facts(projection):
    data = json.loads(export_graph(projection, "json").body)
    labels = {node["id"]: node["label"] for node in data["nodes"]}
    assert HOSTILE in labels.values()
    assert {(labels[link["source"]], link["predicate"], labels[link["target"]]) for link in data["links"]} >= {
        ("alice chen", "works_at", "Acme, Inc."), ("Acme, Inc.", "based_in", "Lisbon")}
    assert all(link["fact_ids"] for link in data["links"])
    assert data["graph"]["digest"] == projection.digest and data["directed"] is True


def test_graphml_parses_and_keeps_hostile_names_as_text(projection):
    root = ElementTree.fromstring(export_graph(projection, "graphml").body)
    ns = {"g": "http://graphml.graphdrawing.org/xmlns"}
    names = [data.text for data in root.iterfind(".//g:node/g:data[@key='label']", ns)]
    assert HOSTILE in names
    assert not [element for element in root.iter() if element.tag.rsplit("}", 1)[-1] == "script"]
    kinds = [edge.find("g:data[@key='link']", ns).text for edge in root.iterfind(".//g:edge", ns)]
    assert kinds.count("relation") == len(projection.relations) and kinds.count("value") == len(projection.attributes)


def test_cypher_escapes_quotes_and_creates_one_statement_per_item(projection):
    script = export_graph(projection, "cypher").body.decode()
    assert "O\\'Brien \\\"Bob\\\" <script>" in script
    statements = [line for line in script.splitlines() if line.strip() and not line.startswith("//")]
    assert len(statements) == len(projection.entities) + len(projection.relations) + len(projection.attributes)


def test_csv_neutralises_formulas_and_round_trips(projection):
    bundle = zipfile.ZipFile(io.BytesIO(export_graph(projection, "csv").body))
    nodes = list(csv.DictReader(io.StringIO(bundle.read("entities.csv").decode())))
    edges = list(csv.DictReader(io.StringIO(bundle.read("relations.csv").decode())))
    assert all(not row["label"].startswith(("=", "+", "-", "@")) for row in nodes)
    assert any(row["label"].startswith("'=cmd") for row in nodes)
    assert len(edges) == len(projection.relations) and all(row["fact_ids"] for row in edges)


def test_json_ld_is_linked_data(projection):
    data = json.loads(export_graph(projection, "jsonld").body)
    assert "@context" in data and data["@graph"]
    assert any(item.get("p:works_at") for item in data["@graph"])


def test_the_obsidian_vault_links_notes_with_safe_file_names(projection):
    vault = zipfile.ZipFile(io.BytesIO(export_graph(projection, "obsidian").body))
    names = vault.namelist()
    assert all("/" not in name.removeprefix("entities/") and ".." not in name for name in names)
    alice = next(name for name in names if name.lower().startswith("entities/alice chen"))
    note = vault.read(alice).decode()
    assert "[[" in note and "works_at" in note and "May 2021" in note and "fact 1" in note


def test_the_same_projection_always_exports_the_same_bytes(projection):
    for format in EXPORT_FORMATS:
        assert export_graph(projection, format).body == export_graph(projection, format).body


@pytest.mark.parametrize("format", ["csv", "obsidian", "wiki"])
def test_zipped_exports_carry_a_fixed_timestamp_not_the_clock(projection, format):
    """A zip entry written by name takes the current time; two exports of
    the same projection a second apart would then differ."""
    bundle = zipfile.ZipFile(io.BytesIO(export_graph(projection, format).body))
    assert {info.date_time for info in bundle.infolist()} == {(1980, 1, 1, 0, 0, 0)}


def test_obsidian_notes_escape_names_in_headings_and_lists():
    """Obsidian renders HTML, links, tags and highlights in a note, so a
    stored name must be escaped where the note shows it."""
    names = ['<img src="https://example.invalid/x.png">', "**bold** [x](https://example.invalid) #tag ==mark=="]
    projection = project_entities("alpha", [fact(1, names[0], "knows", "Bob"), fact(2, names[1], "knows", "Bob")],
                                  revision=1)
    bundle = zipfile.ZipFile(io.BytesIO(export_graph(projection, "obsidian").body))
    bodies = [bundle.read(name).decode().split("---", 2)[-1] for name in bundle.namelist()]
    for body in bodies:
        assert "<img" not in body.replace("\\<", "") and "[x]" not in body.replace("\\[", "").replace("\\]", "")
    shown = re.sub(r"\\([!-/:-@\[-`{-~])", r"\1", "\n".join(bodies))
    assert all(f"# {name}" in shown for name in names)


def test_names_that_clash_once_made_safe_get_their_own_notes_and_every_link_opens_one():
    projection = project_entities("alpha", [fact(1, "a/b", "knows", "a:b"), fact(2, "a:b", "knows", "zed")], revision=1)
    bundle = zipfile.ZipFile(io.BytesIO(export_graph(projection, "obsidian").body))
    notes = {name[len("entities/"):-len(".md")] for name in bundle.namelist() if name.startswith("entities/")}
    assert len(notes) == len(projection.entities) == 3
    links = set(re.findall(r"\[\[([^\]]+)\]\]", "\n".join(bundle.read(name).decode() for name in bundle.namelist())))
    assert len(links) == 3 and links <= notes



def text_of(body: bytes) -> str:
    if body[:2] != b"PK":
        return body.decode()
    bundle = zipfile.ZipFile(io.BytesIO(body))
    return "\n".join(bundle.read(name).decode() for name in bundle.namelist())


@pytest.mark.parametrize("format", ["json", "graphml", "gexf", "cypher", "csv", "jsonld", "obsidian", "wiki"])
def test_every_format_keeps_every_value_and_its_facts(projection, format):
    text = text_of(export_graph(projection, format).body)
    assert "May 2021" in text and "joined_on" in text


def test_graphml_and_cypher_hang_each_value_off_its_entity_with_its_facts(projection):
    ns = {"g": "http://graphml.graphdrawing.org/xmlns"}
    root = ElementTree.fromstring(export_graph(projection, "graphml").body)
    values = {node.get("id"): node.find("g:data[@key='label']", ns).text for node in root.iterfind(".//g:node", ns)
              if node.find("g:data[@key='type']", ns).text == "value"}
    edge = next(edge for edge in root.iterfind(".//g:edge", ns) if values.get(edge.get("target")) == "May 2021")
    assert edge.find("g:data[@key='predicate']", ns).text == "joined_on"
    assert edge.find("g:data[@key='fact_ids']", ns).text == "4"
    script = export_graph(projection, "cypher").body.decode()
    line = next(line for line in script.splitlines() if "'May 2021'" in line)
    assert ":HAS_VALUE" in line and "r.predicate = 'joined_on'" in line and "r.fact_ids = [4]" in line


def test_json_ld_predicates_never_collide_with_its_own_fields():
    """A stored predicate may be called label, key or @id; each lives under
    the predicate prefix, so none overwrites the node's own fields."""
    rows = [fact(1, "alice", "label", "Example"), fact(2, "alice", "@id", "urn:evil"), fact(3, "alice", "key", "k"),
            fact(4, "alice", "works_at", "Acme")]
    data = json.loads(export_graph(project_entities("alpha", rows, revision=1), "jsonld").body)
    alice = next(item for item in data["@graph"] if item.get("key") == "alice")
    assert alice["label"] == "alice" and alice["@id"].startswith("urn:scone:alpha:ent:")
    assert {"p:label", "p:%40id", "p:key", "p:works_at"} <= set(alice)
    assert data["@context"]["p"].endswith("/")


def notes_of(rows) -> list[str]:
    projection = project_entities("alpha", rows, revision=1)
    bundle = zipfile.ZipFile(io.BytesIO(export_graph(projection, "obsidian").body))
    notes = [name[len("entities/"):-len(".md")] for name in bundle.namelist() if name.startswith("entities/")]
    assert len(notes) == len(projection.entities)
    return notes


def raw(number: int, subject: str) -> Fact:
    return Fact(fact_id=number, space="alpha", subject=subject, predicate="status", object="ready",
                valid_from="2025-01-01T00:00:00Z")


def test_a_name_spelling_out_another_notes_disambiguated_name_gets_its_own_note():
    """Two names that are equal once made file-safe; then a third that
    spells out the name the second was given. All three keep a note."""
    pair = [raw(1, "item0/x"), raw(2, "item0:x")]
    given = next(note for note in notes_of(pair) if note != "item0-x")
    notes = notes_of([*pair, raw(3, given)])
    assert len(set(notes)) == 3


def test_names_equal_on_a_disk_that_ignores_case_or_normalisation_get_their_own_notes():
    notes = notes_of([raw(1, "A/b"), raw(2, "a:B"), raw(3, "caf\u00e9"), raw(4, "cafe\u0301")])
    assert len({unicodedata.normalize("NFC", note).casefold() for note in notes}) == 4


def test_no_note_takes_the_index_or_a_windows_device_name():
    notes = notes_of([raw(1, "index"), raw(2, "con"), raw(3, "Con.txt"), raw(4, "LPT1")])
    assert not {note.casefold().split(".")[0] for note in notes} & {"index", "con", "lpt1"}

CONTROLLED = "Alice\x01Admin\x0bNote￾"


def test_graphml_carries_characters_xml_cannot_hold_without_breaking():
    """XML 1.0 has no way to write U+0001, a vertical tab or U+FFFE. The
    label shows each as a visible symbol, and ``exact`` keeps the value."""
    projection = project_entities("alpha", [fact(1, CONTROLLED, "knows", "Bob")], revision=1)
    ns = {"g": "http://graphml.graphdrawing.org/xmlns"}
    root = ElementTree.fromstring(export_graph(projection, "graphml").body)
    node = next(node for node in root.iterfind(".//g:node", ns)
                if "admin" in node.find("g:data[@key='label']", ns).text)
    assert node.find("g:data[@key='label']", ns).text == "alice␁admin␋note�"
    assert json.loads(node.find("g:data[@key='exact']", ns).text)["label"] == CONTROLLED.casefold()
    plain = next(node for node in root.iterfind(".//g:node", ns) if node.find("g:data[@key='label']", ns).text == "Bob")
    assert plain.find("g:data[@key='exact']", ns) is None


def test_cypher_and_markdown_write_control_characters_as_escapes():
    projection = project_entities("alpha", [fact(1, CONTROLLED, "knows", "Bob")], revision=1)
    script = export_graph(projection, "cypher").body.decode()
    assert "\\u0001" in script and "\x01" not in script and "\x0b" not in script
    notes = text_of(export_graph(projection, "obsidian").body)
    assert "\x01" not in notes and "␁" in notes


@pytest.mark.parametrize("format", ["json", "graphml", "gexf", "cypher", "csv", "jsonld", "obsidian", "wiki", "mermaid"])
def test_every_format_writes_whatever_the_ledger_holds(format):
    """The ledger accepts NUL and lone surrogates. No export may fail on
    them or write a file its own parser rejects."""
    names = ["nul\x00here", "half\ud800pair", CONTROLLED]
    projection = project_entities("alpha", [fact(n, name, "knows", "Bob") for n, name in enumerate(names, 1)],
                                  revision=1)
    body = export_graph(projection, format).body
    if format in ("json", "jsonld"):
        keys = {json.dumps(item) for item in json.loads(body).get("nodes", json.loads(body).get("@graph"))}
        assert all(any(json.dumps(name.casefold())[1:-1] in key for key in keys) for name in names)
    elif format in ("graphml", "gexf"):
        ElementTree.fromstring(body)
    elif format == "csv":
        cells = zipfile.ZipFile(io.BytesIO(body)).read("entities.csv").decode()
        assert "nul\u2400here" in cells and "half\ufffdpair" in cells
    else:
        text_of(body)



def walk(item):
    yield item
    children = item.values() if isinstance(item, dict) else item if isinstance(item, list) else ()
    for child in children:
        yield from walk(child)


_VALUE_KEYWORDS = {"@value", "@type", "@language", "@index", "@direction"}


def test_json_ld_value_objects_hold_only_json_ld_keywords(projection):
    """A value object may carry nothing but JSON-LD keywords; provenance
    beside "@value" makes the document invalid to a JSON-LD processor."""
    data = json.loads(export_graph(projection, "jsonld").body)
    assert all(set(item) <= _VALUE_KEYWORDS for item in walk(data) if isinstance(item, dict) and "@value" in item)


def test_json_ld_claims_keep_each_relations_own_facts():
    """Two people who know Bob are two claims, each with its own facts;
    the facts are never gathered onto Bob."""
    rows = [fact(1, "alice", "knows", "Bob"), fact(2, "charlie", "knows", "Bob"), fact(3, "alice", "joined_on", "May 2021")]
    projection = project_entities("alpha", rows, revision=1)
    data = json.loads(export_graph(projection, "jsonld").body)
    ids = {entity.key: f"urn:scone:alpha:{entity.entity_id}" for entity in projection.entities}
    claims = [item for item in data["@graph"] if item.get("@type") == "Claim"]
    knows = {(claim["subject"]["@id"], claim["object"]["@id"], tuple(claim["facts"])) for claim in claims
             if claim["predicate"]["@id"] == "p:knows"}
    assert knows == {(ids["alice"], ids["bob"], (1,)), (ids["charlie"], ids["bob"], (2,))}
    value = next(claim for claim in claims if claim["predicate"]["@id"] == "p:joined_on")
    assert value["value"] == "May 2021" and value["facts"] == [3] and value["subject"]["@id"] == ids["alice"]
    bob = next(item for item in data["@graph"] if item.get("@id") == ids["bob"])
    assert "facts" not in bob


def test_json_ld_takes_any_predicate_the_ledger_holds():
    rows = [fact(1, "bob", "bad\ud800predicate", "a value")]
    data = json.loads(export_graph(project_entities("alpha", rows, revision=1), "jsonld").body)
    assert any(key.startswith("p:bad%ED%A0%80predicate") for item in data["@graph"] for key in item)


def test_graphml_key_ids_are_unique(projection):
    root = ElementTree.fromstring(export_graph(projection, "graphml").body)
    ids = [key.get("id") for key in root.iter("{http://graphml.graphdrawing.org/xmlns}key")]
    assert len(ids) == len(set(ids))


@pytest.mark.parametrize("value", ["first\rsecond", "a\r\nb", "tab\there"])
def test_graphml_keeps_carriage_returns_a_parser_would_fold(value):
    """An XML parser folds a raw CR, and CR LF, into LF; written as a
    character reference, the CR survives."""
    projection = project_entities("alpha", [fact(1, "alice", "description", value)], revision=1)
    ns = {"g": "http://graphml.graphdrawing.org/xmlns"}
    root = ElementTree.fromstring(export_graph(projection, "graphml").body)
    shown = [node.find("g:data[@key='label']", ns).text for node in root.iterfind(".//g:node", ns)
             if node.find("g:data[@key='type']", ns).text == "value"]
    assert shown == [value]



def test_json_ld_puts_every_statement_in_the_default_graph(projection):
    """Properties beside "@graph" would turn the document into a node whose
    statements sit in a named graph; the export's own record is a node."""
    data = json.loads(export_graph(projection, "jsonld").body)
    assert set(data) == {"@context", "@graph"}
    record = next(item for item in data["@graph"] if item.get("@type") == "Export")
    assert record["digest"] == projection.digest and "about" in record


GEXF = {"g": "http://www.gexf.net/1.2draft"}


def timed(number, subject, predicate, object_, valid_from, valid_until=None):
    return Fact(fact_id=number, space="alpha", subject=subject, predicate=predicate, object=object_,
                valid_from=valid_from, valid_until=valid_until, status="closed" if valid_until else "active")


def gexf_items(projection):
    root = ElementTree.fromstring(export_graph(projection, "gexf").body)
    graph = root.find("g:graph", GEXF)
    names = {attribute.get("id"): attribute.get("title") for attribute in graph.iterfind("g:attributes/g:attribute", GEXF)}

    def item(element):
        values = {names[value.get("for")]: value.get("value") for value in element.iterfind("g:attvalues/g:attvalue", GEXF)}
        spells = [(spell.get("start"), spell.get("end"), spell.get("endopen"))
                  for spell in element.iterfind("g:spells/g:spell", GEXF)]
        return {"label": element.get("label"), "spells": spells, "timed": element.get("start") or element.get("end"),
                **values}

    nodes = {node.get("id"): item(node) for node in graph.iterfind("g:nodes/g:node", GEXF)}
    edges = {edge.get("id"): {**item(edge), "source": edge.get("source"), "target": edge.get("target")}
             for edge in graph.iterfind("g:edges/g:edge", GEXF)}
    return root, graph, nodes, edges


def test_gexf_is_a_dynamic_graph_where_each_relation_holds_for_its_valid_time():
    """Alice worked at Acme from 2020 until 2023, then at Beta: on Gephi's
    timeline the Acme edge ends where the Beta edge begins, Alice stays."""
    projection = project_entities("alpha", [
        timed(1, "alice", "works_at", "Acme", "2020-01-01T00:00:00Z", "2023-01-01T00:00:00Z"),
        timed(2, "alice", "works_at", "Beta", "2023-01-01T00:00:00Z"),
        timed(3, "alice", "age", "34", "2021-06-01T00:00:00Z")], revision=1)
    root, graph, nodes, edges = gexf_items(projection)
    assert root.get("version") == "1.2"
    assert (graph.get("mode"), graph.get("timeformat"), graph.get("defaultedgetype")) == ("dynamic", "dateTime", "directed")
    by_label = {node["label"]: node for node in nodes.values()}
    assert by_label["alice"]["spells"] == [("2020-01-01T00:00:00.000Z", None, None)]
    assert by_label["Acme"]["spells"] == [("2020-01-01T00:00:00.000Z", None, "2023-01-01T00:00:00.000Z")]
    assert by_label["Beta"]["spells"] == [("2023-01-01T00:00:00.000Z", None, None)]
    assert by_label["34"]["type"] == "value" and by_label["34"]["spells"] == [("2021-06-01T00:00:00.000Z", None, None)]
    spans = {(nodes[edge["target"]]["label"], *edge["spells"][0]) for edge in edges.values()
             if edge["predicate"] == "works_at"}
    assert spans == {("Acme", "2020-01-01T00:00:00.000Z", None, "2023-01-01T00:00:00.000Z"),
                     ("Beta", "2023-01-01T00:00:00.000Z", None, None)}
    assert not any(item["timed"] for item in [*nodes.values(), *edges.values()]), "presence is in spells alone"
    assert {(edge["link"], edge["fact_ids"]) for edge in edges.values()} == {("relation", "1"), ("relation", "2"),
                                                                             ("value", "3")}


def test_gexf_keeps_hostile_names_as_text_and_what_xml_cannot_hold_exactly(projection):
    root, _, nodes, _ = gexf_items(projection)
    assert HOSTILE in {node["label"] for node in nodes.values()}
    described = json.loads(root.find("g:meta/g:description", GEXF).text)
    assert described["digest"] == projection.digest and described["about"] == {}
    assert export_graph(projection, "gexf").media_type == "application/gexf+xml"
    odd = project_entities("alpha", [fact(1, "nul\x00here\rline", "knows", "Bob")], revision=1)
    _, _, nodes, _ = gexf_items(odd)
    shown = next(node for node in nodes.values() if node["label"].startswith("nul"))
    assert shown["label"] == "nul\u2400here\rline" and shown["key"] == "nul\u2400here line"
    assert json.loads(shown["exact"]) == {"label": "nul\x00here\rline", "key": "nul\x00here line"}


def test_gexf_keeps_each_stretch_a_relation_held_not_one_envelope():
    """Acme from 2020, Globex from 2021, Acme again from 2023: Acme's edge
    and node are absent between 2021 and 2023, and Alice, present all the
    while, is one stretch. Each end is exclusive, as valid_until is: GEXF
    writes that as endopen holding the instant, never beside an end."""
    projection = project_entities("alpha", [
        timed(1, "alice", "works_at", "Acme", "2020-01-01T00:00:00Z", "2021-01-01T00:00:00Z"),
        timed(2, "alice", "works_at", "Globex", "2021-01-01T00:00:00Z", "2023-01-01T00:00:00Z"),
        timed(3, "alice", "works_at", "Acme", "2023-01-01T00:00:00Z")], revision=1)
    _, _, nodes, edges = gexf_items(projection)
    by_label = {node["label"]: node for node in nodes.values()}
    acme = [("2020-01-01T00:00:00.000Z", None, "2021-01-01T00:00:00.000Z"), ("2023-01-01T00:00:00.000Z", None, None)]
    assert by_label["Acme"]["spells"] == acme
    assert next(edge for edge in edges.values() if nodes[edge["target"]]["label"] == "Acme")["spells"] == acme
    assert by_label["alice"]["spells"] == [("2020-01-01T00:00:00.000Z", None, None)]


WIKI_LEDGER = [
    fact(1, "alice chen", "works_at", "Acme Robotics"), fact(2, "bob stone", "works_at", "Acme Robotics"),
    fact(3, "acme robotics", "based_in", "Lisbon"), fact(4, "carol diaz", "works_at", "Globex"),
    fact(5, "dan roe", "works_at", "Globex"), fact(6, "globex", "based_in", "Porto"),
    fact(7, "bob stone", "knows", "Carol Diaz"), fact(8, "alice chen", "age", "34"),
    fact(9, "alice chen", "knows", "[x](http://evil.example)"),
]


def wiki(ledger=WIKI_LEDGER):
    bundle = zipfile.ZipFile(io.BytesIO(export_graph(project_entities("alpha", ledger, revision=1), "wiki").body))
    return {name: bundle.read(name).decode() for name in bundle.namelist()}


_LINK = re.compile(r"\]\(([^)\s]+)\)")


def test_the_wiki_is_an_index_topic_articles_and_entity_articles():
    files = wiki()
    topics = [name for name in files if name.startswith("topics/")]
    entities = [name for name in files if name.startswith("entities/")]
    projected = project_entities("alpha", WIKI_LEDGER, revision=1)
    assert "index.md" in files and len(topics) >= 2 and len(entities) == len(projected.entities)
    assert all(name.endswith(".md") for name in files)


def test_every_link_in_the_wiki_opens_a_page_in_it():
    """An agent crawling from index.md reaches every page, and no link
    points outside the wiki or at a page that is not there."""
    import posixpath
    from urllib.parse import unquote

    files = wiki()
    reached, frontier = {"index.md"}, ["index.md"]
    while frontier:
        page = frontier.pop()
        for target in _LINK.findall(files[page]):
            resolved = posixpath.normpath(posixpath.join(posixpath.dirname(page), unquote(target)))
            assert resolved in files, (page, target)
            if resolved not in reached:
                reached.add(resolved)
                frontier.append(resolved)
    assert reached == set(files)


def test_a_topic_lists_its_members_the_relations_inside_and_those_leading_out():
    """A relation inside one topic is listed there once; one between two
    topics is listed as leading out of both, linking the other."""
    files = wiki()
    topics = {name: text for name, text in files.items() if name.startswith("topics/")}
    assert all("## Entities" in text for text in topics.values())
    for number in range(1, 8):
        inside = sum(text.split("## Leading out")[0].count(f"(fact {number})") for text in topics.values())
        leading = sum(text.split("## Leading out")[1].count(f"(fact {number})") for text in topics.values()
                      if "## Leading out" in text)
        assert (inside, leading) in ((1, 0), (0, 2)), number
    assert any("## Leading out" in text and ", other end in topic [" in text.split("## Leading out")[1]
               for text in topics.values())


def test_an_entity_article_cites_every_statement_and_names_its_topic():
    files = wiki()
    alice = next(text for name, text in files.items() if name.startswith("entities/alice chen"))
    assert "(fact 1)" in alice and "(fact 8)" in alice and "34" in alice
    assert "../topics/" in alice


def test_stored_text_cannot_become_a_link_or_markup_in_the_wiki():
    files = wiki()
    everything = "\n".join(files.values())
    assert "http://evil.example" not in _LINK.findall(everything) and "http://evil.example" in everything


def test_a_crowded_entity_lists_its_first_relations_and_counts_the_rest():
    ledger = [fact(n, f"worker {n:03d}", "works_at", "Acme") for n in range(1, 251)]
    acme = next(text for name, text in wiki(ledger).items() if name.lower() == "entities/acme.md")
    assert acme.count("works\\_at (fact") + acme.count("works_at (fact") == 200
    assert "and 50 more" in acme


def mermaid(ledger=LEDGER):
    return export_graph(project_entities("alpha", ledger, revision=1), "mermaid").body.decode()


def test_mermaid_is_a_flowchart_of_entities_and_the_relations_between_them():
    text = mermaid()
    lines = text.splitlines()
    assert lines[0].startswith("%% ") and lines[1] == "flowchart LR"
    nodes = dict(re.findall(r'^  (n\d+)\["([^"]*)"\]$', text, re.M))
    edges = re.findall(r'^  (n\d+) -->\|"([^"]*)"\| (n\d+)$', text, re.M)
    assert "alice chen" in nodes.values() and len(edges) == len(project_entities("alpha", LEDGER, revision=1).relations)
    assert all("(fact" in label for _, label, _ in edges)
    assert export_graph(project_entities("alpha", LEDGER, revision=1), "mermaid").media_type == "text/vnd.mermaid"


def test_mermaid_text_cannot_close_a_label_or_start_markup():
    text = mermaid([fact(1, 'eve "x"] --> evil["y', "knows", "Bob <b>#1</b>")])
    labels = re.findall(r'\["([^"]*)"\]', text)
    assert len(labels) == 2 and all('"' not in label and "<" not in label for label in labels)
    assert any("#quot;" in label for label in labels) and any("#35;1" in label for label in labels)
    assert text.count("-->") == 1, "one relation, one arrow: a name cannot add an edge"


def test_mermaid_shows_the_most_connected_and_counts_the_rest():
    ledger = [fact(n, f"worker {n:03d}", "works_at", "zenith corp") for n in range(1, 80)]
    text = mermaid(ledger)
    assert len(re.findall(r'^  n\d+\[', text, re.M)) == 60 and '  n1["zenith corp"]' in text
    assert text.splitlines()[0].endswith("20 entities and 20 relations left out of the chart; values are not drawn")


def _reachable(files, links):
    import posixpath
    from urllib.parse import unquote

    reached, frontier = {"index.md"}, ["index.md"]
    while frontier:
        page = frontier.pop()
        for target in links(files[page]):
            resolved = posixpath.normpath(posixpath.join(posixpath.dirname(page), unquote(target)))
            assert resolved in files, (page, target)
            if resolved not in reached:
                reached.add(resolved)
                frontier.append(resolved)
    return reached


def _commonmark_links(text):
    markdown_it = pytest.importorskip("markdown_it")
    return [token.attrGet("href") for block in markdown_it.MarkdownIt().parse(text)
            for token in block.children or [] if token.type == "link_open"]


@pytest.mark.parametrize("ledger", [
    [fact(n, f"person {n:03d}", "age", "34") for n in range(1, 251)],
    [fact(n, f"worker {n:03d}", "works_at", "Acme") for n in range(1, 251)],
], ids=["in_no_topic", "one_big_topic"])
def test_every_wiki_page_is_reachable_however_long_a_list_grows(ledger):
    """A list longer than a page continues on numbered pages linking each
    other, so a capped section never leaves an article unreachable."""
    files = wiki(ledger)
    assert _reachable(files, _LINK.findall) == set(files)
    assert _reachable(files, _commonmark_links) == set(files)
    assert any("part 2" in text for text in files.values())


def test_mermaid_says_what_the_read_left_out_and_when_it_was_read():
    about = {"status": "history", "as_of": "2025-01-01T00:00:00.000Z",
             "coverage": {"facts_read": 1, "facts_counted": 1, "truncated": True, "reasons": ["fact_limit"]}}
    head = export_graph(project_entities("alpha", LEDGER, revision=1), "mermaid", about=about).body.decode().splitlines()[0]
    assert "; history facts as of 2025-01-01T00:00:00.000Z; read limited by fact_limit;" in head
    assert head.endswith("every entity and relation read is drawn; values are not drawn")


def test_mermaid_stays_within_what_mermaid_will_draw():
    """Mermaid refuses a chart over 500 edges or 50,000 characters by
    default; the export stays well inside both and says what it left."""
    many = [fact(n, "alice", f"relation_{n:03d}", "Bob") for n in range(1, 502)]
    text = mermaid(many)
    assert text.count("-->") == 300 and "201 relations left out" in text.splitlines()[0]
    long = [fact(n, f"person {n:02d} " + "x" * 5_000, "knows", "Bob " + "y" * 5_000) for n in range(1, 70)]
    text = mermaid(long)
    assert len(text) < 45_000 and max(len(label) for label in re.findall(r'\["([^"]*)"\]', text)) <= 80


def test_mermaid_keeps_to_its_text_budget_when_escapes_swell_the_labels():
    """A '#' is written as '#35;', so clipped labels can still swell four
    times over; edges give way until the chart fits, and it says so."""
    swollen = [fact(n, "alice", f"p{n:03d}" + "#" * 200, "Bob") for n in range(1, 300)]
    text = mermaid(swollen)
    assert len(text) < 45_000 and text.count("-->") < 300
    assert re.search(r"\d+ relations left out of the chart", text.splitlines()[0])


def test_mermaid_budgets_the_utf16_units_a_browser_counts():
    """An emoji is one character in Python and two UTF-16 units in the
    browser, where Mermaid measures its limit; the whole chart, header
    included, is kept inside it by that count."""
    names = [f"person {n:02d} " + "\U0001F600" * 70 for n in range(60)]
    ledger = [fact(n + 1, names[n % 60], "\U0001F600" * 60 + str(n), names[(n + 1) % 60].title()) for n in range(300)]
    text = mermaid(ledger)
    assert len(text.encode("utf-16-le")) // 2 <= 45_000
    assert re.search(r"\d+ relations left out of the chart", text.splitlines()[0]) and "-->" in text


SVG = "{http://www.w3.org/2000/svg}"


def test_svg_draws_each_entity_and_relation_with_a_title_citing_its_facts(projection):
    exported = export_graph(projection, "svg", about={"status": "current", "as_of": "2025-06-01T00:00:00.000Z"})
    assert exported.media_type == "image/svg+xml" and exported.filename == "graph.svg"
    root = ElementTree.fromstring(exported.body)
    assert root.tag == f"{SVG}svg" and root.get("role") == "img"
    assert root.find(f"{SVG}title").text == "Knowledge graph alpha"
    assert "current facts as of 2025-06-01T00:00:00.000Z" in root.find(f"{SVG}desc").text
    circles = root.findall(f".//{SVG}circle")
    assert len(circles) == len(projection.entities)
    titles = [element.find(f"{SVG}title").text for element in root.iter() if element.tag == f"{SVG}line"]
    assert len(titles) == len(projection.relations)
    assert any(title.startswith("alice chen works_at Acme, Inc.") and title.endswith("(fact 1)") for title in titles)
    names = [text for text in root.iter(f"{SVG}text") if text.get("text-anchor") == "middle"]
    assert names and all(text.get("paint-order") == "stroke" for text in names), "a halo keeps names legible"


def test_svg_keeps_hostile_names_as_text():
    hostile = 'Evil</text><script>alert(1)</script>\x07 & co'
    projection = project_entities("alpha", [fact(1, hostile, "knows", "Alice Chen")], revision=1)
    body = export_graph(projection, "svg").body
    assert b"<script" not in body and b"\x07" not in body
    shown = [element.text for element in ElementTree.fromstring(body).iter(f"{SVG}text")]
    assert any(text and text.startswith("evil</text><script>") for text in shown)


def test_svg_says_what_the_drawing_and_the_read_left_out(monkeypatch):
    from scone_memory.entities import export as export_module

    monkeypatch.setattr(export_module, "_SVG_NODES", 2)
    projection = project_entities("alpha", LEDGER, revision=3)
    about = {"status": "history", "as_of": "2025-06-01T00:00:00.000Z",
             "coverage": {"facts_read": 5, "facts_counted": 5, "truncated": True, "reasons": ["fact_limit"]}}
    desc = ElementTree.fromstring(export_graph(projection, "svg", about=about).body).find(f"{SVG}desc").text
    assert "read limited by fact_limit" in desc and "3 entities and 3 relations left out of the drawing" in desc


def test_canvas_is_a_json_canvas_of_community_groups_cards_and_labelled_edges(projection):
    exported = export_graph(projection, "canvas")
    assert exported.filename == "graph.canvas" and exported.media_type == "application/json"
    canvas = json.loads(exported.body)
    groups = [node for node in canvas["nodes"] if node["type"] == "group"]
    cards = [node for node in canvas["nodes"] if node["type"] == "text" and node["id"] != "about"]
    assert groups and len(cards) == len(projection.entities)
    [about] = [node for node in canvas["nodes"] if node["id"] == "about"]
    assert about["text"].startswith("**Knowledge graph alpha**") and "every entity and relation read is drawn" in about["text"]
    assert all(isinstance(node[side], int) for node in canvas["nodes"] for side in ("x", "y", "width", "height"))
    ids = {node["id"] for node in canvas["nodes"]}
    assert len(ids) == len(canvas["nodes"]) and len(canvas["edges"]) == len(projection.relations)
    assert all(edge["fromNode"] in ids and edge["toNode"] in ids and edge["toEnd"] == "arrow" for edge in canvas["edges"])
    assert any(edge["label"] == "works_at (fact 1)" for edge in canvas["edges"])
    for card in cards:
        assert any(group["x"] <= card["x"] and card["x"] + card["width"] <= group["x"] + group["width"]
                   and group["y"] <= card["y"] and card["y"] + card["height"] <= group["y"] + group["height"]
                   for group in groups), card


def test_canvas_cards_escape_names_as_notes_do():
    names = ['<img src="https://example.invalid/x.png">', "**bold** [x](https://example.invalid)"]
    projection = project_entities("alpha", [fact(1, names[0], "knows", "Bob"), fact(2, names[1], "knows", "Bob")],
                                  revision=1)
    texts = [node["text"] for node in json.loads(export_graph(projection, "canvas").body)["nodes"]
             if node["type"] == "text" and node["id"] != "about"]
    assert all("<img" not in text.replace("\\<", "") and "[x]" not in text.replace("\\[", "").replace("\\]", "")
               for text in texts)


def test_the_obsidian_vault_carries_a_canvas_of_its_notes(projection):
    vault = zipfile.ZipFile(io.BytesIO(export_graph(projection, "obsidian").body))
    canvas = json.loads(vault.read("graph.canvas"))
    files = [node["file"] for node in canvas["nodes"] if node["type"] == "file"]
    assert len(files) == len(projection.entities) and all(name in vault.namelist() for name in files)


def test_svg_box_titles_fit_their_boxes():
    triples = [(f"worker {n}", "works_at", "Acme Robotics International Holdings") for n in range(2)]
    projection = project_entities("alpha", [fact(n + 1, *triple) for n, triple in enumerate(triples)], revision=1)
    root = ElementTree.fromstring(export_graph(projection, "svg").body)
    [box] = root.find(f"{SVG}g[@class='communities']").findall(f"{SVG}rect")
    [title] = root.find(f"{SVG}g[@class='communities']").findall(f"{SVG}text")
    assert len(title.text) * 6.6 <= float(box.get("width")) - 24 and title.text.endswith("…")


def test_svg_dashes_the_arrows_between_communities_and_loops_a_self_relation():
    triples = [(f"worker {n}", "works_at", "Acme") for n in range(3)] + [
        (f"resident {n}", "lives_in", "Porto") for n in range(3)] + [
        ("worker 0", "lives_in", "Porto"), ("worker 1", "knows", "Worker 1")]
    projection = project_entities("alpha", [fact(n + 1, *triple) for n, triple in enumerate(triples)], revision=1)
    root = ElementTree.fromstring(export_graph(projection, "svg").body)
    lines = {element.find(f"{SVG}title").text: element for element in root.iter(f"{SVG}line")}
    across = lines["worker 0 lives_in Porto (fact 7)"]
    inside = lines["worker 2 works_at Acme (fact 3)"]
    assert across.get("stroke-dasharray") == "5 4" and inside.get("stroke-dasharray") is None
    [loop] = [element for element in root.iter(f"{SVG}path") if element.find(f"{SVG}title") is not None]
    assert loop.find(f"{SVG}title").text == "worker 1 knows worker 1 (fact 8)"


def test_canvas_cards_take_their_communitys_colour(projection):
    canvas = json.loads(export_graph(projection, "canvas").body)
    colour = {node["id"].removeprefix("group-"): node.get("color") for node in canvas["nodes"] if node["type"] == "group"}
    cards = [node for node in canvas["nodes"] if node["type"] == "text" and node["id"] != "about"]
    assert any("color" in card for card in cards)
    for card in cards:
        group = next(g for g in canvas["nodes"] if g["type"] == "group" and g["x"] <= card["x"] <= g["x"] + g["width"]
                     and g["y"] <= card["y"] <= g["y"] + g["height"])
        assert card.get("color") == colour[group["id"].removeprefix("group-")]


def _page(body: bytes):
    from html.parser import HTMLParser

    class Page(HTMLParser):
        def __init__(self):
            super().__init__()
            self.tags, self.data, self.inside = [], {}, None

        def handle_starttag(self, tag, attrs):
            self.tags.append((tag, dict(attrs)))
            self.inside = dict(attrs).get("id") if tag == "script" else None

        def handle_data(self, data):
            if self.inside:
                self.data[self.inside] = self.data.get(self.inside, "") + data

    page = Page()
    page.feed(body.decode("utf-8"))
    return page


def test_html_is_one_page_with_the_drawing_its_data_and_nothing_fetched(projection):
    exported = export_graph(projection, "html", about={"status": "current", "as_of": "2025-06-01T00:00:00.000Z"})
    assert exported.media_type == "text/html" and exported.filename == "graph.html"
    page = _page(exported.body)
    tags = [tag for tag, _ in page.tags]
    assert tags.count("svg") == 1 and "input" in tags
    assert not any(attrs.get("src") or (tag == "link" and not attrs.get("href", "").startswith("data:"))
                   for tag, attrs in page.tags), "nothing fetched"
    assert ("link", {"rel": "icon", "href": "data:,"}) in page.tags, "not even an icon"
    data = json.loads(page.data["graph-data"])
    assert {node["id"] for node in data["nodes"]} == {entity.entity_id for entity in projection.entities}
    shapes = {attrs["data-entity"] for tag, attrs in page.tags if tag == "g" and "data-entity" in attrs}
    assert shapes == {node["id"] for node in data["nodes"]}, "every entity's shape can be found by its id"
    works = next(edge for edge in data["edges"] if edge["predicate"] == "works_at")
    assert works["fact_ids"] == [1] and data["about"].startswith("projection ")
    assert b"Content-Security-Policy" in exported.body and b"default-src 'none'" in exported.body


def test_html_keeps_hostile_names_as_data():
    hostile = '</script><script>alert(1)</script><img src=x onerror=alert(2)>'
    projection = project_entities("alpha", [fact(1, hostile, "knows", "Alice Chen")], revision=1)
    body = export_graph(projection, "html").body
    assert body.count(b"<script") == 2, "only the page's own data and code"
    assert b"<img" not in body and b"onerror" not in body.replace(b"onerror=alert", b"")
    data = json.loads(_page(body).data["graph-data"])
    assert any(node["label"] == hostile for node in data["nodes"])


def test_html_code_writes_names_as_text_never_as_markup(projection):
    code = _page(export_graph(projection, "html").body).data["graph-code"]
    assert "textContent" in code and "innerHTML" not in code and "eval(" not in code


def _label_boxes(root):
    """Each name's box as the SVG fixes it: textLength wide, centred on x."""
    boxes = []
    for text in root.iter(f"{SVG}text"):
        if text.get("text-anchor") == "middle":
            width, x, y = float(text.get("textLength")), float(text.get("x")), float(text.get("y"))
            assert text.get("lengthAdjust") == "spacingAndGlyphs"
            boxes.append((x - width / 2, y - 11, x + width / 2, y + 3))
    return boxes


@pytest.mark.parametrize("workers", [1, 20])
def test_svg_names_fit_the_drawing_and_never_overlap(workers):
    """Long names on a crowded ring: every name is fitted to the width the
    layout made room for, inside the drawing and clear of every other."""
    triples = [(f"Alexandria Montgomery Researcher {n}", "works_at", "Wellington International Company")
               for n in range(workers)]
    projection = project_entities("alpha", [fact(n + 1, *triple) for n, triple in enumerate(triples)], revision=1)
    root = ElementTree.fromstring(export_graph(projection, "svg").body)
    _, _, width, height = map(float, root.get("viewBox").split())
    boxes = _label_boxes(root)
    assert len(boxes) == workers + 1
    for left, top, right, bottom in boxes:
        assert 0 <= left and right <= width and 0 <= top and bottom <= height
    for index, a in enumerate(boxes):
        for b in boxes[index + 1:]:
            assert a[2] <= b[0] or b[2] <= a[0] or a[3] <= b[1] or b[3] <= a[1], (a, b)
    circles = [(float(c.get("cx")), float(c.get("cy")), float(c.get("r"))) for c in root.iter(f"{SVG}circle")]
    for left, top, right, bottom in boxes:
        for cx, cy, r in circles:
            nearest = (min(max(cx, left), right), min(max(cy, top), bottom))
            assert math.dist(nearest, (cx, cy)) >= r - 1e-9 or top >= cy + r - 1e-9, "no name over a circle"


def test_svg_box_titles_are_fitted_to_their_boxes():
    triples = [(f"worker {n}", "works_at", "Acme Robotics International Holdings") for n in range(2)]
    projection = project_entities("alpha", [fact(n + 1, *triple) for n, triple in enumerate(triples)], revision=1)
    root = ElementTree.fromstring(export_graph(projection, "svg").body)
    [box] = root.find(f"{SVG}g[@class='communities']").findall(f"{SVG}rect")
    [title] = root.find(f"{SVG}g[@class='communities']").findall(f"{SVG}text")
    assert float(title.get("textLength")) <= float(box.get("width")) - 24
    assert title.get("lengthAdjust") == "spacingAndGlyphs"


def test_a_names_width_is_estimated_from_its_letters():
    """Fitting a name to its estimate distorts it little only if the
    estimate is close: narrow letters, capitals and wide characters differ."""
    from scone_memory.entities.export import _text_width

    assert _text_width("illi") < _text_width("oooo") < _text_width("OOOO") < _text_width("MMMM") < _text_width("漢字漢字")
