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
    assert set(EXPORT_FORMATS) == {"json", "graphml", "cypher", "csv", "jsonld", "obsidian"}


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
    kinds = [edge.find("g:data[@key='kind']", ns).text for edge in root.iterfind(".//g:edge", ns)]
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


@pytest.mark.parametrize("format", ["csv", "obsidian"])
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


@pytest.mark.parametrize("format", ["json", "graphml", "cypher", "csv", "jsonld", "obsidian"])
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
    alice = next(item for item in data["@graph"] if item["key"] == "alice")
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


@pytest.mark.parametrize("format", ["json", "graphml", "cypher", "csv", "jsonld", "obsidian"])
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
    elif format == "graphml":
        ElementTree.fromstring(body)
    elif format == "csv":
        cells = zipfile.ZipFile(io.BytesIO(body)).read("entities.csv").decode()
        assert "nul\u2400here" in cells and "half\ufffdpair" in cells
    else:
        text_of(body)
