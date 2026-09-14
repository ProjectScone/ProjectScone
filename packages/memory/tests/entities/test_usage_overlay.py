"""Which drawn entities recent recalls reach, on the drawing itself.

The report already names the most recalled entities and the central ones
no recall returned. Given the same counts, the SVG and the page say on
each entity how many of the recalls read returned one of its facts, say
over which window, and the page can dim the entities none returned. The
counts are a window the event log keeps, never all time, and an unknown
count is never drawn as zero.
"""
from __future__ import annotations

import json
import xml.etree.ElementTree as ElementTree

from scone_memory.core.models import Fact
from scone_memory.entities.export import export_graph
from scone_memory.entities.project import project_entities
from scone_memory.entities.usage import Usage

SVG = "{http://www.w3.org/2000/svg}"


def fact(number: int, subject: str, predicate: str, object_: str) -> Fact:
    return Fact(fact_id=number, space="alpha", subject=subject.casefold(), predicate=predicate, object=object_,
                valid_from="2025-01-01T00:00:00Z")


LEDGER = [fact(1, "alice chen", "works_at", "Acme"), fact(2, "acme", "based_in", "Lisbon"),
          fact(3, "bob", "knows", "Carol")]
# Two recalls: one returned fact 1, the other facts 1 and 2. Nothing returned fact 3.
READ = Usage(returned=(frozenset({1}), frozenset({1, 2})), recalls_read=2, oldest="2025-05-01T00:00:00.000Z",
             retention={"max_events": 5000})


def _titles(body: bytes) -> dict[str, str]:
    root = ElementTree.fromstring(body)
    return {circle.find(f"{SVG}title").text.split(" (")[0]: circle.find(f"{SVG}title").text
            for circle in root.iter(f"{SVG}circle")}


def _drawing(page: bytes) -> ElementTree.Element:
    text = page.decode()
    start = text.index("<svg", text.index('id="drawing"'))
    return ElementTree.fromstring(text[start:text.index("</svg>", start) + 6])


def test_each_drawn_entity_says_how_many_recalls_returned_it():
    projection = project_entities("alpha", LEDGER, revision=1)
    titles = _titles(export_graph(projection, "svg", usage=READ).body)
    assert titles["alice chen"].endswith("returned by 2 of the 2 recalls read")
    assert titles["Acme"].endswith("returned by 2 of the 2 recalls read"), "each recall counts once for an entity"
    assert titles["Lisbon"].endswith("returned by 1 of the 2 recalls read")
    assert titles["bob"].endswith("returned by 0 of the 2 recalls read")


def test_the_drawing_says_which_recalls_it_counted():
    projection = project_entities("alpha", LEDGER, revision=1)
    desc = ElementTree.fromstring(export_graph(projection, "svg", usage=READ).body).find(f"{SVG}desc").text
    assert "recall use over the 2 recalls the event log keeps, the oldest from 2025-05-01T00:00:00.000Z" in desc
    cut = Usage(returned=(frozenset({1}),), recalls_read=1, truncated=True, since="2025-04-01T00:00:00.000Z")
    desc = ElementTree.fromstring(export_graph(projection, "svg", usage=cut).body).find(f"{SVG}desc").text
    assert "recall use over the 1 recall the event log keeps since 2025-04-01T00:00:00.000Z (older ones unread)" in desc


def test_an_unknown_count_is_never_drawn_as_zero():
    projection = project_entities("alpha", LEDGER, revision=1)
    for usage, said in ((Usage(available=False), "recall use unknown: the engine keeps no events"),
                        (Usage(), "recall use unknown: the event log keeps no recalls")):
        body = export_graph(projection, "svg", usage=usage).body
        assert said in ElementTree.fromstring(body).find(f"{SVG}desc").text
        assert not any("returned by" in title for title in _titles(body).values())


def test_without_usage_the_drawing_is_unchanged():
    projection = project_entities("alpha", LEDGER, revision=1)
    assert export_graph(projection, "svg").body == export_graph(projection, "svg", usage=None).body
    markup = export_graph(projection, "html").body.split(b'<script id="graph-code">')[0]
    assert b"recall use" not in markup and b"data-recalled" not in markup and b"usage-dim" not in markup


def test_the_page_carries_the_counts_and_a_switch_to_dim_what_no_recall_returned():
    projection = project_entities("alpha", LEDGER, revision=1)
    page = export_graph(projection, "html", usage=READ).body
    text = page.decode()
    block = text.split('<script id="graph-data" type="application/json">', 1)[1].split("</script>", 1)[0]
    data = json.loads(block)
    recalled = {node["label"]: node["recalled"] for node in data["nodes"]}
    assert recalled == {"alice chen": 2, "Acme": 2, "Lisbon": 1, "bob": 0, "Carol": 0}
    assert data["usage"]["recalls_read"] == 2 and data["usage"]["available"] is True
    marks = {group.get("data-entity"): group.get("data-recalled") for group in _drawing(page).iter(f"{SVG}g")
             if group.get("data-entity")}
    assert sorted(marks.values()) == ["0", "0", "1", "2", "2"]
    assert '<input type="checkbox" id="usage-dim">' in text and "Dim what no recall returned" in text
    code = text.split('<script id="graph-code">', 1)[1]
    assert "usage-dim" in code and "data-recalled" in code


def test_the_page_offers_no_switch_when_the_count_is_unknown():
    projection = project_entities("alpha", LEDGER, revision=1)
    text = export_graph(projection, "html", usage=Usage(available=False)).body.decode()
    assert 'id="usage-dim"' not in text and "data-recalled" not in text.split('<script id="graph-code">')[0]
    assert "recall use unknown" in text
