"""Every drawn edge says what backs it.

A relation in the entity graph is one or more facts, and those facts
differ in ways a reader has to know before trusting the line: stated by
a person or inferred by the framework, holding or only proposed or
excluded, quoted from a source or resting on no source at all. The
projection already counts all of that per relation. The drawings threw
it away: every arrow looked alike, styled only by whether it crossed
communities and how many facts it had. So an inferred, proposed claim
with no source was drawn exactly like a quoted, stated one.

Each export now carries the backing of every edge -- origin, status and
grounding -- as data, and the visual formats draw the difference.
"""
from __future__ import annotations

import json
import xml.etree.ElementTree as ElementTree

from scone_memory.core.models import Fact
from scone_memory.entities.export import export_graph
from scone_memory.entities.project import backing_of, project_entities

SVG = "{http://www.w3.org/2000/svg}"


def fact(number, subject, predicate, object_, **fields):
    fields.setdefault("valid_from", "2025-01-01T00:00:00Z")
    return Fact(fact_id=number, space="alpha", subject=subject, predicate=predicate, object=object_, **fields)


LEDGER = [
    fact(1, "alice chen", "works_at", "Acme Robotics", source_episode_id=1, quote="Alice works at Acme Robotics"),
    fact(2, "acme robotics", "based_in", "Lisbon", origin="inferred", source_episode_id=2),
    fact(3, "bob stone", "knows", "Alice Chen", status="proposed"),
    fact(4, "bob stone", "lives_in", "Lisbon", source_episode_id=3, quote="Bob lives in Lisbon", excluded_reason="a person said so"),
]


def projected():
    return project_entities("alpha", LEDGER, revision=1)


def by_predicate(projection):
    return {relation.predicate: relation for relation in projection.relations}


def test_backing_is_read_off_the_support_counts():
    relations = by_predicate(projected())
    assert backing_of(relations["works_at"].support) == ("stated", "active", "quoted")
    assert backing_of(relations["based_in"].support) == ("inferred", "active", "unquoted")
    assert backing_of(relations["knows"].support) == ("stated", "proposed", "unsourced")
    assert backing_of(relations["lives_in"].support) == ("stated", "excluded", "quoted")


def test_a_relation_with_one_held_fact_among_excluded_ones_is_active():
    ledger = [fact(1, "alice chen", "works_at", "Acme Robotics", source_episode_id=1, quote="q"),
              fact(2, "alice chen", "works_at", "Acme Robotics", source_episode_id=2, quote="q", excluded_reason="x",
                   valid_from="2025-02-01T00:00:00Z")]
    [relation] = project_entities("alpha", ledger, revision=1).relations
    assert backing_of(relation.support)[1] == "active"


def test_mixed_origins_say_mixed():
    ledger = [fact(1, "alice chen", "works_at", "Acme Robotics", source_episode_id=1, quote="q"),
              fact(2, "alice chen", "works_at", "Acme Robotics", origin="inferred", valid_from="2025-02-01T00:00:00Z")]
    [relation] = project_entities("alpha", ledger, revision=1).relations
    assert backing_of(relation.support)[0] == "mixed"


def test_every_svg_arrow_carries_its_backing_and_draws_the_difference():
    root = ElementTree.fromstring(export_graph(projected(), "svg").body)
    arrows = {element.get("data-relation"): element for element in root.iter()
              if element.get("data-relation") is not None}
    relations = by_predicate(projected())
    stated = arrows[relations["works_at"].relation_id]
    inferred = arrows[relations["based_in"].relation_id]
    proposed = arrows[relations["knows"].relation_id]
    excluded = arrows[relations["lives_in"].relation_id]
    assert (stated.get("data-origin"), stated.get("data-status"), stated.get("data-grounding")) == ("stated", "active", "quoted")
    assert (inferred.get("data-origin"), proposed.get("data-status"), proposed.get("data-grounding")) == ("inferred", "proposed", "unsourced")
    assert excluded.get("data-status") == "excluded"
    assert inferred.get("stroke-dasharray") == "2 3" and stated.get("stroke-dasharray") in (None, "5 4")
    assert float(proposed.get("stroke-opacity")) < float(stated.get("stroke-opacity"))
    assert float(excluded.get("stroke-opacity")) < float(stated.get("stroke-opacity"))
    assert proposed.get("marker-start") == "url(#unsourced)" and stated.get("marker-start") is None
    title = proposed.find(f"{SVG}title").text
    assert "proposed" in title and "stated" in title and "unsourced" in title, title


def test_the_html_page_data_carries_each_edges_backing():
    body = export_graph(projected(), "html").body.decode()
    start = body.index('<script id="graph-data"')
    data = json.loads(body[body.index(">", start) + 1:body.index("</script>", start)])
    backing = {edge["predicate"]: (edge["origin"], edge["status"], edge["grounding"]) for edge in data["edges"]}
    assert backing["based_in"] == ("inferred", "active", "unquoted")
    assert backing["knows"] == ("stated", "proposed", "unsourced")


def test_mermaid_draws_anything_not_stated_and_active_as_a_dotted_arrow():
    text = export_graph(projected(), "mermaid").body.decode()
    lines = {line for line in text.splitlines() if "|" in line}
    assert any("works_at" in line and " -->|" in line for line in lines), lines
    assert any("based_in" in line and " -.->|" in line for line in lines), lines
    assert any("knows" in line and " -.->|" in line for line in lines), lines


def test_wiki_and_canvas_name_a_weak_backing_and_leave_a_strong_one_unmarked():
    import io
    import zipfile

    bundle = zipfile.ZipFile(io.BytesIO(export_graph(projected(), "wiki").body))
    pages = "\n".join(bundle.read(name).decode() for name in bundle.namelist() if name.endswith(".md"))
    assert "[inferred]" in pages and "[proposed]" in pages
    works = [line for line in pages.splitlines() if "works_at" in line and line.startswith("- ")]
    assert works and not any(line.rstrip().endswith("]") for line in works), works
    canvas = json.loads(export_graph(projected(), "canvas").body)
    labels = {edge["label"] for edge in canvas["edges"]}
    assert any(label.startswith("based_in") and label.endswith("[inferred]") for label in labels), labels


def test_an_all_stated_active_quoted_graph_draws_as_before():
    """The other half: nothing about a strong edge changes but its data."""
    ledger = [fact(1, "alice chen", "works_at", "Acme Robotics", source_episode_id=1, quote="q"),
              fact(2, "acme robotics", "based_in", "Lisbon", source_episode_id=1, quote="q")]
    projection = project_entities("alpha", ledger, revision=1)
    text = export_graph(projection, "mermaid").body.decode()
    assert " -.->|" not in text
    root = ElementTree.fromstring(export_graph(projection, "svg").body)
    arrows = [element for element in root.iter() if element.get("data-relation") is not None]
    assert arrows and all(a.get("marker-start") is None and a.get("stroke-dasharray") in (None, "5 4") for a in arrows)


def test_an_edge_is_as_grounded_as_its_best_fact():
    """A quoted fact and an unsourced restatement of the same relation: the
    edge is quoted, and its counts still say one fact has no source."""
    ledger = [fact(1, "alice chen", "works_at", "Acme Robotics"),
              fact(2, "alice chen", "works_at", "Acme Robotics", source_episode_id=1, quote="q", valid_from="2025-02-01T00:00:00Z")]
    [relation] = project_entities("alpha", ledger, revision=1).relations
    assert backing_of(relation.support)[2] == "quoted"
    assert relation.support.unsourced == 1


def test_an_extracted_quoted_claim_is_drawn_solid_like_a_stated_one():
    """A claim read from a source and quoted from it -- every claim in a code
    graph -- is not weaker than a stated one. Drawing it dotted would make
    a code graph uniformly dotted, which says nothing."""
    ledger = [fact(1, "pkg/a.py", "imports", "pkg/b.py", origin="extracted", source_episode_id=1, quote="import b"),
              fact(2, "pkg/b.py", "imports", "pkg/c.py", origin="extracted", source_episode_id=2, quote="import c")]
    projection = project_entities("alpha", ledger, revision=1)
    root = ElementTree.fromstring(export_graph(projection, "svg").body)
    arrows = [element for element in root.iter() if element.get("data-relation") is not None]
    assert arrows and all(a.get("data-origin") == "extracted" and a.get("stroke-dasharray") in (None, "5 4") for a in arrows)
    assert " -.->|" not in export_graph(projection, "mermaid").body.decode()
