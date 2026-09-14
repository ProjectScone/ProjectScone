"""What a graph names and never reads is kept apart: not a community's
name, not a central entity, not a surprise, not the middle of the picture."""
from __future__ import annotations

from scone_memory.core.models import Fact
from scone_memory.entities.analysis import analyze_projection, external_entities
from scone_memory.entities.layout import layout_projection
from scone_memory.entities.project import project_entities
from scone_memory.entities.report import build_report, render_markdown


def fact(number: int, subject: str, predicate: str, object_: str) -> Fact:
    return Fact(fact_id=number, space="alpha", subject=subject.casefold(), predicate=predicate, object=object_,
                valid_from="2025-01-01T00:00:00Z")


def codebase() -> list[Fact]:
    """Two packages of three files each, every file importing `typing`,
    `json` and a third-party class, each package's files importing one
    another, one import across."""
    rows: list[Fact] = []
    number = 0

    def say(subject: str, predicate: str, obj: str) -> None:
        nonlocal number
        number += 1
        rows.append(fact(number, subject, predicate, obj))

    for package in ("app", "lib"):
        files = [f"{package}/{name}.py" for name in ("a", "b", "c")]
        for path in files:
            say(path, "defines", f"{path}:run")
            say(path, "imports", "typing")
            say(path, "imports", "json")
            say(path, "imports", "pydantic.BaseModel")
        for left in files:
            for right in files:
                if left != right:
                    say(left, "imports", right)
    say("app/a.py", "imports", "lib/a.py")
    say("app/a.py", "cites", "ADR-12")
    return rows


def labels(projection):
    return {entity.entity_id: entity.key for entity in projection.entities}


def test_what_is_named_and_never_read_is_external():
    projection = project_entities("alpha", codebase(), revision=1)
    names = labels(projection)
    external = {names[entity_id] for entity_id in external_entities(projection)}
    assert external == {"typing", "json", "pydantic.basemodel"}, "imported everywhere, defines nothing here"
    assert "adr-12" not in labels(projection).values(), "a cited record is a value of the file here, not an entity"
    assert "lib/a.py" not in external, "a module of the codebase is imported too, but it defines its own things"


def test_communities_are_the_codebases_own_and_named_by_it():
    projection = project_entities("alpha", codebase(), revision=1)
    names = labels(projection)
    analysis = analyze_projection(projection)
    assert analysis.coverage.external_entities == 3 and len(analysis.external) == 3
    assert len(analysis.communities) == 2, "two packages, not one blob tied together by `typing`"
    for community in analysis.communities:
        own = {names[member] for member in community.members if member not in analysis.external}
        assert own and all(name.startswith(("app/", "lib/")) for name in own)
        assert all(names[top] not in ("typing", "json", "pydantic.basemodel") for top in community.top_entities)
        assert "typing" not in community.label
    ranked = [names[item.entity_id] for item in analysis.importance if not item.external][:5]
    assert all(name.startswith(("app/", "lib/")) for name in ranked)
    assert {names[item.entity_id] for item in analysis.importance if item.external} == {"typing", "json", "pydantic.basemodel"}
    assert not any(names[surprise.object_id] in ("typing", "json") for surprise in analysis.surprising_connections), \
        "that a second community imports `typing` too is no surprise"
    assert not any(names[suggestion.entity_ids[0]] == "typing" for suggestion in analysis.suggestions
                   if suggestion.kind == "bridge"), "`typing` bridges everything and is no bridge worth asking about"
    attached = {names[node]: community for community in analysis.communities for node in community.members
                if node in analysis.external}
    assert "typing" in attached, "an external is attached, for reading, to the community that names it most"


def test_the_report_leads_with_the_codebases_own_and_lists_the_rest_apart():
    projection = project_entities("alpha", codebase(), revision=1)
    analysis = analyze_projection(projection)
    report = build_report(projection, analysis, meta={"digest": "d", "revision": 1},
                          filters={"status": "current", "as_of": "now"}, coverage={})
    central = [item["label"] for item in report["central_entities"]]
    assert central and all(name.startswith(("app/", "lib/")) for name in central), central
    apart = [item["label"] for item in report["external_dependencies"]]
    assert apart == ["json", "pydantic.BaseModel", "typing"], "by how many things name them, then by name"
    assert all(item["named_by"] == 6 for item in report["external_dependencies"])
    assert report["summary"]["external_entities"] == 3
    text = render_markdown(report)
    assert "## Named but never read" in text and "| typing | 6 |" in text
    assert "3 entities named but never read" in text
    assert not any(name.startswith(("app/", "lib/")) for name in apart)


def test_the_drawing_spends_its_room_on_the_codebase_first():
    projection = project_entities("alpha", codebase(), revision=1)
    names = labels(projection)
    drawing = layout_projection(projection, max_nodes=12, max_edges=100)
    shown = {names[node.entity_id] for node in drawing.nodes}
    assert not shown & {"typing", "json", "pydantic.basemodel"}, "the room went to the code, not to what it imports"
    assert len(shown) == 12 and all(name.startswith(("app/", "lib/")) for name in shown)
    whole = layout_projection(projection, max_nodes=200, max_edges=1000)
    assert "typing" in {names[node.entity_id] for node in whole.nodes}, "with room to spare, the imports are drawn too"


def test_a_graph_of_people_has_nothing_external():
    rows = [fact(1, "Ana", "knows", "Ben"), fact(2, "Ben", "works_at", "Acme"), fact(3, "Ana", "works_at", "Acme")]
    projection = project_entities("alpha", rows, revision=1)
    assert external_entities(projection) == frozenset()
    analysis = analyze_projection(projection)
    assert analysis.coverage.external_entities == 0 and not any(item.external for item in analysis.importance)
