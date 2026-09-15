"""Communities in the Obsidian vault: a hub note each, a tag, a colour.

The vault had a note per entity and an index. It now has a note per
community that the analysis found, listing its members and the
communities it links to. Each member's note carries the community's tag
and links to that note, and the vault's graph view colours each
community by that tag, in the drawings' colours. A tag is built once and
used for both, so a colour group never queries a tag no note carries.
"""
from __future__ import annotations

import io
import json
import re
import zipfile

from scone_memory.core.models import Fact
from scone_memory.entities.analysis import cached_analysis
from scone_memory.entities.export import _PALETTE, export_graph
from scone_memory.entities.project import project_entities


def fact(number: int, subject: str, predicate: str, object_: str) -> Fact:
    return Fact(fact_id=number, space="alpha", subject=subject.casefold(), predicate=predicate, object=object_,
                valid_from="2025-01-01T00:00:00Z")


def ledger():
    triples = [(f"a{k}", "knows", "Hub A") for k in range(5)] + [(f"b{k}", "knows", "Hub B") for k in range(4)] + [
        ("Hub A", "knows", "Hub B"), ("a0", "knows", "b0"), ("Ann", "age", "34")]
    return [fact(n + 1, *triple) for n, triple in enumerate(triples)]


def vault(projection):
    bundle = zipfile.ZipFile(io.BytesIO(export_graph(projection, "obsidian").body))
    return {name: bundle.read(name).decode() for name in bundle.namelist()}


def front(note: str) -> dict[str, str]:
    head = note.split("---", 2)[1]
    return dict(line.split(": ", 1) for line in head.strip().splitlines())


def tags(note: str) -> list[str]:
    value = front(note).get("tags")
    return json.loads(value) if value else []


def test_each_community_has_a_hub_note_listing_its_members():
    projection = project_entities("alpha", ledger(), revision=1)
    files = vault(projection)
    communities = cached_analysis(projection).communities
    hubs = {front(text)["id"]: (name, text) for name, text in files.items() if name.startswith("communities/")}
    assert set(hubs) == {community.community_id for community in communities}
    notes = {front(text)["id"]: name[len("entities/"):-len(".md")] for name, text in files.items()
             if name.startswith("entities/")}
    for community in communities:
        name, text = hubs[community.community_id]
        members = text.split("## Members", 1)[1].split("##", 1)[0]
        assert sorted(re.findall(r"^- \[\[([^\]]+)\]\]$", members, re.M)) == sorted(notes[m] for m in community.members)
        inside = f"{community.internal_links} link{'' if community.internal_links == 1 else 's'} inside"
        assert f"{len(community.members)} entities" in text and inside in text


def test_a_hub_note_lists_the_communities_it_links_to_by_how_many():
    projection = project_entities("alpha", ledger(), revision=1)
    files = vault(projection)
    communities = cached_analysis(projection).communities
    of = {member: community.community_id for community in communities for member in community.members}
    crossing: dict[frozenset[str], int] = {}
    for pair in {frozenset((r.subject_id, r.object_id)) for r in projection.relations if r.subject_id != r.object_id}:
        ends = {of.get(end) for end in pair} - {None}
        if len(ends) == 2:
            crossing[frozenset(ends)] = crossing.get(frozenset(ends), 0) + 1
    assert crossing
    hub_of = {front(text)["id"]: name[:-len(".md")] for name, text in files.items() if name.startswith("communities/")}
    for community in communities:
        text = files[hub_of[community.community_id] + ".md"]
        linked = dict(re.findall(r"^- \[\[(communities/[^\]]+)\]\] \((\d+) links?\)$",
                                 text.split("## Linked communities", 1)[-1], re.M))
        expected = {hub_of[other]: str(count) for pair, count in crossing.items() if community.community_id in pair
                    for other in pair - {community.community_id}}
        assert linked == expected, (community.label, linked, expected)


def test_a_member_note_carries_its_communitys_tag_and_links_its_hub():
    projection = project_entities("alpha", ledger(), revision=1)
    files = vault(projection)
    communities = cached_analysis(projection).communities
    of = {member: community.community_id for community in communities for member in community.members}
    hub = {front(text)["id"]: (name[:-len(".md")], tags(text)) for name, text in files.items()
           if name.startswith("communities/")}
    for name, text in files.items():
        if not name.startswith("entities/"):
            continue
        entity_id = front(text)["id"]
        if entity_id in of:
            hub_name, hub_tags = hub[of[entity_id]]
            assert tags(text) == hub_tags and len(hub_tags) == 1
            assert f"Community: [[{hub_name}]]" in text
        else:
            assert tags(text) == [] and "Community:" not in text, "Ann has no relation, so no community"


def test_the_graph_view_colours_each_community_by_the_tag_its_notes_carry():
    projection = project_entities("alpha", ledger(), revision=1)
    files = vault(projection)
    groups = json.loads(files[".obsidian/graph.json"])["colorGroups"]
    ranked = sorted(cached_analysis(projection).communities, key=lambda c: (-len(c.members), c.community_id))
    hub_tags = {front(text)["id"]: tags(text)[0] for name, text in files.items() if name.startswith("communities/")}
    carried = {tag for text in files.values() if text.startswith("---") for tag in tags(text)}
    assert [group["query"] for group in groups] == [f"tag:#{hub_tags[c.community_id]}" for c in ranked]
    assert all(group["query"].removeprefix("tag:#") in carried for group in groups), "no group queries a missing tag"
    assert [group["color"] for group in groups] == [
        {"a": 1, "rgb": int(_PALETTE[index % (len(_PALETTE) - 1)].lstrip("#"), 16)} for index in range(len(ranked))]
    for tag in carried:
        # The rank first, so no tag is only digits and no two communities share one.
        assert re.fullmatch(r"community/c\d+(?:-[^\W_]+(?:-[^\W_]+)*)?", tag), tag
    assert len(carried) == len(ranked)


def test_a_hostile_community_name_is_escaped_in_its_note_and_made_safe_in_its_tag():
    hostile = '<img src=x> #tag [[link]] ==mark=='
    other = "#second ==name== <b>"
    # Both members are hostile, so whichever names the community comes first in its label.
    projection = project_entities("alpha", [fact(1, hostile, "knows", other), fact(2, other, "knows", hostile)],
                                  revision=1)
    files = vault(projection)
    [(name, text)] = [(name, text) for name, text in files.items() if name.startswith("communities/")]
    assert text.split("# ", 1)[1].split(" · ")[0] != "bob"
    body = text.split("---", 2)[-1]
    assert "<img" not in body.replace("\\<", "") and "[[link]]" not in body
    [tag] = tags(text)
    assert re.fullmatch(r"community/[\w-]+", tag) and "#" not in tag[1:] and " " not in tag


def test_the_index_lists_the_communities_and_every_link_opens_a_note():
    projection = project_entities("alpha", ledger(), revision=1)
    files = vault(projection)
    index = files["index.md"]
    assert "## Communities" in index and "## Entities" in index
    hubs = {name[:-len(".md")] for name in files if name.startswith("communities/")}
    assert set(re.findall(r"\[\[(communities/[^\]]+)\]\]", index)) == hubs
    notes = {name[len("entities/"):-len(".md")] for name in files if name.startswith("entities/")}
    for text in files.values():
        for link in re.findall(r"\[\[([^\]]+)\]\]", text):
            assert link in notes or link in hubs, link


def test_a_hub_note_never_shares_a_name_with_an_entity_note():
    """Obsidian opens a bare [[name]] by file name wherever it sits, so an
    entity named like a community must not leave two notes of one name."""
    base = project_entities("alpha", ledger(), revision=1)
    label = sorted(cached_analysis(base).communities, key=lambda c: (-len(c.members), c.community_id))[0].label
    # An entity with only a value is in no community, so the communities stay as they were.
    clashing = project_entities("alpha", [*ledger(), fact(99, label, "age", "40")], revision=2)
    assert label in {community.label for community in cached_analysis(clashing).communities}
    files = vault(clashing)
    stems = [name.split("/", 1)[1][:-len(".md")].casefold() for name in files
             if name.startswith(("entities/", "communities/"))]
    assert len(stems) == len(set(stems)), sorted(stems)


def test_a_hub_counts_no_link_through_what_the_graph_only_names():
    """`typing`, imported by every module, is attached to one community for
    reading; the analysis counts no link through it, and neither does a hub."""
    rows = []
    for side in ("a", "b"):
        for n in range(4):
            rows += [(f"{side}/m{n}.py", "defines", f"{side}/m{n}.py:run"),
                     (f"{side}/m{n}.py", "imports", f"{side}/m{(n + 1) % 4}.py"),
                     (f"{side}/m{n}.py", "imports", "typing")]
    projection = project_entities("alpha", [fact(n + 1, *row) for n, row in enumerate(rows)], revision=1)
    analysis = cached_analysis(projection)
    assert analysis.external
    files = vault(projection)
    hub_of = {front(text)["id"]: name for name, text in files.items() if name.startswith("communities/")}
    for community in analysis.communities:
        text = files[hub_of[community.community_id]]
        linked = re.findall(r"^- \[\[communities/[^\]]+\]\] \((\d+) links?\)$",
                            text.split("## Linked communities", 1)[1] if "## Linked communities" in text else "", re.M)
        assert sum(map(int, linked)) == community.boundary_links, (community.label, linked)


def test_written_into_a_vault_the_communities_keep_their_tags_and_hubs_and_leave_the_graph_view_alone(tmp_path):
    """Into a vault a person keeps, the notes go under ``scone/``: a graph
    view file there would be read by nothing, and the vault's own
    ``.obsidian/`` is not the writer's, so it is left out. The hubs carry the
    signature, so a second write updates them instead of keeping them as the
    person's."""
    from scone_memory.entities.export import obsidian_files
    from scone_memory.entities.vault import DEFAULT_FOLDER, signed, write_vault

    projection = project_entities("alpha", ledger(), revision=1)
    about = {"status": "current", "as_of": "2025-06-01T00:00:00.000Z"}
    placed = obsidian_files(projection, about, root=f"{DEFAULT_FOLDER}/")
    assert ".obsidian/graph.json" in vault(projection) and ".obsidian/graph.json" not in placed
    receipt = write_vault(placed, tmp_path, projection=projection.digest)
    assert not (tmp_path / DEFAULT_FOLDER / ".obsidian").exists() and not (tmp_path / ".obsidian").exists()
    hubs = sorted((tmp_path / DEFAULT_FOLDER / "communities").iterdir())
    assert len(hubs) == len(cached_analysis(projection).communities) and all(signed(hub) for hub in hubs)
    assert all(tags(hub.read_text(encoding="utf-8")) for hub in hubs)
    again = write_vault(placed, tmp_path, projection=projection.digest)
    assert again.kept_theirs == () and again.unchanged == receipt.written
