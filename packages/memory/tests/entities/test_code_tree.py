"""A codebase in the graph, as the tree of directories, files and declarations it is.

The drawn graph shows code as a web of calls; nobody finds a function by
its place in a web. The reference emits a collapsible tree of its
graph's source paths from a separate viewer. Here the tree is built from
the facts the code readers record -- a file defines a declaration, a
declaration defines a method -- so it holds only what was read, says
what each declaration calls and is called by, and says what it left
out: entities that are not code, declarations named only by a call,
and children past the cap.
"""

from __future__ import annotations

import pytest

from scone_memory.core.models import Fact
from scone_memory.entities.code_tree import code_tree
from scone_memory.entities.project import project_entities


def fact(number: int, subject: str, predicate: str, object_: str) -> Fact:
    return Fact(fact_id=number, space="alpha", subject=subject, predicate=predicate, object=object_,
                valid_from="2025-01-01T00:00:00Z")


WINDOW = "pkg/retrieval/window.py"
ERRORS = "pkg/core/errors.py"
LEDGER = [
    fact(1, WINDOW, "defines", f"{WINDOW}:widen"),
    fact(2, WINDOW, "defines", f"{WINDOW}:Widened"),
    fact(3, f"{WINDOW}:Widened", "defines", f"{WINDOW}:Widened.record"),
    fact(4, WINDOW, "imports", ERRORS),
    fact(5, f"{WINDOW}:widen", "calls", f"{ERRORS}:InvalidInput"),
    fact(6, f"{WINDOW}:widen", "calls", f"{WINDOW}:Widened"),
    fact(7, "pkg/retrieval/empty.py", "imports", "typing"),
    fact(8, "Ann", "works_at", "Acme"),
]


def tree(ledger=LEDGER, **options):
    return code_tree(project_entities("alpha", ledger, revision=1), **options)


def shape(node, depth=0):
    """Each node as indented `kind name`, children in order."""
    lines = [f"{'  ' * depth}{node.kind} {node.name}"]
    for child in node.children:
        lines.extend(shape(child, depth + 1))
    return lines


def find(node, name):
    if node.name == name:
        return node
    for child in node.children:
        if (found := find(child, name)) is not None:
            return found
    return None


def test_directories_files_and_declarations_nest_as_the_paths_and_definitions_say():
    made = tree()
    assert shape(made.root) == [
        "directory pkg",
        "  directory core",
        "    file errors.py",
        "      declaration InvalidInput",
        "  directory retrieval",
        "    file empty.py",
        "    file window.py",
        "      declaration widen",
        "      declaration Widened",
        "        declaration Widened.record",
    ]


def test_a_chain_of_single_directories_is_one_node():
    made = tree([fact(1, "src/app/deep/nested/one.py", "defines", "src/app/deep/nested/one.py:run")])
    assert shape(made.root) == ["directory src/app/deep/nested", "  file one.py", "    declaration run"]


def test_counts_are_declarations_and_files_beneath_a_node():
    made = tree()
    assert (made.root.files, made.root.declarations) == (3, 4)
    assert (find(made.root, "retrieval").files, find(made.root, "retrieval").declarations) == (2, 3)
    assert find(made.root, "empty.py").declarations == 0


def test_what_only_a_call_or_an_import_names_is_marked_as_not_read():
    made = tree()
    # errors.py was imported and InvalidInput called; neither file was read.
    assert find(made.root, "InvalidInput").defined is False and find(made.root, "errors.py").defined is False
    assert find(made.root, "widen").defined is True and find(made.root, "Widened.record").defined is True
    assert find(made.root, "empty.py").defined is True
    assert made.undefined == 2


def test_a_declaration_says_what_it_calls_and_what_calls_it():
    made = tree()
    widen = find(made.root, "widen")
    assert [edge.label for edge in widen.calls] == [f"{ERRORS}:InvalidInput", f"{WINDOW}:Widened"]
    assert [edge.label for edge in find(made.root, "Widened").called_by] == [f"{WINDOW}:widen"]
    assert [edge.label for edge in find(made.root, "window.py").imports] == [ERRORS]
    assert [edge.label for edge in find(made.root, "empty.py").imports] == ["typing"]
    assert [edge.label for edge in find(made.root, "errors.py").imported_by] == [WINDOW]


def test_what_is_not_code_is_left_out_and_counted():
    made = tree()
    assert find(made.root, "Ann") is None and find(made.root, "Acme") is None
    # Ann and Acme are prose; "typing" is a module no path names.
    assert made.not_code == 3
    assert "3 entities" in made.why


def test_a_name_shaped_like_a_file_is_not_code_unless_a_code_relation_holds_it():
    made = tree([*LEDGER, fact(9, "the team", "uses", "Node.js")])
    assert find(made.root, "Node.js") is None


def test_a_path_with_a_space_still_holds_its_declarations():
    made = tree([fact(1, "my module.py", "defines", "my module.py:leaf")])
    assert shape(made.root) == ["file my module.py", "  declaration leaf"]


def test_children_past_the_cap_are_counted_on_their_parent_and_the_tree():
    ledger = [fact(n + 1, "pkg/big.py", "defines", f"pkg/big.py:f{n:02d}") for n in range(12)]
    made = tree(ledger, max_children=5)
    big = find(made.root, "big.py")
    assert [child.name for child in big.children] == ["f00", "f01", "f02", "f03", "f04"]
    assert big.more == 7 and big.declarations == 12 and made.cut == 7
    assert "7" in made.why and "cap" in made.why


def test_a_graph_with_no_code_is_an_empty_tree_that_says_so():
    made = tree([fact(1, "Ann", "works_at", "Acme")])
    assert made.root is None and made.not_code == 2
    assert "no source files" in made.why


@pytest.mark.parametrize("cap", [0, -1, 1.5, True])
def test_a_cap_that_is_not_a_positive_whole_number_is_refused(cap):
    from scone_memory.core.errors import InvalidInput

    with pytest.raises(InvalidInput):
        tree(max_children=cap)


def test_a_declaration_of_a_file_the_graph_holds_no_entity_for_still_sits_in_that_file():
    made = tree([fact(1, "pkg/a.py:run", "calls", "pkg/lib/b.py:helper"),
                 fact(2, "pkg/a.py", "defines", "pkg/a.py:run")])
    b = find(made.root, "b.py")
    assert b is not None and b.entity_id is None and b.defined is False
    assert [child.name for child in b.children] == ["helper"]
    assert made.undefined == 2, "helper was only called, and b.py only named by it"


def test_a_method_whose_class_is_not_in_the_graph_sits_in_its_file():
    made = tree([fact(1, "pkg/a.py", "defines", "pkg/a.py:Shelf.put")])
    # A directory holding one file stays a directory: only directory chains are joined.
    assert shape(made.root) == ["directory pkg", "  file a.py", "    declaration Shelf.put"]


def test_entries_with_no_directory_in_common_sit_under_an_unnamed_root():
    made = tree([fact(1, "a.py", "defines", "a.py:one"), fact(2, "lib/b.py", "defines", "lib/b.py:two")])
    assert shape(made.root) == ["directory ", "  directory lib", "    file b.py", "      declaration two",
                                "  file a.py", "    declaration one"]


def test_links_past_the_cap_in_one_list_are_counted():
    import scone_memory.entities.code_tree as module

    ledger = [fact(n + 1, "pkg/a.py:run", "calls", f"pkg/b.py:f{n}") for n in range(8)]
    original = module.MAX_LINKS
    module.MAX_LINKS = 3
    try:
        made = tree(ledger)
    finally:
        module.MAX_LINKS = original
    run = find(made.root, "run")
    assert len(run.calls) == 3 and run.link_counts["calls"] == 8


def test_a_chain_of_single_directories_below_a_branch_is_one_node_too():
    made = tree([fact(1, "pkg/a.py", "defines", "pkg/a.py:one"),
                 fact(2, "pkg/deep/er/b.py", "defines", "pkg/deep/er/b.py:two")])
    assert shape(made.root) == ["directory pkg", "  directory deep/er", "    file b.py", "      declaration two",
                                "  file a.py", "    declaration one"]


def test_a_package_named_like_a_file_is_not_placed_as_one():
    """`lodash.merge`, `socket.io` and `chart.js` are packages a manifest
    depends on or a file imports, not files in this tree: a name with no
    directory is placed only when the graph read it or it holds a declaration."""
    made = tree([fact(1, "package.json", "depends_on", "lodash.merge"),
                 fact(2, "web/app.js", "imports", "socket.io"),
                 fact(3, "web/app.js", "imports", "chart.js"),
                 fact(4, "web/app.js", "defines", "web/app.js:start"),
                 fact(5, "setup.py", "defines", "setup.py:main"),
                 # A path with a directory is a file even when only imported and holding nothing.
                 fact(6, "web/app.js", "imports", "web/lib/util.js"),
                 # A name with no directory, not read, is still the file that holds what is called in it.
                 fact(7, "web/app.js", "imports", "tasks.py"),
                 fact(8, "web/app.js:start", "calls", "tasks.py:run")])
    assert shape(made.root) == ["directory ", "  directory web", "    directory lib", "      file util.js",
                                "    file app.js", "      declaration start",
                                "  file package.json", "  file setup.py", "    declaration main",
                                "  file tasks.py", "    declaration run"]
    assert made.not_code == 3
    assert find(made.root, "tasks.py").entity_id is not None, "its own entity, not a stand-in"


def test_every_code_predicate_is_read_in_both_directions():
    """The tree reads every relation the classifier calls code, so each needs
    a name in reverse: `references` joined the code predicates when documents
    entered the graph, and a tree of a mapped repository failed on it."""
    from scone_memory.entities.classify import CODE_PREDICATES
    from scone_memory.entities.code_tree import _REVERSED

    assert set(_REVERSED) == set(CODE_PREDICATES)
    assert len(set(_REVERSED.values())) == len(_REVERSED) and not set(_REVERSED.values()) & set(_REVERSED)


def test_a_document_that_links_a_file_sits_in_the_tree_and_the_file_says_what_links_it():
    made = tree([fact(1, "pkg/a.py", "defines", "pkg/a.py:run"),
                 fact(2, "docs/guide.md", "references", "pkg/a.py"),
                 fact(3, "docs/guide.md", "references", "pkg/b.py")])
    assert shape(made.root) == ["directory ", "  directory docs", "    file guide.md",
                                "  directory pkg", "    file a.py", "      declaration run", "    file b.py"]
    guide, linked, unread = find(made.root, "guide.md"), find(made.root, "a.py"), find(made.root, "b.py")
    assert guide.defined and [link.label for link in guide.links["references"]] == ["pkg/a.py", "pkg/b.py"]
    assert [(link.label, link.fact_ids) for link in linked.links["referenced_by"]] == [("docs/guide.md", (2,))]
    assert linked.defined and not unread.defined, "b.py says nothing the graph holds; a link is all that names it"
    assert "known only from a call, an import or a link in a document" in made.why
