"""A document's links and citations become claims like a file's imports:
bound only to files the walk read, quoted from the line, said when not."""
import io
import json

import pytest

from scone_memory import HashEmbedder, InMemoryDocumentStore, InMemoryVectorIndex, MemoryEngine
from scone_memory.ingestion.code_graph import CITES, MAX_LINES, REFERENCES, record_claims
from scone_memory.ingestion.code_resolution import file_resolver
from scone_memory.ingestion.doc_graph import (DOC_SUFFIXES, MAX_LINE_CHARS, doc_claims, doc_links, is_document, link_target,
                                              link_targets)

TREE = {"README.md", "docs/design.md", "docs/adr/ADR-0012-ledger.md", "docs/guide/index.md", "src/pkg/engine.py",
        "src/pkg/util.py", "src/other/util.py", "notes.txt", "CHANGELOG.rst"}


def said(claims, predicate=None):
    return [(c.subject, c.predicate, c.object) for c in claims if predicate is None or c.predicate == predicate]


def test_every_way_a_page_links_a_file_is_one_reference_to_it():
    page = "\n".join([
        "# Design",
        "See [the engine](../src/pkg/engine.py) and the [guide](guide/) for the rest.",
        "The README is [[README]], and this page is [itself](design.md).",
        "It is built on `src/pkg/util.py`, not `x.y` or `a/b`.",
        "Again: [engine](../src/pkg/engine.py#open) and [engine](../src/pkg/engine.py?raw=1).",
        "![a diagram](../src/pkg/engine.py)",
        "[spec]: ../src/pkg/engine.py",
        "Elsewhere: [docs](https://example.org/docs) and <mailto:a@b.c>.",
        "Missing: [gone](gone.md) and [[No Such Page]] and `src/pkg/missing.py`.",
    ])
    links = doc_links(page, "docs/design.md", resolve=file_resolver(TREE))
    assert said(links.claims, REFERENCES) == [
        ("docs/design.md", REFERENCES, "src/pkg/engine.py"),
        ("docs/design.md", REFERENCES, "docs/guide/index.md"),
        ("docs/design.md", REFERENCES, "README.md"),
        ("docs/design.md", REFERENCES, "src/pkg/util.py"),
    ], "one edge per file however often it is linked; a link to itself and an image are not edges"
    assert links.unresolved == ("gone.md", "No Such Page", "src/pkg/missing.py"), "said once each, never guessed"
    assert links.ambiguous == () and links.record()["references"] == 4 and links.record()["unresolved"][0] == "gone.md"
    assert links.outside == 1, "a URL leads outside the tree; a mailto is not a link the page makes"
    first = links.claims[0]
    assert first.quote == "See [the engine](../src/pkg/engine.py) and the [guide](guide/) for the rest." and first.first_line == 2
    assert page.encode()[first.start:first.end].decode() == first.quote


def test_a_wikilink_names_a_page_by_title_and_is_refused_when_two_would_answer():
    assert link_target(TREE, "docs/design.md", "[[readme]]") == "README.md"
    assert link_target(TREE, "docs/design.md", "[[Index]]") == "docs/guide/index.md"
    assert link_target(TREE | {"other/README.md"}, "docs/design.md", "[[README]]") is None, "two READMEs: neither"
    assert link_target(TREE, "docs/design.md", "[[engine]]") is None, "a title names a document, not code"


def test_a_bare_name_is_the_one_file_with_that_name_anywhere():
    assert link_target(TREE, "docs/design.md", "engine.py") == "src/pkg/engine.py"
    assert link_target(TREE, "docs/design.md", "util.py") is None, "two files are called util.py"
    assert link_target(TREE, "docs/design.md", "/src/pkg/util.py") == "src/pkg/util.py", "from the root"
    assert link_target(TREE, "docs/design.md", "adr/ADR-0012-ledger") == "docs/adr/ADR-0012-ledger.md", "suffix supplied"
    assert link_target(TREE, "docs/design.md", "../../etc/passwd") is None, "never above the root"
    assert link_target(TREE, "docs/design.md", "pkg/engine.py") == "src/pkg/engine.py", "the one path that ends so"
    assert link_target(TREE, "docs/design.md", "pkg/util.py") == "src/pkg/util.py", "and src/pkg/util.py alone ends in pkg/util.py"
    assert link_target(TREE | {"lib/pkg/engine.py"}, "docs/design.md", "pkg/engine.py") is None, "two would answer"
    assert link_target(TREE, "docs/design.md", "kg/engine.py") is None, "a tail is whole path segments"
    assert link_targets(TREE | {"lib/pkg/engine.py"}, "docs/design.md", "pkg/engine.py") == ["lib/pkg/engine.py", "src/pkg/engine.py"]
    assert link_target(TREE, "docs/api/x.md", "./README.md") is None, "a link that says where it is is followed only there"
    assert link_target(TREE, "docs/api/x.md", "../design.md") == "docs/design.md"
    assert link_target(TREE, "docs/api/x.md", "README.md") == "README.md", "a bare name is looked for everywhere"


def test_rst_links_and_includes_are_read_too():
    page = "\n".join([
        "Changes",
        "=======",
        "See :doc:`design <docs/design>` and `the engine <src/pkg/engine.py>`_.",
        ".. include:: notes.txt",
        "Follows ADR 12 and RFC-7231; see also adr#12.",
    ])
    links = doc_links(page, "CHANGELOG.rst", resolve=file_resolver(TREE))
    assert said(links.claims, REFERENCES) == [("CHANGELOG.rst", REFERENCES, "docs/design.md"),
                                              ("CHANGELOG.rst", REFERENCES, "src/pkg/engine.py"),
                                              ("CHANGELOG.rst", REFERENCES, "notes.txt")]
    assert said(links.claims, CITES) == [("CHANGELOG.rst", CITES, "ADR-12"), ("CHANGELOG.rst", CITES, "RFC-7231")], \
        "two spellings of ADR 12 are one node, the same one the code cites"


def test_links_inside_fenced_code_are_examples_not_references():
    page = "\n".join(["Run it:", "```md", "[not a link](../src/pkg/engine.py)", "`src/pkg/util.py`", "```",
                      "~~~", "[[README]]", "~~~", "But [this](../src/pkg/engine.py) is."])
    assert said(doc_claims(page, "docs/design.md", resolve=file_resolver(TREE))) == [
        ("docs/design.md", REFERENCES, "src/pkg/engine.py")]
    # A fence closes only with at least as many of its own character and
    # nothing after; a fence with an info string inside a block is content.
    nested = "\n".join(["````text", "```json", "[x](../src/pkg/engine.py)", "````", "[real](../src/pkg/util.py)",
                         "```python", "[y](../src/pkg/engine.py)", "```", "[[README]]"])
    assert said(doc_claims(nested, "docs/design.md", resolve=file_resolver(TREE))) == [
        ("docs/design.md", REFERENCES, "src/pkg/util.py"), ("docs/design.md", REFERENCES, "README.md")]
    indented = "\n".join(["- item", "    ```", "    [x](../src/pkg/engine.py)", "    ```", "[z](../src/pkg/util.py)"])
    assert said(doc_claims(indented, "docs/design.md", resolve=file_resolver(TREE))) == [
        ("docs/design.md", REFERENCES, "src/pkg/util.py")], "an indented fence is a fence"
    unclosed = "\n".join(["```", "[x](../src/pkg/engine.py)"])
    assert doc_claims(unclosed, "docs/design.md", resolve=file_resolver(TREE)) == ()


def test_a_badge_inside_a_link_leaves_the_link_it_wraps():
    page = "[![Docs](https://img.shields.io/badge/docs-blue.svg)](guide/index.md) and ![alone](../src/pkg/engine.py)"
    links = doc_links(page, "docs/design.md", resolve=file_resolver(TREE))
    assert said(links.claims) == [("docs/design.md", REFERENCES, "docs/guide/index.md")] and links.outside == 0


def test_an_ambiguous_name_is_said_apart_from_an_unresolved_one():
    links = doc_links("See `util.py` and `nowhere.py`.", "docs/design.md", resolve=file_resolver(TREE))
    assert links.claims == () and links.ambiguous == ("util.py",) and links.unresolved == ("nowhere.py",)


def test_a_record_does_not_cite_itself():
    page = "# ADR-0012: the ledger\n\nSupersedes ADR-3.\n"
    assert said(doc_links(page, "docs/adr/ADR-0012-ledger.md").claims) == [("docs/adr/ADR-0012-ledger.md", CITES, "ADR-3")]


def test_the_bounds_are_said_not_read_as_no_links(monkeypatch):
    long_line = "[x](../src/pkg/engine.py) " + "a" * MAX_LINE_CHARS
    links = doc_links(long_line + "\n[y](../src/pkg/util.py)", "docs/design.md", resolve=file_resolver(TREE))
    assert links.long_lines == 1 and said(links.claims) == [("docs/design.md", REFERENCES, "src/pkg/util.py")]
    over = doc_links("[x](../src/pkg/engine.py)\n" * (MAX_LINES + 2), "docs/design.md", resolve=file_resolver(TREE))
    assert over.claims == () and over.lines_exceeded is True and over.record()["lines_exceeded"] is True
    monkeypatch.setattr("scone_memory.ingestion.doc_graph.MAX_CLAIMS", 1)
    capped = doc_links("[a](../src/pkg/engine.py) [b](../src/pkg/util.py)", "docs/design.md", resolve=file_resolver(TREE))
    assert len(capped.claims) == 1 and capped.claims_capped is True
    # Adversarial brackets on one line stay linear: a run the length of
    # the bound is read in well under a second.
    import time
    started = time.perf_counter()
    doc_links("[" * MAX_LINE_CHARS + "\n" + "[[" * (MAX_LINE_CHARS // 2), "docs/design.md", resolve=file_resolver(TREE))
    assert time.perf_counter() - started < 1.0


def test_without_a_resolver_links_are_unresolved_and_citations_still_read():
    page = "See [x](../src/pkg/engine.py); follows ADR-3."
    links = doc_links(page, "docs/design.md")
    assert said(links.claims) == [("docs/design.md", CITES, "ADR-3")] and links.unresolved == ("../src/pkg/engine.py",)
    assert doc_links("", "docs/design.md").claims == ()
    assert doc_links("[x](../src/pkg/engine.py)\n" * (MAX_LINES + 2), "docs/design.md", resolve=file_resolver(TREE)).claims == (), \
        "a document past the line bound is not read, like a source file"


def test_what_is_a_document():
    assert all(is_document(f"a/b{suffix}") for suffix in DOC_SUFFIXES)
    assert is_document("README.MD") and not is_document("a/b.py") and not is_document("Makefile")


async def test_a_mapped_tree_puts_its_documents_in_the_graph_and_the_blast_radius(tmp_path):
    from scone_memory.runtime.cli import build_parser, run

    root = tmp_path / "repo"
    (root / "src" / "pkg").mkdir(parents=True)
    (root / "docs").mkdir()
    (root / "src" / "pkg" / "engine.py").write_text("def open_engine():\n    return 1\n", encoding="utf-8")
    (root / "src" / "pkg" / "cli.py").write_text("from .engine import open_engine\n\ndef main():\n    return open_engine()\n",
                                                 encoding="utf-8")
    (root / "docs" / "engine.md").write_text("# The engine\n\nSee `src/pkg/engine.py`; it follows ADR-2.\n"
                                             "The [CLI](../src/pkg/cli.py) calls it.\n", encoding="utf-8")
    (root / "README.md").write_text("Start with [the engine page](docs/engine.md).\n", encoding="utf-8")
    memory = await MemoryEngine(InMemoryDocumentStore(), InMemoryVectorIndex(), HashEmbedder(), code_graph=True).open()
    out = io.StringIO()
    code = await run(build_parser().parse_args(["--json", "map", str(root), "--graph"]), memory, io.StringIO(""), out)
    receipt = json.loads(out.getvalue())
    assert code == 0 and receipt["read"] == 4, "two source files and two documents"
    assert receipt["unresolved_links"] == [] and receipt["ambiguous_links"] == [] and receipt["outside_links"] == 0
    facts = await memory.facts("default")
    references = sorted((f.subject, f.object) for f in facts if f.predicate == REFERENCES)
    # A subject is an entity key, which the ledger folds to lower case, as
    # it does for every file the code graph records.
    assert references == [("docs/engine.md", "src/pkg/cli.py"), ("docs/engine.md", "src/pkg/engine.py"), ("readme.md", "docs/engine.md")]
    assert [(f.subject, f.object) for f in facts if f.predicate == CITES] == [("docs/engine.md", "ADR-2")]
    # What rests on the engine: its caller, and the page that describes it,
    # and through that page the README.
    out = io.StringIO()
    code = await run(build_parser().parse_args(["--json", "graph", "affected", "src/pkg/engine.py"]), memory, io.StringIO(""), out)
    body = json.loads(out.getvalue())
    named = {name.lower() for name in _affected_names(body)}
    assert {"src/pkg/cli.py", "docs/engine.md", "readme.md"} <= named, body
    await memory.close()


async def test_a_document_with_nothing_to_say_is_quiet_and_its_lost_links_are_named(tmp_path):
    from scone_memory.runtime.cli import build_parser, run

    root = tmp_path / "repo"
    (root / "docs").mkdir(parents=True)
    (root / "src").mkdir()
    (root / "src" / "engine.py").write_text("import json\n\ndef open_engine():\n    return 1\n", encoding="utf-8")
    (root / "docs" / "engine.md").write_text("See `src/engine.py`, [gone](gone.md) and <https://example.org>.\n", encoding="utf-8")
    (root / "docs" / "notes.md").write_text("Nothing links anywhere here.\n", encoding="utf-8")
    memory = await MemoryEngine(InMemoryDocumentStore(), InMemoryVectorIndex(), HashEmbedder(), code_graph=True).open()
    out = io.StringIO()
    code = await run(build_parser().parse_args(["--json", "map", str(root), "--graph"]), memory, io.StringIO(""), out)
    receipt = json.loads(out.getvalue())
    assert code == 0 and receipt["quiet"] == 1 and receipt["unread"] == 0, "a document with no links had nothing to say"
    assert receipt["unresolved_links"] == ["gone.md"] and receipt["outside_links"] == 0, "an autolink in angle brackets is not a link the page makes"
    # Said in prose too, on a fresh space: a second pass over an unchanged
    # tree reads nothing again and so has nothing to say about links.
    fresh = await MemoryEngine(InMemoryDocumentStore(), InMemoryVectorIndex(), HashEmbedder(), code_graph=True).open()
    out = io.StringIO()
    code = await run(build_parser().parse_args(["map", str(root), "--graph"]), fresh, io.StringIO(""), out)
    assert "documents: 1 link(s) to files not read, 0 that two files would answer, 0 outside the tree" in out.getvalue()
    await fresh.close()
    # No file here imports another file, yet the page that describes the
    # engine rests on it: the blast radius says so without the notice
    # that nothing does.
    out = io.StringIO()
    code = await run(build_parser().parse_args(["--json", "graph", "affected", "src/engine.py"]), memory, io.StringIO(""), out)
    body = out.getvalue()
    assert code == 0 and "docs/engine.md" in body and "no file in this graph imports another" not in body, body
    await memory.close()


def _affected_names(body):
    """The entity names an affected receipt lists, whatever it nests them under."""
    found = []

    def walk(value):
        if isinstance(value, dict):
            if "entity" in value and isinstance(value["entity"], str):
                found.append(value["entity"])
            for inner in value.values():
                walk(inner)
        elif isinstance(value, list):
            for inner in value:
                walk(inner)
        elif isinstance(value, str):
            found.append(value)
    walk(body)
    return found
