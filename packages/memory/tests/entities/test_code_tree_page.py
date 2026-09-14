"""The code tree as one page that fetches nothing: folds, a filter, and an inspector.

Each directory, file and declaration is a native disclosure, so the tree
opens and closes from the keyboard and reads in order without its script.
The script adds what the markup cannot: a filter that keeps what matches
and the folders holding it, expand and collapse all, and a panel saying
what the chosen node calls, imports and is called by. Every name reaches
the page escaped in markup or in a JSON block no name can close, and the
page's policy lets only its own style and code run.
"""

from __future__ import annotations

import base64
import hashlib
import json
import re
from html.parser import HTMLParser

from scone_memory.core.models import Fact
from scone_memory.entities.export import EXPORT_FORMATS, export_graph
from scone_memory.entities.project import project_entities


def fact(number: int, subject: str, predicate: str, object_: str) -> Fact:
    return Fact(fact_id=number, space="alpha", subject=subject, predicate=predicate, object=object_,
                valid_from="2025-01-01T00:00:00Z")


WINDOW = "pkg/retrieval/window.py"
LEDGER = [fact(1, WINDOW, "defines", f"{WINDOW}:widen"), fact(2, WINDOW, "defines", f"{WINDOW}:Widened"),
          fact(3, f"{WINDOW}:Widened", "defines", f"{WINDOW}:Widened.record"),
          fact(4, f"{WINDOW}:widen", "calls", "pkg/core/errors.py:InvalidInput"),
          fact(5, "Ann", "works_at", "Acme")]


def page(ledger=LEDGER) -> str:
    made = export_graph(project_entities("alpha", ledger, revision=1), "tree")
    assert made.media_type == "text/html" and made.filename == "code-tree.html"
    return made.body.decode()


class Names(HTMLParser):
    """The node names in document order, as the tree's name buttons carry them."""

    def __init__(self):
        super().__init__()
        self.names: list[str] = []
        self.inside = False

    def handle_starttag(self, tag, attrs):
        self.inside = tag == "button" and ("class", "name") in attrs

    def handle_data(self, data):
        if self.inside:
            self.names.append(data)
            self.inside = False


def data(text: str) -> dict:
    return json.loads(text.split('<script id="tree-data" type="application/json">', 1)[1].split("</script>", 1)[0])


def test_the_tree_is_a_format_the_export_offers():
    assert "tree" in EXPORT_FORMATS


def test_the_page_lists_the_tree_in_order_as_nested_disclosures():
    text = page()
    parser = Names()
    parser.feed(text)
    assert parser.names == ["pkg", "core", "errors.py", "InvalidInput", "retrieval", "window.py", "widen", "Widened",
                            "Widened.record"]
    assert text.count("<details") == 6, "every node with children folds; leaves do not"
    assert '<details open' in text


def test_the_page_says_what_it_left_out_and_marks_what_was_not_read():
    markup = page().split('<script id="tree-data"', 1)[0]
    assert "2 entities are not source files or declarations" in markup.split("</header>", 1)[0]
    assert markup.count('class="unread"') == 2, "errors.py and InvalidInput were only named by a call"


def test_the_data_block_carries_each_nodes_links_and_counts_for_the_inspector():
    nodes = {node["path"]: node for node in data(page())["nodes"]}
    widen = nodes[f"{WINDOW}:widen"]
    assert widen["links"]["calls"] == [["pkg/core/errors.py:InvalidInput", nodes["pkg/core/errors.py:InvalidInput"]["id"],
                                        [4]]], "each link carries the ids of the facts behind it"
    assert widen["counts"] == {"calls": 1, "defined_by": 1}
    assert nodes["pkg/retrieval"]["declarations"] == 3 and nodes["pkg/retrieval"]["files"] == 1
    assert nodes[f"{WINDOW}:Widened.record"]["parent"] == nodes[f"{WINDOW}:Widened"]["id"]


def test_hostile_names_are_escaped_in_markup_and_cannot_close_the_data_block():
    # Shapes the code readers' classifier accepts: markup in a path and in a declaration name.
    hostile = "web/</script><svg onload=alert(1)>.js"
    text = page([fact(1, hostile, "defines", f"{hostile}:go"), fact(2, "pkg/x.py", "defines", "pkg/x.py:<b>run")])
    markup = text.split('<script id="tree-data"', 1)[0]
    assert "<svg" not in markup and "<b>" not in markup and "</script><" not in markup
    block = text.split('<script id="tree-data" type="application/json">', 1)[1].split("</script>", 1)[0]
    assert "<" not in block
    names = {node["name"] for node in data(text)["nodes"]}
    assert "<b>run" in names and "script><svg onload=alert(1)>.js" in names


def test_the_page_fetches_nothing_and_runs_only_its_own_style_and_code():
    text = page()
    policy = re.search(r'http-equiv="Content-Security-Policy" content="([^"]+)"', text).group(1)
    style = text.split("<style>", 1)[1].split("</style>", 1)[0]
    code = text.split('<script id="tree-code">', 1)[1].split("</script>", 1)[0]
    pinned = lambda body: "'sha256-" + base64.b64encode(hashlib.sha256(body.encode()).digest()).decode() + "'"
    assert policy.startswith("default-src 'none'") and f"script-src {pinned(code)}" in policy
    assert f"style-src {pinned(style)}" in policy
    assert not re.search(r'\b(?:src|href)="(?!data:)', text), "nothing is loaded from anywhere"
    for control in ('id="filter"', 'id="expand-all"', 'id="collapse-all"', 'id="inspect"'):
        assert control in text


def test_children_past_the_cap_are_said_under_their_parent():
    import scone_memory.entities.code_tree as module

    ledger = [fact(n + 1, "pkg/big.py", "defines", f"pkg/big.py:f{n:02d}") for n in range(7)]
    original = module.MAX_CHILDREN
    module.MAX_CHILDREN = 5
    try:
        text = page(ledger)
    finally:
        module.MAX_CHILDREN = original
    assert "2 more not listed" in text


def test_a_graph_with_no_code_gets_a_page_that_says_so():
    text = page([fact(1, "Ann", "works_at", "Acme")])
    assert "no source files in this graph" in text and 'class="name"' not in text


async def test_a_mapped_directory_exports_as_its_code_tree_from_the_command_line_and_http(tmp_path):
    import io

    from fastapi.testclient import TestClient

    from scone_memory import HashEmbedder, InMemoryDocumentStore, InMemoryVectorIndex, MemoryEngine
    from scone_memory.api import create_app
    from scone_memory.runtime.cli import build_parser, run

    source = tmp_path / "shop" / "store"
    source.mkdir(parents=True)
    (source / "shelf.py").write_text("class Shelf:\n    def put(self, paper):\n        return keep(paper)\n\n\n"
                                     "def keep(paper):\n    return paper\n")
    engine = await MemoryEngine(InMemoryDocumentStore(), InMemoryVectorIndex(), HashEmbedder()).open()
    try:
        assert await run(build_parser().parse_args(["map", str(tmp_path / "shop"), "--graph"]), engine,
                         io.StringIO(""), io.StringIO()) == 0
        target = tmp_path / "code-tree.html"
        out = io.StringIO()
        code = await run(build_parser().parse_args(["graph", "export", "--format", "tree", "--out", str(target)]),
                         engine, io.StringIO(""), out)
        with TestClient(create_app(engine, {"key-a": "default"})) as client:
            served = client.get("/v1/graph/export", params={"format": "tree"},
                                headers={"authorization": "Bearer key-a"})
    finally:
        await engine.close()
    assert code == 0
    names = Names()
    names.feed(target.read_text())
    assert {"shelf.py", "Shelf", "Shelf.put", "keep"} <= set(names.names), names.names
    assert served.status_code == 200 and served.headers["content-type"].startswith("text/html")
    assert 'filename="code-tree.html"' in served.headers["content-disposition"]
    nodes = {node["name"]: node for node in data(served.text)["nodes"]}
    assert [pair[0].rpartition(":")[2] for pair in nodes["Shelf.put"]["links"]["calls"]] == ["keep"]


def test_the_page_names_the_projection_and_the_read_it_lays_out():
    projection = project_entities("alpha", LEDGER, revision=4)
    text = export_graph(projection, "tree", about={"status": "all", "as_of": "2025-06-01T00:00:00Z",
                                                   "coverage": {"reasons": ["max_facts"]}}).body.decode()
    header = text.split("</header>", 1)[0]
    assert f"projection {projection.digest[:12]} at revision 4" in header
    assert "all facts as of 2025-06-01T00:00:00Z" in header and "read limited by max_facts" in header
    assert data(text)["projection"] == projection.digest
