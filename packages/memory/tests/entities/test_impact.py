"""A diff's blast radius: what its hunks touch, what rests on that, and what the answer does not cover."""
from __future__ import annotations

import io
import json

import pytest

from scone_memory import HashEmbedder, InMemoryDocumentStore, InMemoryVectorIndex, MemoryEngine
from scone_memory.core.errors import InvalidInput
from scone_memory.entities import impact as module
from scone_memory.entities.impact import ChangedFile, impact, parse_diff
from tests.entities.test_affected import SOURCES

pytestmark = pytest.mark.asyncio


def diff_of(path, old, new, *, removed=False, added=False):
    """A unified diff in git's shape for one file, hunk lines taken from
    the new text's changed span."""
    old_lines, new_lines = old.split("\n"), new.split("\n")
    head = f"diff --git a/{path} b/{path}\n"
    if removed:
        return head + f"--- a/{path}\n+++ /dev/null\n@@ -1,{len(old_lines)} +0,0 @@\n" + "".join(f"-{l}\n" for l in old_lines)
    if added:
        return head + f"--- /dev/null\n+++ b/{path}\n@@ -0,0 +1,{len(new_lines)} @@\n" + "".join(f"+{l}\n" for l in new_lines)
    first = next((i for i, (a, b) in enumerate(zip(old_lines, new_lines)) if a != b), min(len(old_lines), len(new_lines)))
    tail = 0
    while (tail < min(len(old_lines), len(new_lines)) - first
           and old_lines[-1 - tail] == new_lines[-1 - tail]):
        tail += 1
    old_span, new_span = old_lines[first:len(old_lines) - tail], new_lines[first:len(new_lines) - tail]
    return (head + f"--- a/{path}\n+++ b/{path}\n@@ -{first + 1},{len(old_span)} +{first + 1},{len(new_span)} @@\n"
            + "".join(f"-{l}\n" for l in old_span) + "".join(f"+{l}\n" for l in new_span))


# A second importer of pkg.store, so a listing bound has something to cut.
CLI = "from pkg.store import Shelf\n\n\ndef main() -> str:\n    return Shelf().keep('x')\n"


async def graphed(root, extra=()):
    engine = await MemoryEngine(InMemoryDocumentStore(), InMemoryVectorIndex(), HashEmbedder(), code_graph=True).open()
    for name, text in {**SOURCES, **dict(extra)}.items():
        (root / name).parent.mkdir(parents=True, exist_ok=True)
        (root / name).write_text(text, encoding="utf-8")
        await engine.remember("default", text, source=name)
    return engine


def test_a_diff_is_read_for_its_files_and_the_lines_its_hunks_cover():
    text = ("diff --git a/pkg/store.py b/pkg/store.py\n--- a/pkg/store.py\n+++ b/pkg/store.py\n"
            "@@ -6,3 +6,4 @@ class Shelf:\n     def keep(self, paper: str) -> str:\n-        return x\n+        y = 1\n+        return y\n"
            "@@ -20 +21,0 @@\n-gone\n"
            "diff --git a/old.py b/new.py\nsimilarity index 90%\nrename from old.py\nrename to new.py\n"
            "diff --git a/pkg/dead.py b/pkg/dead.py\ndeleted file mode 100644\n--- a/pkg/dead.py\n+++ /dev/null\n@@ -1,2 +0,0 @@\n-a\n-b\n")
    assert parse_diff(text) == (
        ChangedFile("pkg/store.py", False, ((6, 10), (21, 23))),
        ChangedFile("new.py", False, ()),
        ChangedFile("pkg/dead.py", True, ()))
    with pytest.raises(InvalidInput, match="git diff"):
        parse_diff("not a diff at all\n")
    with pytest.raises(InvalidInput, match="bytes"):
        parse_diff("x" * (module.MAX_DIFF_BYTES + 1))


async def test_a_change_inside_a_module_reaches_what_imports_it_and_says_what_was_asked(tmp_path):
    engine = await graphed(tmp_path)
    try:
        new = SOURCES["pkg/store.py"].replace('return json.dumps({"paper": paper})', 'return json.dumps({"paper": paper.strip()})')
        (tmp_path / "pkg/store.py").write_text(new, encoding="utf-8")
        result = await impact(engine, "default", diff_of("pkg/store.py", SOURCES["pkg/store.py"], new), root=tmp_path)
        assert [(t.name, t.asked, t.status) for t in result.targets] == [
            ("Shelf.keep", "pkg/store.py:Shelf.keep", "nothing"), (None, "pkg/store.py", "nothing"), (None, "pkg.store", "found")]
        assert result.targets[0].lines == (7, 8), "the declaration under the changed line, with its span"
        # pkg/web.py imports pkg.api, and this graph does not yet join the
        # file pkg/api.py to the module pkg.api (recorded in test_affected),
        # so the walk ends at one hop; the answer says what it reached.
        assert [(r.label, r.depth, r.through, r.via) for r in result.reached] == [("pkg/api.py", 1, "imports", "pkg.store")]
        assert result.by_depth == {1: 1} and result.not_listed == 0 and not result.partial_read
        assert (result.files, result.files_examined, result.files_removed, result.files_unread) == (1, 1, 0, 0)
        assert "3 thing(s) asked about (1 with dependants here" in result.why and "1 thing(s) rest on the change" in result.why
        assert "nothing in this graph rests on the change, not that nothing does" in result.why
    finally:
        await engine.close()


async def test_a_change_outside_every_declaration_touches_the_file_alone(tmp_path):
    engine = await graphed(tmp_path)
    try:
        new = "import json\nimport os\n" + SOURCES["pkg/store.py"][len("import json\n"):]
        (tmp_path / "pkg/store.py").write_text(new, encoding="utf-8")
        result = await impact(engine, "default", diff_of("pkg/store.py", SOURCES["pkg/store.py"], new), root=tmp_path)
        assert [t.asked for t in result.targets] == ["pkg/store.py", "pkg.store"]
        assert [r.label for r in result.reached] == ["pkg/api.py"]
    finally:
        await engine.close()


async def test_a_removed_file_and_a_file_the_graph_never_saw_are_both_said(tmp_path):
    engine = await graphed(tmp_path)
    try:
        gone = diff_of("pkg/store.py", SOURCES["pkg/store.py"], "", removed=True)
        (tmp_path / "pkg/store.py").unlink()
        result = await impact(engine, "default", gone, root=tmp_path)
        assert result.files_removed == 1 and [t.asked for t in result.targets] == ["pkg/store.py", "pkg.store"]
        assert [r.label for r in result.reached] == ["pkg/api.py"]
        stranger = diff_of("pkg/new_thing.py", "", "def fresh():\n    return 1\n", added=True)
        (tmp_path / "pkg/new_thing.py").write_text("def fresh():\n    return 1\n", encoding="utf-8")
        result = await impact(engine, "default", stranger, root=tmp_path)
        assert [(t.asked, t.status) for t in result.targets] == [
            ("pkg/new_thing.py:fresh", "unknown"), ("pkg/new_thing.py", "unknown"), ("pkg.new_thing", "unknown")]
        assert result.reached == () and "3 not in this graph" in result.why
    finally:
        await engine.close()


async def test_a_file_without_a_declaration_language_is_asked_about_as_a_file(tmp_path):
    engine = await graphed(tmp_path)
    try:
        (tmp_path / "notes.md").write_text("# notes\nchanged\n", encoding="utf-8")
        result = await impact(engine, "default", diff_of("notes.md", "# notes\n", "# notes\nchanged\n"), root=tmp_path)
        assert result.files_unread == 1 and [t.asked for t in result.targets] == ["notes.md"]
    finally:
        await engine.close()


async def test_every_bound_is_disclosed(tmp_path):
    engine = await graphed(tmp_path, extra={"pkg/cli.py": CLI})
    try:
        new = SOURCES["pkg/store.py"].replace("paper})", "paper.strip()})")
        (tmp_path / "pkg/store.py").write_text(new, encoding="utf-8")
        two = diff_of("pkg/store.py", SOURCES["pkg/store.py"], new) + diff_of("pkg/unrelated.py", SOURCES["pkg/unrelated.py"], SOURCES["pkg/unrelated.py"] + "\n")
        result = await impact(engine, "default", two, root=tmp_path, max_files=1, max_targets=2, limit=1)
        assert (result.files, result.files_examined) == (2, 1) and "1 file(s) not examined (max_files)" in result.why
        assert result.targets_not_asked == 1 and "1 thing(s) not asked about (max_targets)" in result.why
        assert [t.asked for t in result.targets] == ["pkg/store.py:Shelf.keep", "pkg/store.py"]
        result = await impact(engine, "default", diff_of("pkg/store.py", SOURCES["pkg/store.py"], new), root=tmp_path, limit=1)
        assert len(result.reached) == 1 and result.not_listed == 1 and "1 not listed (limit)" in result.why
        assert result.by_depth == {1: 2}, "the shape is whole even when the list is cut"
        for bad in (dict(max_hops=0), dict(limit=0), dict(max_files=0), dict(max_targets=module.MAX_TARGETS + 1)):
            with pytest.raises(InvalidInput):
                await impact(engine, "default", two, root=tmp_path, **bad)
        assert result.record()["entities"] == [["pkg/api.py", 1, "imports", "pkg.store"]]
    finally:
        await engine.close()


async def test_the_command_reads_a_diff_file_or_stdin_and_prints_what_rests_on_it(tmp_path):
    from scone_memory.runtime.cli import build_parser, run

    engine = await graphed(tmp_path)
    try:
        new = SOURCES["pkg/store.py"].replace("paper})", "paper.strip()})")
        (tmp_path / "pkg/store.py").write_text(new, encoding="utf-8")
        patch = diff_of("pkg/store.py", SOURCES["pkg/store.py"], new)
        (tmp_path / "change.diff").write_text(patch, encoding="utf-8")
        out = io.StringIO()
        code = await run(build_parser().parse_args(["graph", "impact", str(tmp_path / "change.diff"), "--root", str(tmp_path)]),
                         engine, io.StringIO(""), out)
        text = out.getvalue()
        assert code == 0 and "touched  pkg/store.py:Shelf.keep (lines 7-8): nothing" in text and "  1  pkg/api.py  (imports pkg.store)" in text
        assert "reached: 1 at 1 hop(s)" in text
        out = io.StringIO()
        code = await run(build_parser().parse_args(["--json", "graph", "impact", "--root", str(tmp_path)]), engine, io.StringIO(patch), out)
        assert code == 0 and json.loads(out.getvalue())["entities"] == [["pkg/api.py", 1, "imports", "pkg.store"]]
        out = io.StringIO()
        with pytest.raises(InvalidInput, match="cannot read a diff"):
            await run(build_parser().parse_args(["graph", "impact", str(tmp_path / "missing.diff")]), engine, io.StringIO(""), out)
    finally:
        await engine.close()
