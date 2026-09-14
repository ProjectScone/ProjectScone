"""What a tree says not to read is left unread, counted and named."""
import io
import json

import pytest

from scone_memory import HashEmbedder, InMemoryDocumentStore, InMemoryVectorIndex, MemoryEngine
from scone_memory.ingestion.ignore import MAX_IGNORE_FILES, MAX_RULES, Ignore, parse_rules, walk_files
from scone_memory.ingestion.sync import sync_directory


def rules(text, base=""):
    return Ignore(parse_rules(text, base=base))


@pytest.mark.parametrize("pattern,path,directory,expected", [
    ("build/", "build", True, True), ("build/", "build", False, False), ("build/", "a/build", True, True),
    ("*.log", "x.log", False, True), ("*.log", "a/b/x.log", False, True), ("*.log", "x.log.txt", False, False),
    ("/todo.txt", "todo.txt", False, True), ("/todo.txt", "docs/todo.txt", False, False),
    ("docs/*.md", "docs/a.md", False, True), ("docs/*.md", "docs/sub/a.md", False, False),
    ("docs/**/*.md", "docs/sub/deep/a.md", False, True), ("docs/**/*.md", "docs/a.md", False, True),
    ("**/logs", "a/b/logs", True, True), ("**/logs", "logs", True, True),
    ("logs/**", "logs/a/b", False, True), ("logs/**", "logs", True, False),
    ("a/**/b", "a/b", False, True), ("a/**/b", "a/x/y/b", False, True),
    ("?.py", "a.py", False, True), ("?.py", "ab.py", False, False), ("?.py", "a/b.py", False, True),
    ("[abc].py", "b.py", False, True), ("[!abc].py", "b.py", False, False), ("[!abc].py", "d.py", False, True),
    ("\\#notes", "#notes", False, True), ("\\!important", "!important", False, True),
    ("trailing   ", "trailing", False, True), ("escaped\\ ", "escaped ", False, True),
    ("*.py", "a/b.py", False, True), ("sub/", "a/sub/x.py", False, True),
    ("src**test.py", "src/x/test.py", False, False), ("src**test.py", "srcxtest.py", False, True),
    ("a/**b", "a/x/b", False, False), ("a/**b", "a/xb", False, True),
    ("[[:alpha:]].py", "a.py", False, True), ("[[:alpha:]].py", "1.py", False, False),
    ("[[:digit:]]*.log", "3x.log", False, True), ("[]]x", "]x", False, True), ("[!]]x", "ax", False, True),
    ("foo\\\\ ", "foo\\", False, True),
])
def test_a_pattern_means_what_git_means_by_it(pattern, path, directory, expected):
    assert rules(pattern).ignored(path, directory=directory) is expected, (pattern, path)


def test_comments_blanks_negation_and_order():
    text = "# build output\n\n*.log\n!keep.log\nbuild/\n!build/\n"
    ignore = rules(text)
    assert ignore.ignored("x.log") and not ignore.ignored("keep.log"), "the last matching pattern wins"
    assert not ignore.ignored("build", directory=True), "a directory re-included by a later pattern is read"
    assert rules("*.log\n!keep.log\n*.log").ignored("keep.log"), "and re-excluded by a later one still"
    assert parse_rules("   \n#x\n") == []


def test_a_file_under_an_excluded_directory_is_never_re_included():
    ignore = rules("build/\n!build/keep.txt\n")
    assert ignore.ignored("build/keep.txt"), "as in git: the directory is not entered"
    assert ignore.ignored("build/deep/x.py")


def test_a_deeper_files_rules_come_after_a_shallower_ones_and_hold_below_their_directory(tmp_path):
    (tmp_path / ".gitignore").write_text("*.log\ndist/\n", encoding="utf-8")
    (tmp_path / "app").mkdir()
    (tmp_path / "app" / ".gitignore").write_text("!important.log\n/local.md\n", encoding="utf-8")
    ignore = Ignore.load(tmp_path)
    assert ignore.files == (".gitignore", "app/.gitignore")
    assert ignore.ignored("app/x.log") and not ignore.ignored("app/important.log"), "the deeper file re-includes"
    assert ignore.ignored("x.log") and ignore.ignored("important.log"), "but only under its own directory"
    assert ignore.ignored("app/local.md") and not ignore.ignored("local.md") and not ignore.ignored("app/sub/local.md"), \
        "an anchored pattern is anchored to its file's directory"
    assert ignore.ignored("dist", directory=True) and ignore.ignored("app/dist", directory=True)
    assert ignore.record() == {"files": [".gitignore", "app/.gitignore"], "rules": 4, "truncated": False,
                               "unusable": [], "links": 0}


def test_a_sconeignore_can_only_exclude_more(tmp_path):
    (tmp_path / ".gitignore").write_text("*.log\n", encoding="utf-8")
    (tmp_path / ".sconeignore").write_text("!*.log\nfixtures/\n", encoding="utf-8")
    ignore = Ignore.load(tmp_path)
    assert ignore.ignored("x.log"), "what .gitignore excludes stays excluded whatever .sconeignore says"
    assert ignore.ignored("fixtures", directory=True), "and .sconeignore excludes what .gitignore allows"
    assert ignore.files == (".gitignore", ".sconeignore")


def test_an_ignore_file_inside_an_excluded_directory_is_not_read(tmp_path):
    (tmp_path / ".gitignore").write_text("node_modules/\n", encoding="utf-8")
    (tmp_path / "node_modules" / "pkg").mkdir(parents=True)
    (tmp_path / "node_modules" / "pkg" / ".gitignore").write_text("!*\n", encoding="utf-8")
    ignore = Ignore.load(tmp_path)
    assert ignore.files == (".gitignore",) and ignore.ignored("node_modules/pkg/index.js")


def test_the_bounds_hold_and_are_said(tmp_path, monkeypatch):
    monkeypatch.setattr("scone_memory.ingestion.ignore.MAX_IGNORE_FILES", 2)
    for name in ("a", "b", "c"):
        (tmp_path / name).mkdir()
        (tmp_path / name / ".gitignore").write_text("*.tmp\n", encoding="utf-8")
    ignore = Ignore.load(tmp_path)
    assert len(ignore.files) == 2 and ignore.truncated is True
    monkeypatch.setattr("scone_memory.ingestion.ignore.MAX_IGNORE_FILES", MAX_IGNORE_FILES)
    monkeypatch.setattr("scone_memory.ingestion.ignore.MAX_RULES", 3)
    (tmp_path / ".gitignore").write_text("a\nb\nc\nd\ne\n", encoding="utf-8")
    ignore = Ignore.load(tmp_path)
    assert ignore.record()["rules"] == 3 and ignore.truncated is True
    monkeypatch.setattr("scone_memory.ingestion.ignore.MAX_RULES", MAX_RULES)
    # Exactly at the bound, nothing was left unread, and the record says so.
    monkeypatch.setattr("scone_memory.ingestion.ignore.MAX_IGNORE_FILES", 4)
    assert Ignore.load(tmp_path).truncated is False
    monkeypatch.setattr("scone_memory.ingestion.ignore.MAX_IGNORE_FILES", MAX_IGNORE_FILES)


def test_an_unusable_pattern_and_a_linked_ignore_file_are_counted(tmp_path):
    (tmp_path / ".gitignore").write_text("*.log\n[abc\n", encoding="utf-8")
    (tmp_path / "real.gitignore").write_text("*.tmp\n", encoding="utf-8")
    (tmp_path / ".sconeignore").symlink_to(tmp_path / "real.gitignore")
    ignore = Ignore.load(tmp_path)
    assert ignore.ignored("x.log") and not ignore.ignored("x.tmp"), "the linked file is not read"
    assert ignore.record()["links"] == 1 and ignore.record()["unusable"] == []
    assert ignore.ignored("[abc"), "an unclosed class is a literal, as in git, not an unusable pattern"
    assert parse_rules("[!]]x\n") and parse_rules("[\\]]x\n"), "classes git accepts are read"


def test_a_sconeignore_negation_cannot_re_include_what_gitignore_excludes_nor_the_other_way(tmp_path):
    (tmp_path / ".gitignore").write_text("*.log\n!x.tmp\n", encoding="utf-8")
    (tmp_path / ".sconeignore").write_text("!*.log\n*.tmp\n", encoding="utf-8")
    ignore = Ignore.load(tmp_path)
    assert ignore.ignored("a.log") and ignore.ignored("x.tmp")


def test_the_walk_prunes_what_is_ignored_and_counts_it(tmp_path):
    (tmp_path / ".gitignore").write_text("node_modules/\n*.min.js\n", encoding="utf-8")
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "app.js").write_text("export const a = 1;\n", encoding="utf-8")
    (tmp_path / "src" / "app.min.js").write_text("export const a=1;\n", encoding="utf-8")
    (tmp_path / "node_modules" / "left-pad").mkdir(parents=True)
    (tmp_path / "node_modules" / "left-pad" / "index.js").write_text("module.exports = 1;\n", encoding="utf-8")
    (tmp_path / ".eslintrc.js").write_text("module.exports = {};\n", encoding="utf-8")
    (tmp_path / "linked.js").symlink_to(tmp_path / "src" / "app.js")
    walked = walk_files(tmp_path, keep=lambda path: path.suffix == ".js", ignore=Ignore.load(tmp_path))
    assert [p.relative_to(tmp_path).as_posix() for p in walked.files] == ["src/app.js"], "a dot-file and a link are not read"
    assert (walked.ignored_files, walked.ignored_directories, walked.links) == (1, 1, 1)
    whole = walk_files(tmp_path, keep=lambda path: path.suffix == ".js", ignore=None)
    assert len(whole.files) == 3 and whole.ignored_files == 0


@pytest.mark.asyncio
async def test_map_and_sync_leave_the_ignored_unread_and_say_so(tmp_path):
    from scone_memory.runtime.cli import build_parser, run

    root = tmp_path / "repo"
    (root / "src").mkdir(parents=True)
    (root / ".gitignore").write_text("build/\n*.generated.py\n", encoding="utf-8")
    (root / "src" / "a.py").write_text("def a():\n    return 1\n", encoding="utf-8")
    (root / "src" / "b.generated.py").write_text("def b():\n    return 2\n", encoding="utf-8")
    (root / "build").mkdir()
    (root / "build" / "c.py").write_text("def c():\n    return 3\n", encoding="utf-8")
    memory = await MemoryEngine(InMemoryDocumentStore(), InMemoryVectorIndex(), HashEmbedder()).open()
    out = io.StringIO()
    code = await run(build_parser().parse_args(["--json", "map", str(root)]), memory, io.StringIO(""), out)
    receipt = json.loads(out.getvalue())
    assert code == 0 and receipt["read"] == 1 and (receipt["ignored"], receipt["ignored_directories"]) == (1, 1)
    assert receipt["ignore_files"] == [".gitignore"]
    out = io.StringIO()
    code = await run(build_parser().parse_args(["map", str(root)]), memory, io.StringIO(""), out)
    assert "1 file(s) and 1 directory(ies) left unread by .gitignore (pass --no-ignore to read them)" in out.getvalue()
    out = io.StringIO()
    code = await run(build_parser().parse_args(["--json", "map", str(root), "--no-ignore"]), memory, io.StringIO(""), out)
    whole = json.loads(out.getvalue())
    assert code == 0 and whole["read"] == 2 and whole["deduplicated"] == 1 and whole["ignored"] == 0
    receipt = await sync_directory(memory, "default", root, marker="repo", suffixes=(".py",))
    assert (receipt.files_found, receipt.ignored, receipt.ignored_directories, receipt.ignore_files) == (1, 1, 1, (".gitignore",))
    assert "1 file(s) and 1 directory(ies) left unread by .gitignore; pass --no-ignore to read them" in receipt.text()
    assert receipt.record()["ignored"] == 1 and receipt.record()["ignore_files"] == [".gitignore"]
    plain = await sync_directory(memory, "default", root, marker="repo", suffixes=(".py",), ignore=False)
    assert plain.files_found == 3 and plain.ignored == 0 and plain.ignore_files == ()
    # A memory for a file the rules now exclude is not a file that is gone.
    applied = await sync_directory(memory, "default", root, marker="repo", suffixes=(".py",), apply=True, ignore=False)
    assert applied.added == 3, "the tree read whole under this marker: a.py, the generated file and build/c.py"
    narrowed = await sync_directory(memory, "default", root, marker="repo", suffixes=(".py",), apply=True, remove=True)
    assert (narrowed.removed, narrowed.forgotten, narrowed.ignored_memories) == (0, 0, 2), "left alone, neither read nor removed"
    assert "2 memory(ies) are for files the ignore rules now exclude; they were left alone" in narrowed.text()
    assert narrowed.record()["ignored_memories"] == 2 and narrowed.record()["ignore_truncated"] is False
    assert len(await memory.episodes("default", {"sync": "repo"})) == 3, "nothing forgotten"
    await memory.close()


@pytest.mark.asyncio
async def test_the_receipts_say_when_the_ignore_bound_bit(tmp_path, monkeypatch):
    from scone_memory.runtime.cli import build_parser, run

    monkeypatch.setattr("scone_memory.ingestion.ignore.MAX_IGNORE_FILES", 1)
    root = tmp_path / "repo"
    (root / "a").mkdir(parents=True)
    (root / ".gitignore").write_text("*.log\n", encoding="utf-8")
    (root / "a" / ".gitignore").write_text("*.tmp\n", encoding="utf-8")
    (root / "a" / "x.py").write_text("x = 1\n", encoding="utf-8")
    memory = await MemoryEngine(InMemoryDocumentStore(), InMemoryVectorIndex(), HashEmbedder()).open()
    out = io.StringIO()
    code = await run(build_parser().parse_args(["--json", "map", str(root)]), memory, io.StringIO(""), out)
    receipt = json.loads(out.getvalue())
    assert code == 0 and receipt["ignore_truncated"] is True and receipt["ignore_files"] == [".gitignore"]
    out = io.StringIO()
    await run(build_parser().parse_args(["map", str(root)]), memory, io.StringIO(""), out)
    assert "the ignore rules were read only as far as the bound allows" in out.getvalue()
    receipt = await sync_directory(memory, "default", root, marker="repo", suffixes=(".py",))
    assert receipt.ignore_truncated is True and "were not read" in receipt.text()
    await memory.close()
