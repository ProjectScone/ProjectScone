"""Pointing the framework at a directory and asking it questions after.

A codebase is the case where "remember this, then ask about it" has an
obvious shape: walk the files, store each one, and — when asked — record
what each says about itself. What it read and what it did not is said
plainly, because a map that quietly skipped half a repository is worse
than no map.
"""

from __future__ import annotations

import io

import pytest

from scone_memory import HashEmbedder, InMemoryDocumentStore, InMemoryVectorIndex, MemoryEngine
from scone_memory.runtime.cli import build_parser, run
from scone_memory.testing import Clock

pytestmark = pytest.mark.asyncio

PLANNER = '''import json


def plan(question: str) -> str:
    return tidy(question)


def tidy(text: str) -> str:
    return text.strip()
'''
STORE = '''def keep(record: str) -> None:
    return None
'''


@pytest.fixture()
def tree(tmp_path):
    (tmp_path / "app").mkdir()
    (tmp_path / "app" / "planner.py").write_text(PLANNER, encoding="utf-8")
    (tmp_path / "app" / "store.py").write_text(STORE, encoding="utf-8")
    (tmp_path / "app" / "broken.py").write_text("def half(:\n", encoding="utf-8")
    (tmp_path / "README.md").write_text("# not code\n", encoding="utf-8")
    return tmp_path


async def memory() -> MemoryEngine:
    return await MemoryEngine(InMemoryDocumentStore(), InMemoryVectorIndex(), HashEmbedder(),
                              clock=Clock("2025-06-01T00:00:00.000Z")).open()


async def mapped(engine, *args) -> str:
    out = io.StringIO()
    code = await run(build_parser().parse_args(["map", *args]), engine, io.StringIO(""), out)
    assert code == 0, out.getvalue()
    return out.getvalue()


async def test_mapping_a_directory_remembers_every_source_file(tree):
    engine = await memory()
    said = await mapped(engine, str(tree))
    counts = await engine.documents.counts("default")
    assert counts.episodes == 3, "three .py files; the readme is not code"
    assert "3 file(s)" in said


async def test_a_file_is_stored_under_the_path_it_was_read_from(tree):
    engine = await memory()
    await mapped(engine, str(tree))
    sources = {episode.source for episode in await engine.documents.recent_episodes("default", 10)}
    assert sources == {"app/planner.py", "app/store.py", "app/broken.py"}


async def test_mapping_can_record_what_each_file_says_about_itself(tree):
    engine = await memory()
    said = await mapped(engine, str(tree), "--graph")
    facts = await engine.documents.list_facts("default", include_closed=True)
    triples = {(f.subject, f.predicate, f.object) for f in facts}
    assert ("app/planner.py:plan", "calls", "app/planner.py:tidy") in triples
    assert ("app/planner.py", "imports", "json") in triples
    assert "claim(s)" in said


async def test_without_the_flag_nothing_is_claimed(tree):
    engine = await memory()
    await mapped(engine, str(tree))
    assert await engine.documents.list_facts("default", include_closed=True) == []


async def test_a_file_that_cannot_be_read_is_counted_apart_from_one_with_nothing_to_say(tree):
    """A file this cannot read and a file that simply holds no claims are
    different things, and a count that adds them together says neither."""
    (tree / "app" / "empty.py").write_text("X = 1\n", encoding="utf-8")
    engine = await memory()
    said = await mapped(engine, str(tree), "--graph")
    assert "1 could not be read" in said, said
    assert "1 had nothing to say" in said, said


async def test_a_directory_that_is_not_one_is_refused(tmp_path):
    from scone_memory.core.errors import InvalidInput

    engine = await memory()
    with pytest.raises(InvalidInput, match="directory"):
        await run(build_parser().parse_args(["map", str(tmp_path / "nowhere")]), engine,
                  io.StringIO(""), io.StringIO())


async def test_mapping_the_same_directory_again_changes_nothing(tree):
    engine = await memory()
    await mapped(engine, str(tree), "--graph")
    facts = len(await engine.documents.list_facts("default", include_closed=True))
    episodes = (await engine.documents.counts("default")).episodes
    said = await mapped(engine, str(tree), "--graph")
    assert (await engine.documents.counts("default")).episodes == episodes
    assert len(await engine.documents.list_facts("default", include_closed=True)) == facts
    assert "3 already here" in said, said


async def test_mapping_resolves_the_imports_between_the_files_it_saw(tmp_path):
    """The edges that matter most in a codebase are the ones inside it."""
    (tmp_path / "pkg").mkdir()
    (tmp_path / "pkg" / "__init__.py").write_text("", encoding="utf-8")
    (tmp_path / "pkg" / "shared.py").write_text("VALUE = 1\n", encoding="utf-8")
    (tmp_path / "pkg" / "worker.py").write_text(
        "from .shared import VALUE\nfrom . import shared\nimport json\n", encoding="utf-8")
    engine = await memory()
    await mapped(engine, str(tmp_path), "--graph")
    triples = {(f.subject, f.predicate, f.object)
               for f in await engine.documents.list_facts("default", include_closed=True)}
    assert ("pkg/worker.py", "imports", "pkg/shared.py") in triples
    assert ("pkg/worker.py", "imports", "json") in triples


async def test_mapping_leaves_out_an_import_of_something_it_did_not_see(tmp_path):
    (tmp_path / "pkg").mkdir()
    (tmp_path / "pkg" / "worker.py").write_text("from .missing import thing\n", encoding="utf-8")
    engine = await memory()
    await mapped(engine, str(tmp_path), "--graph")
    assert not [f for f in await engine.documents.list_facts("default", include_closed=True)
                if f.predicate == "imports"], "nothing points at a file that is not there"


async def test_calls_the_graph_could_not_bind_are_reported(tree):
    """A map that says "12 claims" and stops reads as a complete graph.

    The calls it could not bind are the difference between "nothing calls
    this" and "I could not see what calls this", and the summary is the
    only place a reader finds out. Names, never edges -- the count is a
    disclosure, and no `calls` fact is written for any of them.
    """
    (tree / "app" / "outward.py").write_text(
        "import json\n"
        "\n"
        "def handle(thing):\n"
        "    thing.run()\n"
        "    json.dumps({})\n"
        "    return len(thing)\n",
        encoding="utf-8")
    engine = await memory()
    said = await mapped(engine, str(tree), "--graph")
    assert "call(s) left unbound" in said, said

    facts = await engine.documents.list_facts("default", include_closed=True)
    calls = {(f.subject, f.object) for f in facts if f.predicate == "calls"}
    # `json.dumps` is bound -- the file's own `import json` says where it
    # lives. `thing.run` is a method on a value of unstated type and
    # cannot be, and `len` is a builtin that resolves to nothing here.
    assert ("app/outward.py:handle", "json.dumps") in calls, calls
    assert not any("thing.run" in obj for _, obj in calls), calls
    assert not any(obj == "len" for _, obj in calls), calls


async def test_a_single_matching_declaration_is_reported_and_not_asserted(tree):
    """The corpus can name a candidate. It must not write one down.

    `handler` calls `thing.settle()`, and exactly one `settle` is
    declared in everything mapped. That is a candidate, not evidence --
    nothing in the corpus says what `thing` is, and sampling this rule
    over the real package bound `.items()` to a function of ours five
    times in twelve. So the count is reported and the ledger is left
    alone.
    """
    (tree / "app" / "shelf.py").write_text(
        "class Shelf:\n    def settle(self):\n        return 1\n", encoding="utf-8")
    (tree / "app" / "caller.py").write_text(
        "def handler(thing):\n    return thing.settle()\n", encoding="utf-8")
    engine = await memory()
    said = await mapped(engine, str(tree), "--graph")
    assert "name one declaration here" in said, said

    facts = await engine.documents.list_facts("default", include_closed=True)
    assert not [f for f in facts if f.predicate == "calls"
                and f.subject == "app/caller.py:handler"], "no edge may be written"
    assert not [f for f in facts if f.origin == "inferred"], "nothing here is inferred"


async def test_a_name_a_builtin_also_carries_is_not_even_a_candidate(tree):
    """`.items()` is the collision that made this a report rather than a
    feature: on an unknown receiver it cannot be told from a dictionary's
    own method, so a declaration named `items` is not offered at all."""
    (tree / "app" / "payload.py").write_text(
        "def items():\n    return []\n", encoding="utf-8")
    (tree / "app" / "caller.py").write_text(
        "def handler(thing):\n    return thing.items()\n", encoding="utf-8")
    engine = await memory()
    said = await mapped(engine, str(tree), "--graph")
    assert "0 name one declaration here" in said, said


# --- A credential in the tree is refused, named, and never stored ---------

KEY = "-----BEGIN RSA PRIVATE KEY-----\nMIIEow\n-----END RSA PRIVATE KEY-----\n"


async def test_a_source_holding_a_private_key_is_withheld_and_named(tree):
    """`map` reads source files, and a key pasted into one is still a key.
    It is not remembered, not embedded, not claimed -- and the receipt
    names it, because a map that quietly skipped a file is the thing this
    command exists not to be."""
    (tree / "app" / "deploy.py").write_text('KEY = """' + KEY + '"""\n', encoding="utf-8")
    engine = await memory()
    said = await mapped(engine, str(tree))
    assert "1 withheld as sensitive" in said, said
    assert "app/deploy.py" in said, said
    sources = {e.source for e in await engine.documents.recent_episodes("default", limit=50)}
    assert "app/deploy.py" not in sources
    assert KEY not in "".join(e.content for e in await engine.documents.recent_episodes("default", limit=50))


async def test_including_sensitive_sources_is_an_explicit_choice(tree):
    (tree / "app" / "deploy.py").write_text('KEY = """' + KEY + '"""\n', encoding="utf-8")
    engine = await memory()
    said = await mapped(engine, str(tree), "--include-sensitive")
    assert "withheld" not in said, said
    sources = {e.source for e in await engine.documents.recent_episodes("default", limit=50)}
    assert "app/deploy.py" in sources


async def test_the_json_receipt_carries_what_was_withheld_and_why(tree):
    import json as _json

    (tree / "app" / "deploy.py").write_text('KEY = """' + KEY + '"""\n', encoding="utf-8")
    (tree / "app" / "config.py").write_text('TOKEN = "ghp_' + "b" * 36 + '"\n', encoding="utf-8")
    engine = await memory()
    said = await mapped(engine, str(tree), "--json")
    report = _json.loads(said)
    assert report["withheld"] == [
        {"path": "app/config.py", "reason": "content:secret"},
        {"path": "app/deploy.py", "reason": "content:private_key"},
    ]
    assert report["read"] == 3


async def test_mapping_reads_the_manifests_beside_the_code(tree):
    """A project's dependencies are claims like a file's imports: quoted
    from the manifest's line, held by the project's declared name."""
    (tree / "pyproject.toml").write_text('[project]\nname = "planner"\ndependencies = ["requests>=2.31", "rich"]\n'
                                         '[dependency-groups]\ntest = ["pytest"]\n', encoding="utf-8")
    (tree / "app" / "requirements.txt").write_text("numpy==2.0\n", encoding="utf-8")
    engine = await memory()
    await mapped(engine, str(tree), "--graph")
    held = {(f.subject, f.predicate, f.object) for f in await engine.facts("default")}
    assert ("pyproject.toml", "defines", "planner") in held
    assert ("planner", "depends_on", "requests") in held and ("planner", "depends_on", "rich") in held
    assert ("planner", "develops_with", "pytest") in held
    assert ("app/requirements.txt", "depends_on", "numpy") in held
    requests = next(f for f in await engine.facts("default") if f.object == "requests")
    assert requests.quote == 'dependencies = ["requests>=2.31", "rich"]' and requests.grounded is True


async def test_everything_a_file_says_holds_at_once(tree):
    """A module with three imports imports three modules. Under the
    one-value-at-a-time rule the ledger kept the last and closed the rest
    as superseded, and `list_facts(include_closed=True)` hid it."""
    (tree / "app" / "wide.py").write_text("import os\nimport re\nimport json\n\ndef a():\n    pass\n\ndef b():\n    pass\n",
                                          encoding="utf-8")
    engine = await memory()
    await mapped(engine, str(tree), "--graph")
    current = await engine.facts("default")
    assert sorted(f.object for f in current if f.subject == "app/wide.py" and f.predicate == "imports") == \
        ["json", "os", "re"]
    assert sorted(f.object for f in current if f.subject == "app/wide.py" and f.predicate == "defines") == \
        ["app/wide.py:a", "app/wide.py:b"]
    assert [f for f in await engine.facts("default", status="closed") if f.subject == "app/wide.py"] == []
