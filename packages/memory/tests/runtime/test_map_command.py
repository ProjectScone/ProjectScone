"""Pointing the framework at a directory and asking it questions after.

A codebase is the case where "remember this, then ask about it" has an
obvious shape: walk the files, store each one, and — when asked — record
what each says about itself. What it read and what it did not is said
plainly, because a map that quietly skipped half a repository is worse
than no map.
"""

from __future__ import annotations

import io
import json

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
    # A document is read too (see ingestion/doc_graph.py); one with no links has nothing to say.
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
    assert counts.episodes == 4, "three .py files and the readme, a document"
    assert "4 file(s)" in said


async def test_a_file_is_stored_under_the_path_it_was_read_from(tree):
    engine = await memory()
    await mapped(engine, str(tree))
    sources = {episode.source for episode in await engine.documents.recent_episodes("default", 10)}
    assert sources == {"app/planner.py", "app/store.py", "app/broken.py", "README.md"}


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
    assert "2 had nothing to say" in said, said  # empty.py and the readme, which links nothing


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
    assert "4 already here" in said, said


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
    assert report["read"] == 4


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


# --- a map reflects the tree as it is now -----------------------------------

async def test_a_changed_file_replaces_the_memory_of_its_earlier_version(tree):
    """Mapping again after an edit used to add a second memory of the
    file beside the first, and the first version's claims stood. A map is
    of the tree as it is: the file's memory is updated under the same
    identity `sync` uses, and what the new version no longer says is
    closed, naming the file."""
    engine = await memory()
    await mapped(engine, str(tree), "--graph")
    (tree / "app" / "planner.py").write_text(PLANNER.replace("import json", "import csv"), encoding="utf-8")
    said = await mapped(engine, str(tree), "--graph")
    assert "1 updated" in said and "1 claim(s) closed" in said, said
    held = [e for e in await engine.episodes("default", {"sync": str(tree.resolve())}) if e.source == "app/planner.py"]
    assert len(held) == 1, "one memory of the file, not one per version"
    current = {(f.subject, f.predicate, f.object) for f in await engine.facts("default")}
    assert ("app/planner.py", "imports", "csv") in current
    assert ("app/planner.py", "imports", "json") not in current
    [closed] = [f for f in await engine.facts("default", status="closed") if f.object == "json"]
    assert closed.closed_reason == "no longer stated by app/planner.py"


async def test_the_json_receipt_counts_updates_and_closed_claims(tree):
    import json as _json

    engine = await memory()
    await mapped(engine, str(tree), "--graph")
    (tree / "app" / "planner.py").write_text(PLANNER.replace("import json", "import csv"), encoding="utf-8")
    report = _json.loads(await mapped(engine, str(tree), "--graph", "--json"))
    assert report["read"] == 1 and report["updated"] == 1 and report["claims_closed"] == 1
    assert report["deduplicated"] >= 1


async def test_map_and_sync_share_one_identity_for_a_directory(tree):
    """What `map` remembers, `sync` recognises as its own, and the other
    way round: neither adds a second memory of a file the other stored."""
    from scone_memory.ingestion.sync import sync_directory

    engine = await memory()
    await mapped(engine, str(tree))
    receipt = await sync_directory(engine, "default", str(tree), apply=True)
    assert receipt.updated == 0 and receipt.unchanged >= 3, receipt.text()
    (tree / "app" / "store.py").write_text(STORE + "\nEXTRA = 1\n", encoding="utf-8")
    await sync_directory(engine, "default", str(tree), apply=True)
    said = await mapped(engine, str(tree))
    assert "0 file(s) read" in said or "updated" not in said, said
    held = [e for e in await engine.episodes("default", {"sync": str(tree.resolve())}) if e.source == "app/store.py"]
    assert len(held) == 1


LIB_UTIL = "def tidy(text: str) -> str:\n    return text.strip()\n"
APP_MAIN = "import libpkg.util\nfrom libpkg.util import tidy\n\n\ndef run(text: str) -> str:\n    return tidy(text)\n"


def two_repositories(tmp_path):
    lib, app = tmp_path / "lib", tmp_path / "app"
    (lib / "src" / "libpkg").mkdir(parents=True)
    (lib / "pyproject.toml").write_text('[project]\nname = "libpkg"\nversion = "1.0"\n', encoding="utf-8")
    (lib / "src" / "libpkg" / "__init__.py").write_text("", encoding="utf-8")
    (lib / "src" / "libpkg" / "util.py").write_text(LIB_UTIL, encoding="utf-8")
    (lib / "src" / "main.py").write_text("import libpkg.util\n\n\ndef go() -> None:\n    libpkg.util.tidy('x')\n", encoding="utf-8")
    (app / "src").mkdir(parents=True)
    (app / "src" / "main.py").write_text(APP_MAIN, encoding="utf-8")
    return lib, app


async def test_two_repositories_in_one_space_keep_their_files_apart_and_an_import_reaches_the_other_s_file(tmp_path):
    from scone_memory.core.errors import InvalidInput
    from scone_memory.entities.read import load_projection

    lib, app = two_repositories(tmp_path)
    engine = await memory()
    said = await mapped(engine, str(lib), "--graph", "--repo", "lib")
    assert "1 import(s) reach another repository's file" not in said, "its own package is its own, not another's"
    said = await mapped(engine, str(app), "--graph", "--repo", "app")
    assert "2 import(s) reach another repository's file" in said, said
    projection, _ = await load_projection(engine, "default", mode="current")
    keys = {entity.key for entity in projection.entities}
    assert {"lib/src/main.py", "app/src/main.py", "lib/src/libpkg/util.py"} <= keys, "two main.py, two entities"
    by_id = {entity.entity_id: entity.key for entity in projection.entities}
    edges = {(by_id[r.subject_id], r.predicate, by_id[r.object_id]) for r in projection.relations}
    assert ("app/src/main.py", "imports", "lib/src/libpkg/util.py") in edges, "the import reaches the library's file"
    assert ("app/src/main.py:run", "calls", "lib/src/libpkg/util.py:tidy") in edges, "and the call reaches its function"
    assert ("lib/src/main.py", "imports", "lib/src/libpkg/util.py") in edges, "a repository's own package resolves too"
    assert ("lib/pyproject.toml", "defines", "libpkg") in edges
    out = io.StringIO()
    code = await run(build_parser().parse_args(["map", str(app), "--graph", "--repo", "app", "--json"]), engine,
                     io.StringIO(""), out)
    receipt = json.loads(out.getvalue())
    assert code == 0 and receipt["repository"] == "app" and receipt["published"] == ["libpkg"]
    assert receipt["cross_repository_imports"] == 0, "a map that changed nothing records nothing, and counts so"
    with pytest.raises(InvalidInput, match="one path segment"):
        await run(build_parser().parse_args(["map", str(app), "--repo", "a/b"]), engine, io.StringIO(""), io.StringIO())
    await engine.close()


async def test_without_a_repository_name_a_package_nobody_here_publishes_stays_a_name(tmp_path):
    from scone_memory.entities.read import load_projection

    lib, app = two_repositories(tmp_path)
    engine = await memory()
    await mapped(engine, str(app), "--graph")
    projection, _ = await load_projection(engine, "default", mode="current")
    by_id = {entity.entity_id: entity.key for entity in projection.entities}
    edges = {(by_id[r.subject_id], r.predicate, by_id[r.object_id]) for r in projection.relations}
    assert ("src/main.py", "imports", "libpkg.util") in edges, "nobody here publishes libpkg: the module is a name"
    await engine.close()


async def test_an_unnamed_map_links_its_own_package_and_never_reports_another_repository(tmp_path):
    from scone_memory.entities.read import load_projection

    root = tmp_path / "flat"
    (root / "mypkg").mkdir(parents=True)
    (root / "pyproject.toml").write_text('[project]\nname = "mypkg"\nversion = "1.0"\n', encoding="utf-8")
    (root / "mypkg" / "__init__.py").write_text("", encoding="utf-8")
    (root / "mypkg" / "util.py").write_text(LIB_UTIL, encoding="utf-8")
    (root / "main.py").write_text("import mypkg.util\n\n\ndef go() -> None:\n    mypkg.util.tidy('x')\n", encoding="utf-8")
    engine = await memory()
    await mapped(engine, str(root), "--graph")
    projection, _ = await load_projection(engine, "default", mode="current")
    by_id = {entity.entity_id: entity.key for entity in projection.entities}
    edges = {(by_id[r.subject_id], r.predicate, by_id[r.object_id]) for r in projection.relations}
    assert ("main.py", "imports", "mypkg/util.py") in edges, "a package published at the root resolves below it"
    (root / "main.py").write_text("import mypkg.util\n\n\ndef go() -> None:\n    return mypkg.util.tidy('y')\n", encoding="utf-8")
    out = io.StringIO()
    code = await run(build_parser().parse_args(["map", str(root), "--graph", "--json"]), engine, io.StringIO(""), out)
    receipt = json.loads(out.getvalue())
    assert code == 0 and receipt["updated"] == 1 and receipt["cross_repository_imports"] == 0, \
        "a map without a name has no other repository: its own files, however stale, never count as one"
    await engine.close()


async def test_a_go_import_reaches_another_repository_s_package_directory_and_a_manifest_like_name_is_refused(tmp_path):
    from scone_memory.core.errors import InvalidInput
    from scone_memory.entities.read import load_projection

    svc, app = tmp_path / "svc", tmp_path / "app"
    (svc / "pkg" / "store").mkdir(parents=True)
    (svc / "go.mod").write_text("module example.com/svc\n\ngo 1.22\n", encoding="utf-8")
    (svc / "pkg" / "store" / "store.go").write_text("package store\n\nfunc Open() {}\n", encoding="utf-8")
    app.mkdir()
    (app / "main.go").write_text('package main\n\nimport "example.com/svc/pkg/store"\n\nfunc main() { store.Open() }\n', encoding="utf-8")
    engine = await memory()
    await mapped(engine, str(svc), "--graph", "--repo", "svc")
    said = await mapped(engine, str(app), "--graph", "--repo", "app")
    assert "1 import(s) reach another repository's file" in said, said
    projection, _ = await load_projection(engine, "default", mode="current")
    by_id = {entity.entity_id: entity.key for entity in projection.entities}
    edges = {(by_id[r.subject_id], r.predicate, by_id[r.object_id]) for r in projection.relations}
    assert ("app/main.go", "imports", "svc/pkg/store") in edges, "a Go package is its directory of files"
    with pytest.raises(InvalidInput, match="package manifests"):
        await run(build_parser().parse_args(["map", str(app), "--repo", "requirements"]), engine, io.StringIO(""), io.StringIO())
    await engine.close()
