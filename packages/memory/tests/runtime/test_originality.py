"""Nothing we ship is lifted from the reference trees.

The scan reports any run of identical normalised code lines between our
sources and ``reference/``, so a copied block fails a test instead of
relying on a reviewer's memory. Shared idiom (import blocks, one-line
boilerplate) is not a finding; a long identical run is.
"""

from __future__ import annotations

from ..paths import REPO_ROOT

from pathlib import Path

import pytest

from scone_memory.testing.originality import Overlap, normalise, scan, sources

REPO = REPO_ROOT
REFERENCE = REPO / "reference"


def write(root: Path, name: str, body: str) -> Path:
    path = root / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(body)
    return path


THEIRS = '''\
"""Their module docstring."""
import asyncio


class Ladder:
    def __init__(self, rungs):
        self.rungs = rungs
        self.height = 0

    async def climb(self, steps):
        for step in range(steps):
            self.height += self.rungs[step]
            await asyncio.sleep(0)
        return self.height
'''


def test_a_long_identical_run_is_reported_with_both_locations(tmp_path):
    theirs = write(tmp_path / "reference", "theirs.py", THEIRS)
    lifted = THEIRS.splitlines()[3:]  # the class body, comments and docstring dropped
    ours = write(tmp_path / "ours", "copied.py", "# our header\n" + "\n".join(lifted) + "\n")

    [found] = scan([ours], [theirs], window=8)
    assert isinstance(found, Overlap)
    assert (found.ours, found.theirs) == (ours, theirs)
    assert found.lines == 9, "the run extends past the window to the whole lifted class"
    assert found.text[0] == "class Ladder:" and found.text[-1] == "return self.height"
    assert found.our_line == 3 and found.their_line == 5, "1-based lines in each file, blanks and comments skipped"


def test_shared_idiom_is_not_a_finding(tmp_path):
    # Each case isolates one rule: long import lines, long repeated lines,
    # and short distinct lines. None of them says who wrote the file.
    imports = "\n".join(f"from very.long.package.path.number_{i} import SomethingNamedAtLength as alias_{i}" for i in range(12))
    theirs = write(tmp_path / "reference", "theirs.py", imports + "\n")
    ours = write(tmp_path / "ours", "mine.py", imports + "\n\ndef mine():\n    return 1\n")
    assert scan([ours], [theirs], window=8) == [], "an import block is idiom, not a lifted design"

    repeated = "\n".join(["result = compute_the_whole_thing(alpha, beta, gamma)",
                          "checked = validate_every_argument(result, strictly=True)",
                          "recorded = write_the_outcome_somewhere(checked, when=now)"] * 3)
    theirs0 = write(tmp_path / "reference", "repeated.py", repeated + "\n")
    ours0 = write(tmp_path / "ours", "repeated.py", repeated + "\n")
    assert scan([ours0], [theirs0], window=8) == [], "three long lines in a loop are repetition, not a design"

    tiny = "\n".join(["x = 1", "y = 2", "z = 3", "a = x + y", "b = y + z", "c = a + b", "d = c * 2", "e = d - 1"])
    theirs1 = write(tmp_path / "reference", "tiny.py", tiny + "\n")
    ours1 = write(tmp_path / "ours", "tiny.py", tiny + "\n")
    assert scan([ours1], [theirs1], window=8) == [], "eight short distinct lines are still boilerplate, not a design"

    short = "\n".join(["try:", "    pass", "except Exception:", "    pass"] * 3)
    theirs2 = write(tmp_path / "reference", "short.py", short + "\n")
    ours2 = write(tmp_path / "ours", "short.py", short + "\n")
    assert scan([ours2], [theirs2], window=8) == [], "one-line boilerplate is not a lifted design"


def test_normalise_drops_comments_blanks_and_keeps_line_numbers():
    numbered = normalise("import os\n\n# a comment\ndef f():\n    return  1\n")
    assert numbered == [(1, "import os"), (4, "def f():"), (5, "return 1")], "collapsed, renumbered to the source"


def test_sources_walks_only_code_and_skips_caches(tmp_path):
    write(tmp_path, "pkg/mod.py", "x = 1\n")
    write(tmp_path, "pkg/__pycache__/mod.cpython-313.pyc", "binary")
    write(tmp_path, "pkg/notes.md", "prose")
    write(tmp_path, "node_modules/dep/index.py", "y = 2\n")
    assert [p.name for p in sources(tmp_path)] == ["mod.py"]


@pytest.mark.skipif(not REFERENCE.exists(), reason="the reference clones are local only")
def test_what_we_ship_shares_no_long_run_with_the_reference_trees():
    found = scan(sources(REPO / "packages/memory/src"), sources(REFERENCE), window=8)
    assert found == [], "\n".join(f.report() for f in found[:5])
