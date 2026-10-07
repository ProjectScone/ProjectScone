"""The proprietary trees stay out of the repository.

The knowledge base, the design docs, the capability ledger, the vendor
references and the benchmark run artefacts are the private part of this
project. The public benchmark datasets in `bench-data` are the exception,
and only those its manifest lists with a licence. `.gitignore` keeps them out, and an ignore rule is a promise
that holds right up until somebody runs `git add -f`, or adds a path the
rule does not quite cover, or commits from a tool with its own idea of
what is staged.

None of those leaves a mark anyone would notice in review. This does.
"""

from __future__ import annotations

from ..paths import REPO_ROOT

import pathlib
import shutil
import subprocess

import pytest

REPO = REPO_ROOT

#: Anchored at the repository root, so `packages/memory` is untouched by the
#: rule that hides `memory`.
PRIVATE = (
    "memory",          # the knowledge base: decisions, experiments, mailboxes
    "docs",            # design documents and specifications
    "CAPABILITIES.md",  # the capability ledger
    "supermemory",     # vendored upstream reference, never redistributed
    "pipecat",         # the same
    "reference",       # every upstream reference tree
    "bench-runs",      # run artefacts, which can hold sampled corpus text
)

#: `bench-data` is public since 2026-10-06, by the owner's decision: it holds only datasets whose licences allow
#: redistribution, each listed with its source and licence in the manifest. Anything else there stays out.
BENCH_DATA = "bench-data"
BENCH_DATA_OWN_FILES = {"MANIFEST.json", "README.md", "prepare.py"}


def git(*args: str) -> str:
    return subprocess.run(["git", "-C", str(REPO), *args],
                          capture_output=True, text=True, check=False).stdout


@pytest.fixture(scope="module", autouse=True)
def in_a_checkout():
    if shutil.which("git") is None or not (REPO / ".git").exists():
        pytest.skip("not a git checkout; nothing to guard")


@pytest.mark.parametrize("path", PRIVATE)
def test_a_private_tree_is_not_tracked(path):
    tracked = [line for line in git("ls-files", "--", f"{path}").splitlines() if line.strip()]
    assert tracked == [], f"{path} is proprietary and must not be committed: {tracked[:5]}"


@pytest.mark.parametrize("path", PRIVATE)
def test_a_private_tree_has_never_been_committed(path):
    """Untracked today is not enough. A file removed in a later commit is
    still readable in every clone that has the history, so the question is
    whether it was ever there at all."""
    touched = [line for line in git("log", "--oneline", "--all", "--", f"{path}").splitlines() if line.strip()]
    assert touched == [], f"{path} appears in history: {touched[:3]}"


def test_bench_data_tracks_only_the_datasets_its_manifest_licenses():
    """A corpus without a recorded licence must not ride in beside the ones that have one: OntoNotes 5, for one,
    may not be redistributed, and `prepare.py` downloads it instead."""
    import json

    manifest = REPO / BENCH_DATA / "MANIFEST.json"
    if not manifest.exists():
        pytest.skip("no bench-data in this checkout")
    listed = json.loads(manifest.read_text())["files"]
    tracked = {line.removeprefix(BENCH_DATA + "/") for line in git("ls-files", "--", BENCH_DATA).splitlines() if line.strip()}
    assert tracked - BENCH_DATA_OWN_FILES <= set(listed), \
        f"tracked in bench-data without a manifest entry: {sorted(tracked - BENCH_DATA_OWN_FILES - set(listed))}"
    assert all(entry.get("license") and entry.get("source") for entry in listed.values()), \
        "every bench-data file needs a licence and a source in MANIFEST.json"


def test_the_ignore_rules_still_cover_them():
    """The rules and this list have to agree, or the next person adds a
    directory beside one of these and it is tracked from its first day."""
    present = [p for p in PRIVATE if (REPO / p).exists()]
    if not present:
        pytest.skip("none of the private trees exist in this checkout")
    # No "--": check-ignore treats it as a path to test and then matches
    # nothing, which would make this pass while checking absolutely nothing.
    ignored = set(git("check-ignore", *present).split())
    assert set(present) <= {i.rstrip("/") for i in ignored}, \
        f"present but not ignored: {sorted(set(present) - {i.rstrip('/') for i in ignored})}"
