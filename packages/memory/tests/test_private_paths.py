"""The proprietary trees stay out of the repository.

The knowledge base, the design docs, the capability ledger, the vendor
references and the benchmark corpora are the private part of this
project. `.gitignore` keeps them out, and an ignore rule is a promise
that holds right up until somebody runs `git add -f`, or adds a path the
rule does not quite cover, or commits from a tool with its own idea of
what is staged.

None of those leaves a mark anyone would notice in review. This does.
"""

from __future__ import annotations

import pathlib
import shutil
import subprocess

import pytest

REPO = pathlib.Path(__file__).resolve().parents[3]

#: Anchored at the repository root, so `packages/memory` is untouched by the
#: rule that hides `memory`.
PRIVATE = (
    "memory",          # the knowledge base: decisions, experiments, mailboxes
    "docs",            # design documents and specifications
    "CAPABILITIES.md",  # the capability ledger
    "supermemory",     # vendored upstream reference, never redistributed
    "pipecat",         # the same
    "reference",       # every upstream reference tree
    "bench-data",      # licensed benchmark corpora
    "bench-runs",      # run artefacts, which can hold sampled corpus text
)


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
