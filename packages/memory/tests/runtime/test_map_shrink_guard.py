"""A watch pass that suddenly sees far fewer files refuses to forget the rest.

A directory that cannot be read this minute -- a permission changed, a
volume not mounted, a checkout half done -- looks exactly like a directory
whose files were deleted. Before this, the pass forgot every memory it
could not see again. The guard refuses that forgetting, keeps writing what
it did read, and says what it would have forgotten, so the next pass (or a
caller who means it) can go on.
"""

from __future__ import annotations

import asyncio
import io
import json

import pytest

from scone_memory import HashEmbedder, InMemoryDocumentStore, InMemoryVectorIndex, MemoryEngine
from scone_memory.core.errors import InvalidInput
from scone_memory.runtime.cli import build_parser, run


def json_stream(text: str) -> list[dict]:
    decoder, at, values = json.JSONDecoder(), 0, []
    while at < len(text):
        while at < len(text) and text[at].isspace():
            at += 1
        if at >= len(text):
            break
        value, at = decoder.raw_decode(text, at)
        values.append(value)
    return values


def tree(root, names: list[str]) -> None:
    for name in names:
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(f"def {path.stem}():\n    return {len(name)}\n", encoding="utf-8")


async def watch(root, memory, between, extra: list[str] | None = None) -> list[dict]:
    """Two watch passes over `root`, with `between()` run in between."""
    out = io.StringIO()
    asked = ["--json", "map", str(root), "--watch", "--every", "0.4", "--rounds", "2"]

    async def change():
        await asyncio.sleep(0.2)
        between()

    changing = asyncio.create_task(change())
    code = await run(build_parser().parse_args(asked + (extra or [])), memory, io.StringIO(""), out)
    await changing
    assert code == 0, out.getvalue()
    return [line for line in json_stream(out.getvalue()) if "pass" not in line]


async def engine() -> MemoryEngine:
    return await MemoryEngine(InMemoryDocumentStore(), InMemoryVectorIndex(), HashEmbedder()).open()


async def test_a_subtree_that_cannot_be_read_is_not_forgotten(tmp_path):
    root = tmp_path / "repo"
    tree(root, ["keep.py"] + [f"pkg/mod{n}.py" for n in range(9)])
    memory = await engine()
    try:
        first, second = await watch(root, memory, lambda: (root / "pkg").chmod(0o000))
        (root / "pkg").chmod(0o755)
        assert first["read"] == 10
        assert second["removed"] == 0, "a pass that suddenly sees 1 of 10 files forgets none of them"
        assert second["shrink_refused"]["would_forget"] == 9
        assert second["shrink_refused"]["seen_before"] == 10
        assert "pkg/mod0.py" in second["shrink_refused"]["files"]
        assert (await memory.status("default")).episodes == 10, "every memory is still here"
    finally:
        await memory.close()


async def test_the_files_kept_are_still_watched_after_a_refusal(tmp_path):
    root = tmp_path / "repo"
    tree(root, ["keep.py"] + [f"pkg/mod{n}.py" for n in range(9)])
    memory = await engine()
    try:
        await watch(root, memory, lambda: (root / "pkg").chmod(0o000))
        (root / "pkg").chmod(0o755)
        # A later watch, with the subtree readable again and one file truly
        # deleted, forgets that one and no more.
        first, second = await watch(root, memory, lambda: (root / "pkg" / "mod0.py").unlink())
        assert first["deduplicated"] == 10, "the kept memories are found again, not written twice"
        assert second["removed"] == 1
        assert "shrink_refused" not in second
        assert (await memory.status("default")).episodes == 9
    finally:
        await memory.close()


async def test_a_file_deleted_during_an_outage_is_forgotten_when_the_tree_returns(tmp_path):
    """The refusal keeps watching what it kept, so a real deletion under it
    is still noticed. Were the pass to take its shrunken view as the new
    truth, those files would leave the watch quietly and their memories
    would outlive the files for good."""
    root = tmp_path / "repo"
    tree(root, ["keep.py"] + [f"pkg/mod{n}.py" for n in range(9)])
    memory = await engine()
    out = io.StringIO()
    asked = ["--json", "map", str(root), "--watch", "--every", "0.4", "--rounds", "3"]

    async def script() -> None:
        await asyncio.sleep(0.2)          # before pass two: the subtree goes dark
        (root / "pkg").chmod(0o000)
        await asyncio.sleep(0.4)          # before pass three: it returns, one file gone for real
        (root / "pkg").chmod(0o755)
        (root / "pkg" / "mod0.py").unlink()

    try:
        running = asyncio.create_task(script())
        code = await run(build_parser().parse_args(asked), memory, io.StringIO(""), out)
        await running
        assert code == 0, out.getvalue()
        first, second, third = [line for line in json_stream(out.getvalue()) if "pass" not in line]
        assert first["read"] == 10
        assert second["removed"] == 0 and second["shrink_refused"]["would_forget"] == 9
        assert third["removed"] == 1, "the one file truly deleted is forgotten once the tree is readable"
        assert "shrink_refused" not in third
        assert (await memory.status("default")).episodes == 9
    finally:
        (root / "pkg").chmod(0o755)
        await memory.close()


async def test_a_deletion_the_caller_declares_is_forgotten(tmp_path):
    root = tmp_path / "repo"
    tree(root, ["keep.py"] + [f"pkg/mod{n}.py" for n in range(9)])
    memory = await engine()
    try:
        def drop() -> None:
            for n in range(9):
                (root / "pkg" / f"mod{n}.py").unlink()

        _, second = await watch(root, memory, drop, extra=["--allow-shrink"])
        assert second["removed"] == 9 and "shrink_refused" not in second
        assert (await memory.status("default")).episodes == 1
    finally:
        await memory.close()


async def test_an_ordinary_deletion_stays_under_the_guard(tmp_path):
    root = tmp_path / "repo"
    tree(root, [f"mod{n}.py" for n in range(10)])
    memory = await engine()
    try:
        _, second = await watch(root, memory, lambda: (root / "mod0.py").unlink())
        assert second["removed"] == 1, "one file of ten gone is a deletion, not a shrink"
        assert "shrink_refused" not in second
        assert (await memory.status("default")).episodes == 9
    finally:
        await memory.close()


async def test_a_small_tree_emptied_is_forgotten_without_asking(tmp_path):
    # Below the floor a share tells a reader nothing: two files of three
    # gone is 67%, and a tree that small is edited that way every day.
    root = tmp_path / "repo"
    tree(root, ["a.py", "b.py", "c.py"])
    memory = await engine()
    try:
        def drop() -> None:
            (root / "b.py").unlink()
            (root / "c.py").unlink()

        _, second = await watch(root, memory, drop)
        assert second["removed"] == 2 and "shrink_refused" not in second
    finally:
        await memory.close()


@pytest.mark.parametrize("share", ["0", "1.5"])
async def test_a_share_outside_its_bounds_is_refused(tmp_path, share):
    root = tmp_path / "repo"
    tree(root, ["a.py"])
    memory = await engine()
    try:
        with pytest.raises(InvalidInput, match="--shrink-share"):
            await watch(root, memory, lambda: None, extra=["--shrink-share", share])
    finally:
        await memory.close()


async def test_a_share_set_low_shows_the_guard_biting(tmp_path):
    """Half a tree gone passes the default share and fails a low one."""
    def drop(root):
        def go() -> None:
            for n in range(5):
                (root / f"mod{n}.py").unlink()
        return go

    plain = tmp_path / "plain"
    tree(plain, [f"mod{n}.py" for n in range(10)])
    memory = await engine()
    try:
        _, second = await watch(plain, memory, drop(plain))
        assert second["removed"] == 5 and "shrink_refused" not in second, "5 of 10 is not over half"
    finally:
        await memory.close()

    strict = tmp_path / "strict"
    tree(strict, [f"mod{n}.py" for n in range(10)])
    memory = await engine()
    try:
        _, second = await watch(strict, memory, drop(strict), extra=["--shrink-share", "0.05"])
        assert second["removed"] == 0, "the same deletion, refused under a share set low"
        assert second["shrink_refused"] == {"would_forget": 5, "seen_before": 10, "share": 0.05,
                                            "files": [f"mod{n}.py" for n in range(5)], "files_truncated": False}
        assert (await memory.status("default")).episodes == 10
    finally:
        await memory.close()
