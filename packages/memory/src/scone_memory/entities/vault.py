"""Write the Obsidian export into a vault that already exists, and keep the vault's own notes.

The ``obsidian`` export is a zip: one note per entity, an index, a canvas.
Unzipping it into a vault a person already keeps is the wrong tool -- it
would write over a note of theirs that happens to share an entity's name,
and a second export would leave the first one's notes for entities since
forgotten. This writes the same notes under one folder of the vault
(``scone/`` unless named otherwise) by three rules:

- **A file this did not write is never written over.** Every note this
  writes opens with a frontmatter line ``scone_note: 1``; a folder
  manifest (``scone/.scone-vault.json``) lists what the last write left.
  A note at a target path is this writer's only if it still carries the
  signature (a person who took the signature out has taken the note
  back), and a file that is not a note (the canvas) only if the manifest
  lists it; anything else is the person's, kept and counted as
  ``kept_theirs``, and the wiki links to that name then reach their note,
  which is about the same thing. Names are compared the way a disk that
  ignores case and Unicode normalisation compares them, so a note whose
  spelling changed case between two writes is one note, updated and not
  removed.
- **What this wrote last time and does not write now is removed**, so a
  forgotten entity does not keep a note; a person's own file is never
  removed, and nothing outside the folder is read or touched --
  ``.obsidian/`` least of all.
- **The receipt says what happened**: written, updated, unchanged, kept
  as theirs, removed, and the projection the notes were written from.

The manifest is written before the notes, naming what is about to be
written, so a crash between the two leaves nothing orphaned: a listed
note that is missing is written next time, and the canvas stays this
writer's. Writes are atomic per file (a temporary name in the same
directory, then a rename, the mode kept), so a crash leaves whole notes
or none. A note's text carries nothing that changes when the rest of
the graph does beyond its community (the tag and the hub it links to),
so a note whose entity and community did not change reads as
``unchanged``; the projection digest lives in ``index.md`` alone. A
community's hub note is named for its members, so a change in them
renames the hub and the old one is removed as stale.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import json
import os
from pathlib import Path
import tempfile
from typing import Mapping
import unicodedata

from ..core.errors import InvalidInput

#: The frontmatter key every note this writes carries.
SIGNATURE = "scone_note"
#: The folder of the vault the notes go under.
DEFAULT_FOLDER = "scone"
MANIFEST = ".scone-vault.json"
#: Bytes of a file read to look for the signature: a frontmatter block is short.
_PROBE = 4_096


@dataclass(frozen=True)
class VaultReceipt:
    into: str
    folder: str
    projection: str
    written: int
    updated: int
    unchanged: int
    removed: int
    kept_theirs: tuple[str, ...] = field(default_factory=tuple)

    def record(self) -> dict[str, object]:
        return {"into": self.into, "folder": self.folder, "projection": self.projection, "written": self.written,
                "updated": self.updated, "unchanged": self.unchanged, "removed": self.removed,
                "kept_theirs": list(self.kept_theirs)}


def signed(path: Path) -> bool:
    """Whether a note carries this writer's frontmatter signature: a block
    that opens the file and holds the key before it closes. A note that
    merely mentions the key in its prose is not adopted."""
    try:
        with path.open("r", encoding="utf-8-sig", errors="replace") as file:
            head = file.read(_PROBE)
    except OSError:
        return False
    if not head.startswith("---"):
        return False
    for line in head.splitlines()[1:]:
        if line.strip() == "---":
            return False
        if line.strip().startswith(SIGNATURE + ":"):
            return True
    return False


def _folded(name: str) -> str:
    """How a disk that ignores case and Unicode normalisation sees a name."""
    return unicodedata.normalize("NFC", name).casefold()


def _atomic_write(target: Path, content: str) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    try:
        mode = target.stat().st_mode & 0o777
    except OSError:
        mode = 0o644
    handle, temporary = tempfile.mkstemp(prefix=".scone-", suffix=".tmp", dir=target.parent)
    try:
        with os.fdopen(handle, "w", encoding="utf-8") as file:
            file.write(content)
        os.chmod(temporary, mode)
        os.replace(temporary, target)
    except BaseException:
        try:
            os.unlink(temporary)
        except OSError:
            pass
        raise


def _inside(home: Path, name: str) -> Path | None:
    """The path a folder-relative name means, or None when it escapes."""
    target = home / name
    try:
        if not target.resolve().is_relative_to(home.resolve()):
            return None
    except OSError:
        return None
    return target


def write_vault(files: Mapping[str, str], into: str | Path, *, projection: str, folder: str = DEFAULT_FOLDER) -> VaultReceipt:
    """Write ``files`` (folder-relative path -> text) under ``into/folder``
    by the rules above, and say what happened."""
    root = Path(into)
    if root.exists() and not root.is_dir():
        raise InvalidInput(f"{root} is not a directory")
    if not folder or "/" in folder or "\\" in folder or folder in (".", "..") or folder.startswith("."):
        raise InvalidInput("the folder must be one plain name inside the vault")
    if any(not name or name.startswith(("/", "\\")) or ".." in Path(name).parts or name.endswith("/") for name in files):
        raise InvalidInput("a note's path must be relative and inside the folder")
    home = root / folder
    manifest_path = home / MANIFEST
    owned: dict[str, str] = {}
    try:
        listed = json.loads(manifest_path.read_text(encoding="utf-8"))
        owned = {_folded(str(name)): str(name) for name in listed.get("files", [])} if isinstance(listed, dict) else {}
    except (OSError, ValueError):
        owned = {}
    # Decide first, write after: what is the person's, what is this
    # writer's to update, what is new.
    kept: list[str] = []
    to_write: list[tuple[str, bool]] = []
    for name in sorted(files):
        target = _inside(home, name)
        if target is None:
            raise InvalidInput("a note's path must be relative and inside the folder")
        if target.exists():
            listed_here = _folded(name) in owned
            ours = signed(target) if target.suffix == ".md" else listed_here
            if not ours:
                kept.append(name)
                continue
            to_write.append((name, True))
        else:
            to_write.append((name, False))
    # The manifest names what is about to be written, before it is, so a
    # crash between the two orphans nothing.
    _atomic_write(manifest_path, json.dumps({"projection": projection, "files": [name for name, _ in to_write]}, indent=1) + "\n")
    written = updated = unchanged = 0
    for name, existed in to_write:
        target = home / name
        if existed:
            try:
                current = target.read_text(encoding="utf-8")
            except (OSError, UnicodeDecodeError):
                current = None
            if current == files[name]:
                unchanged += 1
                continue
            _atomic_write(target, files[name])
            updated += 1
        else:
            _atomic_write(target, files[name])
            written += 1
    # What was ours last time and is not written now goes, if it is still
    # ours; a note whose spelling only changed case is not stale.
    removed = 0
    keeping = {_folded(name) for name in files}
    for folded, name in sorted(owned.items()):
        if folded in keeping:
            continue
        stale = _inside(home, name)
        if stale is not None and stale.is_file() and not stale.is_symlink() and (stale.suffix != ".md" or signed(stale)):
            stale.unlink()
            removed += 1
    return VaultReceipt(str(root), folder, projection, written, updated, unchanged, removed, tuple(kept))
