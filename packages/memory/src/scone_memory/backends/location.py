"""Where a local store lives, compared as the file it is rather than as a path's spelling."""

from __future__ import annotations

import os
from pathlib import Path


def local_identity(path: str | Path) -> object:
    """The device and inode of the file or directory at ``path``, which every
    path to it shares: another letter case on a case-insensitive filesystem
    (macOS's by default), a hard link, a symbolic link, ``..``. The resolved
    path when nothing is there yet. The caller expands ``~`` when its store does."""
    try:
        found = os.stat(path)
    except OSError:
        return str(Path(path).resolve())
    return ("file", found.st_dev, found.st_ino)
