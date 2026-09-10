"""Where an attachment's bytes live.

Bytes are addressed by their SHA-256 and kept out of the document store:
a database that holds screenshots stops being cheap to back up or copy,
and every backend would otherwise need its own blob path. The port is
small on purpose, so a bucket implementation is a day's work.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Optional, Protocol, Sequence

from ..core.errors import NotFound
from ..core.models import Attachment


def digest_of(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


class BlobStore(Protocol):
    """Bytes in, bytes out, and which episode asked for them."""

    name: str

    async def put(self, space: str, data: bytes, media_type: str, filename: Optional[str] = None) -> Attachment: ...
    async def get(self, space: str, attachment_id: str) -> tuple[Attachment, bytes]: ...
    async def link(self, space: str, attachment_id: str, episode_id: int) -> None: ...
    async def for_episode(self, space: str, episode_id: int) -> list[Attachment]: ...
    async def held(self, space: str) -> list[str]:
        """Every attachment id the space holds, linked or not."""
        ...
    async def linked(self, space: str) -> set[str]:
        """The attachment ids some episode of the space carries."""
        ...
    async def released_by(self, space: str, episode_id: int) -> list[str]:
        """The attachment ids no other episode of the space carries."""
        ...
    async def unlink(self, space: str, episode_id: int) -> list[str]:
        """Drop the episode's references; release the space's hold on what
        nothing else there carries, and the bytes when no space holds them."""
        ...
    async def release_space(self, space: str, *, preview: bool = False) -> tuple[list[str], list[str]]:
        """(released, kept): the attachment ids whose bytes go with the
        space's holds, and those another space still holds. Removes the
        holds and the orphaned bytes unless ``preview``."""
        ...


class InMemoryBlobStore:
    """The reference implementation, and what tests and an in-memory
    engine use. Nothing survives the process."""

    name = "memory"

    def __init__(self) -> None:
        self._bytes: dict[str, bytes] = {}
        self._held: dict[tuple[str, str], Attachment] = {}
        self._links: dict[tuple[str, int], list[str]] = {}

    async def put(self, space: str, data: bytes, media_type: str, filename: Optional[str] = None) -> Attachment:
        attachment_id = digest_of(data)
        self._bytes[attachment_id] = data
        held = self._held.get((space, attachment_id))
        if held is not None:
            return held
        stored = Attachment(attachment_id=attachment_id, media_type=media_type,
                            bytes=len(data), filename=filename)
        self._held[(space, attachment_id)] = stored
        return stored

    async def get(self, space: str, attachment_id: str) -> tuple[Attachment, bytes]:
        held = self._held.get((space, attachment_id))
        if held is None:
            raise NotFound(f"attachment {attachment_id} not found in {space!r}")
        return held, self._bytes[attachment_id]

    async def link(self, space: str, attachment_id: str, episode_id: int) -> None:
        await self.get(space, attachment_id)
        held = self._links.setdefault((space, episode_id), [])
        if attachment_id not in held:
            held.append(attachment_id)

    async def for_episode(self, space: str, episode_id: int) -> list[Attachment]:
        return [self._held[(space, i)] for i in self._links.get((space, episode_id), [])]

    async def held(self, space: str) -> list[str]:
        return sorted(i for (s, i) in self._held if s == space)

    async def linked(self, space: str) -> set[str]:
        return {i for (s, _), ids in self._links.items() if s == space for i in ids}

    async def released_by(self, space: str, episode_id: int) -> list[str]:
        mine = self._links.get((space, episode_id), [])
        elsewhere = {i for (s, e), ids in self._links.items() if s == space and e != episode_id for i in ids}
        return [i for i in mine if i not in elsewhere]

    async def release_space(self, space: str, *, preview: bool = False) -> tuple[list[str], list[str]]:
        mine = sorted(i for (s, i) in self._held if s == space)
        elsewhere = {i for (s, i) in self._held if s != space}
        released = [i for i in mine if i not in elsewhere]
        kept = [i for i in mine if i in elsewhere]
        if not preview:
            for key in [k for k in self._links if k[0] == space]:
                del self._links[key]
            for attachment_id in mine:
                self._held.pop((space, attachment_id), None)
            for attachment_id in released:
                self._bytes.pop(attachment_id, None)
        return released, kept

    async def unlink(self, space: str, episode_id: int) -> list[str]:
        released = await self.released_by(space, episode_id)
        self._links.pop((space, episode_id), None)
        for attachment_id in released:
            self._held.pop((space, attachment_id), None)
            if not any(i == attachment_id for (_, i) in self._held):
                self._bytes.pop(attachment_id, None)
        return released


class FileBlobStore:
    """Bytes in a content-addressed directory, membership and links as
    small files beside them.

    ``<root>/blobs/<ab>/<digest>`` holds the bytes once, whichever space
    stored them. ``<root>/spaces/<space>/`` records what that space may
    read and which episode asked for it, so one space cannot reach
    another's attachment by knowing its digest.
    """

    name = "file"

    def __init__(self, root: Path | str) -> None:
        self.root = Path(root)

    def _blob(self, attachment_id: str) -> Path:
        return self.root / "blobs" / attachment_id[:2] / attachment_id

    def _held(self, space: str, attachment_id: str) -> Path:
        return self.root / "spaces" / space / "attachments" / f"{attachment_id}.json"

    def _links(self, space: str, episode_id: int) -> Path:
        return self.root / "spaces" / space / "episodes" / f"{episode_id}.json"

    async def put(self, space: str, data: bytes, media_type: str, filename: Optional[str] = None) -> Attachment:
        attachment_id = digest_of(data)
        blob = self._blob(attachment_id)
        if not blob.exists():
            blob.parent.mkdir(parents=True, exist_ok=True)
            # Write beside and rename, so a reader never sees half a file.
            partial = blob.with_suffix(".partial")
            partial.write_bytes(data)
            partial.replace(blob)
        held = self._held(space, attachment_id)
        if held.exists():
            return Attachment(**json.loads(held.read_text()))
        stored = Attachment(attachment_id=attachment_id, media_type=media_type,
                            bytes=len(data), filename=filename)
        held.parent.mkdir(parents=True, exist_ok=True)
        held.write_text(stored.model_dump_json())
        return stored

    async def get(self, space: str, attachment_id: str) -> tuple[Attachment, bytes]:
        held = self._held(space, attachment_id)
        blob = self._blob(attachment_id)
        if not held.exists() or not blob.exists():
            raise NotFound(f"attachment {attachment_id} not found in {space!r}")
        return Attachment(**json.loads(held.read_text())), blob.read_bytes()

    async def link(self, space: str, attachment_id: str, episode_id: int) -> None:
        await self.get(space, attachment_id)
        path = self._links(space, episode_id)
        path.parent.mkdir(parents=True, exist_ok=True)
        held: list[str] = json.loads(path.read_text()) if path.exists() else []
        if attachment_id not in held:
            held.append(attachment_id)
            path.write_text(json.dumps(held))

    async def for_episode(self, space: str, episode_id: int) -> list[Attachment]:
        path = self._links(space, episode_id)
        if not path.exists():
            return []
        found: list[Attachment] = []
        for attachment_id in json.loads(path.read_text()):
            held = self._held(space, attachment_id)
            if held.exists():
                found.append(Attachment(**json.loads(held.read_text())))
        return found

    async def held(self, space: str) -> list[str]:
        folder = self.root / "spaces" / space / "attachments"
        return sorted(p.stem for p in folder.glob("*.json")) if folder.exists() else []

    async def linked(self, space: str) -> set[str]:
        folder = self.root / "spaces" / space / "episodes"
        out: set[str] = set()
        if folder.exists():
            for path in folder.glob("*.json"):
                out.update(json.loads(path.read_text()))
        return out

    async def released_by(self, space: str, episode_id: int) -> list[str]:
        path = self._links(space, episode_id)
        if not path.exists():
            return []
        mine: list[str] = json.loads(path.read_text())
        elsewhere: set[str] = set()
        for other in path.parent.glob("*.json"):
            if other != path:
                elsewhere.update(json.loads(other.read_text()))
        return [i for i in mine if i not in elsewhere]

    async def release_space(self, space: str, *, preview: bool = False) -> tuple[list[str], list[str]]:
        mine = await self.held(space)
        spaces = self.root / "spaces"

        def held_elsewhere(attachment_id: str) -> bool:
            return any(p.parent.parent.name != space for p in spaces.glob(f"*/attachments/{attachment_id}.json"))

        released = [i for i in mine if not held_elsewhere(i)]
        kept = [i for i in mine if held_elsewhere(i)]
        if not preview:
            import shutil

            shutil.rmtree(spaces / space, ignore_errors=True)
            for attachment_id in released:
                blob = self._blob(attachment_id)
                if blob.exists():
                    blob.unlink()
        return released, kept

    async def unlink(self, space: str, episode_id: int) -> list[str]:
        released = await self.released_by(space, episode_id)
        path = self._links(space, episode_id)
        if path.exists():
            path.unlink()
        for attachment_id in released:
            held = self._held(space, attachment_id)
            if held.exists():
                held.unlink()
            # The bytes serve every space that stored them; they go only
            # when no space holds them any more.
            if not any((self.root / "spaces").glob(f"*/attachments/{attachment_id}.json")):
                blob = self._blob(attachment_id)
                if blob.exists():
                    blob.unlink()
        return released


def attachments_of(ids: Sequence[str]) -> list[str]:
    """The ids an episode asked for, each once, in the order asked."""
    return list(dict.fromkeys(ids))


__all__ = ["BlobStore", "FileBlobStore", "InMemoryBlobStore", "attachments_of", "digest_of"]
