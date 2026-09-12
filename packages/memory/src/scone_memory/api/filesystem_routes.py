"""The tree over HTTP: list, read, search, and write a note.

Refusals keep their meanings as status codes. A path that cannot mean
anything here is a 400, because it is the request that is wrong; a write
to a tree nobody made writable is a 403, because it is the asking that is
refused rather than the wording; and a write onto a note that moved since
it was read is a 409, which is what a conflict is for.
"""

from __future__ import annotations

from typing import Optional

from fastapi import Depends, FastAPI, Query
from pydantic import BaseModel, ConfigDict, Field

from ..core.errors import Conflict, InvalidInput
from ..filesystem import (DEFAULT_ENTRIES, DEFAULT_HITS, MAX_ENTRIES, MAX_FILE_BYTES,
                          MAX_FILE_BYTES_LIMIT, MAX_HITS, MAX_NOTE_BYTES_LIMIT, FilesystemPolicy,
                          MemoryFilesystem, PathConflict, PathRefused)
from ..memory.engine import MemoryEngine

#: The longest path a request may carry, matching the tree's own bound.
MAX_PATH_QUERY = 512


class NoteBody(BaseModel):
    """A note to write, and the version of the note it was written on."""

    model_config = ConfigDict(extra="forbid")

    path: str = Field(min_length=2, max_length=MAX_PATH_QUERY)
    text: str = Field(min_length=1, max_length=MAX_NOTE_BYTES_LIMIT)
    #: The version a read gave. Without it the newest note stands.
    if_version: Optional[int] = Field(default=None, ge=1)


def mount_filesystem_routes(app: FastAPI, engine: MemoryEngine, space_for, policy: FilesystemPolicy,
                            forbidden: type[Exception]) -> None:
    """The tree, under one policy for the whole app. A space's tree is
    opened per request from the key's own space, so no request can name
    another space's tree."""

    def tree(space: str) -> MemoryFilesystem:
        return MemoryFilesystem(engine, space, policy)

    def refused(error: Exception) -> Exception:
        """The refusal a caller gets, with its meaning kept."""
        if isinstance(error, PathConflict):
            # The version it now stands at goes with the refusal, so the
            # caller re-reads rather than guessing.
            return Conflict(str(error), error.version)
        said = str(error)
        if "read only" in said or "may be written" in said:
            return forbidden(said)
        return InvalidInput(said)

    @app.get("/v1/fs")
    async def get_listing(
        path: str = Query(default="/", min_length=1, max_length=MAX_PATH_QUERY),
        limit: int = Query(default=DEFAULT_ENTRIES, ge=1, le=MAX_ENTRIES),
        offset: int = Query(default=0, ge=0),
        space: str = Depends(space_for),
    ) -> dict:
        """What is under a path. Listing changes nothing."""
        try:
            return (await tree(space).list(path, limit=limit, offset=offset)).record()
        except (PathRefused, PathConflict) as error:
            raise refused(error) from None

    @app.get("/v1/fs/read")
    async def get_file(
        path: str = Query(min_length=2, max_length=MAX_PATH_QUERY),
        max_bytes: int = Query(default=MAX_FILE_BYTES, ge=1, le=MAX_FILE_BYTES_LIMIT),
        space: str = Depends(space_for),
    ) -> dict:
        """One file of the tree, exactly as the ledger holds it."""
        try:
            return (await tree(space).read(path, max_bytes=max_bytes)).record()
        except (PathRefused, PathConflict) as error:
            raise refused(error) from None

    @app.get("/v1/fs/search")
    async def get_search(
        query: str = Query(min_length=1, max_length=1000),
        under: str = Query(default="/", min_length=1, max_length=MAX_PATH_QUERY),
        limit: int = Query(default=DEFAULT_HITS, ge=1, le=MAX_HITS),
        space: str = Depends(space_for),
    ) -> dict:
        """Paths whose content answers a query, using ordinary recall."""
        try:
            return (await tree(space).search(query, under=under, limit=limit)).record()
        except (PathRefused, PathConflict) as error:
            raise refused(error) from None

    @app.post("/v1/fs/notes")
    async def post_note(body: NoteBody, space: str = Depends(space_for)) -> dict:
        """Write a note. Refused unless this app allows writing, unless the
        path is under /notes, and unless the note is as it was read."""
        try:
            return (await tree(space).write(body.path, body.text, if_version=body.if_version)).record()
        except (PathRefused, PathConflict) as error:
            raise refused(error) from None
