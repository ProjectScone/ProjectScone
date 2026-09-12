"""A space's relation vocabulary, kept somewhere that promises to keep it.

What a predicate means to another predicate decides which edges a reader
is shown, so two processes configured differently hand two readers
different graphs of one space. The fix is for the space to hold its own
vocabulary — but only if it is held somewhere that will still have it
later.

The event log could not be that place, and the reason is worth keeping:
**retention is a property of the store, not of the handle that opened
it.** A Mongo TTL index outlives the handle that created it; two SQLite
handles on one file disagree, the one opened without expiry promising to
keep what the one opened with expiry sweeps; and SQLite persists nothing
about retention for the first handle to inspect. No promise made from one
constructor's arguments can describe what a shared store will do.

So this is a store of its own, on the pattern of the plan and run
stores. Nothing sweeps it, nothing else writes it, and what it holds can
be **read back and checked** rather than inferred from whoever opened a
handle. It is local SQLite and directly testable, and it needs nothing
from the six document backends — configuration does not belong in each of
them.

Three things it keeps straight:

- **An explicit "no meanings" is not the absence of a record.** "This
  space has no relation meanings" is a thing to say, and a space that has
  never been configured is saying nothing. ``RelationMeanings()`` is
  falsy, which is exactly how that distinction gets lost.
- **A stale writer is refused, not allowed to win.** Saves compare and
  swap on a revision, so a process that never saw the current vocabulary
  cannot overwrite it.
- **The bytes are sealed**, like the plan and run stores, because a
  vocabulary names a space's internal predicates.

No model is invoked and nothing is fetched. Authorization belongs to the
owning application, as with the other local stores.
"""

from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import hmac
from pathlib import Path
from typing import Optional, Self

from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

from ..agents._encrypted_store import EncryptedRecordStore
from ..agents.workflow import WorkflowError, _integer
from ..core.validation import check_space
from .meanings import RelationMeanings

#: Distinguishes this store's files from the other local stores'.
_APP_ID = 0x53434656
#: Revisions a vocabulary may go through. A space's vocabulary changes
#: rarely; this is a ceiling that keeps a runaway writer bounded.
MAX_REVISION = 1_000_000


class VocabularyConflict(WorkflowError):
    """A save whose expected revision is not the one the store holds."""

    def __init__(self) -> None:
        super().__init__("vocabulary_revision_conflict")


class SavedVocabulary(BaseModel):
    """What the store holds for one space."""

    model_config = ConfigDict(frozen=True, strict=True, extra="forbid")

    space: str
    revision: int = Field(ge=1, le=MAX_REVISION)
    #: The meanings in force, or None when the space was explicitly
    #: cleared. Absence of a record altogether means never configured,
    #: which is a different fact and is why ``cleared`` is stored.
    inverse: dict[str, str] = Field(default_factory=dict)
    symmetric: list[str] = Field(default_factory=list)
    transitive: list[str] = Field(default_factory=list)
    cleared: bool = False
    saved_at: datetime

    @model_validator(mode="after")
    def valid(self) -> Self:
        check_space(self.space)
        if self.cleared and (self.inverse or self.symmetric or self.transitive):
            raise ValueError("a cleared vocabulary holds no meanings")
        return self

    @property
    def meanings(self) -> Optional[RelationMeanings]:
        """The vocabulary, or None when the space was explicitly cleared."""
        if self.cleared:
            return None
        return RelationMeanings(inverse=dict(self.inverse), symmetric=list(self.symmetric),
                                transitive=list(self.transitive))


class VocabularyStore:
    """Local SQLite storage for per-space relation vocabularies.

    Space names are HMAC-indexed and records are authenticated
    ciphertext. Saves compare and swap on a revision across local
    connections. The host owns key storage and backup retention. The
    containing directory must be owned by this user and not writable by
    others. No model is invoked.

    **Lifecycle, stated because it is a contract and not an accident:**
    this store is the host's, separate from the document store, so
    ``MemoryEngine.delete_space`` does **not** clear a space's vocabulary
    and cannot -- it has no handle on this file. The hazard that follows is
    real: a space recreated under a deleted name **inherits the old
    vocabulary**. A host that deletes spaces should :meth:`clear` or
    delete the corresponding record here, and a host that does not should
    know that it has not.
    """

    def __init__(self, path: str | Path, *, key: bytes) -> None:
        self._key = key
        self._storage = EncryptedRecordStore(
            path, key=key, table="space_vocabulary", metadata="space_vocabulary_meta",
            application_id=_APP_ID, domain="scone-space-vocabulary-v1", label="vocabulary")

    def _token(self, space: str) -> str:
        check_space(space)
        return hmac.new(self._key, b"vocabulary-space:" + space.encode(),
                        hashlib.sha256).hexdigest()

    def _decode(self, token: str, payload: object, space: str) -> SavedVocabulary:
        try:
            saved = SavedVocabulary.model_validate_json(self._storage._unseal(token, payload))
        except ValidationError:
            raise WorkflowError("vocabulary_key_or_integrity") from None
        if saved.space != space:
            raise WorkflowError("vocabulary_key_or_integrity")
        return saved

    def get(self, space: str) -> Optional[SavedVocabulary]:
        """What this space holds, or None when it has never been set.

        None means never configured. A record whose ``cleared`` is true
        means configured to have none, which is a different answer.
        """
        token = self._token(space)
        with self._storage._access() as db:
            row = db.execute("SELECT payload FROM space_vocabulary WHERE token=?",
                             (token,)).fetchone()
            return None if row is None else self._decode(token, row[0], space)

    def save(self, space: str, meanings: RelationMeanings, *,
             expected_revision: int) -> SavedVocabulary:
        """Set this space's vocabulary, refusing a stale writer.

        ``meanings`` may be empty: that records "this space has no
        relation meanings", which is not the same as never setting any and
        not the same as clearing.
        """
        if not isinstance(meanings, RelationMeanings):
            raise WorkflowError("invalid_vocabulary")
        # Checked against the constructor before anything is committed. A
        # `RelationMeanings` can arrive here mutated past its own
        # validation -- a predicate made its own inverse, say -- and a save
        # that succeeds and then cannot be read back is worse than a
        # refusal: it leaves the space holding something that breaks every
        # later reader, including the next engine to open.
        try:
            RelationMeanings(inverse=dict(meanings.inverse), symmetric=list(meanings.symmetric),
                             transitive=list(meanings.transitive))
        except Exception as broken:
            raise WorkflowError("invalid_vocabulary") from broken
        return self._write(space, meanings, expected_revision=expected_revision, cleared=False)

    def clear(self, space: str, *, expected_revision: int) -> SavedVocabulary:
        """Record that this space holds no vocabulary of its own.

        Recorded rather than deleted, so a reader can tell "cleared on
        Tuesday" from "never had one".
        """
        return self._write(space, None, expected_revision=expected_revision, cleared=True)

    def _write(self, space: str, meanings: Optional[RelationMeanings], *,
               expected_revision: int, cleared: bool) -> SavedVocabulary:
        _integer(expected_revision, 0, MAX_REVISION - 1)
        token = self._token(space)
        saved = SavedVocabulary(
            space=space, revision=expected_revision + 1,
            inverse={} if meanings is None else dict(meanings.inverse),
            symmetric=[] if meanings is None else list(meanings.symmetric),
            transitive=[] if meanings is None else list(meanings.transitive),
            cleared=cleared, saved_at=datetime.now(timezone.utc))
        with self._storage._access(write=True) as db:
            row = db.execute("SELECT payload FROM space_vocabulary WHERE token=?",
                             (token,)).fetchone()
            held = 0 if row is None else self._decode(token, row[0], space).revision
            if held != expected_revision:
                raise VocabularyConflict()
            db.execute("INSERT OR REPLACE INTO space_vocabulary (token, payload) VALUES (?, ?)",
                       (token, self._storage._seal(token, saved.model_dump_json().encode())))
        return saved

    def close(self) -> None:
        self._storage.close()
