"""Private, authenticated transitions for a single local directory collection."""
from __future__ import annotations

from contextlib import contextmanager
import fcntl
import json
import os
from pathlib import Path
import stat
from typing import Iterator, Literal, Self
import uuid

from cryptography.exceptions import InvalidTag
from cryptography.hazmat.primitives.ciphers.aead import AESGCM
from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

from ..core.errors import InvalidInput
from ..core.validation import check_space
from .document_source import DocumentSource

_HEADER = b'SCONE-SOURCES-1\n'
_MAX_BYTES = 8_000_000


class SourceRevision(BaseModel):
    model_config = ConfigDict(frozen=True, strict=True, extra='forbid', revalidate_instances='always')
    original_sha256: str = Field(pattern=r'^[a-f0-9]{64}$')
    manifest_sha256: str = Field(pattern=r'^[a-f0-9]{64}$')
    parser_revision: str = Field(min_length=1, max_length=128)
    generation: int = Field(default=0, ge=0, lt=2**63)
    episode_id: int | None = Field(default=None, gt=0)


class SourceEntry(BaseModel):
    model_config = ConfigDict(frozen=True, strict=True, extra='forbid', revalidate_instances='always')
    state: Literal['active', 'replace', 'retire', 'delete', 'absent', 'suppress', 'suppressed']
    current: SourceRevision | None = None
    pending: SourceRevision | None = None

    @model_validator(mode='after')
    def transition_shape(self) -> Self:
        if self.state in ('replace', 'retire'):
            if self.pending is None or (self.state == 'retire' and self.pending.episode_id is None):
                raise ValueError('replacement requires its pending revision')
        elif self.state == 'suppress':
            if self.current is None and self.pending is None:
                raise ValueError('suppression requires a known revision')
        elif self.pending is not None:
            raise ValueError('only replacement transitions carry a pending revision')
        if self.state in ('active', 'delete', 'absent') and (self.current is None or self.current.episode_id is None):
            raise ValueError('active and deletion entries require a current episode')
        if self.state == 'suppressed' and self.current is None:
            raise ValueError('suppression requires the forgotten revision')
        if self.current is not None and self.state != 'suppressed' and self.current.episode_id is None:
            raise ValueError('a previous revision requires an episode')
        return self


class SourceState(BaseModel):
    model_config = ConfigDict(frozen=True, strict=True, extra='forbid', revalidate_instances='always')
    schema_version: Literal[1] = 1
    collection_id: str = Field(pattern=r'^[a-f0-9]{32}$')
    entries: dict[str, SourceEntry] = Field(default_factory=dict, max_length=10000)

    @model_validator(mode='after')
    def source_identities(self) -> Self:
        for path, entry in self.entries.items():
            DocumentSource(self.collection_id, path, 'journal')
            for revision in (entry.current, entry.pending):
                if revision is not None:
                    DocumentSource(self.collection_id, path, revision.parser_revision, revision.generation)
        return self


class SourceJournal:
    """POSIX, one process at a time; store_id is the caller's stable catalog identity.

    The existing parent directory must be owned by the caller and not writable by
    others. Reads and atomic saves use its open descriptor. Each operation must
    hold locked(); the separate lock inode survives replacement of the journal.
    """

    def __init__(self, path: str | Path, *, root: str | Path, space: str, store_id: str, key: bytes):
        check_space(space)
        if type(key) is not bytes or len(key) != 32:
            raise InvalidInput('directory journal requires a 32-byte encryption key')
        if not isinstance(store_id, str) or not 1 <= len(store_id) <= 1024:
            raise InvalidInput('directory journal requires a bounded stable store identity')
        try:
            target = Path(path).absolute()
            self.path = target.parent.resolve(strict=True) / target.name
            self.root = Path(root).resolve(strict=True)
            if not self.root.is_dir() or target.name in ('', '.', '..'):
                raise ValueError('invalid path')
            self._binding = json.dumps(['directory-sync-v1', str(self.root), space, store_id],
                                       ensure_ascii=False, separators=(',', ':')).encode('utf-8')
        except (OSError, ValueError, UnicodeError):
            raise InvalidInput('directory journal requires existing root and parent directories') from None
        self._cipher = AESGCM(key)
        self._directory_fd: int | None = None
        self._collection: str | None = None

    def _fd(self) -> int:
        if self._directory_fd is None:
            raise InvalidInput('directory journal operation requires its lock')
        return self._directory_fd

    def _open(self, name: str, flags: int) -> int:
        fd = os.open(name, flags | os.O_NOFOLLOW | os.O_NONBLOCK, 0o600, dir_fd=self._fd())
        info = os.fstat(fd)
        if (not stat.S_ISREG(info.st_mode) or info.st_uid != os.getuid()
                or info.st_mode & 0o077 or info.st_nlink != 1):
            os.close(fd)
            raise InvalidInput('directory journal requires private regular files with one link')
        return fd

    @contextmanager
    def locked(self) -> Iterator[Self]:
        if self._directory_fd is not None:
            raise InvalidInput('directory journal is already in use')
        directory_fd: int | None = None
        lock_fd: int | None = None
        try:
            directory_fd = os.open(self.path.parent, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
            info = os.fstat(directory_fd)
            if info.st_uid != os.getuid() or info.st_mode & 0o022:
                raise InvalidInput('directory journal parent must be owned and not writable by others')
            self._directory_fd = directory_fd
            lock_fd = self._open(self.path.name + '.lock', os.O_RDWR | os.O_CREAT)
            fcntl.flock(lock_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except (OSError, InvalidInput) as error:
            if lock_fd is not None:
                os.close(lock_fd)
            if directory_fd is not None:
                os.close(directory_fd)
            self._directory_fd = None
            raise InvalidInput('directory journal lock unavailable or unsafe') from error
        try:
            yield self
        finally:
            self._collection = None
            self._directory_fd = None
            os.close(lock_fd)
            os.close(directory_fd)

    def load(self) -> SourceState:
        self._fd()
        try:
            try:
                fd = self._open(self.path.name, os.O_RDONLY)
            except FileNotFoundError:
                if self._collection is not None:
                    raise InvalidInput('directory journal disappeared during use')
                state = SourceState(collection_id=uuid.uuid4().hex)
                self._collection = state.collection_id
                return state
            with os.fdopen(fd, 'rb') as stream:
                encoded = stream.read(_MAX_BYTES + 1)
            if len(encoded) > _MAX_BYTES or not encoded.startswith(_HEADER):
                raise InvalidInput('invalid directory journal envelope')
            envelope = encoded[len(_HEADER):]
            decoded = self._cipher.decrypt(envelope[:12], envelope[12:], self._binding)
            state = SourceState.model_validate_json(decoded)
        except (OSError, ValueError, InvalidTag, InvalidInput):
            raise InvalidInput('directory journal is unreadable, invalid, or bound to another scope or key') from None
        if self._collection is not None and self._collection != state.collection_id:
            raise InvalidInput('directory journal collection changed during use')
        self._collection = state.collection_id
        return state

    def save(self, state: SourceState) -> None:
        directory_fd = self._fd()
        try:
            checked = SourceState.model_validate(state)
            if self._collection is None or checked.collection_id != self._collection:
                raise InvalidInput('directory journal must save its loaded collection')
            plaintext = checked.model_dump_json().encode('utf-8')
            if len(plaintext) + len(_HEADER) + 28 > _MAX_BYTES:
                raise InvalidInput('directory journal exceeds its byte limit')
            nonce = os.urandom(12)
            encoded = _HEADER + nonce + self._cipher.encrypt(nonce, plaintext, self._binding)
        except (ValidationError, ValueError, TypeError):
            raise InvalidInput('directory journal state is invalid') from None
        temporary = f'.{self.path.name}.{uuid.uuid4().hex}.tmp'
        try:
            fd = self._open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL)
            with os.fdopen(fd, 'wb') as stream:
                stream.write(encoded)
                stream.flush()
                os.fsync(stream.fileno())
            # Refuse an unsafe existing target even if it changed after load().
            try:
                existing = self._open(self.path.name, os.O_RDONLY)
            except FileNotFoundError:
                pass
            else:
                os.close(existing)
            os.replace(temporary, self.path.name, src_dir_fd=directory_fd, dst_dir_fd=directory_fd)
            os.fsync(directory_fd)
        except OSError:
            raise InvalidInput('directory journal save failed') from None
        finally:
            try:
                os.unlink(temporary, dir_fd=directory_fd)
            except FileNotFoundError:
                pass
