"""Private local encrypted rows shared by agent plan and run registries.

Schemas and domains are fixed by trusted callers. The containing directory must
be owned by this OS user and not writable by others; concurrent same-user path
replacement is unsupported. No inference or network operations occur here.
"""
from collections.abc import Iterator
from contextlib import contextmanager
import os
from pathlib import Path
import re
import sqlite3
import stat
from cryptography.exceptions import InvalidTag
from cryptography.hazmat.primitives.ciphers.aead import AESGCM
from .workflow import WorkflowError, _integer, _private_file

_MAX_BYTES = 128000


class EncryptedRecordStore:
    def __init__(self, path: str | Path, *, key: bytes, table: str, metadata: str,
                 application_id: int, domain: str, label: str) -> None:
        if type(key) is not bytes or len(key) != 32:
            raise WorkflowError('key_must_be_32_bytes')
        if any(not re.fullmatch(r'[a-z][a-z0-9_]{0,63}', value) for value in (table, metadata, label)):
            raise ValueError('invalid encrypted store schema')
        if not re.fullmatch(r'[a-z0-9-]{1,96}', domain):
            raise ValueError('invalid encrypted store domain')
        _integer(application_id, 1, 2**31 - 1)
        self._cipher, self._domain, self._label = AESGCM(key), domain, label
        self._closed = False
        descriptor = -1
        try:
            target = Path(path).absolute()
            parent = target.parent.stat(follow_symlinks=False)
            if (not stat.S_ISDIR(parent.st_mode) or parent.st_uid != os.getuid()
                    or parent.st_mode & 0o022):
                raise WorkflowError('private_directory_required')
            descriptor = _private_file(target)

            def unchanged_file() -> None:
                opened = os.fstat(descriptor)
                current = target.stat(follow_symlinks=False)
                if (not stat.S_ISREG(current.st_mode) or current.st_uid != os.getuid()
                        or current.st_mode & 0o077 or current.st_nlink != 1
                        or (opened.st_dev, opened.st_ino) != (current.st_dev, current.st_ino)):
                    raise WorkflowError('private_file_required')

            unchanged_file()
            self._db = sqlite3.connect(target, timeout=1, isolation_level=None)
            unchanged_file()
            self._db.execute('PRAGMA synchronous=FULL')
            with self._access(write=True) as db:
                app = db.execute('PRAGMA application_id').fetchone()[0]
                version = db.execute('PRAGMA user_version').fetchone()[0]
                names = {row[0] for row in db.execute("SELECT name FROM sqlite_master WHERE name NOT GLOB 'sqlite_*'")}
                if not app and not version and not names:
                    db.execute(f'CREATE TABLE {table} (token TEXT PRIMARY KEY, payload BLOB NOT NULL)')
                    db.execute(f'CREATE TABLE {metadata} (payload BLOB NOT NULL)')
                    db.execute(f'INSERT INTO {metadata} VALUES (?)', (self._seal('key-check', domain.encode()),))
                    db.execute(f'PRAGMA application_id={application_id}')
                    db.execute('PRAGMA user_version=1')
                elif app != application_id or version != 1 or names != {table, metadata}:
                    raise WorkflowError(f'foreign_{self._label}_store')
                markers = db.execute(f'SELECT payload FROM {metadata} LIMIT 2').fetchall()
                if len(markers) != 1 or self._unseal('key-check', markers[0][0]) != domain.encode():
                    raise WorkflowError(f'{self._label}_key_or_integrity')
        except BaseException as error:
            if hasattr(self, '_db'):
                self._db.close()
            if isinstance(error, (OSError, sqlite3.Error)):
                raise WorkflowError(f'{self._label}_store_unavailable') from None
            raise
        finally:
            if descriptor >= 0:
                os.close(descriptor)

    @contextmanager
    def _access(self, *, write: bool = False) -> Iterator[sqlite3.Connection]:
        if self._closed:
            raise WorkflowError(f'{self._label}_store_closed')
        try:
            self._db.execute('BEGIN IMMEDIATE' if write else 'BEGIN')
            try:
                yield self._db
                self._db.execute('COMMIT')
            except BaseException:
                self._db.execute('ROLLBACK')
                raise
        except sqlite3.Error:
            raise WorkflowError(f'{self._label}_store_unavailable') from None

    def _seal(self, token: str, payload: bytes) -> bytes:
        if len(payload) > _MAX_BYTES:
            raise WorkflowError(f'{self._label}_payload_limit')
        nonce = os.urandom(12)
        return nonce + self._cipher.encrypt(nonce, payload, (self._domain + ':' + token).encode())

    def _unseal(self, token: str, payload: object) -> bytes:
        if not isinstance(payload, bytes) or not 28 <= len(payload) <= _MAX_BYTES + 28:
            raise WorkflowError(f'{self._label}_key_or_integrity')
        try:
            return self._cipher.decrypt(payload[:12], payload[12:], (self._domain + ':' + token).encode())
        except (InvalidTag, ValueError):
            raise WorkflowError(f'{self._label}_key_or_integrity') from None

    def close(self) -> None:
        if not self._closed:
            self._db.close()
            self._closed = True
