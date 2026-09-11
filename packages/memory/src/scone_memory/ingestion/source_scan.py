"""Bounded local inventories without symlink traversal or special-file reads."""
from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
import hashlib
import os
from pathlib import Path, PurePosixPath
import re
import stat
from time import monotonic
from typing import Iterator

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from ..core.errors import InvalidInput
from .document_source import DocumentSource


class ScanLimits(BaseModel):
    model_config = ConfigDict(frozen=True, strict=True, extra='forbid', allow_inf_nan=False,
                              revalidate_instances='always')
    max_files: int = Field(default=1000, ge=1, le=10000)
    max_entries: int = Field(default=20000, ge=1, le=100000)
    max_file_bytes: int = Field(default=25_000_000, ge=1, le=25_000_000)
    max_total_bytes: int = Field(default=256_000_000, ge=1, le=1_000_000_000)
    max_depth: int = Field(default=32, ge=1, le=64)
    timeout_seconds: float = Field(default=30.0, gt=0, le=300)


@dataclass(frozen=True)
class ScanIssue:
    path: str
    code: str


@dataclass(frozen=True)
class FileStamp:
    device: int
    inode: int
    size: int
    modified_ns: int
    changed_ns: int

    @classmethod
    def of(cls, info: os.stat_result) -> FileStamp:
        return cls(info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns, info.st_ctime_ns)


@dataclass(frozen=True)
class ScannedFile:
    path: str
    sha256: str
    stamp: FileStamp


@dataclass(frozen=True)
class SourceSnapshot:
    files: tuple[ScannedFile, ...]
    directories: tuple[tuple[str, FileStamp], ...]
    issues: tuple[ScanIssue, ...]
    skipped: int

    @property
    def complete(self) -> bool:
        return not self.issues

    def same_inventory(self, other: SourceSnapshot) -> bool:
        return (self.complete and other.complete and self.files == other.files
                and self.directories == other.directories and self.skipped == other.skipped)


class _ScanFault(Exception):
    def __init__(self, code: str):
        self.code = code
        super().__init__(code)


def default_extensions() -> frozenset[str]:
    from .formats.converters import CONVERTER_EXTENSIONS
    from .formats.office import OFFICE_EXTENSIONS
    from .formats.text import TEXT_EXTENSIONS
    return TEXT_EXTENSIONS | CONVERTER_EXTENSIONS | frozenset('.' + value for value in OFFICE_EXTENSIONS) | {'.pdf'}


class DirectoryScanner:
    def __init__(self, root: str | Path, *, limits: ScanLimits = ScanLimits(),
                 extensions: frozenset[str] | None = None):
        try:
            self.root = Path(root).resolve(strict=True)
            info = self.root.stat()
            if not stat.S_ISDIR(info.st_mode):
                raise ValueError('directory required')
        except (OSError, ValueError):
            raise InvalidInput('source root must be an existing local directory') from None
        self._root_identity = (info.st_dev, info.st_ino)
        try:
            self.limits = ScanLimits.model_validate(limits)
        except ValidationError:
            raise InvalidInput('directory scan limits are invalid') from None
        self.extensions = default_extensions() if extensions is None else extensions
        if (not isinstance(self.extensions, frozenset) or not self.extensions or len(self.extensions) > 256
                or any(not isinstance(value, str) or not re.fullmatch(r'\.[a-z0-9_+-]{1,32}', value)
                       for value in self.extensions)):
            raise InvalidInput('directory source extensions must be bounded lowercase dotted suffixes')

    @contextmanager
    def _root(self) -> Iterator[int]:
        fd = os.open(self.root, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
        try:
            info = os.fstat(fd)
            if (info.st_dev, info.st_ino) != self._root_identity:
                raise _ScanFault('root_changed')
            yield fd
        finally:
            os.close(fd)

    def _path(self, path: str) -> None:
        try:
            DocumentSource('0' * 32, path, 'scan')
            if len(path.encode('utf-8')) > 1024:
                raise InvalidInput('path byte limit')
        except InvalidInput:
            raise _ScanFault('invalid_path') from None

    def _read(self, parent: int, name: str, expected: FileStamp, deadline: float) -> bytes:
        fd = os.open(name, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK, dir_fd=parent)
        try:
            before = os.fstat(fd)
            if not stat.S_ISREG(before.st_mode) or FileStamp.of(before) != expected:
                raise _ScanFault('file_changed')
            if before.st_size < 1 or before.st_size > self.limits.max_file_bytes:
                raise _ScanFault('file_byte_limit')
            chunks: list[bytes] = []
            size = 0
            while True:
                if monotonic() > deadline:
                    raise _ScanFault('time_limit')
                chunk = os.read(fd, min(65536, self.limits.max_file_bytes + 1 - size))
                if not chunk:
                    break
                chunks.append(chunk)
                size += len(chunk)
                if size > self.limits.max_file_bytes:
                    raise _ScanFault('file_byte_limit')
            if FileStamp.of(os.fstat(fd)) != expected or size != expected.size:
                raise _ScanFault('file_changed')
            return b''.join(chunks)
        finally:
            os.close(fd)

    def read(self, item: ScannedFile) -> bytes:
        """Reopen beneath the pinned root, refusing replaced parents or changed bytes."""
        try:
            self._path(item.path)
            with self._root() as root_fd:
                parent = os.dup(root_fd)
                try:
                    parts = item.path.split('/')
                    for name in parts[:-1]:
                        child = os.open(name, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=parent)
                        os.close(parent)
                        parent = child
                    data = self._read(parent, parts[-1], item.stamp, monotonic() + self.limits.timeout_seconds)
                finally:
                    os.close(parent)
            if hashlib.sha256(data).hexdigest() != item.sha256:
                raise _ScanFault('file_changed')
            return data
        except (OSError, _ScanFault):
            raise InvalidInput('directory source changed or cannot be read safely') from None

    def scan(self) -> SourceSnapshot:
        files: list[ScannedFile] = []
        directories: list[tuple[str, FileStamp]] = []
        issues: list[ScanIssue] = []
        entries = total_bytes = skipped = 0
        deadline = monotonic() + self.limits.timeout_seconds

        def walk(fd: int, prefix: str, depth: int) -> None:
            nonlocal entries, total_bytes, skipped
            before = FileStamp.of(os.fstat(fd))
            directories.append((prefix, before))
            names: list[str] = []
            with os.scandir(fd) as iterator:
                for entry in iterator:
                    entries += 1
                    if entries > self.limits.max_entries:
                        raise _ScanFault('entry_limit')
                    if monotonic() > deadline:
                        raise _ScanFault('time_limit')
                    names.append(entry.name)
            for name in sorted(names):
                path = prefix + '/' + name if prefix else name
                try:
                    self._path(path)
                    if monotonic() > deadline:
                        raise _ScanFault('time_limit')
                    info = os.stat(name, dir_fd=fd, follow_symlinks=False)
                    if stat.S_ISDIR(info.st_mode):
                        if depth >= self.limits.max_depth:
                            raise _ScanFault('depth_limit')
                        child = os.open(name, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=fd)
                        try:
                            if FileStamp.of(os.fstat(child)) != FileStamp.of(info):
                                raise _ScanFault('directory_changed')
                            walk(child, path, depth + 1)
                        finally:
                            os.close(child)
                    elif not stat.S_ISREG(info.st_mode):
                        raise _ScanFault('special_file')
                    elif PurePosixPath(name).suffix.lower() not in self.extensions:
                        skipped += 1
                    else:
                        if len(files) >= self.limits.max_files:
                            raise _ScanFault('file_limit')
                        total_bytes += info.st_size
                        if total_bytes > self.limits.max_total_bytes:
                            raise _ScanFault('total_byte_limit')
                        stamp = FileStamp.of(info)
                        data = self._read(fd, name, stamp, deadline)
                        files.append(ScannedFile(path, hashlib.sha256(data).hexdigest(), stamp))
                except _ScanFault as error:
                    issues.append(ScanIssue(path, error.code))
                except OSError:
                    issues.append(ScanIssue(path, 'unreadable'))
            if FileStamp.of(os.fstat(fd)) != before:
                issues.append(ScanIssue(prefix, 'directory_changed'))

        try:
            with self._root() as root_fd:
                walk(root_fd, '', 0)
        except _ScanFault as error:
            issues.append(ScanIssue('', error.code))
        except OSError:
            issues.append(ScanIssue('', 'unreadable_root'))
        return SourceSnapshot(tuple(sorted(files, key=lambda item: item.path)), tuple(sorted(directories)),
                              tuple(issues), skipped)
