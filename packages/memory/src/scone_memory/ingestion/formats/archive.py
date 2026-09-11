"""Read Office/EPUB containers without extraction, expansion bombs or XML entities."""
from __future__ import annotations

from io import BytesIO
from pathlib import PurePosixPath
from types import TracebackType
from xml.etree.ElementTree import Element
from zipfile import BadZipFile, ZipFile

from ...core.errors import InvalidInput
from .types import DocumentLimits


class SafeArchive:
    def __init__(self, data: bytes, limits: DocumentLimits):
        if len(data) > limits.max_input_bytes:
            raise InvalidInput('document exceeds its input byte limit')
        self._limit = limits.max_archive_bytes
        try:
            self._zip = ZipFile(BytesIO(data))
            entries = self._zip.infolist()
            self.names = tuple(item.filename for item in entries)
            if len(entries) > limits.max_archive_entries or len(set(self.names)) != len(entries):
                raise InvalidInput('archive has too many or duplicate entries')
            if sum(item.file_size for item in entries) > self._limit:
                raise InvalidInput('archive exceeds its expanded byte limit')
            for item in entries:
                path = PurePosixPath(item.filename)
                if (path.is_absolute() or '..' in path.parts or '\\' in item.filename
                        or item.flag_bits & 1 or (item.external_attr >> 16) & 0o170000 == 0o120000):
                    raise InvalidInput('archive contains an unsafe or encrypted entry')
        except (BadZipFile, OSError, ValueError):
            if hasattr(self, '_zip'):
                self._zip.close()
            raise InvalidInput('document is not a valid ZIP container') from None
        except InvalidInput:
            self._zip.close()
            raise

    def __enter__(self) -> SafeArchive:
        return self

    def __exit__(self, *_exc: object) -> None:
        self._zip.close()

    def read(self, name: str) -> bytes:
        try:
            with self._zip.open(name) as stream:
                data = stream.read(self._limit + 1)
            if len(data) > self._limit:
                raise InvalidInput('archive member exceeds its byte limit')
            return data
        except (KeyError, BadZipFile, RuntimeError, OSError, NotImplementedError):
            raise InvalidInput('archive member is missing or unreadable') from None

    def xml(self, name: str) -> Element:
        try:
            from defusedxml.ElementTree import fromstring
        except ImportError:
            raise InvalidInput('XML documents require scone-memory[documents]') from None
        try:
            return fromstring(self.read(name), forbid_dtd=True)
        except InvalidInput:
            raise
        except Exception:
            raise InvalidInput('document contains malformed or unsafe XML') from None
