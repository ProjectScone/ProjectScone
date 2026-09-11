"""Read Office/EPUB containers without extraction, expansion bombs or XML entities."""
from __future__ import annotations

from io import BytesIO
from pathlib import PurePosixPath
from xml.etree.ElementTree import Element
from zipfile import ZIP_DEFLATED, ZIP_STORED, BadZipFile, ZipFile

from ...core.errors import InvalidInput
from .types import DocumentLimits
from .bounded_xml import XML_MAX_BYTES, parse_xml

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
                if item.compress_type not in (ZIP_STORED, ZIP_DEFLATED):
                    raise InvalidInput('archive compression must be stored or deflated')
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
        return self._read(name, self._limit, 'archive member')

    def _read(self, name: str, maximum: int, label: str) -> bytes:
        try:
            if self._zip.getinfo(name).file_size > maximum:
                raise InvalidInput(f'{label} exceeds its byte limit')
            with self._zip.open(name) as stream:
                data = stream.read(maximum + 1)
            if len(data) > maximum:
                raise InvalidInput(f'{label} exceeds its byte limit')
            return data
        except (KeyError, BadZipFile, RuntimeError, OSError, NotImplementedError):
            raise InvalidInput('archive member is missing or unreadable') from None

    def xml(self, name: str, *, allow_doctype: bool = False,
            supported_namespaces: frozenset[str] | None = None) -> Element:
        data = self._read(name, min(self._limit, XML_MAX_BYTES), 'XML member')
        return parse_xml(data, allow_doctype=allow_doctype, supported_namespaces=supported_namespaces)
