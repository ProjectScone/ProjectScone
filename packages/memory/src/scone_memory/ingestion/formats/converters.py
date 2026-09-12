"""Optional document converters; no automatic service or converter startup.

An operator may set SCONE_MEMORY_DOCUMENT_CONVERTER to an absolute executable
path. Its interface is INPUT_PATH OUTPUT_TXT_PATH, and it must write UTF-8 text.
The operator is responsible for running that converter without network access
and with macros/external links disabled (for example an isolated LibreOffice
wrapper). No arbitrary command string, shell expansion, or download is used.
"""
from __future__ import annotations

from collections.abc import Iterator
from importlib import import_module
from io import BytesIO
import os
from pathlib import Path
import platform
import re
import stat
import subprocess
from tempfile import TemporaryDirectory
from typing import Protocol, cast
from urllib.parse import quote

from ...core.errors import InvalidInput
from ...ocr.process import worker_environment
from .archive import SafeArchive
from .types import DocumentLimits, DocumentSegment, ParsedDocument, validate_document

CONVERTER_EXTENSIONS = frozenset({'.rtf', '.xls', '.xlsb', '.msg', '.doc', '.ppt'})
_OLE_SIGNATURE = bytes.fromhex('d0cf11e0a1b11ae1')


class _RtfModule(Protocol):
    def rtf_to_text(self, text: str, encoding: str = 'cp1252', errors: str = 'strict') -> str: ...


class _Sheet(Protocol):
    @property
    def start(self) -> tuple[int, int] | None: ...
    @property
    def total_height(self) -> int: ...
    @property
    def total_width(self) -> int: ...
    def iter_rows(self) -> Iterator[list[object]]: ...


class _Workbook(Protocol):
    @property
    def sheet_names(self) -> list[str]: ...
    def get_sheet_by_name(self, name: str) -> _Sheet: ...
    def close(self) -> None: ...


class _WorkbookFactory(Protocol):
    def from_filelike(self, file: BytesIO) -> _Workbook: ...


class _CalamineModule(Protocol):
    @property
    def CalamineWorkbook(self) -> _WorkbookFactory: ...


class _Message(Protocol):
    def getStringStream(self, filename: str) -> str | None: ...
    def close(self) -> None: ...


class _MsgModule(Protocol):
    def MSGFile(self, path: bytes, *, delayAttachments: bool) -> _Message: ...


class _Segments:
    def __init__(self, limits: DocumentLimits) -> None:
        self.items: list[DocumentSegment] = []
        self.limits = limits
        self.bytes = 0

    def add(self, text: str, locator: str) -> None:
        text = text.strip(' \t\r\n\x00')
        if not text:
            return
        if len(self.items) >= self.limits.max_segments:
            raise InvalidInput('document exceeds its segment limit')
        self.bytes += len(text.encode('utf-8')) + (2 if self.items else 0)
        if self.bytes > self.limits.max_text_bytes:
            raise InvalidInput('document exceeds its extracted text byte limit')
        if len(locator) > 4096:
            raise InvalidInput('document source locator exceeds its limit')
        self.items.append(DocumentSegment(text=text, locator=locator))


def _optional(name: str) -> object:
    try:
        return import_module(name)
    except ImportError:
        raise InvalidInput('no parser available; install scone-memory[document-converters]') from None


def parse_converted(data: bytes, filename: str, limits: DocumentLimits) -> ParsedDocument:
    if len(data) > limits.max_input_bytes:
        raise InvalidInput('document exceeds its input byte limit')
    extension = Path(filename).suffix.lower()
    if extension not in CONVERTER_EXTENSIONS:
        raise InvalidInput('no parser available for this document extension')
    segments = _Segments(limits)
    try:
        if extension == '.rtf':
            parser, metadata = _rtf(data, segments)
        elif extension in {'.xls', '.xlsb'}:
            parser, metadata = _spreadsheet(data, extension, segments)
        elif extension == '.msg':
            parser, metadata = _message(data, segments)
        else:
            parser, metadata = _converted(data, extension, segments)
    except InvalidInput:
        raise
    except Exception:
        raise InvalidInput(f'invalid or unsupported {extension[1:]} document') from None
    if not segments.items:
        raise InvalidInput('document contains no extractable text')
    result = ParsedDocument(
        format=extension[1:], parser=parser,
        segments=tuple(segments.items), metadata=metadata,
    )
    validate_document(result, limits)
    return result


def _rtf(data: bytes, segments: _Segments) -> tuple[str, dict[str, str]]:
    if not data.startswith(b'{\\rtf'):
        raise InvalidInput('invalid RTF document signature')
    module = cast(_RtfModule, _optional('striprtf.striprtf'))
    # Normalize raw high bytes into RTF escapes so the reader applies the
    # declared document/font code page instead of treating them as Latin-1.
    rtf = re.sub(r'[\x80-\xff]', lambda match: f"\\'{ord(match[0]):02x}", data.decode('latin-1'))
    text = module.rtf_to_text(rtf)
    text = text.encode('utf-16-le', errors='surrogatepass').decode('utf-16-le')
    for index, line in enumerate(text.splitlines(), 1):
        segments.add(line, f'paragraph:{index}')
    return 'striprtf', {'limitations': 'Text only; embedded objects and visual layout are not extracted.'}


def _spreadsheet(data: bytes, extension: str, segments: _Segments) -> tuple[str, dict[str, str]]:
    if extension == '.xlsb':
        with SafeArchive(data, segments.limits):
            pass
    elif not data.startswith(_OLE_SIGNATURE):
        raise InvalidInput('invalid XLS compound document signature')
    module = cast(_CalamineModule, _optional('python_calamine'))
    workbook = module.CalamineWorkbook.from_filelike(BytesIO(data))
    try:
        if len(workbook.sheet_names) > segments.limits.max_archive_entries:
            raise InvalidInput('workbook exceeds its sheet limit')
        for name in workbook.sheet_names:
            sheet = workbook.get_sheet_by_name(name)
            if sheet.total_height * sheet.total_width > segments.limits.max_archive_bytes // 8:
                raise InvalidInput('workbook exceeds its expanded cell limit')
            for index, row in enumerate(sheet.iter_rows(), 1):
                values = [_cell(value) for value in row]
                while values and not values[-1]:
                    values.pop()
                # Retain interior gaps; leading empty columns carry no row text.
                text = '\t'.join(values).lstrip('\t')
                segments.add(text, f'sheet:{quote(name, safe="")}:row:{index}')
    finally:
        workbook.close()
    return 'python-calamine', {'limitations': 'Cell values only; formulas are not recalculated and charts are not extracted.'}


def _cell(value: object) -> str:
    if value is None:
        return ''
    if isinstance(value, float) and value.is_integer():
        return str(int(value))
    return str(value)


def _message(data: bytes, segments: _Segments) -> tuple[str, dict[str, str]]:
    if not data.startswith(_OLE_SIGNATURE):
        raise InvalidInput('invalid MSG compound document signature')
    module = cast(_MsgModule, _optional('extract_msg'))
    message = module.MSGFile(data, delayAttachments=True)
    try:
        # Read raw text properties: do not initialize attachments, decompress
        # RTF bodies, render HTML, execute links, or save message resources.
        for name, property_id in (
            ('subject', '0037'), ('sender', '0C1A'), ('to', '0E04'),
            ('cc', '0E03'), ('headers', '007D'), ('body', '1000'),
        ):
            text = message.getStringStream(f'__substg1.0_{property_id}')
            if text:
                segments.add(text, f'mailpart:{name}')
    finally:
        message.close()
    return 'extract-msg', {'limitations': 'Plain message properties only; attachments, HTML and compressed RTF bodies are not extracted.'}


def _converted(data: bytes, extension: str, segments: _Segments) -> tuple[str, dict[str, str]]:
    converter = os.environ.get('SCONE_MEMORY_DOCUMENT_CONVERTER', '')
    textutil = not converter and extension == '.doc' and platform.system() == 'Darwin'
    executable = Path('/usr/bin/textutil') if textutil else Path(converter)
    if (not converter and not textutil) or not executable.is_absolute() or not executable.is_file() or not os.access(executable, os.X_OK):
        raise InvalidInput('no parser available; configure an offline SCONE_MEMORY_DOCUMENT_CONVERTER executable')
    if not data.startswith(_OLE_SIGNATURE):
        raise InvalidInput('invalid binary Office compound document signature')
    with TemporaryDirectory(prefix='scone-convert-') as directory:
        source = Path(directory) / f'input{extension}'
        destination = Path(directory) / 'output.txt'
        source.write_bytes(data)
        if textutil:
            command = [str(executable), '-convert', 'txt', '-format', 'doc', '-noload', '-nostore', '-encoding', 'UTF-8', '-output', str(destination), str(source)]
        else:
            command = [str(executable), str(source), str(destination)]
        try:
            subprocess.run(
                command, stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL, timeout=segments.limits.timeout_seconds,
                check=True, cwd=directory, env=worker_environment(),
            )
        except (OSError, subprocess.SubprocessError):
            raise InvalidInput('document conversion failed or timed out') from None
        try:
            descriptor = os.open(destination, os.O_RDONLY | getattr(os, 'O_NOFOLLOW', 0) | getattr(os, 'O_NONBLOCK', 0))
            with os.fdopen(descriptor, 'rb') as output:
                status = os.fstat(output.fileno())
                if not stat.S_ISREG(status.st_mode):
                    raise InvalidInput('document converter output is not a regular file')
                content = output.read(segments.limits.max_text_bytes + 1)
            if len(content) > segments.limits.max_text_bytes:
                raise InvalidInput('document exceeds its extracted text byte limit')
            text = content.decode('utf-8-sig')
        except (OSError, UnicodeError):
            raise InvalidInput('document conversion produced no valid UTF-8 text output') from None
        for index, line in enumerate(text.splitlines(), 1):
            segments.add(line, f'converted-paragraph:{index}')
    return ('textutil' if textutil else 'configured-offline-converter'), {
        'limitations': 'Converted text only; original page/slide boundaries, tables and visual layout may be lost.',
    }
