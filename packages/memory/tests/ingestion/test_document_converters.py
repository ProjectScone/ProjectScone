"""Document converters use generated, non-user document fixtures."""
from __future__ import annotations

from importlib import import_module
from io import BytesIO
from pathlib import Path
import platform
import struct
import subprocess
from typing import Protocol, cast

import pytest

from scone_memory.core.errors import InvalidInput
from scone_memory.ingestion.formats.converters import parse_converted
from scone_memory.ingestion.formats.types import DocumentLimits


class _SheetWriter(Protocol):
    def write(self, row: int, column: int, value: str | int) -> None: ...


class _WorkbookWriter(Protocol):
    def add_sheet(self, name: str) -> _SheetWriter: ...
    def save(self, destination: BytesIO) -> None: ...


class _Xlwt(Protocol):
    def Workbook(self) -> _WorkbookWriter: ...


def _xls() -> bytes:
    pytest.importorskip('xlwt')
    writer = cast(_Xlwt, import_module('xlwt')).Workbook()
    sheet = writer.add_sheet('Budget')
    sheet.write(2, 1, 'Equipment')
    sheet.write(2, 2, 150)
    sheet.write(4, 1, 'Travel')
    sheet.write(4, 2, 200)
    stream = BytesIO()
    writer.save(stream)
    return stream.getvalue()


def _msg() -> bytes:
    """Minimal CFB v3 with regular streams; generated public-domain content."""
    free, end, fat = 0xFFFFFFFF, 0xFFFFFFFE, 0xFFFFFFFD
    header = bytearray(512)
    header[:8] = bytes.fromhex('d0cf11e0a1b11ae1')
    struct.pack_into('<HHHHH', header, 24, 0x3E, 3, 0xFFFE, 9, 6)
    struct.pack_into('<IIIIIIIII', header, 40, 0, 1, 0, 0, 4096, end, 0, end, 0)
    struct.pack_into('<109I', header, 76, 25, *([free] * 108))

    def entry(name: str, kind: int, right: int, child: int, start: int, size: int) -> bytes:
        value = bytearray(128)
        encoded = (name + '\0').encode('utf-16-le')
        value[:len(encoded)] = encoded
        struct.pack_into('<HBBIII', value, 64, len(encoded), kind, 1, free, right, child)
        struct.pack_into('<IQ', value, 116, start, size)
        return bytes(value)

    directory = b''.join([
        entry('Root Entry', 5, free, 1, end, 0),
        entry('__properties_version1.0', 2, 2, free, 1, 4096),
        entry('__substg1.0_0037001F', 2, 3, free, 9, 4096),
        entry('__substg1.0_1000001F', 2, free, free, 17, 4096),
    ])
    chains = [end]
    for first in (1, 9, 17):
        chains.extend([*range(first + 1, first + 8), end])
    chains.append(fat)
    chains.extend([free] * (128 - len(chains)))
    return b''.join([
        header, directory, bytes(4096),
        'Quarterly review'.ljust(2048).encode('utf-16-le'),
        'The launch is scheduled for Monday.'.ljust(2048).encode('utf-16-le'),
        struct.pack('<128I', *chains),
    ])


def test_rtf_extracts_paragraphs_unicode_and_escapes() -> None:
    pytest.importorskip('striprtf')
    parsed = parse_converted(
        br'{\rtf1\ansi First \b bold\b0.\par Caf\u233? costs \'a35.\par Literal \{brace\}.}',
        'notes.RTF', DocumentLimits(),
    )
    assert [s.text for s in parsed.segments] == ['First bold.', 'Café costs £5.', 'Literal {brace}.']
    assert [s.locator for s in parsed.segments] == ['paragraph:1', 'paragraph:2', 'paragraph:3']


def test_rtf_honors_raw_codepage_bytes_and_surrogate_pairs() -> None:
    pytest.importorskip('striprtf')
    parsed = parse_converted(
        b'{\\rtf1\\ansi\\ansicpg1251 ' + 'Привет'.encode('cp1251')
        + br' \u-10179?\u-8704?}', 'cyrillic.rtf', DocumentLimits(),
    )
    assert parsed.segments[0].text == 'Привет 😀'


def test_xls_preserves_sheet_and_original_row_numbers() -> None:
    pytest.importorskip('python_calamine')
    parsed = parse_converted(_xls(), 'budget.xls', DocumentLimits())
    assert [s.text for s in parsed.segments] == ['Equipment\t150', 'Travel\t200']
    assert [s.locator for s in parsed.segments] == ['sheet:Budget:row:3', 'sheet:Budget:row:5']


def test_msg_extracts_subject_and_plain_body_without_attachments() -> None:
    pytest.importorskip('extract_msg')
    parsed = parse_converted(_msg(), 'review.msg', DocumentLimits())
    assert [(s.locator, s.text) for s in parsed.segments] == [
        ('mailpart:subject', 'Quarterly review'),
        ('mailpart:body', 'The launch is scheduled for Monday.'),
    ]
    assert 'attachments' in parsed.metadata['limitations']


@pytest.mark.parametrize('name', ['bad.rtf', 'bad.xls', 'bad.xlsb', 'bad.msg'])
def test_converter_rejects_malformed_data(name: str) -> None:
    with pytest.raises(InvalidInput):
        parse_converted(b'not a document', name, DocumentLimits())


def test_rtf_enforces_output_and_segment_limits() -> None:
    pytest.importorskip('striprtf')
    with pytest.raises(InvalidInput, match='text byte limit'):
        parse_converted(br'{\rtf1 Too much text}', 'x.rtf', DocumentLimits(max_text_bytes=5))
    with pytest.raises(InvalidInput, match='segment limit'):
        parse_converted(br'{\rtf1 One\par Two}', 'x.rtf', DocumentLimits(max_segments=1))


def test_converter_checks_input_before_optional_dependencies() -> None:
    with pytest.raises(InvalidInput, match='input byte limit'):
        parse_converted(b'oversized', 'x.xls', DocumentLimits(max_input_bytes=3))


def test_missing_optional_dependency_is_actionable(monkeypatch: pytest.MonkeyPatch) -> None:
    import scone_memory.ingestion.formats.converters as converters

    def missing(name: str) -> object:
        raise ImportError(name)

    monkeypatch.setattr(converters, 'import_module', missing)
    with pytest.raises(InvalidInput, match='no parser available.*document-converters'):
        parse_converted(br'{\rtf1 Text}', 'x.rtf', DocumentLimits())


def test_ppt_does_not_autostart_an_unconfigured_converter(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv('SCONE_MEMORY_DOCUMENT_CONVERTER', raising=False)
    with pytest.raises(InvalidInput, match='no parser available'):
        parse_converted(bytes.fromhex('d0cf11e0a1b11ae1'), 'slides.ppt', DocumentLimits())


def test_configured_converter_keeps_source_paragraph_limitations(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    converter = tmp_path / 'offline-converter'
    converter.write_text('#!/bin/sh\nprintf "Converted title\\nConverted details\\n" > "$2"\n')
    converter.chmod(0o700)
    monkeypatch.setenv('SCONE_MEMORY_DOCUMENT_CONVERTER', str(converter))
    parsed = parse_converted(bytes.fromhex('d0cf11e0a1b11ae1'), 'slides.ppt', DocumentLimits())
    assert [s.text for s in parsed.segments] == ['Converted title', 'Converted details']
    assert [s.locator for s in parsed.segments] == ['converted-paragraph:1', 'converted-paragraph:2']
    assert 'slide' in parsed.metadata['limitations']


def test_converter_failure_does_not_expose_command_output(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    converter = tmp_path / 'offline-converter'
    converter.write_text('#!/bin/sh\necho private-data >&2\nexit 1\n')
    converter.chmod(0o700)
    monkeypatch.setenv('SCONE_MEMORY_DOCUMENT_CONVERTER', str(converter))
    with pytest.raises(InvalidInput, match='conversion failed') as error:
        parse_converted(bytes.fromhex('d0cf11e0a1b11ae1'), 'slides.ppt', DocumentLimits())
    assert 'private-data' not in str(error.value)


@pytest.mark.parametrize('action', [
    'printf "123456789" > "$2"',
    'ln -s "$1" "$2"',
    'mkfifo "$2"',
])
def test_converter_rejects_large_or_unsafe_output(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, action: str,
) -> None:
    converter = tmp_path / 'offline-converter'
    converter.write_text('#!/bin/sh\n' + action + '\n')
    converter.chmod(0o700)
    monkeypatch.setenv('SCONE_MEMORY_DOCUMENT_CONVERTER', str(converter))
    with pytest.raises(InvalidInput):
        parse_converted(bytes.fromhex('d0cf11e0a1b11ae1'), 'x.ppt', DocumentLimits(max_text_bytes=8))


def test_converter_timeout_is_bounded(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    converter = tmp_path / 'offline-converter'
    converter.write_text('#!/bin/sh\nexec sleep 5\n')
    converter.chmod(0o700)
    monkeypatch.setenv('SCONE_MEMORY_DOCUMENT_CONVERTER', str(converter))
    with pytest.raises(InvalidInput, match='timed out'):
        parse_converted(bytes.fromhex('d0cf11e0a1b11ae1'), 'x.ppt', DocumentLimits(timeout_seconds=0.05))


@pytest.mark.skipif(platform.system() != 'Darwin', reason='macOS textutil adapter')
def test_doc_uses_real_textutil_conversion(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.delenv('SCONE_MEMORY_DOCUMENT_CONVERTER', raising=False)
    source = tmp_path / 'sample.txt'
    destination = tmp_path / 'sample.doc'
    source.write_text('Project status\nReady for review.\n')
    subprocess.run([
        '/usr/bin/textutil', '-convert', 'doc', '-noload', '-output',
        str(destination), str(source),
    ], check=True, capture_output=True, timeout=10)
    parsed = parse_converted(destination.read_bytes(), 'sample.doc', DocumentLimits())
    assert [s.text for s in parsed.segments] == ['Project status', 'Ready for review.']
    assert parsed.parser == 'textutil'
