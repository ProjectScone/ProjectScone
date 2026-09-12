from __future__ import annotations

from io import BytesIO
from zipfile import ZipFile

import pytest

from scone_memory.core.errors import InvalidInput
from scone_memory.ingestion.formats.office import parse_office
from scone_memory.ingestion.formats.types import DocumentLimits

W = 'http://schemas.openxmlformats.org/wordprocessingml/2006/main'
S = 'http://schemas.openxmlformats.org/spreadsheetml/2006/main'
P = 'http://schemas.openxmlformats.org/presentationml/2006/main'
A = 'http://schemas.openxmlformats.org/drawingml/2006/main'
R = 'http://schemas.openxmlformats.org/officeDocument/2006/relationships'
REL = 'http://schemas.openxmlformats.org/package/2006/relationships'
O = 'urn:oasis:names:tc:opendocument:xmlns:office:1.0'
T = 'urn:oasis:names:tc:opendocument:xmlns:text:1.0'
TABLE = 'urn:oasis:names:tc:opendocument:xmlns:table:1.0'
D = 'urn:oasis:names:tc:opendocument:xmlns:drawing:1.0'


def archive(parts: dict[str, str]) -> bytes:
    output = BytesIO()
    with ZipFile(output, 'w') as bundle:
        for name, content in parts.items():
            bundle.writestr(name, content)
    return output.getvalue()


def package_relationship(main_part: str) -> str:
    return f'<Relationships xmlns="{REL}"><Relationship Id="main" Target="{main_part}" Type="{R}/officeDocument"/></Relationships>'


def ooxml_archive(parts: dict[str, str], *, main_part: str) -> bytes:
    return archive({'_rels/.rels': package_relationship(main_part), **parts})


def docx(text: str = 'Hello') -> bytes:
    return ooxml_archive({'word/document.xml': f'<w:document xmlns:w="{W}"><w:body><w:p><w:r><w:t>{text}</w:t></w:r></w:p><w:tbl><w:tr><w:tc><w:p><w:r><w:t>First</w:t></w:r></w:p></w:tc><w:tc><w:p><w:r><w:t>Second</w:t></w:r></w:p></w:tc></w:tr></w:tbl></w:body></w:document>'}, main_part='word/document.xml')


def test_docx_preserves_paragraph_table_order_and_locators() -> None:
    result = parse_office(docx(), 'report.docx', DocumentLimits())
    assert [item.text for item in result.segments] == ['Hello', 'First\tSecond']
    assert [item.locator for item in result.segments] == ['paragraph:1', 'table:1/row:1']


def test_xlsx_sheet_order_shared_inline_strings_cached_formula() -> None:
    data = ooxml_archive({
        'xl/workbook.xml': f'<workbook xmlns="{S}" xmlns:r="{R}"><sheets><sheet name="Budget" sheetId="2" r:id="r2"/><sheet name="Names" sheetId="1" r:id="r1"/></sheets></workbook>',
        'xl/_rels/workbook.xml.rels': f'<Relationships xmlns="{REL}"><Relationship Id="r1" Target="worksheets/sheet1.xml" Type="{R}/worksheet"/><Relationship Id="r2" Target="worksheets/sheet2.xml" Type="{R}/worksheet"/><Relationship Id="strings" Target="sharedStrings.xml" Type="{R}/sharedStrings"/></Relationships>',
        'xl/sharedStrings.xml': f'<sst xmlns="{S}"><si><r><t>Ada</t></r><r><t> Lovelace</t></r></si></sst>',
        'xl/worksheets/sheet1.xml': f'<worksheet xmlns="{S}"><sheetData><row r="1"><c r="A1" t="s"><v>0</v></c></row></sheetData></worksheet>',
        'xl/worksheets/sheet2.xml': f'<worksheet xmlns="{S}"><sheetData><row r="3"><c r="B3" t="inlineStr"><is><t>Cost</t></is></c><c r="C3"><f>1+1</f><v>2</v></c><c r="D3"><f>WEBSERVICE("https://invalid")</f></c></row></sheetData></worksheet>',
    }, main_part='xl/workbook.xml')
    result = parse_office(data, 'report.xlsx', DocumentLimits())
    assert [item.text for item in result.segments] == ['Cost', '2', 'Ada Lovelace']
    assert [item.locator for item in result.segments] == ['sheet:Budget/cell:B3', 'sheet:Budget/cell:C3', 'sheet:Names/cell:A1']
    assert result.segments[1].metadata['formula'] == 'cached-value'


def test_pptx_uses_presentation_order_and_includes_notes() -> None:
    data = ooxml_archive({
        'ppt/presentation.xml': f'<p:presentation xmlns:p="{P}" xmlns:r="{R}"><p:sldIdLst><p:sldId id="300" r:id="r2"/><p:sldId id="200" r:id="r1"/></p:sldIdLst></p:presentation>',
        'ppt/_rels/presentation.xml.rels': f'<Relationships xmlns="{REL}"><Relationship Id="r1" Target="slides/slide1.xml" Type="{R}/slide"/><Relationship Id="r2" Target="slides/slide2.xml" Type="{R}/slide"/></Relationships>',
        'ppt/slides/slide1.xml': f'<p:sld xmlns:p="{P}" xmlns:a="{A}"><p:cSld><p:spTree><p:sp><p:txBody><a:p><a:r><a:t>Last</a:t></a:r></a:p></p:txBody></p:sp></p:spTree></p:cSld></p:sld>',
        'ppt/slides/slide2.xml': f'<p:sld xmlns:p="{P}" xmlns:a="{A}"><p:cSld><p:spTree><p:graphicFrame><a:tbl><a:tr><a:tc><a:txBody><a:p><a:r><a:t>First</a:t></a:r></a:p></a:txBody></a:tc></a:tr></a:tbl></p:graphicFrame></p:spTree></p:cSld></p:sld>',
        'ppt/slides/_rels/slide2.xml.rels': f'<Relationships xmlns="{REL}"><Relationship Id="n1" Target="../notesSlides/notesSlide1.xml" Type="{R}/notesSlide"/></Relationships>',
        'ppt/notesSlides/notesSlide1.xml': f'<p:notes xmlns:p="{P}" xmlns:a="{A}"><p:cSld><p:spTree><p:sp><p:txBody><a:p><a:r><a:t>Speaker detail</a:t></a:r></a:p></p:txBody></p:sp></p:spTree></p:cSld></p:notes>',
    }, main_part='ppt/presentation.xml')
    result = parse_office(data, 'slides.pptx', DocumentLimits())
    assert [item.text for item in result.segments] == ['First', 'Speaker detail', 'Last']
    assert result.segments[0].locator.startswith('slide:1/')
    assert result.segments[1].locator.startswith('slide:1/notes/')
    assert result.segments[2].locator.startswith('slide:2/')


@pytest.mark.parametrize(('extension', 'body', 'expected'), [
    ('odt', '<office:text><text:h>Heading</text:h><text:p>Hello<text:s text:c="2"/>world</text:p></office:text>', ['Heading', 'Hello  world']),
    ('odp', '<office:presentation><draw:page draw:name="Intro"><draw:frame><draw:text-box><text:p>Welcome</text:p></draw:text-box></draw:frame></draw:page></office:presentation>', ['Welcome']),
    ('ods', '<office:spreadsheet><table:table table:name="Budget"><table:table-row><table:table-cell office:value-type="float" office:value="42" table:number-columns-repeated="2"/></table:table-row></table:table></office:spreadsheet>', ['42']),
])
def test_open_document(extension: str, body: str, expected: list[str]) -> None:
    data = archive({'content.xml': f'<office:document-content xmlns:office="{O}" xmlns:text="{T}" xmlns:table="{TABLE}" xmlns:draw="{D}"><office:body>{body}</office:body></office:document-content>'})
    result = parse_office(data, f'document.{extension}', DocumentLimits())
    assert [item.text for item in result.segments] == expected
    if extension == 'ods':
        assert result.segments[0].metadata['column_repeat'] == '2'
        assert result.segments[0].locator == 'sheet:Budget/cell:A1'


def test_epub_spine_order_ignores_script_and_preserves_inline_spacing() -> None:
    data = archive({
        'META-INF/container.xml': '<container xmlns="urn:oasis:names:tc:opendocument:xmlns:container"><rootfiles><rootfile full-path="OPS/book.opf"/></rootfiles></container>',
        'OPS/book.opf': '<package xmlns="http://www.idpf.org/2007/opf"><manifest><item id="a" href="a.xhtml" media-type="application/xhtml+xml"/><item id="b" href="b.xhtml" media-type="application/xhtml+xml"/></manifest><spine><itemref idref="b"/><itemref idref="a"/></spine></package>',
        'OPS/a.xhtml': '<html xmlns="http://www.w3.org/1999/xhtml"><body><p>Last</p></body></html>',
        'OPS/b.xhtml': '<html xmlns="http://www.w3.org/1999/xhtml"><head><title>Ignore</title></head><body><p>Hello <em>reader</em>!</p><script>ignore()</script></body></html>',
    })
    result = parse_office(data, 'book.epub', DocumentLimits())
    assert [item.text for item in result.segments] == ['Hello reader!', 'Last']
    assert 'OPS/b.xhtml' in result.segments[0].locator


@pytest.mark.parametrize('data', [b'not a zip', b'\xd0\xcf\x11\xe0encrypted', ooxml_archive({'word/document.xml': '<broken>'}, main_part='word/document.xml'), ooxml_archive({'word/document.xml': f'<w:document xmlns:w="{W}"><w:body/></w:document>'}, main_part='word/document.xml'), ooxml_archive({'word/document.xml': '<!DOCTYPE x [<!ENTITY x SYSTEM "file:///etc/passwd">]><x>&x;</x>'}, main_part='word/document.xml')])
def test_rejects_malformed_empty_or_encrypted_documents(data: bytes) -> None:
    with pytest.raises(InvalidInput):
        parse_office(data, 'bad.docx', DocumentLimits())


def test_enforces_output_bounds_while_extracting() -> None:
    with pytest.raises(InvalidInput):
        parse_office(docx('x' * 100), 'big.docx', DocumentLimits(max_text_bytes=32))
    with pytest.raises(InvalidInput):
        parse_office(docx(), 'many.docx', DocumentLimits(max_segments=1))


def test_rejects_oversized_odf_repeat_without_expansion() -> None:
    data = archive({'content.xml': f'<office:document-content xmlns:office="{O}" xmlns:text="{T}"><office:body><office:text><text:p>Hello<text:s text:c="999999999999999"/></text:p></office:text></office:body></office:document-content>'})
    with pytest.raises(InvalidInput):
        parse_office(data, 'repeat.odt', DocumentLimits())


def test_rejects_external_required_relationship() -> None:
    data = ooxml_archive({
        'xl/workbook.xml': f'<workbook xmlns="{S}" xmlns:r="{R}"><sheets><sheet name="Bad" r:id="r1"/></sheets></workbook>',
        'xl/_rels/workbook.xml.rels': f'<Relationships xmlns="{REL}"><Relationship Id="r1" Target="https://example.com/sheet.xml" TargetMode="External" Type="{R}/worksheet"/></Relationships>',
    }, main_part='xl/workbook.xml')
    with pytest.raises(InvalidInput):
        parse_office(data, 'bad.xlsx', DocumentLimits())


def test_docx_table_separates_paragraphs_within_cells() -> None:
    data = ooxml_archive({'word/document.xml': f'<w:document xmlns:w="{W}"><w:body><w:tbl><w:tr><w:tc><w:p><w:r><w:t>Line one</w:t></w:r></w:p><w:p><w:r><w:t>Line two</w:t></w:r></w:p></w:tc></w:tr></w:tbl></w:body></w:document>'}, main_part='word/document.xml')
    result = parse_office(data, 'table.docx', DocumentLimits())
    assert result.segments[0].text == 'Line one\nLine two'


def test_odt_table_has_row_locator_and_cell_boundaries() -> None:
    data = archive({'content.xml': f'<office:document-content xmlns:office="{O}" xmlns:text="{T}" xmlns:table="{TABLE}"><office:body><office:text><text:p>Before</text:p><table:table><table:table-row><table:table-cell><text:p>Left</text:p></table:table-cell><table:table-cell><text:p>Right</text:p></table:table-cell></table:table-row></table:table><text:p>After</text:p></office:text></office:body></office:document-content>'})
    result = parse_office(data, 'table.odt', DocumentLimits())
    assert [item.text for item in result.segments] == ['Before', 'Left\tRight', 'After']
    assert result.segments[1].locator == 'table:1/row:1'


def test_epub_keeps_bare_body_and_div_text_with_paragraph_boundaries() -> None:
    data = archive({
        'META-INF/container.xml': '<container><rootfile full-path="book.opf"/></container>',
        'book.opf': '<package><manifest><item id="c" href="chapter.xhtml" media-type="application/xhtml+xml"/></manifest><spine><itemref idref="c"/></spine></package>',
        'chapter.xhtml': '<html><body>Opening<div>Detail<p>Paragraph <em>one</em>.</p><p>Paragraph two.</p></div>Ending<script><p>Never include</p></script></body></html>',
    })
    result = parse_office(data, 'book.epub', DocumentLimits())
    combined = '\n'.join(item.text for item in result.segments)
    assert combined == 'Opening\nDetail\nParagraph one.\nParagraph two.\nEnding'


@pytest.mark.parametrize(('extension', 'parts'), [
    ('odt', {'META-INF/manifest.xml': '<manifest><encryption-data/></manifest>'}),
    ('epub', {'META-INF/encryption.xml': '<encryption/>'}),
])
def test_rejects_encryption_metadata(extension: str, parts: dict[str, str]) -> None:
    with pytest.raises(InvalidInput, match='encrypted'):
        parse_office(archive(parts), f'locked.{extension}', DocumentLimits())


def test_rejects_input_archive_entry_and_expansion_limits() -> None:
    data = docx()
    for limits in [DocumentLimits(max_input_bytes=10), DocumentLimits(max_archive_bytes=10)]:
        with pytest.raises(InvalidInput):
            parse_office(data, 'big.docx', limits)
    assert parse_office(data, 'small.docx', DocumentLimits(max_archive_entries=2)).segments[0].text == 'Hello'
    extra = ooxml_archive({'word/document.xml': f'<w:document xmlns:w="{W}"><w:body><w:p><w:r><w:t>Visible</w:t></w:r></w:p></w:body></w:document>', 'extra': 'x'}, main_part='word/document.xml')
    with pytest.raises(InvalidInput, match='too many'):
        parse_office(extra, 'big.docx', DocumentLimits(max_archive_entries=2))


def test_rejects_escaping_epub_relationship() -> None:
    data = archive({'META-INF/container.xml': '<container><rootfile full-path="../book.opf"/></container>'})
    with pytest.raises(InvalidInput, match='escapes'):
        parse_office(data, 'unsafe.epub', DocumentLimits())


def test_rejects_legacy_office_in_native_reader() -> None:
    with pytest.raises(InvalidInput, match='unsupported'):
        parse_office(b'\xd0\xcf\x11\xe0', 'legacy.ppt', DocumentLimits())


def test_timeout_is_checked_during_native_extraction(monkeypatch: pytest.MonkeyPatch) -> None:
    from scone_memory.ingestion.formats import office

    readings = iter([0.0, 2.0])
    monkeypatch.setattr(office, 'monotonic', lambda: next(readings))
    with pytest.raises(InvalidInput, match='timeout'):
        parse_office(docx(), 'slow.docx', DocumentLimits(timeout_seconds=1.0))
