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


@pytest.mark.parametrize('extension', ['docm', 'dotx', 'dotm'])
def test_word_family_reads_the_same_package_under_its_own_extension_and_says_when_macros_were_there(extension: str) -> None:
    result = parse_office(docx('Macro text'), f'memo.{extension}', DocumentLimits())
    assert [item.text for item in result.segments] == ['Macro text', 'First\tSecond'] and result.format == extension
    assert 'macros' not in result.metadata, "a package without vbaProject.bin carried no macros"
    with_macros = ooxml_archive({
        'word/document.xml': f'<w:document xmlns:w="{W}"><w:body><w:p><w:r><w:t>Macro text</w:t></w:r></w:p></w:body></w:document>',
        'word/vbaProject.bin': 'Sub Auto_Open() MsgBox "never run" End Sub',
    }, main_part='word/document.xml')
    result = parse_office(with_macros, f'memo.{extension}', DocumentLimits())
    assert [item.text for item in result.segments] == ['Macro text'], "the macro member is neither read nor run"
    assert result.metadata == {'macros': 'present, not read'}


@pytest.mark.parametrize('extension', ['xlsm', 'xltx', 'xltm'])
def test_sheet_family_reads_as_a_workbook(extension: str) -> None:
    data = ooxml_archive({
        'xl/workbook.xml': f'<workbook xmlns="{S}" xmlns:r="{R}"><sheets><sheet name="Budget" sheetId="1" r:id="r1"/></sheets></workbook>',
        'xl/_rels/workbook.xml.rels': f'<Relationships xmlns="{REL}"><Relationship Id="r1" Target="worksheets/sheet1.xml" Type="{R}/worksheet"/></Relationships>',
        'xl/worksheets/sheet1.xml': f'<worksheet xmlns="{S}"><sheetData><row r="1"><c r="A1" t="inlineStr"><is><t>Cost</t></is></c></row></sheetData></worksheet>',
        'xl/vbaProject.bin': 'binary',
    }, main_part='xl/workbook.xml')
    result = parse_office(data, f'report.{extension}', DocumentLimits())
    assert [(item.text, item.locator) for item in result.segments] == [('Cost', 'sheet:Budget/cell:A1')]
    assert result.format == extension and result.metadata == {'macros': 'present, not read'}


@pytest.mark.parametrize('extension', ['pptm', 'potx', 'potm', 'ppsx', 'ppsm'])
def test_slides_family_reads_as_a_presentation(extension: str) -> None:
    data = ooxml_archive({
        'ppt/presentation.xml': f'<p:presentation xmlns:p="{P}" xmlns:r="{R}"><p:sldIdLst><p:sldId id="200" r:id="r1"/></p:sldIdLst></p:presentation>',
        'ppt/_rels/presentation.xml.rels': f'<Relationships xmlns="{REL}"><Relationship Id="r1" Target="slides/slide1.xml" Type="{R}/slide"/></Relationships>',
        'ppt/slides/slide1.xml': f'<p:sld xmlns:p="{P}" xmlns:a="{A}"><p:cSld><p:spTree><p:sp><p:txBody><a:p><a:r><a:t>Only slide</a:t></a:r></a:p></p:txBody></p:sp></p:spTree></p:cSld></p:sld>',
    }, main_part='ppt/presentation.xml')
    result = parse_office(data, f'deck.{extension}', DocumentLimits())
    assert [item.text for item in result.segments] == ['Only slide'] and result.segments[0].locator.startswith('slide:1/')
    assert result.format == extension and 'macros' not in result.metadata


FAMILY_MEMBERS = {'docx', 'docm', 'dotx', 'dotm', 'xlsx', 'xlsm', 'xltx', 'xltm',
                  'pptx', 'pptm', 'potx', 'potm', 'ppsx', 'ppsm'}


def test_the_capability_listing_the_source_scan_and_the_media_types_name_every_office_family_member() -> None:
    from scone_memory.ingestion.files import FILE_MEDIA_TYPES
    from scone_memory.ingestion.formats.capabilities import document_formats
    from scone_memory.ingestion.formats.office import OFFICE_FAMILIES
    from scone_memory.ingestion.source_scan import default_extensions
    from scone_memory.memory.engine import ATTACHMENT_TYPES

    assert set().union(*OFFICE_FAMILIES.values()) == FAMILY_MEMBERS
    dotted = {'.' + member for member in FAMILY_MEMBERS}
    assert dotted <= set(document_formats()), "the capability listing names every member"
    assert dotted <= set(default_extensions()), "a directory scan reads every member"
    assert dotted <= set(FILE_MEDIA_TYPES), "every member has a media type"
    assert set(FILE_MEDIA_TYPES.values()) <= set(ATTACHMENT_TYPES), "every document media type is one the engine attaches"


async def test_a_macro_enabled_document_is_ingested_end_to_end_under_its_own_media_type() -> None:
    from scone_memory import HashEmbedder, InMemoryDocumentStore, InMemoryVectorIndex, MemoryEngine
    from scone_memory.ingestion.files import ingest_document

    memory = await MemoryEngine(InMemoryDocumentStore(), InMemoryVectorIndex(), HashEmbedder()).open()
    try:
        ingested = await ingest_document(memory, 'docs', docx('Quarterly memo'), filename='memo.docm')
        assert ingested.format == 'docm' and ingested.original.media_type == 'application/vnd.ms-word.document.macroEnabled.12'
        episode = await memory.episode('docs', ingested.added.episode_id)
        assert 'Quarterly memo' in episode.content and episode.metadata['document_format'] == 'docm'
    finally:
        await memory.close()


HP = 'http://www.hancom.co.kr/hwpml/2011/paragraph'
HS = 'http://www.hancom.co.kr/hwpml/2011/section'
OPF = 'http://www.idpf.org/2007/opf/'


def hwpx_section(*paragraphs: str) -> str:
    return f'<hs:sec xmlns:hs="{HS}" xmlns:hp="{HP}">' + ''.join(paragraphs) + '</hs:sec>'


def hwpx_p(*runs: str) -> str:
    return '<hp:p>' + ''.join(f'<hp:run>{run}</hp:run>' for run in runs) + '</hp:p>'


def hwpx_package(sections: dict[str, str], *, spine: list[str] | None = None) -> bytes:
    parts = dict(sections)
    if spine is not None:
        items = ''.join(f'<opf:item id="{name}" href="Contents/{name}.xml" media-type="application/xml"/>' for name in spine)
        refs = ''.join(f'<opf:itemref idref="{name}"/>' for name in spine)
        parts['Contents/content.hpf'] = f'<opf:package xmlns:opf="{OPF}"><opf:manifest>{items}</opf:manifest><opf:spine>{refs}</opf:spine></opf:package>'
    return archive(parts)


def test_hwpx_reads_paragraphs_in_spine_order_with_tabs_breaks_and_table_cells_after_their_paragraph() -> None:
    second = hwpx_section(hwpx_p('<hp:t>Second section</hp:t>'))
    first = hwpx_section(
        hwpx_p('<hp:t>Harbour</hp:t><hp:t> notes</hp:t>'),
        hwpx_p('<hp:t>Tide</hp:t><hp:tab/><hp:t>high</hp:t><hp:lineBreak/><hp:t>Wind</hp:t><hp:tab/><hp:t>west</hp:t>'),
        hwpx_p('<hp:t>Berths:</hp:t>',
               '<hp:tbl><hp:tr><hp:tc><hp:cellAddr colAddr="0" rowAddr="0"/><hp:subList><hp:p><hp:run><hp:t>A1</hp:t></hp:run></hp:p></hp:subList></hp:tc>'
               '<hp:tc><hp:cellAddr colAddr="1" rowAddr="0"/><hp:subList><hp:p><hp:run><hp:t>rope</hp:t></hp:run></hp:p><hp:p><hp:run><hp:t>chain</hp:t></hp:run></hp:p></hp:subList></hp:tc></hp:tr></hp:tbl>'),
        '<hp:p><hp:run><hp:secPr/></hp:run></hp:p>',
    )
    data = hwpx_package({'Contents/section0.xml': second, 'Contents/section1.xml': first}, spine=['section1', 'section0'])
    result = parse_office(data, 'notes.hwpx', DocumentLimits())
    assert [item.text for item in result.segments] == ['Harbour notes', 'Tide\thigh\nWind\twest', 'Berths:', 'A1', 'rope\nchain', 'Second section']
    assert [item.locator for item in result.segments] == [
        'section:1/paragraph:1', 'section:1/paragraph:2', 'section:1/paragraph:3',
        'section:1/paragraph:3/table:1/cell:1', 'section:1/paragraph:3/table:1/cell:2', 'section:2/paragraph:1'], "the spine, not the file names, orders the sections"
    cell = result.segments[3]
    assert cell.metadata == {'content_role': 'table_cell', 'parent_locator': 'section:1/paragraph:3', 'row': '0', 'column': '0'}
    assert result.format == 'hwpx' and result.segments[0].metadata['member'] == 'Contents/section1.xml'
    assert 'secPr' not in ''.join(item.text for item in result.segments), "a paragraph with only a control has no text and is not a segment"


def test_hwpx_reads_nested_tables_under_their_cell_once_and_the_characters_that_stand_inside_a_text() -> None:
    inner = ('<hp:tbl><hp:tr><hp:tc><hp:cellAddr colAddr="0" rowAddr="0"/><hp:subList><hp:p><hp:run><hp:t>inner</hp:t></hp:run></hp:p></hp:subList></hp:tc></hp:tr></hp:tbl>')
    outer = ('<hp:tbl><hp:tr>'
             f'<hp:tc><hp:cellAddr colAddr="0" rowAddr="0"/><hp:subList><hp:p><hp:run>{inner}</hp:run></hp:p></hp:subList></hp:tc>'
             '<hp:tc><hp:cellAddr colAddr="1" rowAddr="0"/><hp:subList><hp:p><hp:run><hp:t>outer2</hp:t></hp:run></hp:p></hp:subList></hp:tc>'
             '</hp:tr></hp:tbl>')
    section = hwpx_section(
        hwpx_p(f'<hp:t>Rooms:</hp:t>{outer}'),
        '<hp:p>\n  <hp:run>\n    <hp:t>서울<hp:fwSpace/>특별시<hp:nbSpace/>강남구<hp:markpenBegin/>tail kept<hp:markpenEnd/>.</hp:t>\n  </hp:run>\n</hp:p>',
    )
    result = parse_office(hwpx_package({'Contents/section0.xml': section}), 'rooms.hwpx', DocumentLimits())
    assert [(item.locator, item.text) for item in result.segments] == [
        ('section:1/paragraph:1', 'Rooms:'),
        ('section:1/paragraph:1/table:1/cell:1/table:1/cell:1', 'inner'),
        ('section:1/paragraph:1/table:1/cell:2', 'outer2'),
        ('section:1/paragraph:2', '서울\u3000특별시\u00a0강남구tail kept.'),
    ], "a nested cell is read once under its cell, the outer cells keep their numbers, and the whitespace between elements is not text"


def test_hwpx_reads_the_paragraphs_a_header_footnote_or_caption_holds_as_their_own_segments() -> None:
    section = hwpx_section(
        '<hp:p><hp:run><hp:ctrl><hp:header><hp:subList><hp:p><hp:run><hp:t>Running head</hp:t></hp:run></hp:p></hp:subList></hp:header></hp:ctrl>'
        '<hp:t>Body text</hp:t>'
        '<hp:ctrl><hp:footNote><hp:subList><hp:p><hp:run><hp:t>A note</hp:t></hp:run></hp:p></hp:subList></hp:footNote></hp:ctrl>'
        '<hp:rect><hp:drawText><hp:subList><hp:p><hp:run><hp:t>In a box</hp:t></hp:run></hp:p></hp:subList></hp:drawText></hp:rect></hp:run></hp:p>',
    )
    result = parse_office(hwpx_package({'Contents/section0.xml': section}), 'controls.hwpx', DocumentLimits())
    assert [(item.locator, item.text, item.metadata.get('content_role')) for item in result.segments] == [
        ('section:1/paragraph:1', 'Body text', None),
        ('section:1/paragraph:1/header:1/paragraph:1', 'Running head', 'header'),
        ('section:1/paragraph:1/footnote:1/paragraph:1', 'A note', 'footnote'),
        ('section:1/paragraph:1/textbox:1/paragraph:1', 'In a box', 'textbox'),
    ]
    assert all(item.metadata['parent_locator'] == 'section:1/paragraph:1' for item in result.segments[1:])
    only_header = hwpx_section('<hp:p><hp:run><hp:ctrl><hp:header><hp:subList><hp:p><hp:run><hp:t>Only a head</hp:t></hp:run></hp:p></hp:subList></hp:header></hp:ctrl></hp:run></hp:p>')
    assert [item.text for item in parse_office(hwpx_package({'Contents/section0.xml': only_header}), 'h.hwpx', DocumentLimits()).segments] == ['Only a head']


def test_hwpx_refuses_a_spine_that_names_what_the_package_lacks_and_a_section_without_text() -> None:
    dangling = hwpx_package({'Contents/section0.xml': hwpx_section(hwpx_p('<hp:t>x</hp:t>'))}, spine=['section0', 'section9'])
    with pytest.raises(InvalidInput, match='missing manifest item|section the package does not hold'):
        parse_office(dangling, 'dangling.hwpx', DocumentLimits())
    empty = hwpx_package({'Contents/section0.xml': hwpx_section('<hp:p><hp:run><hp:secPr/></hp:run></hp:p>')}, spine=['section0'])
    with pytest.raises(InvalidInput, match='no extractable text'):
        parse_office(empty, 'empty.hwpx', DocumentLimits())


def test_hwpx_without_a_package_manifest_reads_sections_by_number_and_refuses_a_package_without_any() -> None:
    data = hwpx_package({'Contents/section10.xml': hwpx_section(hwpx_p('<hp:t>eleventh</hp:t>')),
                         'Contents/section2.xml': hwpx_section(hwpx_p('<hp:t>third</hp:t>')),
                         'Contents/header.xml': f'<hh:head xmlns:hh="{HS}"/>'})
    result = parse_office(data, 'plain.hwpx', DocumentLimits())
    assert [item.text for item in result.segments] == ['third', 'eleventh'], "numeric order, not lexical"
    with pytest.raises(InvalidInput, match='HWPX package has no sections'):
        parse_office(archive({'Contents/header.xml': f'<hh:head xmlns:hh="{HS}"/>'}), 'empty.hwpx', DocumentLimits())


async def test_an_hwpx_document_is_ingested_end_to_end_and_the_listings_name_it() -> None:
    from scone_memory import HashEmbedder, InMemoryDocumentStore, InMemoryVectorIndex, MemoryEngine
    from scone_memory.ingestion.files import FILE_MEDIA_TYPES, ingest_document
    from scone_memory.ingestion.formats.capabilities import document_formats
    from scone_memory.ingestion.source_scan import default_extensions
    from scone_memory.memory.engine import ATTACHMENT_TYPES

    assert '.hwpx' in document_formats() and '.hwpx' in default_extensions() and FILE_MEDIA_TYPES['.hwpx'] in ATTACHMENT_TYPES
    memory = await MemoryEngine(InMemoryDocumentStore(), InMemoryVectorIndex(), HashEmbedder()).open()
    try:
        data = hwpx_package({'Contents/section0.xml': hwpx_section(hwpx_p('<hp:t>항구는 11월에 닫힌다</hp:t>'))}, spine=['section0'])
        ingested = await ingest_document(memory, 'docs', data, filename='harbour.hwpx')
        assert ingested.format == 'hwpx' and ingested.original.media_type == 'application/hwp+zip'
        assert '항구는 11월에 닫힌다' in (await memory.episode('docs', ingested.added.episode_id)).content
    finally:
        await memory.close()
