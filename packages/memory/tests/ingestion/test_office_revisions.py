"""Current document text must not inherit deleted revision-history content."""
import pytest

from scone_memory.ingestion import BuiltinDocumentParser
from .test_office_formats import O, T, TABLE, W, S, R, REL, archive


def odt(body):
    return archive({'content.xml': f'<office:document-content xmlns:office="{O}" xmlns:text="{T}" xmlns:table="{TABLE}"><office:body><office:text>{body}</office:text></office:body></office:document-content>'})


async def test_odt_deleted_revision_does_not_become_current_text_or_shift_locators():
    data = odt('''<text:tracked-changes><text:changed-region text:id="removed">
        <text:deletion><office:change-info><text:p>Old reviewer comment</text:p></office:change-info>
        <text:p>The contract value is 5 million.</text:p>
        <table:table><table:table-row><table:table-cell><text:p>Deleted table</text:p></table:table-cell></table:table-row></table:table>
        </text:deletion></text:changed-region>
        <text:changed-region text:id="added"><text:insertion><office:change-info><text:p>Insertion comment</text:p></office:change-info></text:insertion></text:changed-region>
        </text:tracked-changes>
        <text:p>The contract value is <text:change-start text:change-id="added"/>2 million<text:change-end text:change-id="added"/>.</text:p>
        <text:p>Next live paragraph.</text:p>
        <table:table><table:table-row><table:table-cell><text:p>Current table</text:p></table:table-cell></table:table-row></table:table>''')
    parsed = await BuiltinDocumentParser().parse(data, 'contract.odt')
    assert [(segment.locator, segment.text) for segment in parsed.segments] == [
        ('paragraph:1', 'The contract value is 2 million.'),
        ('paragraph:2', 'Next live paragraph.'),
        ('table:1/row:1', 'Current table'),
    ]


async def test_odt_inline_revision_metadata_is_omitted_without_losing_following_text():
    data = odt('<text:p>Current <office:change-info><text:p>Revision metadata</text:p></office:change-info>claim.</text:p>')
    parsed = await BuiltinDocumentParser().parse(data, 'contract.odt')
    assert [segment.text for segment in parsed.segments] == ['Current claim.']


async def test_odt_unrelated_namespace_is_not_treated_as_revision_history():
    data = odt('<text:p>Current <x:tracked-changes xmlns:x="urn:example"><text:span>visible extension</text:span></x:tracked-changes> text.</text:p>')
    parsed = await BuiltinDocumentParser().parse(data, 'contract.odt')
    assert [segment.text for segment in parsed.segments] == ['Current visible extension text.']


async def test_docx_tracks_current_moves_and_omits_hidden_runs_and_properties():
    data = archive({'word/document.xml': f'''<w:document xmlns:w="{W}"><w:body>
        <w:moveFrom><w:p><w:r><w:t>Old moved paragraph</w:t></w:r></w:p></w:moveFrom>
        <w:p><w:pPr><w:tabs><w:tab w:pos="720"/></w:tabs></w:pPr>
        <w:r><w:t>Current </w:t></w:r><w:moveFrom><w:r><w:t>old location </w:t></w:r></w:moveFrom>
        <w:del><w:r><w:t>deleted content </w:t></w:r></w:del>
        <w:moveTo><w:r><w:t>moved claim</w:t></w:r></w:moveTo>
        <w:r><w:rPr><w:vanish/></w:rPr><w:t>hidden instruction</w:t></w:r>
        <w:r><w:rPr><w:vanish w:val="0"/></w:rPr><w:t> visible.</w:t></w:r></w:p>
        </w:body></w:document>'''})
    parsed = await BuiltinDocumentParser().parse(data, 'source.docx')
    assert [(s.locator, s.text) for s in parsed.segments] == [('paragraph:1', 'Current moved claim visible.')]


async def test_docx_keeps_nonbreaking_hyphens_position_tabs_and_ruby_base_text():
    data = archive({'word/document.xml': f'''<w:document xmlns:w="{W}"><w:body><w:p>
        <w:r><w:t>non</w:t><w:noBreakHyphen/><w:t>breaking</w:t><w:ptab/>
        <w:ruby><w:rt><w:r><w:t>phonetic</w:t></w:r></w:rt>
        <w:rubyBase><w:r><w:t>base</w:t></w:r></w:rubyBase></w:ruby></w:r>
        </w:p></w:body></w:document>'''})
    parsed = await BuiltinDocumentParser().parse(data, 'source.docx')
    assert parsed.segments[0].text == 'non\u2011breaking\tbase'


async def test_xlsx_phonetic_hints_do_not_change_cell_values():
    data = archive({
        'xl/workbook.xml': f'<workbook xmlns="{S}" xmlns:r="{R}"><sheets><sheet name="Words" r:id="s"/></sheets></workbook>',
        'xl/_rels/workbook.xml.rels': f'<Relationships xmlns="{REL}"><Relationship Id="s" Target="worksheets/s.xml" Type="{R}/worksheet"/></Relationships>',
        'xl/worksheets/s.xml': f'<worksheet xmlns="{S}"><sheetData><row r="1"><c r="A1" t="inlineStr"><is><t>東京</t><rPh><t>とうきょう</t></rPh></is></c></row></sheetData></worksheet>',
    })
    parsed = await BuiltinDocumentParser().parse(data, 'source.xlsx')
    assert parsed.segments[0].text == '東京'


@pytest.mark.parametrize('namespace', [W, 'http://purl.oclc.org/ooxml/wordprocessingml/main'], ids=['transitional', 'strict'])
@pytest.mark.parametrize('visible', ['0', 'false', 'off'])
async def test_docx_table_revision_filter_keeps_explicitly_visible_runs(namespace, visible):
    data = archive({'word/document.xml': f'''<w:document xmlns:w="{namespace}"><w:body>
        <w:del><w:tbl><w:tr><w:tc><w:p><w:r><w:t>Old table</w:t></w:r></w:p></w:tc></w:tr></w:tbl></w:del>
        <w:tbl><w:tr><w:tc><w:p><w:r><w:t>Current </w:t></w:r>
        <w:r><w:rPr><w:vanish w:val="true"/></w:rPr><w:t>hidden</w:t></w:r>
        <w:ins><w:r><w:rPr><w:vanish w:val="{visible}"/></w:rPr><w:t>insertion</w:t></w:r></w:ins>
        </w:p></w:tc></w:tr></w:tbl></w:body></w:document>'''})
    parsed = await BuiltinDocumentParser().parse(data, 'source.docx')
    assert [(s.locator, s.text) for s in parsed.segments] == [('table:1/row:1', 'Current insertion')]


async def test_docx_revision_names_in_other_namespaces_do_not_hide_content():
    data = archive({'word/document.xml': f'''<w:document xmlns:w="{W}" xmlns:x="urn:example"><w:body><w:p>
        <x:del><w:r><w:t>Visible </w:t></w:r></x:del>
        <w:r><w:rPr><x:vanish/></w:rPr><w:t>extension</w:t></w:r>
        </w:p></w:body></w:document>'''})
    parsed = await BuiltinDocumentParser().parse(data, 'source.docx')
    assert parsed.segments[0].text == 'Visible extension'


async def test_xlsx_shared_rich_text_preserves_base_text_without_pronunciation():
    data = archive({
        'xl/workbook.xml': f'<workbook xmlns="{S}" xmlns:r="{R}"><sheets><sheet name="Words" r:id="s"/></sheets></workbook>',
        'xl/_rels/workbook.xml.rels': f'<Relationships xmlns="{REL}"><Relationship Id="s" Target="worksheets/s.xml" Type="{R}/worksheet"/><Relationship Id="strings" Target="sharedStrings.xml" Type="{R}/sharedStrings"/></Relationships>',
        'xl/sharedStrings.xml': f'<sst xmlns="{S}"><si><r><t>東京</t></r><r><t>駅</t></r><rPh sb="0" eb="2"><t>とうきょう</t></rPh></si></sst>',
        'xl/worksheets/s.xml': f'<worksheet xmlns="{S}"><sheetData><row r="1"><c r="A1" t="s"><v>0</v></c></row></sheetData></worksheet>',
    })
    parsed = await BuiltinDocumentParser().parse(data, 'source.xlsx')
    assert parsed.segments[0].text == '東京駅'
