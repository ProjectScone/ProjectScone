"""Word drawing alternatives represent one text box, with its own evidence."""
import pytest

from scone_memory.core.errors import InvalidInput
from scone_memory.ingestion import BuiltinDocumentParser
from .test_docx_parts import document, relation
from .test_office_formats import W

MC = 'http://schemas.openxmlformats.org/markup-compatibility/2006'
SHAPE = 'http://schemas.microsoft.com/office/word/2010/wordprocessingShape'


def box(text):
    return f'<w:txbxContent><w:p><w:r><w:t>{text}</w:t></w:r></w:p></w:txbxContent>'


def alternatives(*, prefix='shape', namespace=SHAPE, requires=None, fallback=True):
    return (f'<mc:AlternateContent xmlns:mc="{MC}" xmlns:{prefix}="{namespace}">'
            f'<mc:Choice Requires="{requires or prefix}"><{prefix}:txbx>{box("Box text")}</{prefix}:txbx></mc:Choice>'
            + (f'<mc:Fallback><w:pict>{box("Fallback box")}</w:pict></mc:Fallback>' if fallback else '')
            + '</mc:AlternateContent>')


@pytest.mark.parametrize('prefix', ['wps', 'renamed'])
async def test_docx_selects_one_textbox_alternative_and_separates_anchor(prefix):
    data = document(f'<w:p><w:r><w:t>Anchor.</w:t>{alternatives(prefix=prefix)}</w:r></w:p>')
    parsed = await BuiltinDocumentParser().parse(data, 'boxes.docx')
    assert [(s.locator, s.text) for s in parsed.segments] == [
        ('paragraph:1', 'Anchor.'), ('paragraph:1/textbox:1/paragraph:1', 'Box text'),
    ]
    assert parsed.segments[1].metadata == {
        'member': 'content/main.xml', 'content_role': 'textbox', 'parent_locator': 'paragraph:1',
    }


async def test_docx_unknown_requirement_uses_fallback_and_respects_prefix_shadowing():
    data = document(f'<w:p xmlns:shape="{SHAPE}"><w:r><w:t>Anchor.</w:t>{alternatives(namespace="urn:unsupported")}</w:r></w:p>')
    parsed = await BuiltinDocumentParser().parse(data, 'boxes.docx')
    assert [s.text for s in parsed.segments] == ['Anchor.', 'Fallback box']


async def test_docx_first_supported_choice_wins_without_duplicate_fallback():
    content = f'<mc:AlternateContent xmlns:mc="{MC}" xmlns:s="{SHAPE}" xmlns:u="urn:unknown">'
    content += f'<mc:Choice Requires="s u">{box("Unsupported")}</mc:Choice>'
    content += f'<mc:Choice Requires="s">{box("Chosen")}</mc:Choice><mc:Choice Requires="s">{box("Later")}</mc:Choice>'
    content += f'<mc:Fallback>{box("Fallback")}</mc:Fallback></mc:AlternateContent>'
    parsed = await BuiltinDocumentParser().parse(document(f'<w:p><w:r><w:t>Anchor.</w:t>{content}</w:r></w:p>'), 'boxes.docx')
    assert [s.text for s in parsed.segments] == ['Anchor.', 'Chosen']


@pytest.mark.parametrize('content', [alternatives(namespace='urn:unsupported', fallback=False),
                                     alternatives(requires='undeclared')])
async def test_docx_unreadable_or_malformed_alternatives_are_explicit_errors(content):
    with pytest.raises(InvalidInput, match='alternate|namespace'):
        await BuiltinDocumentParser().parse(document(f'<w:p><w:r><w:t>Anchor.</w:t>{content}</w:r></w:p>'), 'boxes.docx')


async def test_docx_textbox_in_table_has_own_rows_and_reference_parent():
    textbox = '<w:txbxContent><w:p><w:r><w:t>Sidebar.</w:t><w:footnoteReference w:id="1"/></w:r></w:p>'
    textbox += '<w:tbl><w:tr><w:tc><w:p><w:r><w:t>Cell</w:t></w:r></w:p></w:tc></w:tr></w:tbl></w:txbxContent>'
    data = document(f'<w:tbl><w:tr><w:tc><w:p><w:r><w:t>Anchor.</w:t>{textbox}</w:r></w:p></w:tc></w:tr></w:tbl>',
        relationships=relation('footnotes', '../notes.xml'), extra={
            'notes.xml': f'<w:footnotes xmlns:w="{W}"><w:footnote w:id="1"><w:p><w:r><w:t>Source.</w:t></w:r></w:p></w:footnote></w:footnotes>',
        })
    parsed = await BuiltinDocumentParser().parse(data, 'boxes.docx')
    assert [(s.locator, s.text) for s in parsed.segments] == [
        ('table:1/row:1', 'Anchor.'),
        ('table:1/row:1/textbox:1/paragraph:1', 'Sidebar.'),
        ('table:1/row:1/textbox:1/table:1/row:1', 'Cell'),
        ('table:1/row:1/textbox:1/paragraph:1/footnote:1/paragraph:1', 'Source.'),
    ]
    assert parsed.segments[-1].metadata['parent_locator'] == 'table:1/row:1/textbox:1/paragraph:1'


async def test_docx_multiple_and_nested_boxes_follow_source_order():
    inner = f'<w:txbxContent><w:p><w:r><w:t>First.</w:t>{box("Nested.")}</w:r></w:p></w:txbxContent>'
    data = document(f'<w:p><w:r><w:t>Anchor.</w:t><w:pict>{inner}</w:pict>{box("Second.")}</w:r></w:p>')
    parsed = await BuiltinDocumentParser().parse(data, 'boxes.docx')
    assert [(s.locator, s.text) for s in parsed.segments] == [
        ('paragraph:1', 'Anchor.'), ('paragraph:1/textbox:1/paragraph:1', 'First.'),
        ('paragraph:1/textbox:2/paragraph:1', 'Second.'),
        ('paragraph:1/textbox:1/paragraph:1/textbox:1/paragraph:1', 'Nested.'),
    ]


async def test_docx_namespace_shadow_ends_before_next_choice():
    content = alternatives(namespace='urn:unknown') + alternatives()
    parsed = await BuiltinDocumentParser().parse(document(f'<w:p><w:r><w:t>Anchor.</w:t>{content}</w:r></w:p>'), 'boxes.docx')
    assert [s.text for s in parsed.segments] == ['Anchor.', 'Fallback box', 'Box text']


async def test_docx_deleted_textboxes_and_unselected_dangling_references_are_ignored():
    content = alternatives().replace('Fallback box', 'Unused</w:t><w:footnoteReference w:id="99"/><w:t>')
    parsed = await BuiltinDocumentParser().parse(document(f'<w:del><w:p><w:r>{box("Deleted")}</w:r></w:p></w:del><w:p><w:r><w:t>Anchor.</w:t>{content}</w:r></w:p>'), 'boxes.docx')
    assert [s.text for s in parsed.segments] == ['Anchor.', 'Box text']


def test_docx_discarded_alternatives_still_charge_xml_node_budget(monkeypatch):
    from scone_memory.ingestion.formats import bounded_xml
    from scone_memory.ingestion.formats.office import parse_office
    from scone_memory.ingestion.formats.types import DocumentLimits

    content = alternatives().replace('Fallback box', '</w:t>' + '<w:t>unused</w:t>' * 80 + '<w:t>')
    monkeypatch.setattr(bounded_xml, 'XML_MAX_NODES', 60)
    with pytest.raises(InvalidInput, match='node or depth'):
        parse_office(document(f'<w:p><w:r><w:t>Anchor.</w:t>{content}</w:r></w:p>'), 'boxes.docx', DocumentLimits())


async def test_docx_unused_fallback_does_not_require_support_for_nested_alternatives():
    nested = alternatives(namespace='urn:unsupported', fallback=False)
    content = alternatives().replace('Fallback box', f'</w:t>{nested}<w:t>')
    parsed = await BuiltinDocumentParser().parse(document(f'<w:p><w:r><w:t>Anchor.</w:t>{content}</w:r></w:p>'), 'boxes.docx')
    assert [s.text for s in parsed.segments] == ['Anchor.', 'Box text']
