"""Comments and notes retain provenance separately from document assertions."""
import pytest

from scone_memory.core.errors import InvalidInput
from scone_memory.ingestion import BuiltinDocumentParser
from scone_memory.ingestion.formats.types import DocumentLimits
from .test_office_formats import O, T, TABLE, archive
from .test_office_revisions import odt

DC = 'http://purl.org/dc/elements/1.1/'


async def test_odt_comment_keeps_author_and_date_out_of_body_text():
    data = odt(f'''<text:p>Revenue grew<office:annotation office:name="review-1" xmlns:dc="{DC}">
        <dc:creator>Alice</dc:creator><dc:date>2026-02-03T10:00:00</dc:date>
        <text:p>Verify this</text:p><text:p>Check the source.</text:p></office:annotation> 4%.</text:p>''')
    parsed = await BuiltinDocumentParser().parse(data, 'report.odt')
    assert [(s.locator, s.text) for s in parsed.segments] == [
        ('paragraph:1', 'Revenue grew 4%.'),
        ('paragraph:1/comment:1', 'Verify this\nCheck the source.'),
    ]
    assert parsed.segments[1].metadata == {
        'content_role': 'comment', 'parent_locator': 'paragraph:1',
        'comment_name': 'review-1', 'author': 'Alice', 'date': '2026-02-03T10:00:00',
    }


@pytest.mark.parametrize('kind', ['footnote', 'endnote'])
async def test_odt_note_keeps_body_and_reference_separate(kind):
    data = odt(f'''<text:p>Rates rose<text:note text:id="note-a" text:note-class="{kind}">
        <text:note-citation>1</text:note-citation><text:note-body><text:p>Source: central bank.</text:p>
        </text:note-body></text:note> last year.</text:p><text:p>Next paragraph.</text:p>''')
    parsed = await BuiltinDocumentParser().parse(data, 'report.odt')
    assert [(s.locator, s.text) for s in parsed.segments] == [
        ('paragraph:1', 'Rates rose last year.'),
        (f'paragraph:1/{kind}:1', 'Source: central bank.'),
        ('paragraph:2', 'Next paragraph.'),
    ]
    assert parsed.segments[1].metadata == {
        'content_role': kind, 'parent_locator': 'paragraph:1', 'note_id': 'note-a', 'citation': '1',
    }


async def test_ods_comment_is_not_a_cell_value():
    data = archive({'content.xml': f'''<office:document-content xmlns:office="{O}" xmlns:text="{T}" xmlns:table="{TABLE}">
        <office:body><office:spreadsheet><table:table table:name="Budget"><table:table-row>
        <table:table-cell office:value-type="float" office:value="100"><office:annotation>
        <text:p>Estimate only</text:p></office:annotation><text:p>100</text:p></table:table-cell>
        </table:table-row></table:table></office:spreadsheet></office:body></office:document-content>'''})
    parsed = await BuiltinDocumentParser().parse(data, 'report.ods')
    assert [(s.locator, s.text) for s in parsed.segments] == [
        ('sheet:Budget/cell:A1', '100'),
        ('sheet:Budget/cell:A1/comment:1', 'Estimate only'),
    ]
    assert parsed.segments[1].metadata['content_role'] == 'comment'
    assert parsed.segments[1].metadata['cell'] == 'A1'


async def test_odt_body_comments_do_not_renumber_paragraphs():
    data = odt('''<office:annotation><text:p>First comment</text:p></office:annotation>
        <text:p>First assertion.</text:p><office:annotation><text:p>Second comment</text:p></office:annotation>
        <text:p>Second assertion.</text:p>''')
    parsed = await BuiltinDocumentParser().parse(data, 'report.odt')
    assert [(s.locator, s.text) for s in parsed.segments] == [
        ('body/comment:1', 'First comment'), ('paragraph:1', 'First assertion.'),
        ('body/comment:2', 'Second comment'), ('paragraph:2', 'Second assertion.'),
    ]


async def test_odt_table_comment_is_separate_from_row_text():
    data = odt('''<table:table><table:table-row><table:table-cell><text:p>Approved<office:annotation>
        <text:p>Needs review</text:p></office:annotation>.</text:p></table:table-cell>
        <table:table-cell><text:p>10</text:p></table:table-cell></table:table-row></table:table>''')
    parsed = await BuiltinDocumentParser().parse(data, 'report.odt')
    assert [(s.locator, s.text) for s in parsed.segments] == [
        ('table:1/row:1', 'Approved.\t10'), ('table:1/row:1/comment:1', 'Needs review'),
    ]


async def test_odt_comment_segments_count_toward_output_limits():
    data = odt('<text:p>Main<office:annotation><text:p>Comment</text:p></office:annotation></text:p>')
    with pytest.raises(InvalidInput, match='segment limit'):
        await BuiltinDocumentParser().parse(data, 'report.odt', limits=DocumentLimits(max_segments=1))


async def test_nested_comment_does_not_inherit_parent_author_or_note_reference():
    data = odt(f'''<text:p>Main<text:note text:id="note-1" text:note-class="footnote">
        <text:note-citation>1</text:note-citation><text:note-body><text:p>Source<office:annotation xmlns:dc="{DC}">
        <dc:creator>Alice</dc:creator><text:p>Check<office:annotation><text:p>Nested</text:p></office:annotation>.</text:p>
        </office:annotation>.</text:p></text:note-body></text:note>.</text:p>''')
    parsed = await BuiltinDocumentParser().parse(data, 'report.odt')
    assert [s.text for s in parsed.segments] == ['Main.', 'Source.', 'Check.', 'Nested']
    assert parsed.segments[2].metadata == {
        'content_role': 'comment', 'parent_locator': 'paragraph:1/footnote:1', 'author': 'Alice',
    }
    assert parsed.segments[3].metadata == {
        'content_role': 'comment', 'parent_locator': 'paragraph:1/footnote:1/comment:1',
    }


async def test_unrelated_annotation_namespace_remains_inline():
    data = odt('<text:p>Current <x:annotation xmlns:x="urn:example">extension</x:annotation>.</text:p>')
    parsed = await BuiltinDocumentParser().parse(data, 'report.odt')
    assert [(s.locator, s.text) for s in parsed.segments] == [('paragraph:1', 'Current extension.')]


async def test_odt_annotations_in_deleted_history_are_not_indexed():
    data = odt('''<text:tracked-changes><text:changed-region><text:deletion><text:p>Old<office:annotation>
        <text:p>Obsolete comment</text:p></office:annotation></text:p></text:deletion></text:changed-region>
        </text:tracked-changes><text:p>Current</text:p>''')
    parsed = await BuiltinDocumentParser().parse(data, 'report.odt')
    assert [(s.locator, s.text) for s in parsed.segments] == [('paragraph:1', 'Current')]
