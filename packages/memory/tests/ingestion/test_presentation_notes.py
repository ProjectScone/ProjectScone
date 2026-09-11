"""Speaker notes are searchable evidence with their own source locations."""
import pytest

from scone_memory.core.errors import InvalidInput
from scone_memory.ingestion import BuiltinDocumentParser
from scone_memory.ingestion.formats.types import DocumentLimits
from .test_office_formats import O, T, TABLE, D, archive

PRESENTATION = 'urn:oasis:names:tc:opendocument:xmlns:presentation:1.0'


def odp(pages):
    return archive({'content.xml': f'''<office:document-content xmlns:office="{O}" xmlns:text="{T}"
        xmlns:table="{TABLE}" xmlns:draw="{D}" xmlns:presentation="{PRESENTATION}">
        <office:body><office:presentation>{pages}</office:presentation></office:body></office:document-content>'''})


async def test_odp_notes_do_not_become_visible_slide_paragraphs():
    data = odp('''<draw:page draw:name="Intro"><presentation:notes><draw:frame><draw:text-box>
        <text:p>Explain the uncertainty<office:annotation><text:p>Check estimate</text:p></office:annotation>.</text:p>
        </draw:text-box></draw:frame></presentation:notes><draw:frame><draw:text-box>
        <text:p>Forecast</text:p><text:p>Revenue rises</text:p></draw:text-box></draw:frame></draw:page>''')
    parsed = await BuiltinDocumentParser().parse(data, 'forecast.odp')
    assert [(s.locator, s.text) for s in parsed.segments] == [
        ('slide:1/paragraph:1', 'Forecast'), ('slide:1/paragraph:2', 'Revenue rises'),
        ('slide:1/notes/paragraph:1', 'Explain the uncertainty.'),
        ('slide:1/notes/paragraph:1/comment:1', 'Check estimate'),
    ]
    assert parsed.segments[2].metadata == {
        'slide_name': 'Intro', 'content_role': 'speaker_notes', 'parent_locator': 'slide:1',
    }
    assert parsed.segments[3].metadata['content_role'] == 'comment'
    assert parsed.segments[3].metadata['parent_locator'] == 'slide:1/notes/paragraph:1'


async def test_odp_note_only_slides_and_note_tables_keep_slide_identity():
    data = odp('''<draw:page draw:name="First"><presentation:notes><text:p>First note</text:p></presentation:notes></draw:page>
        <draw:page draw:name="Second"><presentation:notes><table:table><table:table-row>
        <table:table-cell><text:p>Cost</text:p></table:table-cell><table:table-cell><text:p>42</text:p></table:table-cell>
        </table:table-row></table:table></presentation:notes></draw:page>''')
    parsed = await BuiltinDocumentParser().parse(data, 'forecast.odp')
    assert [(s.locator, s.text) for s in parsed.segments] == [
        ('slide:1/notes/paragraph:1', 'First note'), ('slide:2/notes/table:1/row:1', 'Cost\t42'),
    ]
    assert [s.metadata['parent_locator'] for s in parsed.segments] == ['slide:1', 'slide:2']


async def test_odp_unrelated_notes_namespace_does_not_hide_slide_text():
    data = odp('<draw:page><x:notes xmlns:x="urn:example"><text:p>Visible extension</text:p></x:notes></draw:page>')
    parsed = await BuiltinDocumentParser().parse(data, 'forecast.odp')
    assert [(s.locator, s.text) for s in parsed.segments] == [('slide:1/paragraph:1', 'Visible extension')]


async def test_odp_note_content_uses_shared_text_budget():
    data = odp('<draw:page><text:p>Visible</text:p><presentation:notes><text:p>Speaker note</text:p></presentation:notes></draw:page>')
    with pytest.raises(InvalidInput, match='text byte limit'):
        await BuiltinDocumentParser().parse(data, 'forecast.odp', limits=DocumentLimits(max_text_bytes=16))
