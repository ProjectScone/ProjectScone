"""Referenced Word parts retain their role and their source-part identity."""
import pytest

from scone_memory.core.errors import InvalidInput
from scone_memory.ingestion import BuiltinDocumentParser
from .test_office_formats import W, R, REL, ooxml_archive


def document(body, *, relationships='', extra=None, namespace=W):
    members = {
        'content/main.xml': f'<w:document xmlns:w="{namespace}" xmlns:r="{R}"><w:body>{body}</w:body></w:document>',
        'content/_rels/main.xml.rels': f'<Relationships xmlns="{REL}">{relationships}</Relationships>',
        **(extra or {}),
    }
    return ooxml_archive(members, main_part='content/main.xml')


def relation(kind, target):
    return f'<Relationship Id="{kind}" Target="{target}" Type="{R}/{kind}"/>'


@pytest.mark.parametrize('kind', ['footnote', 'endnote'])
async def test_docx_referenced_notes_have_separate_locations_and_part_metadata(kind):
    data = document(f'''<w:p><w:r><w:t>The estimate is provisional.</w:t><w:{kind}Reference w:id="2"/></w:r></w:p>
        <w:p><w:r><w:t>Next paragraph.</w:t></w:r></w:p>''',
        relationships=relation(kind + 's', '../notes/notes.xml'), extra={
        'notes/notes.xml': f'''<w:{kind}s xmlns:w="{W}"><w:{kind} w:id="-1" w:type="separator"><w:p><w:r><w:t>Separator</w:t></w:r></w:p></w:{kind}>
        <w:{kind} w:id="2"><w:p><w:r><w:t>Source is unaudited.</w:t></w:r></w:p><w:p><w:r><w:t>Check before release.</w:t></w:r></w:p></w:{kind}>
        <w:{kind} w:id="3"><w:p><w:r><w:t>Unreferenced old note</w:t></w:r></w:p></w:{kind}></w:{kind}s>''',
    })
    parsed = await BuiltinDocumentParser().parse(data, 'report.docx')
    assert [(s.locator, s.text) for s in parsed.segments] == [
        ('paragraph:1', 'The estimate is provisional.'), ('paragraph:2', 'Next paragraph.'),
        (f'paragraph:1/{kind}:2/paragraph:1', 'Source is unaudited.'),
        (f'paragraph:1/{kind}:2/paragraph:2', 'Check before release.'),
    ]
    assert parsed.segments[2].metadata == {
        'member': 'notes/notes.xml', 'content_role': kind, 'parent_locator': 'paragraph:1', 'note_id': '2',
    }


async def test_docx_comment_ranges_emit_once_with_author_date_and_first_reference():
    data = document('''<w:p><w:commentRangeStart w:id="7"/><w:r><w:t>First statement.</w:t></w:r></w:p>
        <w:p><w:r><w:t>Second statement.</w:t></w:r><w:commentRangeEnd w:id="7"/><w:r><w:commentReference w:id="7"/></w:r></w:p>''',
        relationships=relation('comments', '../annotations/comments.xml'), extra={
        'annotations/comments.xml': f'''<w:comments xmlns:w="{W}"><w:comment w:id="7" w:author="Alice" w:date="2026-09-11T10:00:00Z" w:initials="AL">
        <w:p><w:r><w:t>Verify both statements.</w:t></w:r></w:p></w:comment><w:comment w:id="8"><w:p><w:r><w:t>Orphan comment</w:t></w:r></w:p></w:comment></w:comments>''',
    })
    parsed = await BuiltinDocumentParser().parse(data, 'report.docx')
    assert [s.text for s in parsed.segments] == ['First statement.', 'Second statement.', 'Verify both statements.']
    assert parsed.segments[2].locator == 'paragraph:1/comment:7/paragraph:1'
    assert parsed.segments[2].metadata == {
        'member': 'annotations/comments.xml', 'content_role': 'comment', 'parent_locator': 'paragraph:1',
        'comment_id': '7', 'author': 'Alice', 'date': '2026-09-11T10:00:00Z', 'initials': 'AL',
    }


async def test_docx_table_note_reference_and_note_table_have_precise_row_locations():
    data = document('''<w:tbl><w:tr><w:tc><w:p><w:r><w:t>Budget</w:t><w:footnoteReference w:id="1"/></w:r></w:p></w:tc></w:tr></w:tbl>''',
        relationships=relation('footnotes', '../footnotes.xml'), extra={
        'footnotes.xml': f'''<w:footnotes xmlns:w="{W}"><w:footnote w:id="1"><w:tbl><w:tr><w:tc><w:p><w:r><w:t>Estimate</w:t></w:r></w:p></w:tc>
        <w:tc><w:p><w:r><w:t>42</w:t></w:r></w:p></w:tc></w:tr></w:tbl></w:footnote></w:footnotes>''',
    })
    parsed = await BuiltinDocumentParser().parse(data, 'report.docx')
    assert [(s.locator, s.text) for s in parsed.segments] == [
        ('table:1/row:1', 'Budget'), ('table:1/row:1/footnote:1/table:1/row:1', 'Estimate\t42'),
    ]


async def test_docx_deleted_reference_does_not_extract_an_orphan_part():
    data = document('''<w:del><w:p><w:r><w:footnoteReference w:id="9"/></w:r></w:p></w:del>
        <w:p><w:r><w:t>Current statement.</w:t></w:r></w:p>''')
    parsed = await BuiltinDocumentParser().parse(data, 'report.docx')
    assert [s.text for s in parsed.segments] == ['Current statement.']


@pytest.mark.parametrize('failure', ['relationship', 'member', 'note', 'duplicate'])
async def test_docx_dangling_or_ambiguous_note_references_are_rejected(failure):
    notes = f'<w:footnotes xmlns:w="{W}"><w:footnote w:id="2"><w:p><w:r><w:t>Note</w:t></w:r></w:p></w:footnote></w:footnotes>'
    if failure == 'note':
        notes = notes.replace('w:id="2"', 'w:id="3"')
    if failure == 'duplicate':
        notes = notes.replace('</w:footnotes>', '<w:footnote w:id="02"/></w:footnotes>')
    data = document('<w:p><w:r><w:t>Statement.</w:t><w:footnoteReference w:id="2"/></w:r></w:p>',
        relationships='' if failure == 'relationship' else relation('footnotes', '../footnotes.xml'),
        extra={} if failure == 'member' else {'footnotes.xml': notes})
    with pytest.raises(InvalidInput):
        await BuiltinDocumentParser().parse(data, 'report.docx')


@pytest.mark.parametrize('relationship', [
    relation('footnotes', '../footnotes.xml#'), relation('footnotes', '../footnotes.xml?'),
    relation('footnotes', '../foot&#xA;notes.xml'),
    relation('footnotes', '../footnotes.xml').replace(R + '/footnotes', 'urn:unrelated/footnotes'),
])
async def test_docx_note_relationships_cannot_be_repaired_into_another_target(relationship):
    data = document('<w:p><w:r><w:t>Statement.</w:t><w:footnoteReference w:id="1"/></w:r></w:p>',
        relationships=relationship, extra={
        'footnotes.xml': f'<w:footnotes xmlns:w="{W}"><w:footnote w:id="1"><w:p><w:r><w:t>Note</w:t></w:r></w:p></w:footnote></w:footnotes>',
    })
    with pytest.raises(InvalidInput):
        await BuiltinDocumentParser().parse(data, 'report.docx')


async def test_docx_empty_note_chain_still_bounds_source_location_growth():
    notes = ''.join(f'<w:footnote w:id="{number}"><w:p><w:r><w:footnoteReference w:id="{number + 1}"/></w:r></w:p></w:footnote>'
                    for number in range(1, 1000)) + '<w:footnote w:id="1000"/>'
    data = document('<w:p><w:r><w:t>Statement.</w:t><w:footnoteReference w:id="1"/></w:r></w:p>',
        relationships=relation('footnotes', '../footnotes.xml'),
        extra={'footnotes.xml': f'<w:footnotes xmlns:w="{W}">{notes}</w:footnotes>'})
    with pytest.raises(InvalidInput, match='source locator'):
        await BuiltinDocumentParser().parse(data, 'report.docx')


async def test_docx_strict_notes_apply_current_text_filtering():
    strict = 'http://purl.oclc.org/ooxml/wordprocessingml/main'
    data = document('<w:p><w:r><w:t>Statement.</w:t><w:footnoteReference w:id="1"/></w:r></w:p>', namespace=strict,
        relationships=relation('footnotes', '../footnotes.xml').replace(R, 'http://purl.oclc.org/ooxml/officeDocument/relationships'),
        extra={'footnotes.xml': f'''<w:footnotes xmlns:w="{strict}"><w:footnote w:id="1"><w:p>
        <w:del><w:r><w:t>Old claim</w:t></w:r></w:del><w:r><w:rPr><w:vanish/></w:rPr><w:t>Hidden</w:t></w:r>
        <w:ins><w:r><w:t>Current note.</w:t></w:r></w:ins></w:p></w:footnote></w:footnotes>'''})
    parsed = await BuiltinDocumentParser().parse(data, 'report.docx')
    assert [s.text for s in parsed.segments] == ['Statement.', 'Current note.']


async def test_docx_reference_cycles_do_not_duplicate_notes():
    data = document('<w:p><w:r><w:t>Statement.</w:t><w:footnoteReference w:id="1"/></w:r></w:p>',
        relationships=relation('footnotes', '../footnotes.xml'), extra={
        'footnotes.xml': f'<w:footnotes xmlns:w="{W}"><w:footnote w:id="1"><w:p><w:r><w:t>Note.</w:t><w:footnoteReference w:id="1"/></w:r></w:p></w:footnote></w:footnotes>',
    })
    parsed = await BuiltinDocumentParser().parse(data, 'report.docx')
    assert [s.text for s in parsed.segments] == ['Statement.', 'Note.']
