"""Readers say which blocks the source declares as headings, list items, captions and code.

Nothing here is inferred from how text looks: a Word paragraph is a heading
because its outline level or its style says so, and an HTML line is a list
item because it sits in an ``<li>``. A document that declares none of it
parses exactly as it did before.
"""
from __future__ import annotations

from scone_memory.ingestion.formats.office import parse_office
from scone_memory.ingestion.formats.text import parse_text
from scone_memory.ingestion.formats.types import DocumentLimits
from .test_office_formats import W, R, REL, ooxml_archive

MEMBER = {'member': 'word/document.xml'}

STYLES = '''
<w:style w:type="paragraph" w:default="1" w:styleId="Normal"><w:name w:val="Normal"/></w:style>
<w:style w:type="paragraph" w:styleId="Berschrift1"><w:name w:val="heading 1"/><w:basedOn w:val="Normal"/></w:style>
<w:style w:type="paragraph" w:styleId="Chapter"><w:name w:val="Chapter"/><w:basedOn w:val="Berschrift1"/></w:style>
<w:style w:type="paragraph" w:styleId="Deep"><w:name w:val="Deep"/><w:pPr><w:outlineLvl w:val="1"/></w:pPr></w:style>
<w:style w:type="paragraph" w:styleId="Loop"><w:name w:val="Loop"/><w:basedOn w:val="Loop"/></w:style>
<w:style w:type="paragraph" w:styleId="Beschriftung"><w:name w:val="caption"/></w:style>
<w:style w:type="paragraph" w:styleId="Titel"><w:name w:val="Title"/></w:style>
<w:style w:type="paragraph" w:styleId="Heading2"><w:name w:val="Quote"/></w:style>
<w:style w:type="paragraph" w:styleId="Bulleted"><w:name w:val="Bulleted"/><w:pPr><w:numPr><w:numId w:val="1"/></w:numPr></w:pPr></w:style>
'''

NUMBERING = '''
<w:abstractNum w:abstractNumId="10"><w:lvl w:ilvl="0"><w:numFmt w:val="bullet"/></w:lvl><w:lvl w:ilvl="1"><w:numFmt w:val="bullet"/></w:lvl></w:abstractNum>
<w:abstractNum w:abstractNumId="20"><w:lvl w:ilvl="0"><w:numFmt w:val="decimal"/></w:lvl></w:abstractNum>
<w:num w:numId="1"><w:abstractNumId w:val="10"/></w:num>
<w:num w:numId="2"><w:abstractNumId w:val="20"/></w:num>
<w:num w:numId="3"><w:abstractNumId w:val="10"/><w:lvlOverride w:ilvl="0"><w:lvl w:ilvl="0"><w:numFmt w:val="lowerLetter"/></w:lvl></w:lvlOverride></w:num>
<w:num w:numId="4"><w:abstractNumId w:val="20"/></w:num>
<w:abstractNum w:abstractNumId="30"><w:lvl w:ilvl="0"><w:numFmt w:val="none"/></w:lvl></w:abstractNum>
<w:num w:numId="5"><w:abstractNumId w:val="30"/></w:num>
'''


def word(body: str, *, styles: str | None = None, numbering: str | None = None) -> bytes:
    relationships = ''
    parts: dict[str, str] = {}
    for kind, content, root in (('styles', styles, 'styles'), ('numbering', numbering, 'numbering')):
        if content is not None:
            relationships += f'<Relationship Id="{kind}" Target="{kind}.xml" Type="{R}/{kind}"/>'
            parts[f'word/{kind}.xml'] = content if content.startswith('<?') or content.startswith('<!') else (
                f'<w:{root} xmlns:w="{W}">{content}</w:{root}>')
    return ooxml_archive({
        'word/document.xml': f'<w:document xmlns:w="{W}"><w:body>{body}</w:body></w:document>',
        'word/_rels/document.xml.rels': f'<Relationships xmlns="{REL}">{relationships}</Relationships>',
        **parts,
    }, main_part='word/document.xml')


def paragraph(text: str, *, style: str | None = None, outline: int | None = None,
              number: int | None = None, level: str | None = None, extra: str = '') -> str:
    properties = f'<w:pStyle w:val="{style}"/>' if style else ''
    if number is not None:
        properties += '<w:numPr>' + (f'<w:ilvl w:val="{level}"/>' if level is not None else '')
        properties += f'<w:numId w:val="{number}"/></w:numPr>'
    if outline is not None:
        properties += f'<w:outlineLvl w:val="{outline}"/>'
    properties += extra
    return f'<w:p>{"<w:pPr>" + properties + "</w:pPr>" if properties else ""}<w:r><w:t>{text}</w:t></w:r></w:p>'


def roles(data: bytes, filename: str = 'report.docx') -> list[tuple[str, dict[str, str]]]:
    parsed = parse_office(data, filename, DocumentLimits())
    return [(segment.text, segment.metadata) for segment in parsed.segments]


def heading(level: int, basis: str) -> dict[str, str]:
    return {**MEMBER, 'block_role': 'heading', 'heading_level': str(level), 'heading_basis': basis}


def item(identifier: str, level: str = '0', kind: str | None = None) -> dict[str, str]:
    return {**MEMBER, 'block_role': 'list_item', 'list_id': identifier, 'list_level': level,
            **({'list_kind': kind} if kind else {})}


def test_word_headings_lists_and_captions_come_from_outline_levels_styles_and_numbering() -> None:
    body = ''.join([
        paragraph('Annual report', style='Titel'),
        paragraph('Overview', style='Berschrift1'),
        paragraph('Background', style='Chapter'),
        paragraph('Details', style='Deep'),
        paragraph('Aside', style='Deep', outline=9),
        paragraph('Scope', outline=2),
        paragraph('Quoted', style='Heading2'),
        paragraph('Circular', style='Loop'),
        paragraph('Figure 1: Layout', style='Beschriftung'),
        paragraph('Apples', number=1, level='0'),
        paragraph('Green', number=1, level='1'),
        paragraph('Level out of range', number=1, level='9'),
        paragraph('First', number=2),
        paragraph('Lettered', number=3),
        paragraph('Styled bullet', style='Bulleted'),
        paragraph('Unnumbered', style='Bulleted', number=0),
        paragraph('Unknown list', number=7),
        paragraph('Unmarked', number=5),
        paragraph('Plain'),
    ])
    assert roles(word(body, styles=STYLES, numbering=NUMBERING)) == [
        ('Annual report', heading(1, 'style')),
        ('Overview', heading(1, 'style')),
        ('Background', heading(1, 'style')),
        ('Details', heading(2, 'style')),
        ('Aside', MEMBER),
        ('Scope', heading(3, 'outline_level')),
        ('Quoted', MEMBER),
        ('Circular', {**MEMBER, 'heading_level_unresolved': 'style_chain'}),
        ('Figure 1: Layout', {**MEMBER, 'block_role': 'caption'}),
        ('Apples', item('1', '0', 'bullet')),
        ('Green', item('1', '1', 'bullet')),
        ('Level out of range', {**MEMBER, 'block_role': 'list_item', 'list_id': '1'}),
        ('First', item('2', '0', 'ordered')),
        ('Lettered', item('3', '0', 'ordered')),
        ('Styled bullet', item('1', '0', 'bullet')),
        ('Unnumbered', MEMBER),
        ('Unknown list', item('7')),
        ('Unmarked', item('5')),
        ('Plain', MEMBER),
    ]


def test_word_style_ids_speak_only_when_no_styles_part_names_them() -> None:
    body = ''.join([
        paragraph('Report', style='Title'),
        paragraph('Intro', style='Heading2'),
        paragraph('Table 1', style='Caption'),
        paragraph('Bullet without numbering part', number=1),
        paragraph('Centered', extra='<w:jc w:val="center"/>'),
        paragraph('Outline out of range', outline=12),
    ])
    assert parse_office(word(body), 'report.docx', DocumentLimits()).metadata == {}
    assert roles(word(body)) == [
        ('Report', heading(1, 'style_id')),
        ('Intro', heading(2, 'style_id')),
        ('Table 1', {**MEMBER, 'block_role': 'caption'}),
        ('Bullet without numbering part', item('1')),
        ('Centered', MEMBER),
        ('Outline out of range', MEMBER),
    ]


def test_an_unreadable_styles_part_is_noted_and_style_ids_still_speak() -> None:
    parsed = parse_office(word(paragraph('Intro', style='Heading1'), styles='<w:styles'), 'report.docx', DocumentLimits())
    assert parsed.segments[0].metadata == heading(1, 'style_id')
    assert parsed.metadata == {'structure_notes': 'styles_unreadable'}
    unnumbered = parse_office(word(paragraph('Item', number=1), numbering='<w:numbering'), 'report.docx', DocumentLimits())
    assert unnumbered.segments[0].metadata == item('1')
    assert unnumbered.metadata == {'structure_notes': 'numbering_unreadable'}


def test_a_textbox_keeps_the_role_its_paragraph_declared_before_run_metadata_is_pruned() -> None:
    box = f'<w:txbxContent>{paragraph("Boxed step", number=2)}</w:txbxContent>'
    body = f'<w:p><w:r><w:t>Anchor.</w:t><w:pict>{box}</w:pict></w:r></w:p>'
    [(_, anchor), (text, metadata)] = roles(word(body, numbering=NUMBERING))
    assert anchor == MEMBER
    assert text == 'Boxed step'
    assert metadata == {**item('2', '0', 'ordered'), 'content_role': 'textbox', 'parent_locator': 'paragraph:1'}


def test_a_style_chain_longer_than_its_bound_is_cut_and_noted(monkeypatch) -> None:
    from scone_memory.ingestion.formats import word_structure

    body = paragraph('Background', style='Chapter')
    assert roles(word(body, styles=STYLES)) == [('Background', heading(1, 'style'))]
    circular = parse_office(word(paragraph('Circular', style='Loop'), styles=STYLES), 'report.docx', DocumentLimits())
    assert circular.metadata == {}, 'a style based on itself ends its chain; nothing was cut'
    monkeypatch.setattr(word_structure, 'MAX_STYLE_DEPTH', 1)
    parsed = parse_office(word(body, styles=STYLES), 'report.docx', DocumentLimits())
    assert parsed.segments[0].metadata == {**MEMBER, 'heading_level_unresolved': 'style_chain'}
    assert parsed.metadata == {'structure_notes': 'style_chain_cut'}


def test_the_default_paragraph_style_does_not_relabel_unstyled_text() -> None:
    styles = '<w:style w:type="paragraph" w:default="1" w:styleId="Normal"><w:name w:val="heading 1"/></w:style>'
    assert roles(word(paragraph('Body text'), styles=styles)) == [('Body text', MEMBER)]


def test_a_word_document_that_declares_nothing_parses_as_it_did() -> None:
    body = paragraph('Hello', extra='<w:jc w:val="center"/>') + paragraph('World', style='BodyText')
    assert roles(word(body, styles=STYLES)) == [('Hello', MEMBER), ('World', MEMBER)]


def html_roles(raw: str) -> list[tuple[str, str, dict[str, str]]]:
    parsed = parse_text(raw.encode(), 'page.html', DocumentLimits())
    return [(segment.locator, segment.text, segment.metadata) for segment in parsed.segments]


def test_html_elements_declare_headings_lists_captions_and_code() -> None:
    page = ('<title>Guide</title>\n<h1>Install</h1>\n<p>Run the <code>setup</code> step.</p>\n'
            '<ul><li>Linux<ul><li>Debian</li></ul>kernels</li><li>macOS</ul>\n'
            '<ol><li>Download<li>Verify</ol>\n'
            '<figure><img src="a.png"><figcaption>Figure 1: Layout</figcaption></figure>\n'
            '<pre>x = 1\n  y = 2</pre>\n<li>Stray item</li>\n<h4>Deep <em>heading</em></h4>\n'
            '<table><caption>Ports</caption><tr><th>Name</th><th>Port</th></tr><tr><td>web</td><td>80</td></tr></table>')
    found = [(text, metadata) for _, text, metadata in html_roles(page)]
    assert found == [
        ('Guide', {}),
        ('Install', {'block_role': 'heading', 'heading_level': '1'}),
        ('Run the setup step.', {}),
        ('Linux', {'block_role': 'list_item', 'list_id': '1', 'list_level': '0', 'list_kind': 'bullet', 'list_item_id': '1'}),
        ('Debian', {'block_role': 'list_item', 'list_id': '1', 'list_level': '1', 'list_kind': 'bullet', 'list_item_id': '2'}),
        ('kernels', {'block_role': 'list_item', 'list_id': '1', 'list_level': '0', 'list_kind': 'bullet', 'list_item_id': '1'}),
        ('macOS', {'block_role': 'list_item', 'list_id': '1', 'list_level': '0', 'list_kind': 'bullet', 'list_item_id': '3'}),
        ('Download', {'block_role': 'list_item', 'list_id': '2', 'list_level': '0', 'list_kind': 'ordered', 'list_item_id': '4'}),
        ('Verify', {'block_role': 'list_item', 'list_id': '2', 'list_level': '0', 'list_kind': 'ordered', 'list_item_id': '5'}),
        ('Figure 1: Layout', {'block_role': 'caption'}),
        ('x = 1\n  y = 2', {'block_role': 'code'}),
        ('Stray item', {'block_role': 'list_item', 'list_level': '0', 'list_item_id': '6'}),
        ('Deep heading', {'block_role': 'heading', 'heading_level': '4'}),
        ('Ports', {'block_role': 'caption', 'caption_target': 'table:1'}),
        ('Name Port', {'table_locator': 'table:1', 'table_status': 'structured'}),
        ('Name: web\nPort: 80', {'table_locator': 'table:1', 'table_status': 'structured'}),
    ]


def test_html_that_declares_nothing_parses_as_it_did() -> None:
    assert html_roles('<p>Friday launch</p><div>x<br>y</div>') == [('line:1', 'Friday launch', {}), ('line:1', 'x\ny', {})]


def test_a_block_inside_an_html_list_item_names_the_item_it_sits_in() -> None:
    page = ('<ol start="4"><li>Install<pre>pip install x</pre><h3>Check</h3>'
            '<figure><figcaption>F</figcaption></figure>'
            '<table><caption>T</caption><tr><th>K</th></tr><tr><td>v</td></tr></table></li></ol>'
            '<pre>outside</pre><ol start="x"><li>a</li></ol><ol start="-1"><li>b</li></ol><ul start="3"><li>c</li></ul>')
    item = {'list_id': '1', 'list_level': '0', 'list_kind': 'ordered', 'list_item_id': '1', 'list_start': '4'}
    assert [(text, metadata) for _, text, metadata in html_roles(page)] == [
        ('Install', {'block_role': 'list_item', **item}),
        ('pip install x', {'block_role': 'code', **item}),
        ('Check', {'block_role': 'heading', 'heading_level': '3', **item}),
        ('F', {'block_role': 'caption', **item}),
        ('T', {**item, 'block_role': 'caption', 'caption_target': 'table:1'}),
        ('K', {**item, 'table_locator': 'table:1', 'table_status': 'structured'}),
        ('K: v', {**item, 'table_locator': 'table:1', 'table_status': 'structured'}),
        ('outside', {'block_role': 'code'}),
        ('a', {'block_role': 'list_item', 'list_id': '2', 'list_level': '0', 'list_kind': 'ordered', 'list_item_id': '2'}),
        ('b', {'block_role': 'list_item', 'list_id': '3', 'list_level': '0', 'list_kind': 'ordered', 'list_item_id': '3'}),
        ('c', {'block_role': 'list_item', 'list_id': '4', 'list_level': '0', 'list_kind': 'bullet', 'list_item_id': '4'}),
    ]


def test_html_list_ids_name_the_mail_part_they_come_from() -> None:
    mail = (b'From: a@example.com\nSubject: x\nMIME-Version: 1.0\nContent-Type: multipart/mixed; boundary=B\n\n'
            b'--B\nContent-Type: text/html\n\n<ul><li>first</li></ul>\n'
            b'--B\nContent-Type: text/html\n\n<ul><li>second<p>more</p></li></ul>\n--B--\n')
    parsed = parse_text(mail, 'm.eml', DocumentLimits())
    assert [(s.text, s.metadata['list_id'], s.metadata['list_item_id']) for s in parsed.segments if 'list_id' in s.metadata] == [
        ('first', 'mime:1.1/1', 'mime:1.1/1'), ('second', 'mime:1.2/1', 'mime:1.2/1'), ('more', 'mime:1.2/1', 'mime:1.2/1')]


def test_a_short_style_chain_is_unresolved_only_where_nothing_in_it_said_heading_or_body() -> None:
    styles = STYLES + ('<w:style w:type="paragraph" w:styleId="BodyLoop"><w:name w:val="BodyLoop"/>'
                       '<w:pPr><w:outlineLvl w:val="9"/></w:pPr><w:basedOn w:val="Loop"/></w:style>'
                       '<w:style w:type="paragraph" w:styleId="TitleLoop"><w:name w:val="Title"/>'
                       '<w:basedOn w:val="TitleLoop"/></w:style>')
    body = ''.join([
        paragraph('Body by style', style='BodyLoop'),
        paragraph('Body by paragraph', style='Loop', outline=9),
        paragraph('Titled', style='TitleLoop'),
        paragraph('Listed loop', style='Loop', number=2),
    ])
    assert roles(word(body, styles=styles, numbering=NUMBERING)) == [
        ('Body by style', MEMBER),
        ('Body by paragraph', MEMBER),
        ('Titled', heading(1, 'style')),
        ('Listed loop', {**item('2', '0', 'ordered'), 'heading_level_unresolved': 'style_chain'}),
    ]
