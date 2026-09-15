from __future__ import annotations

import json
from email.message import EmailMessage

import pytest

from scone_memory.core.errors import InvalidInput
from scone_memory.ingestion.formats.text import parse_text
from scone_memory.ingestion.formats.types import DocumentLimits


@pytest.mark.parametrize("suffix", ["txt", "md", "markdown", "py", "ts", "yaml", "toml"])
def test_plain_text_preserves_source_and_line_locations(suffix: str) -> None:
    doc = parse_text(b"\xef\xbb\xbf# heading\n\n    indented\n", f"source.{suffix}", DocumentLimits())
    assert [s.text for s in doc.segments] == ["# heading", "    indented"]
    assert [s.locator for s in doc.segments] == ["line:1", "line:3"]


@pytest.mark.parametrize("encoding", ["utf-16", "utf-32"])
def test_explicit_bom_decodes_without_guessing(encoding: str) -> None:
    doc = parse_text("café".encode(encoding), "note.txt", DocumentLimits())
    assert doc.segments[0].text == "café"


@pytest.mark.parametrize("data", [b"\xfftext", b"binary\x00data"])
def test_binary_or_undecodable_text_is_rejected(data: bytes) -> None:
    with pytest.raises(InvalidInput):
        parse_text(data, "note.txt", DocumentLimits())


def test_csv_rows_keep_headers_multiline_values_and_physical_lines() -> None:
    doc = parse_text(b'name,note\r\nAda,"first\nsecond"\r\nBob,last\r\n', "data.csv", DocumentLimits())
    assert [s.locator for s in doc.segments] == ["row:2", "row:3"]
    assert doc.segments[0].text == "name: Ada\nnote: first\nsecond"
    assert doc.segments[0].metadata["line_start"] == "2"
    assert doc.segments[0].metadata["line_end"] == "3"


def test_tsv_duplicate_and_empty_headers_do_not_overwrite_columns() -> None:
    doc = parse_text(b"x\tx\t\n1\t2\t3\n", "data.tsv", DocumentLimits())
    assert doc.segments[0].text == "x [column 1]: 1\nx [column 2]: 2\ncolumn 3: 3"


@pytest.mark.parametrize("data", [b'a,b\n1\n', b'a,b\n1,2,3\n', b'a,b\n"unclosed,2'])
def test_malformed_csv_is_rejected(data: bytes) -> None:
    with pytest.raises(InvalidInput):
        parse_text(data, "data.csv", DocumentLimits())


def test_json_preserves_scalars_empty_containers_and_escaped_pointers() -> None:
    raw = b'{"a/b":{"~key":[0,false,null,"",{},[]]},"n":12345678901234567890.123456789,"huge":1e999}'
    doc = parse_text(raw, "data.json", DocumentLimits())
    assert [s.metadata["json_pointer"] for s in doc.segments] == [
        "/a~1b/~0key/0", "/a~1b/~0key/1", "/a~1b/~0key/2", "/a~1b/~0key/3",
        "/a~1b/~0key/4", "/a~1b/~0key/5", "/n", "/huge",
    ]
    assert [s.text.split(": ", 1)[1] for s in doc.segments] == [
        "0", "false", "null", '""', "{}", "[]", "12345678901234567890.123456789", "1e999",
    ]


@pytest.mark.parametrize("scalar", ["null", "false", "0", '""', "{}", "[]"])
def test_json_root_scalar_is_not_dropped(scalar: str) -> None:
    doc = parse_text(scalar.encode(), "data.json", DocumentLimits())
    assert doc.segments[0].text == scalar
    assert doc.segments[0].metadata["json_pointer"] == ""


@pytest.mark.parametrize("suffix", ["jsonl", "ndjson"])
def test_json_lines_retain_physical_line_and_pointer(suffix: str) -> None:
    doc = parse_text(b'\n{"ok":false}\n0\n', f"data.{suffix}", DocumentLimits())
    assert [s.locator for s in doc.segments] == ["line:2#/ok", "line:3#"]


@pytest.mark.parametrize("raw", [b'{"a":1,"a":2}', b'NaN', b'Infinity', b'{broken}', b'"\\ud800"'])
def test_json_rejects_invalid_or_lossy_documents(raw: bytes) -> None:
    with pytest.raises(InvalidInput):
        parse_text(raw, "data.json", DocumentLimits())


def test_json_lines_do_not_silently_skip_malformed_record() -> None:
    with pytest.raises(InvalidInput):
        parse_text(b'{"a":1}\nbroken\n', "data.jsonl", DocumentLimits())


def test_json_excessive_nesting_is_rejected() -> None:
    with pytest.raises(InvalidInput, match="depth|nested"):
        parse_text(("[" * 150 + "0" + "]" * 150).encode(), "data.json", DocumentLimits())


def test_html_extracts_title_inline_spacing_and_visible_blocks() -> None:
    raw = b'''<html><head><title>A &amp; B</title><style>hidden CSS</style></head>
    <body><p>Hello <b>world</b>!</p><p>Next<br>line</p><script>secret()</script>
    <div hidden>hidden text</div><div style="display: none">hidden style</div>
    <template>hidden template</template><img src="http://example.invalid/a"></body></html>'''
    doc = parse_text(raw, "page.html", DocumentLimits())
    assert doc.metadata["title"] == "A & B"
    assert [s.text for s in doc.segments] == ["A & B", "Hello world!", "Next\nline"]


def test_xml_preserves_attributes_mixed_text_and_element_paths() -> None:
    doc = parse_text(b'<root lang="en">Before<item>one</item>between<item>two</item>after</root>', "data.xml", DocumentLimits())
    assert [s.text for s in doc.segments] == ["@lang: en", "Before", "one", "between", "two", "after"]
    assert any("/root[1]/item[2]" in s.locator for s in doc.segments)


@pytest.mark.parametrize("raw", [
    b'<root><x></root>',
    b'<!DOCTYPE root [<!ENTITY x SYSTEM "file:///etc/passwd">]><root>&x;</root>',
    b'<!DOCTYPE root [<!ENTITY x "expanded">]><root>&x;</root>',
    ("<a>" * 150 + "x" + "</a>" * 150).encode(),
])
def test_xml_rejects_malformed_entities_and_excessive_depth(raw: bytes) -> None:
    with pytest.raises(InvalidInput):
        parse_text(raw, "data.xml", DocumentLimits())


def test_email_keeps_headers_and_mime_locators_but_skips_attachments() -> None:
    email = EmailMessage()
    email["Subject"] = "Café update"
    email["From"] = "ada@example.test"
    email["To"] = "bob@example.test"
    email.set_content("Plain body")
    email.add_alternative("<p>HTML body</p><script>never run</script>", subtype="html")
    email.add_attachment(b"attachment secret", maintype="text", subtype="plain", filename="secret.txt")
    doc = parse_text(email.as_bytes(), "mail.eml", DocumentLimits())
    joined = "\n".join(s.text for s in doc.segments)
    assert "Subject: Café update" in joined
    assert "Plain body" in joined and "HTML body" in joined
    assert "attachment secret" not in joined and "never run" not in joined
    assert any(s.locator.startswith("mime:1.1") for s in doc.segments)
    assert doc.metadata["attachments_skipped"] == "1"


def test_email_respects_declared_charset_and_rejects_bad_charset() -> None:
    raw = b'Subject: hi\nContent-Type: text/plain; charset=iso-8859-1\n\ncaf\xe9'
    assert any("café" in s.text for s in parse_text(raw, "mail.eml", DocumentLimits()).segments)
    with pytest.raises(InvalidInput):
        parse_text(raw.replace(b"iso-8859-1", b"not-a-charset"), "mail.eml", DocumentLimits())


@pytest.mark.parametrize("suffix,data", [("txt", b"one\ntwo"), ("json", b"[1,2]"), ("csv", b"a\n1\n2"), ("html", b"<p>one</p><p>two</p>"), ("xml", b"<r><a>one</a><a>two</a></r>")])
def test_segment_limits_fail_instead_of_truncating(suffix: str, data: bytes) -> None:
    with pytest.raises(InvalidInput):
        parse_text(data, f"data.{suffix}", DocumentLimits(max_segments=1))


@pytest.mark.parametrize("limits", [DocumentLimits(max_input_bytes=3), DocumentLimits(max_text_bytes=3)])
def test_byte_limits_are_enforced(limits: DocumentLimits) -> None:
    with pytest.raises(InvalidInput):
        parse_text("ééé".encode(), "note.txt", limits)


def test_json_pointer_amplification_is_bounded() -> None:
    raw = json.dumps({"x" * 1000: list(range(100))}).encode()
    with pytest.raises(InvalidInput):
        parse_text(raw, "data.json", DocumentLimits(max_text_bytes=5000))


def test_csv_physical_lines_include_bare_carriage_returns_in_quoted_fields() -> None:
    doc = parse_text(b'name,note\rAda,"first\rsecond"\rBob,last\r', 'data.csv', DocumentLimits())
    assert {'line_start': '2', 'line_end': '3'}.items() <= doc.segments[0].metadata.items()
    assert {'line_start': '4', 'line_end': '4'}.items() <= doc.segments[1].metadata.items()


def test_xml_text_node_ordinals_count_only_existing_text_nodes() -> None:
    doc = parse_text(b'<r><a>one</a><b>two</b>tail</r>', 'data.xml', DocumentLimits())
    assert doc.segments[-1].locator == 'xml:/r[1]/text()[1]'


def test_email_rejects_malformed_base64_instead_of_returning_partial_body() -> None:
    raw = b'Content-Type: text/plain\nContent-Transfer-Encoding: base64\n\naGVsbG8=@@@@'
    with pytest.raises(InvalidInput):
        parse_text(raw, 'mail.eml', DocumentLimits())


def test_html_table_cells_are_separated_and_hidden_script_is_not_extracted() -> None:
    doc = parse_text(b'<table><tr><td>A</td><td>B</td></tr></table><script>secret</script>',
                     'page.htm', DocumentLimits())
    assert [s.text for s in doc.segments] == ['A B']


def test_html_depth_limit_is_enforced() -> None:
    with pytest.raises(InvalidInput, match='depth'):
        parse_text(b'<div>' * 150 + b'content' + b'</div>' * 150, 'page.html', DocumentLimits())


@pytest.mark.parametrize('suffix', ['txt', 'html', 'xml', 'csv'])
def test_empty_documents_fail_cleanly(suffix: str) -> None:
    with pytest.raises(InvalidInput):
        parse_text(b'', f'empty.{suffix}', DocumentLimits())


@pytest.mark.parametrize('opening,item,closing', [
    ('<ul>', '<li>', '</ul>'),
    ('<body>', '<p>', '</body>'),
    ('<table>', '<tr><td>', '</table>'),
    ('<table><tr>', '<td>', '</tr></table>'),
])
def test_html_optional_end_tags_do_not_count_as_nested_elements(
    opening: str, item: str, closing: str,
) -> None:
    raw = opening + ''.join(f'{item}Item {i}.' for i in range(101)) + closing
    doc = parse_text(raw.encode(), 'page.html', DocumentLimits())
    joined = '\n'.join(segment.text for segment in doc.segments)
    assert all(f'Item {i}.' in joined for i in range(101))


@pytest.mark.parametrize('raw', [
    '<p>Intro</p><ul><li hidden>secret<li>Visible item</ul>',
    '<p>Intro</p><p hidden>secret<p>Visible item',
    '<p>Intro</p><p hidden>secret<div>Visible item</div>',
    '<p>Intro</p><table><tr hidden><td>secret<tr><td>Visible item</table>',
    '<p>Intro</p><table><tr><th hidden>secret<td>Visible item</table>',
    '<p>Intro</p><table><thead hidden><tr><th>secret<tbody><tr><td>Visible item</table>',
    '<p>Intro</p><table><tbody hidden><tr><td>secret<tfoot><tr><td>Visible item</table>',
])
def test_html_implied_closures_end_hidden_sibling_visibility(raw: str) -> None:
    doc = parse_text(raw.encode(), 'page.html', DocumentLimits())
    assert [segment.text for segment in doc.segments] == ['Intro', 'Visible item']


def test_html_implied_list_closure_stays_within_nearest_list() -> None:
    raw = (b'<p>Intro</p><ul><li hidden>secret<ul><li>nested secret<li>more secret</ul>'
           b'<li>Visible item<ul><li hidden>inner secret<li>Visible nested</ul>Visible tail</ul>')
    doc = parse_text(raw, 'page.html', DocumentLimits())
    assert [segment.text for segment in doc.segments] == [
        'Intro', 'Visible item', 'Visible nested', 'Visible tail',
    ]


def test_html_implied_cell_closure_stays_within_nearest_table() -> None:
    raw = (b'<p>Intro</p><table><tr><td hidden>secret<table><tr><td>nested secret</table>'
           b'<td>Visible item</table>')
    doc = parse_text(raw, 'page.html', DocumentLimits())
    assert [segment.text for segment in doc.segments] == ['Intro', 'Visible item']


def _mailbox_bytes() -> bytes:
    first = EmailMessage()
    first["Subject"] = "Harbour closing"
    first["From"] = "ada@example.test"
    first["Date"] = "Mon, 04 Nov 2024 09:00:00 +0000"
    first.set_content("The harbour closes to sailing boats every November.\nFrom the mole you can see it.")
    second = EmailMessage()
    second["Subject"] = "Re: Harbour closing"
    second["From"] = "bob@example.test"
    second["Date"] = "Tue, 05 Nov 2024 10:00:00 +0000"
    second.set_content("Plain reply")
    second.add_alternative("<p>HTML reply</p><script>never run</script>", subtype="html")
    second.add_attachment(b"attachment secret", maintype="text", subtype="plain", filename="secret.txt")
    body_first = first.as_bytes().replace(b"\nFrom the mole", b"\n>From the mole")  # mbox quoting of a body line
    return (b"From ada@example.test Mon Nov  4 09:00:00 2024\n" + body_first + b"\n"
            b"From bob@example.test Tue Nov  5 10:00:00 2024\n" + second.as_bytes() + b"\n")


def test_a_mailbox_reads_each_message_as_an_email_and_says_which_mail_a_passage_came_from() -> None:
    doc = parse_text(_mailbox_bytes(), "inbox.mbox", DocumentLimits())
    assert doc.format == "mbox" and doc.metadata == {"messages": "2", "messages_unread": "0", "attachments_skipped": "1"}
    joined = "\n".join(s.text for s in doc.segments)
    assert "Subject: Harbour closing" in joined and "Subject: Re: Harbour closing" in joined
    assert "From the mole you can see it." in joined, "mbox quoting of a body line is undone"
    assert ">From the mole" not in joined
    assert "Plain reply" in joined and "HTML reply" in joined
    assert "attachment secret" not in joined and "never run" not in joined
    first = [s for s in doc.segments if s.locator.startswith("message:1/")]
    second = [s for s in doc.segments if s.locator.startswith("message:2/")]
    assert first and second and all(s.metadata["message"] == "1" for s in first)
    assert all(s.metadata["from"] == "bob@example.test" and s.metadata["date"].startswith("Tue, 05 Nov 2024") for s in second)
    assert any(s.locator.startswith("message:2/mime:1.1") and s.locator.endswith("/line:1") for s in second), \
        "the same locators an .eml gets, under the message"


def test_a_bare_message_saved_as_a_mailbox_is_one_message_and_the_bound_counts_the_rest(monkeypatch) -> None:
    email = EmailMessage()
    email["Subject"] = "alone"
    email.set_content("one message, no From line")
    doc = parse_text(email.as_bytes(), "one.mbox", DocumentLimits())
    assert doc.metadata["messages"] == "1" and any("no From line" in s.text for s in doc.segments)
    from scone_memory.ingestion.formats import text as text_formats

    monkeypatch.setattr(text_formats, "MAX_MAILBOX_MESSAGES", 1)
    doc = parse_text(_mailbox_bytes(), "inbox.mbox", DocumentLimits())
    assert doc.metadata["messages"] == "1" and doc.metadata["messages_unread"] == "1"
    assert not any(s.locator.startswith("message:2/") for s in doc.segments)


def test_a_mailbox_with_a_malformed_message_names_it() -> None:
    raw = b"From a@b Mon Jan 1 00:00:00 2024\nSubject: ok\n\nfine\n\nFrom c@d Mon Jan 1 00:00:00 2024\nContent-Type: text/plain; charset=not-a-charset\n\ncaf\xe9\n"
    with pytest.raises(InvalidInput, match="mailbox message 2: "):
        parse_text(raw, "inbox.mbox", DocumentLimits())


def test_a_body_line_that_begins_with_from_is_not_a_message_and_a_bom_is_not_a_preamble() -> None:
    raw = (b"\xef\xbb\xbfFrom alice@example.com Mon Jan  1 00:00:00 2024\nFrom: alice@example.com\nSubject: Notes\n\n"
           b"Hello.\nFrom now on I will write more.\nBye.\n"
           b"From bob@example.com Tue Jan  2 10:00:00 2024\nFrom: bob@example.com\nSubject: Second\n\nSecond body.\n")
    doc = parse_text(raw, "inbox.mbox", DocumentLimits())
    assert doc.metadata["messages"] == "2", "an unquoted body line is not an envelope line, and the BOM is not text before the first"
    joined = "\n".join(s.text for s in doc.segments)
    assert "From now on I will write more." in joined and "Bye." in joined
    assert {s.metadata["message"] for s in doc.segments if "Second body" in s.text} == {"2"}


def test_an_html_table_in_a_mail_carries_the_mails_metadata_and_a_bare_file_is_not_unquoted() -> None:
    email = EmailMessage()
    email["Subject"] = "Table"
    email["From"] = "ada@example.test"
    email.set_content("plain")
    email.add_alternative("<table><caption>Costs</caption><tr><th>item</th><th>cost</th></tr><tr><td>rope</td><td>3</td></tr></table>",
                          subtype="html")
    raw = b"From ada@example.test Mon Nov  4 09:00:00 2024\n" + email.as_bytes() + b"\n"
    doc = parse_text(raw, "inbox.mbox", DocumentLimits())
    rows = [s for s in doc.segments if "table_locator" in s.metadata or s.locator.endswith("/caption")]
    assert rows and all(s.metadata.get("message") == "1" and s.metadata.get("from") == "ada@example.test" for s in rows), \
        "every segment, table rows and captions included, says which mail it came from"
    bare = EmailMessage()
    bare["Subject"] = "reply"
    bare.set_content(">From my notes, the harbour closes in November.")
    doc = parse_text(bare.as_bytes(), "one.mbox", DocumentLimits())
    assert any(s.text == ">From my notes, the harbour closes in November." for s in doc.segments), \
        "a file no mbox writer made was never quoted, so nothing is unquoted"


async def test_a_mailbox_file_is_ingested_end_to_end_and_its_original_kept_under_its_own_type() -> None:
    """The reader is not enough: ingesting a file keeps its original bytes as
    an attachment under the file's media type, and the engine refuses a type
    it does not attach."""
    from scone_memory import HashEmbedder, InMemoryDocumentStore, InMemoryVectorIndex, MemoryEngine
    from scone_memory.ingestion.files import ingest_document

    memory = await MemoryEngine(InMemoryDocumentStore(), InMemoryVectorIndex(), HashEmbedder()).open()
    try:
        ingested = await ingest_document(memory, "mail", _mailbox_bytes(), filename="inbox.mbox")
        assert ingested.format == "mbox" and ingested.original.media_type == "application/mbox"
        episode = await memory.episode("mail", ingested.added.episode_id)
        assert "The harbour closes to sailing boats every November." in episode.content
    finally:
        await memory.close()


def test_csv_rows_carry_their_cells_with_spans_and_the_column_names() -> None:
    import json

    from scone_memory.ingestion.formats.table_types import validate_tables

    doc = parse_text(b'region,revenue\r\nWest,"1,250.50"\r\nEast,35\r\n', "sales.csv", DocumentLimits())
    validate_tables(doc.segments)
    first, second = doc.segments
    assert first.text == "region: West\nrevenue: 1,250.50" and json.loads(first.metadata["table_columns"]) == ["region", "revenue"]
    assert [(c.row, c.column, c.text) for c in first.table_cells] == [(0, 0, "West"), (0, 1, "1,250.50")]
    for cell in (*first.table_cells, *second.table_cells):
        assert cell.table_locator == "delimited" and not cell.headers
    assert first.text.encode()[first.table_cells[1].start:first.table_cells[1].end] == b"1,250.50"
    assert [c.locator for c in second.table_cells] == ["row:3/column:1", "row:3/column:2"]
    assert doc.parser == "scone-text-tables-v1"


def test_a_json_array_of_flat_objects_is_a_table_with_cells() -> None:
    import json

    from scone_memory.ingestion.formats.table_types import validate_tables

    raw = json.dumps({"title": "Sales", "rows": [{"region": "West", "revenue": "1,250.50"}, {"region": "East", "revenue": 35}],
                      "tags": ["a", "b"], "nested": [{"x": {"y": 1}}]}).encode()
    doc = parse_text(raw, "sales.json", DocumentLimits())
    validate_tables(doc.segments)
    cells = [(c.table_locator, c.row, c.column, c.text) for s in doc.segments for c in s.table_cells]
    assert cells == [("json:/rows", 0, 0, "West"), ("json:/rows", 0, 1, "1,250.50"), ("json:/rows", 1, 0, "East"), ("json:/rows", 1, 1, "35")]
    west = next(s for s in doc.segments if s.table_cells and s.table_cells[0].text == "West")
    assert west.text == '/rows/0/region: "West"' and west.text.encode()[west.table_cells[0].start:west.table_cells[0].end] == b"West"
    assert json.loads(west.metadata["table_columns"]) == ["region", "revenue"] and west.metadata["header_basis"] == "json_object_keys"
    plain = [s for s in doc.segments if s.metadata["json_pointer"] in ("/title", "/tags/0", "/nested/0/x/y")]
    assert plain and all(not s.table_cells for s in plain), "scalars, arrays of scalars and nested objects are not tables"
