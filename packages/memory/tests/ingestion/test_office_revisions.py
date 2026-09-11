"""Current document text must not inherit deleted revision-history content."""
from scone_memory.ingestion import BuiltinDocumentParser
from .test_office_formats import O, T, TABLE, archive


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
