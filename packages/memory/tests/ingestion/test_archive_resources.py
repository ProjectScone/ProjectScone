"""Container limits reject unsafe codecs and bound XML construction itself."""
from io import BytesIO
from zipfile import ZIP_BZIP2, ZIP_DEFLATED, ZIP_LZMA, ZIP_STORED, ZipFile

import pytest

from scone_memory.core.errors import InvalidInput
from scone_memory.ingestion import BuiltinDocumentParser, DocumentLimits
from scone_memory.ingestion.formats.archive import SafeArchive
from scone_memory.ingestion.formats.office import parse_office
from .test_office_formats import W, archive, package_relationship


def container(xml, compression=ZIP_DEFLATED):
    stream = BytesIO()
    with ZipFile(stream, 'w', compression=compression) as bundle:
        bundle.writestr('_rels/.rels', package_relationship('word/document.xml'))
        bundle.writestr('word/document.xml', xml)
    return stream.getvalue()


@pytest.mark.parametrize('compression', [ZIP_BZIP2, ZIP_LZMA])
def test_nonstandard_office_codecs_are_refused_before_member_decompression(compression, monkeypatch):
    data = container('<document/>', compression)
    def forbidden_open(*args, **kwargs):
        pytest.fail('archive decoder must not run for an unbounded compression method')
    monkeypatch.setattr(ZipFile, 'open', forbidden_open)
    with pytest.raises(InvalidInput, match='compression'):
        with SafeArchive(data, DocumentLimits()):
            pass


@pytest.mark.parametrize('compression', [ZIP_STORED, ZIP_DEFLATED])
def test_standard_office_codecs_remain_supported(compression):
    with SafeArchive(container('<document/>', compression), DocumentLimits()) as bundle:
        assert bundle.xml('word/document.xml').tag == 'document'


@pytest.mark.parametrize('xml', [
    '<root>' + '<n/>' * 200_000 + '</root>',
    '<n>' * 129 + '</n>' * 129,
    '<root>' + 'x' * (16 * 1024 * 1024) + '</root>',
])
async def test_xml_member_node_depth_and_byte_limits_apply_in_actual_worker(xml):
    # Every case contains valid DOCX body text: rejection cannot be an empty-document accident.
    wrapped = f'<w:document xmlns:w="{W}"><w:body><w:p><w:r><w:t>Visible</w:t></w:r></w:p>{xml}</w:body></w:document>'
    with pytest.raises(InvalidInput, match='XML.*limit'):
        await BuiltinDocumentParser().parse(container(wrapped), 'source.docx')


def epub(chapter):
    return archive({'META-INF/container.xml': '<container><rootfile full-path="book.opf"/></container>',
        'book.opf': '<package><manifest><item id="c" href="chapter.xhtml" media-type="application/xhtml+xml"/></manifest><spine><itemref idref="c"/></spine></package>',
        'chapter.xhtml': chapter})


@pytest.mark.parametrize('doctype', ['<!DOCTYPE html>',
    '<!DOCTYPE html PUBLIC "-//W3C//DTD XHTML 1.1//EN" "http://www.w3.org/TR/xhtml11/DTD/xhtml11.dtd">'])
def test_epub_doctype_is_allowed_without_fetching_external_dtd(doctype):
    parsed = parse_office(epub(doctype + '<html><body><p>Café chapter</p></body></html>'), 'book.epub', DocumentLimits())
    assert parsed.segments[0].text == 'Café chapter'


def test_epub_standard_named_entities_do_not_require_an_external_dtd():
    chapter = '<!DOCTYPE html PUBLIC "-//W3C//DTD XHTML 1.1//EN" "http://www.w3.org/TR/xhtml11/DTD/xhtml11.dtd"><html><body><p>Café&nbsp;&copy; &NotEqualTilde;</p></body></html>'
    parsed = parse_office(epub(chapter), 'book.epub', DocumentLimits())
    assert parsed.segments[0].text == 'Café\u00a0© ≂̸'


@pytest.mark.parametrize('declaration', ['<!ENTITY secret "sentinel-expanded">',
    '<!ENTITY secret SYSTEM "file:///must-not-read">'])
def test_entity_guard_rejects_otherwise_valid_document_content(declaration):
    chapter = '<!DOCTYPE html [' + declaration + ']><html><body><p>&secret;</p></body></html>'
    with pytest.raises(InvalidInput, match='unsafe XML'):
        parse_office(epub(chapter), 'book.epub', DocumentLimits())


def test_epub_internal_attribute_defaults_are_rejected():
    chapter = '<!DOCTYPE html [<!ATTLIST p copied CDATA "' + 'x' * 8192 + '">]><html><body><p>Visible</p></body></html>'
    with pytest.raises(InvalidInput, match='internal DTD'):
        parse_office(epub(chapter), 'book.epub', DocumentLimits())


def test_xml_expanded_namespace_names_have_an_aggregate_budget():
    # Under 300 KB serialized, but over 32 MiB of expanded names without a construction budget.
    xml = '<root xmlns="' + 'x' * 1000 + '">' + ''.join(f'<n{i}/>' for i in range(34000)) + '</root>'
    with SafeArchive(container(xml), DocumentLimits()) as bundle:
        with pytest.raises(InvalidInput, match='XML.*expanded.*limit'):
            bundle.xml('word/document.xml')


@pytest.mark.parametrize('xml', [
    '<n xmlns="' + 'u' * 1025 + '"/>',
    '<n ' + ' '.join(f'a{i}="v"' for i in range(257)) + '/>',
])
def test_raw_namespace_and_attribute_limits_run_before_namespace_expansion(xml, monkeypatch):
    import defusedxml.ElementTree
    def forbidden_parser(*args, **kwargs):
        pytest.fail('namespace-aware parser must not run before raw-name preflight succeeds')
    monkeypatch.setattr(defusedxml.ElementTree, 'DefusedXMLParser', forbidden_parser)
    with SafeArchive(container(xml), DocumentLimits()) as bundle:
        with pytest.raises(InvalidInput, match='XML.*limit'):
            bundle.xml('word/document.xml')


def test_plain_epub_doctype_uses_only_known_local_entity_names():
    parsed = parse_office(epub('<!DOCTYPE html><html><body><p>&copy; Café</p></body></html>'), 'book.epub', DocumentLimits())
    assert parsed.segments[0].text == '© Café'
    with pytest.raises(InvalidInput, match='unsafe XML'):
        parse_office(epub('<!DOCTYPE html><html><body><p>&scone_unknown;</p></body></html>'), 'book.epub', DocumentLimits())


@pytest.mark.parametrize('attribute', ['title="&scone_unknown;"', 'xmlns="urn:&scone_unknown;"',
    'title="&scone:unknown;"', 'title="&inconnué;"'])
def test_epub_unknown_entities_in_attributes_are_not_silently_dropped(attribute):
    with pytest.raises(InvalidInput, match='unsafe XML'):
        parse_office(epub(f'<!DOCTYPE html><html><body><p {attribute}>Visible</p></body></html>'), 'book.epub', DocumentLimits())


@pytest.mark.parametrize('encoding', ['utf-8', 'utf-16', 'iso-8859-1'])
def test_known_attribute_entities_preserve_values_without_changing_cdata_or_comments(encoding):
    xml = f'<?xml version="1.0" encoding="{encoding}"?><!DOCTYPE html><html title="&copy; Café"><!-- &scone_unknown; --><body><p><![CDATA[&copy; literal]]></p></body></html>'
    with SafeArchive(container(xml.encode(encoding)), DocumentLimits()) as bundle:
        root = bundle.xml('word/document.xml', allow_doctype=True)
    assert root.attrib['title'] == '© Café'
    assert root.find('body/p').text == '&copy; literal'
