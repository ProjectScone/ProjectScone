"""Package relationships choose the actual document, not conventional filenames."""
import pytest

from scone_memory.core.errors import InvalidInput
from scone_memory.ingestion import BuiltinDocumentParser
from .test_office_formats import W, S, P, A, R, REL, archive


def main_relationship(target, *, relationship_type=R + '/officeDocument', mode='Internal'):
    return f'<Relationships xmlns="{REL}"><Relationship Id="main" Target="{target}" Type="{relationship_type}" TargetMode="{mode}"/></Relationships>'


def parts(extension, label='Referenced content'):
    if extension == 'docx':
        return {'custom/main.xml': f'<w:document xmlns:w="{W}"><w:body><w:p><w:r><w:t>{label}</w:t></w:r></w:p></w:body></w:document>'}
    if extension == 'xlsx':
        return {
            'custom/main.xml': f'<workbook xmlns="{S}" xmlns:r="{R}"><sheets><sheet name="Sheet" r:id="s"/></sheets></workbook>',
            'custom/_rels/main.xml.rels': f'<Relationships xmlns="{REL}"><Relationship Id="s" Target="../content/sheet.xml" Type="{R}/worksheet"/></Relationships>',
            'content/sheet.xml': f'<worksheet xmlns="{S}"><sheetData><row><c t="inlineStr"><is><t>{label}</t></is></c></row></sheetData></worksheet>',
        }
    return {
        'custom/main.xml': f'<p:presentation xmlns:p="{P}" xmlns:r="{R}"><p:sldIdLst><p:sldId id="256" r:id="s"/></p:sldIdLst></p:presentation>',
        'custom/_rels/main.xml.rels': f'<Relationships xmlns="{REL}"><Relationship Id="s" Target="../content/slide.xml" Type="{R}/slide"/></Relationships>',
        'content/slide.xml': f'<p:sld xmlns:p="{P}" xmlns:a="{A}"><p:cSld><p:spTree><p:sp><p:txBody><a:p><a:r><a:t>{label}</a:t></a:r></a:p></p:txBody></p:sp></p:spTree></p:cSld></p:sld>',
    }


@pytest.mark.parametrize('extension', ['docx', 'xlsx', 'pptx'])
@pytest.mark.parametrize('target', ['custom/main.xml', '/custom/main.xml', 'custom/%6dain.xml'])
async def test_nonstandard_main_parts_and_their_relative_relationships(extension, target):
    data = archive({'_rels/.rels': main_relationship(target), **parts(extension)})
    parsed = await BuiltinDocumentParser().parse(data, f'report.{extension}')
    assert [s.text for s in parsed.segments] == ['Referenced content']


async def test_unreferenced_conventional_docx_part_never_supplies_body_text():
    data = archive({'_rels/.rels': main_relationship('custom/main.xml'), **parts('docx'),
        'word/document.xml': parts('docx', 'Unreferenced decoy')['custom/main.xml']})
    parsed = await BuiltinDocumentParser().parse(data, 'report.docx')
    assert [s.text for s in parsed.segments] == ['Referenced content']


@pytest.mark.parametrize('target', ['../main.xml', 'https://example.test/main.xml', 'custom/main.xml?x=1', 'custom/main.xml#fragment',
    'custom/main.xml?', 'custom/main.xml#', 'custom/ma&#xA;in.xml', 'custom/ma&#x9;in.xml', ' custom/main.xml'])
async def test_main_part_must_be_an_unambiguous_local_member(target):
    data = archive({'_rels/.rels': main_relationship(target),
        'word/document.xml': parts('docx')['custom/main.xml'], **parts('docx')})
    with pytest.raises(InvalidInput):
        await BuiltinDocumentParser().parse(data, 'report.docx')


@pytest.mark.parametrize('relationships', [None, f'<Relationships xmlns="{REL}"/>',
    main_relationship('word/document.xml', mode='External'),
    main_relationship('word/document.xml', relationship_type='urn:unrelated/officeDocument'),
    main_relationship('word/document.xml').replace('</Relationships>', f'<Relationship Id="other" Target="custom/main.xml" Type="{R}/officeDocument"/></Relationships>'),
], ids=['missing', 'empty', 'external', 'unrelated-type', 'ambiguous'])
async def test_missing_external_or_ambiguous_main_relationships_are_rejected(relationships):
    members = {'word/document.xml': parts('docx')['custom/main.xml'], **parts('docx')}
    if relationships is not None:
        members['_rels/.rels'] = relationships
    with pytest.raises(InvalidInput):
        await BuiltinDocumentParser().parse(archive(members), 'report.docx')


async def test_strict_office_document_relationship_is_supported():
    data = archive({'_rels/.rels': main_relationship('custom/main.xml',
        relationship_type='http://purl.oclc.org/ooxml/officeDocument/relationships/officeDocument'), **parts('docx')})
    parsed = await BuiltinDocumentParser().parse(data, 'report.docx')
    assert parsed.segments[0].text == 'Referenced content'


@pytest.mark.parametrize('relationships', [
    main_relationship('custom/main.xml').replace(REL, 'urn:foreign'),
    main_relationship('custom/main.xml').replace('Id="main"', 'Id=""'),
    main_relationship('custom/main.xml').replace('</Relationships>', '<Relationship Id="main" Type="urn:other" Target="other.xml"/></Relationships>'),
], ids=['foreign-namespace', 'empty-id', 'duplicate-id'])
async def test_invalid_package_relationship_structure_is_rejected(relationships):
    with pytest.raises(InvalidInput, match='package relationship'):
        await BuiltinDocumentParser().parse(archive({'_rels/.rels': relationships, **parts('docx')}), 'report.docx')


async def test_percent_encoded_space_in_main_part_name_is_preserved():
    members = {'_rels/.rels': main_relationship('custom/main%20file.xml'),
        'custom/main file.xml': parts('docx')['custom/main.xml']}
    parsed = await BuiltinDocumentParser().parse(archive(members), 'report.docx')
    assert parsed.segments[0].text == 'Referenced content'
