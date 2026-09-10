"""Real generated PDFs exercise extraction, source identity and recall provenance."""
from io import BytesIO
import hashlib
import json
import pytest
from scone_memory import HashEmbedder, InMemoryDocumentStore, InMemoryVectorIndex, MemoryEngine
from scone_memory.core.errors import InvalidInput, NotFound
pypdf = pytest.importorskip('pypdf')
from pypdf.generic import DecodedStreamObject, DictionaryObject, NameObject


def pdf_bytes(pages=("Café calibration uses Polaris.", "Juniper ships on Friday."), *, password=None, title="Fixture"):
    writer = pypdf.PdfWriter()
    font = DictionaryObject({NameObject('/Type'): NameObject('/Font'), NameObject('/Subtype'): NameObject('/Type1'),
        NameObject('/BaseFont'): NameObject('/Helvetica'), NameObject('/Encoding'): NameObject('/WinAnsiEncoding')})
    for text in pages:
        page = writer.add_blank_page(width=612, height=792)
        page[NameObject('/Resources')] = DictionaryObject({NameObject('/Font'): DictionaryObject({NameObject('/F1'): writer._add_object(font)})})
        stream = DecodedStreamObject()
        stream.set_data(f'BT /F1 12 Tf 72 720 Td <{text.encode("cp1252").hex()}> Tj ET'.encode())
        page[NameObject('/Contents')] = writer._add_object(stream)
    writer.add_metadata({'/Title': title})
    if password:
        writer.encrypt(password)
    target = BytesIO()
    writer.write(target)
    return target.getvalue()


@pytest.fixture(params=["memory", "sqlite"])
async def memory(request, tmp_path):
    blobs = None
    if request.param == "sqlite":
        from scone_memory.backends.blobs import FileBlobStore
        blobs = FileBlobStore(tmp_path / "originals")
        from scone_memory.backends import SqliteDocumentStore, SqliteVectorIndex
        path = tmp_path / "pdf-memory.db"
        documents, vectors = SqliteDocumentStore(path), SqliteVectorIndex(path)
    else:
        documents, vectors = InMemoryDocumentStore(), InMemoryVectorIndex()
    engine = await MemoryEngine(documents, vectors, HashEmbedder(), blobs=blobs).open()
    yield engine
    await engine.close()


async def test_pdf_text_is_searchable_and_recall_spans_resolve_to_original_pages(memory):
    import scone_memory.ingestion as ingestion
    assert hasattr(ingestion, 'ingest_pdf'), 'native PDF ingestion is missing'
    from scone_memory.ingestion import ingest_pdf, pdf_provenance
    raw = pdf_bytes()
    result = await ingest_pdf(memory, 'alpha', raw, filename='calibration.pdf')
    episode = await memory.episode('alpha', result.added.episode_id)
    assert episode.kind == 'file' and 'Café calibration uses Polaris.' in episode.content
    assert (await memory.attachment('alpha', result.original.attachment_id))[1] == raw
    assert result.original.attachment_id == hashlib.sha256(raw).hexdigest()
    recalled = await memory.recall('alpha', 'Polaris', limit=5)
    assert any(item.episode_id == episode.episode_id for item in recalled.items)
    start = episode.content.encode().index(b'Juniper')
    provenance = await pdf_provenance(memory, 'alpha', episode.episode_id, start=start, end=start+7)
    assert [page.number for page in provenance.pages] == [2]
    assert provenance.pages[0].width_points == 612
    assert provenance.pages[0].height_points == 792
    assert provenance.original.attachment_id == result.original.attachment_id
    assert len(episode.metadata) < 16


async def test_original_identity_prevents_identical_text_from_collapsing_sources(memory):
    from scone_memory.ingestion import ingest_pdf
    a = await ingest_pdf(memory, 'alpha', pdf_bytes(title='A'))
    b = await ingest_pdf(memory, 'alpha', pdf_bytes(title='B'))
    again = await ingest_pdf(memory, 'alpha', pdf_bytes(title='A'))
    assert a.added.episode_id != b.added.episode_id
    assert again.added.episode_id == a.added.episode_id and again.added.deduplicated


@pytest.mark.parametrize(('raw', 'message'), [(b'not a PDF', 'PDF'), (b'%PDF-1.7\ncorrupt', 'corrupt'),
    (pdf_bytes(password='secret'), 'encrypted'), (pdf_bytes(pages=('',)), 'OCR')])
async def test_unsupported_documents_fail_explicitly_before_creating_memory(memory, raw, message):
    from scone_memory.ingestion import ingest_pdf
    with pytest.raises(InvalidInput, match=message):
        await ingest_pdf(memory, 'alpha', raw)
    assert (await memory.documents.counts('alpha')).episodes == 0


@pytest.mark.parametrize(('options', 'message'), [({'max_input_bytes': 100}, 'input'),
    ({'max_pages': 1}, 'page'), ({'max_text_bytes': 8}, 'text')])
async def test_parser_limits_refuse_incomplete_ingestion(memory, options, message):
    from scone_memory.ingestion import ingest_pdf, PdfLimits
    with pytest.raises(InvalidInput, match=message):
        await ingest_pdf(memory, 'alpha', pdf_bytes(), limits=PdfLimits(**options))
    assert (await memory.documents.counts('alpha')).episodes == 0


async def test_provenance_refuses_foreign_scope_and_unlinked_or_modified_evidence(memory):
    from scone_memory.ingestion import ingest_pdf, pdf_provenance
    result = await ingest_pdf(memory, 'alpha', pdf_bytes())
    with pytest.raises(NotFound):
        await pdf_provenance(memory, 'beta', result.added.episode_id)
    episode = await memory.episode('alpha', result.added.episode_id)
    forged = await memory.remember('alpha', 'Forged text', metadata=episode.metadata)
    with pytest.raises(InvalidInput, match='linked'):
        await pdf_provenance(memory, 'alpha', forged.episode_id)
    forged = await memory.remember('alpha', 'Other forged text', metadata=episode.metadata,
        attachment_ids=[a.attachment_id for a in episode.attachments])
    with pytest.raises(InvalidInput, match='text'):
        await pdf_provenance(memory, 'alpha', forged.episode_id)
    with pytest.raises(InvalidInput, match='span'):
        await pdf_provenance(memory, 'alpha', episode.episode_id, start=0, end=999999)


async def test_mixed_text_and_empty_pages_report_partial_extraction(memory):
    from scone_memory.ingestion import ingest_pdf
    result = await ingest_pdf(memory, 'alpha', pdf_bytes(pages=('Polaris.', '', 'Juniper.')))
    assert result.empty_pages == (2,)
    episode = await memory.episode('alpha', result.added.episode_id)
    assert episode.metadata['pdf_coverage'] == 'partial'
    manifest = json.loads((await memory.attachment('alpha', result.manifest.attachment_id))[1])
    assert manifest['offset_unit'] == 'extracted_text_utf8_bytes'
    assert manifest['pages'][2]['number'] == 3


async def test_deadline_kills_and_reaps_the_real_parser_process(monkeypatch):
    import scone_memory.ingestion.pdf as module
    created = []
    original = module.asyncio.create_subprocess_exec
    async def capture(*args, **kwargs):
        process = await original(*args, **kwargs)
        created.append(process)
        return process
    monkeypatch.setattr(module.asyncio, 'create_subprocess_exec', capture)
    with pytest.raises(InvalidInput, match='wall time'):
        await module.PypdfParser().parse(pdf_bytes(), module.PdfLimits(timeout_seconds=0.000001))
    assert created and created[0].returncode is not None


async def test_cancelling_ingestion_reaps_parser_and_does_not_write(memory, monkeypatch):
    import asyncio
    import scone_memory.ingestion.pdf as module
    from scone_memory.ingestion import ingest_pdf
    created = []
    ready = asyncio.Event()
    original = module.asyncio.create_subprocess_exec
    async def capture(*args, **kwargs):
        process = await original(*args, **kwargs)
        created.append(process)
        ready.set()
        return process
    monkeypatch.setattr(module.asyncio, 'create_subprocess_exec', capture)
    task = asyncio.create_task(ingest_pdf(memory, 'alpha', pdf_bytes()))
    await ready.wait()
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert created[0].returncode is not None
    assert (await memory.documents.counts('alpha')).episodes == 0


async def test_missing_optional_parser_names_the_extra(monkeypatch):
    import scone_memory.ingestion.pdf as module
    monkeypatch.setattr(module, 'find_spec', lambda name: None)
    with pytest.raises(InvalidInput, match=r'scone-memory\[pdf\]'):
        await module.PypdfParser().parse(pdf_bytes())


async def test_real_image_only_pdf_requires_ocr(memory):
    from PIL import Image
    from scone_memory.ingestion import ingest_pdf
    output = BytesIO()
    Image.new('RGB', (32, 32), color='white').save(output, format='PDF')
    with pytest.raises(InvalidInput, match='OCR'):
        await ingest_pdf(memory, 'alpha', output.getvalue())
    assert (await memory.documents.counts('alpha')).episodes == 0


async def test_failed_link_is_not_provenance_and_retry_repairs_it(memory, monkeypatch):
    from scone_memory.ingestion import ingest_pdf, pdf_provenance
    original = memory.blobs.link
    calls = 0
    async def fail_second(space, attachment_id, episode_id):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise OSError('injected storage failure')
        await original(space, attachment_id, episode_id)
    monkeypatch.setattr(memory.blobs, 'link', fail_second)
    with pytest.raises(OSError, match='storage failure'):
        await ingest_pdf(memory, 'alpha', pdf_bytes())
    [episode] = await memory.episodes('alpha', {'document_format': 'pdf'})
    with pytest.raises(InvalidInput, match='linked'):
        await pdf_provenance(memory, 'alpha', episode.episode_id)
    monkeypatch.setattr(memory.blobs, 'link', original)
    result = await ingest_pdf(memory, 'alpha', pdf_bytes())
    assert result.added.deduplicated and result.added.episode_id == episode.episode_id
    assert len((await pdf_provenance(memory, 'alpha', episode.episode_id)).pages) == 2


async def test_page_rotation_layout_and_unicode_spans_are_retained(memory):
    from scone_memory.ingestion import ingest_pdf, pdf_provenance
    reader = pypdf.PdfReader(BytesIO(pdf_bytes()))
    reader.pages[0].rotate(90)
    writer = pypdf.PdfWriter()
    writer.append_pages_from_reader(reader)
    target = BytesIO()
    writer.write(target)
    result = await ingest_pdf(memory, 'alpha', target.getvalue())
    episode = await memory.episode('alpha', result.added.episode_id)
    provenance = await pdf_provenance(memory, 'alpha', episode.episode_id)
    assert provenance.pages[0].rotation == 90
    assert b'Caf\xc3\xa9' in episode.content.encode()
    inside_character = episode.content.encode().index(b'\xc3\xa9') + 1
    with pytest.raises(InvalidInput, match='UTF-8'):
        await pdf_provenance(memory, 'alpha', episode.episode_id, start=inside_character)


async def test_forgotten_document_cannot_resolve_provenance(memory):
    from scone_memory.ingestion import ingest_pdf, pdf_provenance
    result = await ingest_pdf(memory, 'alpha', pdf_bytes())
    await memory.forget('alpha', result.added.episode_id)
    with pytest.raises(NotFound):
        await pdf_provenance(memory, 'alpha', result.added.episode_id)


async def test_pdf_dimensions_use_physical_points_when_user_unit_is_scaled(memory):
    from pypdf.generic import NumberObject
    from scone_memory.ingestion import ingest_pdf, pdf_provenance
    reader = pypdf.PdfReader(BytesIO(pdf_bytes()))
    reader.pages[0][NameObject('/UserUnit')] = NumberObject(2)
    writer = pypdf.PdfWriter()
    writer.append_pages_from_reader(reader)
    target = BytesIO()
    writer.write(target)
    result = await ingest_pdf(memory, 'alpha', target.getvalue())
    provenance = await pdf_provenance(memory, 'alpha', result.added.episode_id)
    assert provenance.pages[0].width_points == 1224
    assert provenance.pages[0].height_points == 1584


async def test_incompatible_retained_original_type_is_refused_before_episode_write(memory):
    from scone_memory.ingestion import ingest_pdf
    raw = pdf_bytes()
    await memory.attach('alpha', raw, media_type='text/plain')
    with pytest.raises(InvalidInput, match='media type'):
        await ingest_pdf(memory, 'alpha', raw)
    assert (await memory.documents.counts('alpha')).episodes == 0


async def test_incompatible_retained_manifest_type_is_refused_before_episode_write(memory):
    from scone_memory.ingestion import ingest_pdf
    raw = pdf_bytes()
    result = await ingest_pdf(memory, 'alpha', raw)
    _, manifest = await memory.attachment('alpha', result.manifest.attachment_id)
    await memory.attach('beta', manifest, media_type='text/plain')
    with pytest.raises(InvalidInput, match='media type'):
        await ingest_pdf(memory, 'beta', raw)
    assert (await memory.documents.counts('beta')).episodes == 0


async def test_originals_and_provenance_survive_fresh_engine_and_store_instances(tmp_path):
    from scone_memory.backends import SqliteDocumentStore, SqliteVectorIndex
    from scone_memory.backends.blobs import FileBlobStore
    from scone_memory.ingestion import ingest_pdf, pdf_provenance
    async def open_engine():
        path = tmp_path / 'memory.db'
        return await MemoryEngine(SqliteDocumentStore(path), SqliteVectorIndex(path), HashEmbedder(),
            blobs=FileBlobStore(tmp_path / 'originals')).open()
    raw = pdf_bytes()
    first = await open_engine()
    try:
        result = await ingest_pdf(first, 'alpha', raw)
    finally:
        await first.close()
    second = await open_engine()
    try:
        provenance = await pdf_provenance(second, 'alpha', result.added.episode_id)
        assert [page.number for page in provenance.pages] == [1, 2]
        assert (await second.attachment('alpha', provenance.original.attachment_id))[1] == raw
        recall = await second.recall('alpha', 'Polaris')
        assert any(item.episode_id == result.added.episode_id for item in recall.items)
    finally:
        await second.close()


async def test_recalled_chunk_id_resolves_its_own_pages_and_rejects_other_episode(memory):
    from scone_memory.ingestion import ingest_pdf, pdf_provenance
    memory.chunk_target = 20
    result = await ingest_pdf(memory, 'alpha', pdf_bytes())
    recalled = await memory.recall('alpha', 'Friday', limit=10)
    item = next(item for item in recalled.items if 'Friday' in item.text)
    provenance = await pdf_provenance(memory, 'alpha', result.added.episode_id, chunk_id=item.chunk_id)
    # The chunker merges this short tail, so this hit spans both pages.
    assert 'Polaris' in item.text and 'Friday' in item.text
    assert [page.number for page in provenance.pages] == [1, 2]
    other = await ingest_pdf(memory, 'alpha', pdf_bytes(title='other'))
    with pytest.raises(InvalidInput, match='chunk'):
        await pdf_provenance(memory, 'alpha', other.added.episode_id, chunk_id=item.chunk_id)
