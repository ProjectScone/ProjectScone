"""Declared table evidence improves embedding inputs without rewriting citations."""
import json

import pytest

from scone_memory import HashEmbedder, InMemoryDocumentStore, InMemoryVectorIndex, MemoryEngine
from scone_memory.core.errors import InvalidInput
from scone_memory.ingestion.files import ingest_document, document_provenance


class ObservedEmbedder(HashEmbedder):
    def __init__(self):
        super().__init__()
        self.inputs = []

    async def embed(self, texts):
        self.inputs.extend(texts)
        return await super().embed(texts)


def table():
    return ('<table><tr><th>Region</th><th>Revenue</th><th>Cost</th></tr>' +
        ''.join(f'<tr><th scope="row">District {i} Café</th><td>{100+i}</td><td>{40+i}</td></tr>' for i in range(30)) +
        '</table>').encode()


async def open_memory(enabled=True, **kwargs):
    model = ObservedEmbedder()
    memory = await MemoryEngine(InMemoryDocumentStore(), InMemoryVectorIndex(), model,
        table_context_embeddings=enabled, chunk_target=120, **kwargs).open()
    return memory, model


async def test_declared_headers_reach_embeddings_while_chunks_and_provenance_stay_exact():
    memory, model = await open_memory()
    try:
        saved = await ingest_document(memory, 'alpha', table(), filename='finances.html')
        episode = await memory.episode('alpha', saved.added.episode_id)
        chunks = await memory.documents.chunks_of('alpha', saved.added.episode_id)
        assert len(model.inputs) == len(chunks) > 2
        assert all(episode.content.encode()[c.start:c.end] == c.text.encode() for c in chunks)
        assert any('Revenue' in text and 'District 20 Café' in text for text in model.inputs)
        for chunk, text in zip(chunks, model.inputs):
            assert text.endswith(chunk.text)
        evidence = await document_provenance(memory, 'alpha', saved.added.episode_id)
        assert '\n\n'.join(segment.text for segment in evidence.segments) == episode.content
        assert 'Table context' not in episode.content
    finally:
        await memory.close()


async def test_disabled_and_plain_documents_keep_the_original_embedding_inputs():
    for enabled in (False, True):
        memory, model = await open_memory(enabled)
        try:
            await ingest_document(memory, 'alpha', b'Ordinary document without table evidence.', filename='notes.txt')
            assert model.inputs == ['Ordinary document without table evidence.']
            if not enabled:
                model.inputs.clear()
                saved = await ingest_document(memory, 'alpha', table(), filename='finances.html')
                chunks = await memory.documents.chunks_of('alpha', saved.added.episode_id)
                assert model.inputs == [chunk.text for chunk in chunks]
        finally:
            await memory.close()


async def test_context_policy_is_immutable_and_has_a_distinct_vector_writer():
    from scone_memory.memory.vector_identity import writer_of
    enabled, _ = await open_memory()
    disabled, _ = await open_memory(False)
    try:
        assert writer_of(enabled) != writer_of(disabled)
        assert writer_of(disabled) == f'{disabled.embedder.id};contextual=0'
        with pytest.raises(AttributeError):
            enabled.table_context_embeddings = False
    finally:
        await enabled.close()
        await disabled.close()


@pytest.mark.parametrize('value', [1, 'yes', None])
def test_context_policy_requires_explicit_boolean(value):
    with pytest.raises(InvalidInput, match='table_context_embeddings'):
        MemoryEngine(InMemoryDocumentStore(), InMemoryVectorIndex(), HashEmbedder(), table_context_embeddings=value)


async def test_recovery_and_explicit_rebuild_reproduce_the_same_embedding_inputs():
    memory, model = await open_memory()
    try:
        saved = await ingest_document(memory, 'alpha', table(), filename='finances.html')
        original = list(model.inputs)
        episode = await memory.episode('alpha', saved.added.episode_id)
        chunks = await memory.documents.chunks_of('alpha', saved.added.episode_id)
        await memory.documents.mark_inflight('alpha', episode.content_hash)
        await memory.vectors.delete([chunk.chunk_id for chunk in chunks])
        model.inputs.clear()
        assert (await memory.recover()).completed == 1
        assert model.inputs == original
        model.inputs.clear()
        report = await memory.reembed_vectors()
        assert report.chunks == len(chunks)
        assert model.inputs == original
    finally:
        await memory.close()


@pytest.mark.parametrize('damage', ['original', 'manifest', 'missing'])
async def test_source_damage_refuses_context_before_any_embedding(monkeypatch, damage):
    from scone_memory.core.errors import NotFound
    memory, model = await open_memory()
    get = memory.blobs.get

    async def damaged(space, identifier):
        attachment, raw = await get(space, identifier)
        if attachment.media_type == 'application/json':
            if damage == 'missing':
                raise NotFound('missing source')
            if damage == 'manifest':
                return attachment, raw + b' '
        elif damage == 'original':
            return attachment, raw + b' '
        return attachment, raw

    monkeypatch.setattr(memory.blobs, 'get', damaged)
    try:
        with pytest.raises((InvalidInput, NotFound)):
            await ingest_document(memory, 'alpha', table(), filename='finances.html')
        assert model.inputs == []
        assert (await memory.documents.counts('alpha')).episodes == 0
    finally:
        await memory.close()


async def test_context_refuses_wrong_content_and_cross_space_manifest():
    from dataclasses import replace
    from scone_memory.core.errors import NotFound
    from scone_memory.ingestion.batch import validated_record
    from scone_memory.ingestion.records import Record
    from scone_memory.ingestion.table_context import embedding_inputs
    memory, model = await open_memory()
    try:
        saved = await ingest_document(memory, 'alpha', table(), filename='finances.html')
        episode = await memory.episode('alpha', saved.added.episode_id)
        new = validated_record('alpha', Record(episode.content, kind='file', source=episode.source,
                                               metadata=episode.metadata), episode.created_at)
        model.inputs.clear()
        with pytest.raises(InvalidInput, match='episode'):
            await embedding_inputs(replace(new, content='Different text'), [(0, 4)], blobs=memory.blobs)
        with pytest.raises(NotFound):
            await embedding_inputs(replace(new, space='beta'), [(0, 4)], blobs=memory.blobs)
        assert model.inputs == []
    finally:
        await memory.close()


async def test_context_budget_refuses_excess_before_provider_calls(monkeypatch):
    from scone_memory.ingestion import table_context
    memory, model = await open_memory()
    monkeypatch.setattr(table_context, 'MAX_CONTEXT_BYTES', 20)
    try:
        with pytest.raises(InvalidInput, match='byte limit'):
            await ingest_document(memory, 'alpha', long_cell_table(), filename='finances.html')
        assert model.inputs == []
    finally:
        await memory.close()


async def test_process_death_reuses_contextual_embedding_batches_without_read_replay(tmp_path):
    import asyncio
    import os
    from pathlib import Path
    import sys
    import scone_memory

    raw = ('<table><tr><th>Region</th><th>Revenue</th></tr>' + ''.join(
        f'<tr><th scope="row">District {i} Café</th><td>{100+i} '+ 'long cell detail ' * 20 + '</td></tr>' for i in range(400)) + '</table>').encode()
    (tmp_path / 'input.html').write_bytes(raw)
    worker = Path(__file__).parent / 'fixtures' / 'table_context_worker.py'
    env = {key: value for key, value in os.environ.items() if key in {'PATH', 'TMPDIR', 'LANG'}}
    env['PYTHONPATH'] = str(Path(scone_memory.__file__).resolve().parent.parent)
    child = None

    async def launch(phase):
        return await asyncio.create_subprocess_exec(sys.executable, str(worker), str(tmp_path), phase,
            env=env, stdin=asyncio.subprocess.DEVNULL, stdout=asyncio.subprocess.DEVNULL, stderr=asyncio.subprocess.PIPE)

    try:
        child = await launch('hold')

        async def entered():
            while not (tmp_path / 'entered').exists():
                if child.returncode is not None:
                    raise AssertionError((await child.stderr.read()).decode())
                await asyncio.sleep(0.02)

        await asyncio.wait_for(entered(), 20)
        child.kill()
        await asyncio.wait_for(child.wait(), 10)
        before = (tmp_path / 'calls.jsonl').read_text()
        child = await launch('inspect')
        await asyncio.wait_for(child.wait(), 20)
        assert child.returncode == 0, (await child.stderr.read()).decode()
        assert (tmp_path / 'calls.jsonl').read_text() == before
        child = await launch('resume')
        await asyncio.wait_for(child.wait(), 20)
        assert child.returncode == 0, (await child.stderr.read()).decode()
        result = json.loads((tmp_path / 'result.json').read_text())
        calls = [json.loads(line) for line in (tmp_path / 'calls.jsonl').read_text().splitlines()]
        assert calls[0]['count'] == 64 and all(call['context'] for call in calls)
        assert sum(call['count'] for call in calls if call['phase'] == 'resume') == result['chunks'] - 64
        assert b'Table context' not in (tmp_path / 'jobs.db').read_bytes()
    finally:
        if child is not None and child.returncode is None:
            child.kill()
            await child.wait()


def long_cell_table():
    return ('<table><tr><th>Regional revenue</th></tr><tr><td>' +
            'Shared long description with source detail. ' * 80 +
            'violet launch milestone</td></tr></table>').encode()


async def test_long_cell_chunks_restore_missing_headers_without_duplicating_present_context():
    memory, model = await open_memory()
    try:
        saved = await ingest_document(memory, 'alpha', long_cell_table(), filename='long.html')
        chunks = await memory.documents.chunks_of('alpha', saved.added.episode_id)
        assert len(chunks) > 5
        for chunk, embedded in zip(chunks, model.inputs):
            if 'Regional revenue' in chunk.text:
                assert embedded == chunk.text
            else:
                assert 'Regional revenue' in embedded
                assert embedded.endswith(chunk.text)
        assert 'Table context:' in model.inputs[-1]
    finally:
        await memory.close()


async def test_present_declared_context_is_not_embedded_twice():
    memory, model = await open_memory()
    try:
        await ingest_document(memory, 'alpha', b'<table><tr><th>Revenue</th></tr><tr><td>42</td></tr></table>', filename='short.html')
        assert all('Table context:' not in text for text in model.inputs)
    finally:
        await memory.close()


@pytest.mark.parametrize('operation', ['recover', 'rebuild'])
async def test_context_refuses_stored_chunk_text_that_contradicts_its_span(monkeypatch, operation):
    memory, model = await open_memory()
    try:
        saved = await ingest_document(memory, 'alpha', long_cell_table(), filename='long.html')
        episode = await memory.episode('alpha', saved.added.episode_id)
        get = memory.documents.chunks_of

        async def corrupt(space, episode_id):
            chunks = await get(space, episode_id)
            return [chunks[0].model_copy(update={'text': 'FORGED CHUNK TEXT'}), *chunks[1:]]

        monkeypatch.setattr(memory.documents, 'chunks_of', corrupt)
        model.inputs.clear()
        if operation == 'recover':
            await memory.documents.mark_inflight('alpha', episode.content_hash)
        with pytest.raises(InvalidInput, match='source span'):
            await (memory.recover() if operation == 'recover' else memory.reembed_vectors())
        assert model.inputs == []
    finally:
        await memory.close()


@pytest.mark.parametrize('suffix', ['docx', 'xlsx'])
async def test_office_cells_restore_declared_headers_and_word_spanning_row_context(suffix):
    long_text = 'Shared cell text with details. ' * 80 + 'violet milestone'
    if suffix == 'docx':
        from .test_word_tables import cell, document, row
        raw = document(row(cell('Region') + cell('Revenue'), '<w:tblHeader/>') +
            row(cell('West', '<w:vMerge w:val="restart"/>') + cell('10')) +
            row(cell('', '<w:vMerge/>') + cell(long_text)))
    else:
        from .test_xlsx_tables import workbook, cell, row
        raw = workbook(row(1, cell('A1', 'Revenue')) + row(2, cell('A2', long_text)),
            'id="1" name="RevenueTable" ref="A1:A2"><tableColumns count="1">'
            '<tableColumn id="1" name="Revenue"/></tableColumns>')
    memory, model = await open_memory()
    try:
        saved = await ingest_document(memory, 'alpha', raw, filename='source.' + suffix)
        chunks = await memory.documents.chunks_of('alpha', saved.added.episode_id)
        found = [(chunk, embedded) for chunk, embedded in zip(chunks, model.inputs) if 'violet milestone' in chunk.text]
        assert len(found) == 1
        chunk, embedded = found[0]
        assert embedded.startswith('Table context: ') and 'Revenue' in embedded
        assert embedded.endswith(chunk.text) and 'Revenue' not in chunk.text
        if suffix == 'docx':
            assert 'West' in embedded and 'West' not in chunk.text
        evidence = await document_provenance(memory, 'alpha', saved.added.episode_id)
        assert all(header.text == 'Revenue' for segment in evidence.segments for c in segment.table_cells
                   if 'violet milestone' in c.text for header in c.headers)
    finally:
        await memory.close()


@pytest.mark.parametrize('label, word', [('US', 'RUSSIA'), ('US', 'US\u0301'), ('e', 'e\u0301')])
async def test_header_name_inside_an_unrelated_word_does_not_count_as_present_context(label, word):
    memory, model = await open_memory()
    try:
        raw = ('<table><tr><th>' + label + '</th></tr><tr><td>' + (word + ' ') * 200 + '</td></tr></table>').encode()
        saved = await ingest_document(memory, 'alpha', raw, filename='locations.html')
        chunks = await memory.documents.chunks_of('alpha', saved.added.episode_id)
        assert label in chunks[-1].text and label + ':' not in chunks[-1].text
        assert model.inputs[-1].startswith('Table context: ' + label + '\n\n')
    finally:
        await memory.close()
