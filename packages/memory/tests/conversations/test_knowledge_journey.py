"""A retained file's knowledge journey uses no generation or remote service.

Historical conversation replies remain retained conversation records. Forgetting
source evidence does not promise to erase those replies; file-only recall must
nevertheless abstain rather than treating a prior reply as independent support.
"""
import pytest

from scone_memory import HashEmbedder, MemoryEngine
from scone_memory.backends import SqliteDocumentStore, SqliteVectorIndex
from scone_memory.backends.blobs import FileBlobStore
from scone_memory.core.errors import NotFound
from scone_memory.ingestion import document_provenance, ingest_document
from scone_memory.realtime.evidence_answer import EvidenceSelection
from scone_memory.realtime.text import TextConversation


class FirstCheckedCard:
    async def select(self, question, cards):
        card = next(iter(cards), None)
        return EvidenceSelection(card_ids=(card.id,) if card else ())


def forbidden_generation():
    raise AssertionError('source-backed journey must not instantiate a generation model')


async def open_memory(path):
    return await MemoryEngine(SqliteDocumentStore(path / 'memory.db'),
        SqliteVectorIndex(path / 'memory.db'), HashEmbedder(),
        blobs=FileBlobStore(path / 'blobs')).open()


async def answer(memory, session):
    conversation = TextConversation(memory, 'alpha', session, forbidden_generation,
        evidence_selector=FirstCheckedCard(), evidence_answer_policy='required', kind='file')
    try:
        return await conversation.reply('Which star does Juniper use for calibration?')
    finally:
        await conversation.close()


async def assert_citation(memory, result, episode_id, expected):
    receipt = result['evidence_answer']
    assert receipt['status'] == 'selected' and receipt['source_status'] == 'retained'
    assert receipt['verified_accuracy'] is False
    assert expected in result['text']
    chunks = {f'chunk:{chunk.chunk_id}': chunk for chunk in
        await memory.documents.chunks_of('alpha', episode_id)}
    assert receipt['evidence_ids']
    for identifier in receipt['evidence_ids']:
        if identifier.startswith('chunk:'):
            assert identifier in chunks and expected in chunks[identifier].text
        else:
            assert identifier.startswith('fact:')
            claim = await memory.fact('alpha', int(identifier.split(':')[1]))
            assert claim.source_episode_id == episode_id and claim.quote == expected
    evidence = await document_provenance(memory, 'alpha', episode_id)
    assert any(expected in segment.text and segment.locator for segment in evidence.segments)
    return evidence


async def test_file_answer_correction_forgetting_and_reopen_keep_source_boundaries(tmp_path):
    original_text = 'Juniper uses Polaris for calibration.'
    corrected_text = 'Juniper uses Vega for calibration.'
    memory = await open_memory(tmp_path)
    try:
        imported = await ingest_document(memory, 'alpha', original_text.encode(), filename='calibration.txt')
        original_id = imported.added.episode_id
        first = await answer(memory, 'before-correction')
        evidence = await assert_citation(memory, first, original_id, original_text)
        assert (await memory.attachment('alpha', evidence.original.attachment_id))[1] == original_text.encode()
        old_claim = await memory.assert_fact('alpha', 'Juniper', 'uses', 'Polaris',
            source_episode_id=original_id, quote=original_text, valid_from='2026-01-01T00:00:00Z')

        corrected = await ingest_document(memory, 'alpha', corrected_text.encode(), filename='calibration-corrected.txt')
        corrected_id = corrected.added.episode_id
        replacement = await memory.assert_fact('alpha', 'Juniper', 'uses', 'Vega',
            source_episode_id=corrected_id, quote=corrected_text, valid_from='2026-02-01T00:00:00Z')
        assert (await memory.fact('alpha', old_claim.fact_id)).superseded_by == replacement.fact_id
        retired = await memory.forget('alpha', original_id, with_claims='exclude')
        assert old_claim.fact_id in retired.claims_excluded
        await assert_citation(memory, await answer(memory, 'after-correction'), corrected_id, corrected_text)
        await ingest_document(memory, 'beta', b'Juniper uses PRIVATE_OTHER_SPACE for calibration.', filename='private.txt')
    finally:
        await memory.close()

    memory = await open_memory(tmp_path)
    try:
        with pytest.raises(NotFound):
            await memory.episode('alpha', original_id)
        assert (await memory.episode('alpha', corrected_id)).episode_id == corrected_id
        assert (await memory.fact('alpha', old_claim.fact_id)).excluded
        resumed = await answer(memory, 'after-reopen')
        await assert_citation(memory, resumed, corrected_id, corrected_text)
        assert 'Polaris' not in resumed['text'] and 'PRIVATE_OTHER_SPACE' not in resumed['text']
        removed = await memory.forget('alpha', corrected_id, with_claims='exclude')
        assert replacement.fact_id in removed.claims_excluded
        history = await memory.episodes('alpha', {'session_id': 'before-correction'})
        assert any(original_text in episode.content for episode in history)
        abstained = await answer(memory, 'after-forget')
        assert abstained['evidence_answer']['status'] == 'no_selection'
        assert abstained['evidence_answer']['source_status'] == 'none'
        assert abstained['evidence_answer']['evidence_ids'] == []
        assert 'Polaris' not in abstained['text'] and 'Vega' not in abstained['text']
    finally:
        await memory.close()

    memory = await open_memory(tmp_path)
    try:
        recall = await memory.recall('alpha', 'Juniper calibration', kind='file')
        assert recall.items == [] and recall.facts == []
        assert (await memory.fact('alpha', replacement.fact_id)).excluded
        abstained = await answer(memory, 'forgotten-after-reopen')
        assert abstained['evidence_answer']['source_status'] == 'none'
        assert abstained['evidence_answer']['evidence_ids'] == []
        assert (await memory.recall('beta', 'Juniper calibration', kind='file')).items
    finally:
        await memory.close()
