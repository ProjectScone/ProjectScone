"""Combined ingestion paths preserve checkpoint, code and retirement behavior."""
import pytest
from scone_memory import HashEmbedder, InMemoryDocumentStore, InMemoryVectorIndex, MemoryEngine
from scone_memory.ingestion.records import Record
from scone_memory.observability.events import InMemoryEventLog

class Receipts:
    def __init__(self):
        self.saved = {}
    def get(self, key):
        return self.saved.get(key)
    def put(self, key, value):
        self.saved[key] = value

SOURCE = 'def plan(question):\n    return tidy(question)\n\ndef tidy(question):\n    return question.strip()\n'

async def test_partial_code_batch_preserves_checkpoint_and_extracted_claims():
    memory = await MemoryEngine(InMemoryDocumentStore(), InMemoryVectorIndex(), HashEmbedder(), code_graph=True).open()
    receipts = Receipts()
    try:
        outcomes = await memory.remember_many('alpha', [Record('invalid source', source=17),
            Record(SOURCE, kind='file', source='planner.py')], partial=True, embedding_checkpoint=receipts)
        assert [item.outcome for item in outcomes] == ['failed', 'accepted']
        assert receipts.saved
        claims = await memory.documents.list_facts('alpha', include_closed=True)
        assert any(f.predicate == 'calls' and f.source_episode_id == outcomes[1].episode_id for f in claims)
    finally:
        await memory.close()

async def test_keyed_replacement_still_extracts_the_new_code_graph():
    memory = await MemoryEngine(InMemoryDocumentStore(), InMemoryVectorIndex(), HashEmbedder(), code_graph=True).open()
    try:
        first = await memory.replace('alpha', Record(SOURCE, kind='file', source='planner.py', dedup_key='planner'))
        second = await memory.replace('alpha', Record(SOURCE.replace('tidy', 'clean'), kind='file', source='planner.py', dedup_key='planner'))
        claims = await memory.documents.list_facts('alpha', include_closed=True)
        assert any(f.predicate == 'calls' and f.source_episode_id == first.added.episode_id for f in claims)
        assert any(f.predicate == 'calls' and f.source_episode_id == second.added.episode_id and f.object.endswith(':clean') for f in claims)
        assert await memory.documents.get_episode('alpha', first.added.episode_id) is None
    finally:
        await memory.close()

async def test_retirement_keeps_affirmation_impact_in_its_durable_event():
    events = InMemoryEventLog()
    memory = await MemoryEngine(InMemoryDocumentStore(), InMemoryVectorIndex(), HashEmbedder(), events=events).open()
    try:
        source = await memory.remember('alpha', 'Alice still works at Acme.')
        await memory.assert_fact('alpha', 'Alice', 'works_at', 'Acme', valid_from='2020-01-01T00:00:00Z')
        await memory.assert_fact('alpha', 'Alice', 'works_at', 'Acme', valid_from='2024-01-01T00:00:00Z', source_episode_id=source.episode_id)
        receipt = await memory.forget('alpha', source.episode_id)
        assert len(receipt.affirmations_citing) == 1
        records = await events.query('alpha', limit=100)
        forgets = [r for r in records if r.kind == 'forget']
        assert len(forgets) == 1 and forgets[0].payload['affirmations_citing'] == 1
        assert await memory.documents.retirement('alpha', source.episode_id) is None
    finally:
        await memory.close()

async def test_recovery_accepts_a_premerge_forget_event_without_affirmation_count():
    class LostCleanup(InMemoryDocumentStore):
        fail_cleanup = True
        async def clear_retirement(self, space, episode_id):
            if self.fail_cleanup:
                self.fail_cleanup = False
                raise RuntimeError('cleanup acknowledgement lost')
            await super().clear_retirement(space, episode_id)
    documents, events = LostCleanup(), InMemoryEventLog()
    memory = await MemoryEngine(documents, InMemoryVectorIndex(), HashEmbedder(), events=events).open()
    try:
        source = await memory.remember('alpha', 'Source stored before the merge.')
        with pytest.raises(RuntimeError, match='acknowledgement'):
            await memory.forget('alpha', source.episode_id)
        [original_event] = await events.query('alpha', kind='forget')
        original_event.payload.pop('affirmations_citing', None)
        report = await memory.recover()
        assert not report.retirements_pending
        assert await documents.retirement('alpha', source.episode_id) is None
        assert len(await events.query('alpha', kind='forget')) == 1
    finally:
        await memory.close()
