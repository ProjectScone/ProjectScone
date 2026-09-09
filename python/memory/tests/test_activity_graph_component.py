"""Recorded graph construction uses document/event ports without an engine."""
import asyncio

import pytest

from scone_memory import InMemoryDocumentStore, InMemoryEventLog
from scone_memory.core.ports import NewChunk, NewEpisode, NewEvent, NewFact

STAMP = '2026-01-01T00:00:00.000Z'


async def source(documents):
    episode = await documents.insert_episode(NewEpisode('alpha', 'note', 'Juniper uses Polaris.',
        'source', STAMP, STAMP))
    [chunk] = await documents.insert_chunks([NewChunk(episode.episode_id, 'alpha', 0,
        0, len(episode.content.encode()), episode.content, STAMP)])
    fact = await documents.insert_fact(NewFact('alpha', 'juniper', 'uses', 'Polaris', STAMP,
        source_episode_id=episode.episode_id))
    return episode, chunk, fact


async def test_builder_preserves_capture_source_retrieval_and_feedback_edges():
    from scone_memory.retrieval.activity_graph import build_activity_graph
    documents, events = InMemoryDocumentStore(), InMemoryEventLog()
    episode, chunk, fact = await source(documents)
    capture = await events.append(NewEvent(space='alpha', kind='agent', ts=STAMP, payload={
        'agent':'codex', 'session_id':'recorded', 'event':'prompt', 'episode_id':episode.episode_id}))
    recall = await events.append(NewEvent(space='alpha', kind='recall', ts=STAMP, payload={'query':'Juniper?',
        'items':[{'chunk_id':chunk.chunk_id, 'lanes':{'vector':1}, 'score':0.7}],
        'fact_ids':[fact.fact_id]}))
    feedback = await events.append(NewEvent(space='alpha', kind='feedback', ts=STAMP, payload={'recall_event_id':recall.event_id,
        'chunk_id':chunk.chunk_id, 'useful':True}))
    graph = await build_activity_graph(documents, events, 'alpha')
    edges = {(e.source, e.target, e.kind) for e in graph.edges}
    assert edges == {
        ('session:codex:recorded', f'turn:{capture.event_id}', 'has'),
        (f'turn:{capture.event_id}', f'episode:{episode.episode_id}', 'captured_as'),
        (f'episode:{episode.episode_id}', f'chunk:{chunk.chunk_id}', 'chunked_into'),
        (f'episode:{episode.episode_id}', f'claim:{fact.fact_id}', 'source_of'),
        (f'recall:{recall.event_id}', f'chunk:{chunk.chunk_id}', 'returned'),
        (f'recall:{recall.event_id}', f'claim:{fact.fact_id}', 'held'),
        (f'feedback:{feedback.event_id}', f'recall:{recall.event_id}', 'judged'),
        (f'feedback:{feedback.event_id}', f'chunk:{chunk.chunk_id}', 'judged'),
    }
    [returned] = [edge for edge in graph.edges if edge.kind == 'returned']
    assert returned.label == 'retrieval evidence: vector rank 1'
    assert returned.data['score'] == 0.7
    assert not graph.truncated and graph.provenance_omitted == graph.provenance_missing == 0


async def test_builder_hydrates_old_capture_without_guessing_a_session():
    from scone_memory.retrieval.activity_graph import build_activity_graph
    documents, events = InMemoryDocumentStore(), InMemoryEventLog()
    episode, _, _ = await source(documents)
    await events.append(NewEvent(space='alpha', kind='agent', ts=STAMP, payload={'agent':'codex', 'session_id':'old',
        'event':'prompt', 'episode_id':episode.episode_id}))
    await events.append(NewEvent(space='beta', kind='agent', ts=STAMP, payload={'agent':'codex', 'session_id':'foreign',
        'event':'prompt', 'episode_id':episode.episode_id}))
    await events.append(NewEvent(space='alpha', kind='agent', ts=STAMP, payload={'agent':'codex', 'session_id':'new', 'event':'prompt'}))
    graph = await build_activity_graph(documents, events, 'alpha', session_id='old', limit=1)
    assert 'session:codex:old' in graph.nodes
    assert 'session:codex:new' not in graph.nodes and 'session:codex:foreign' not in graph.nodes
    assert graph.truncated


async def test_no_event_log_returns_empty_graph_without_document_reads():
    from scone_memory.retrieval.activity_graph import build_activity_graph
    class UnavailableDocuments(InMemoryDocumentStore):
        async def list_facts(self, *args, **kwargs):
            pytest.fail('no storage read when activity log absent')
    graph = await build_activity_graph(UnavailableDocuments(), None, 'alpha')
    assert graph.as_dict() == {'nodes':[], 'edges':[], 'truncated':False,
        'provenance_omitted':0, 'provenance_missing':0, 'counts':{}}


async def test_cancel_during_source_hydration_propagates_without_mutation():
    from scone_memory.retrieval.activity_graph import build_activity_graph
    entered = asyncio.Event()
    class WaitingDocuments(InMemoryDocumentStore):
        async def get_episode(self, *args):
            entered.set()
            await asyncio.Event().wait()
    documents, events = WaitingDocuments(), InMemoryEventLog()
    await source(documents)
    before = await documents.counts('alpha')
    task = asyncio.create_task(build_activity_graph(documents, events, 'alpha'))
    await asyncio.wait_for(entered.wait(), 2)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert await documents.counts('alpha') == before
