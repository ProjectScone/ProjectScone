"""Conversation-facing metadata for a completed native tool turn."""
from __future__ import annotations

import hashlib
import json
from uuid import uuid4

from ..agents.evidence_loop import ToolLoopResult
from .answer_requirements import AnswerRequirements
from .answer_review import AnswerReviewer, AnswerReviewLimits, ReviewedAnswer, review_answer
from .context import ContextReceipt


async def review_tool_answer(reviewer: AnswerReviewer, question: str, result: ToolLoopResult,
                             limits: AnswerReviewLimits, *, requirements: AnswerRequirements | None = None) -> ReviewedAnswer:
    """Review exact retained tool packets within the original tool deadline."""
    evidence = json.dumps({'tool_results':[json.loads(packet) for packet in result.evidence_packets],
        'complete':False}, ensure_ascii=False, separators=(',', ':'), allow_nan=False)
    return await review_answer(reviewer, question, result.text, evidence, result.evidence_ids,
        limits=limits, validate_evidence=result.validate, deadline=result.deadline, requirements=requirements)


def _fingerprint(record: object) -> str:
    return hashlib.sha256(json.dumps(record, ensure_ascii=False, sort_keys=True,
                                    separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def tool_context_receipt(result: ToolLoopResult, session_id: str) -> ContextReceipt:
    """Keep transient source packets separate from cacheable identifiers/hashes."""
    packets = [json.loads(packet) for packet in result.evidence_packets]
    references = {(row['episode_id'], row['chunk_id']) for packet in packets for row in packet.get('items', [])}
    sources = {episode_id for episode_id, _ in references}
    chunk_hashes = {str(row['chunk_id']): hashlib.sha256(row['text'].encode()).hexdigest()
                    for packet in packets for row in packet.get('items', [])}
    claim_hashes = {str(row['fact_id']): _fingerprint(row) for packet in packets
                   for key in ('facts', 'claims') for row in packet.get(key, [])}
    relation_hashes = {str(row['link_id']): _fingerprint(row) for packet in packets for row in packet.get('relations', [])}
    for packet in packets:
        for key in ('facts', 'claims', 'relations'):
            sources.update(row['source_episode_id'] for row in packet.get(key, []) if row.get('source_episode_id'))
    digest = hashlib.sha256()
    for packet in result.evidence_packets:
        raw = packet.encode()
        digest.update(len(raw).to_bytes(8, 'big'))
        digest.update(raw)
    return ContextReceipt(
        request_id=uuid4().hex, session_id=session_id,
        status='prepared' if result.evidence_ids else 'skipped', retrieval_mode='native_tools',
        recall_event_id=None, references=[{'episode_id': episode, 'chunk_id': chunk} for episode, chunk in sorted(references)],
        context_sha256=digest.hexdigest() if result.evidence_ids else None,
        context_bytes=sum(len(packet.encode()) for packet in result.evidence_packets),
        omitted_count=sum(packet.get('coverage', {}).get('output_omitted_count', 0) for packet in packets),
        degraded=[], low_confidence=None, error_type=None,
        evidence_graph_status='unavailable',
        evidence_fingerprints=chunk_hashes, claim_fingerprints=claim_hashes, relation_fingerprints=relation_hashes,
        claim_count=len(claim_hashes), relation_count=len(relation_hashes),
        tool_retrieval={'model_calls': result.model_calls, 'tool_calls': result.tool_calls,
                        'outcomes': [outcome.model_dump(mode='json') for outcome in result.tool_outcomes],
                        'source_status': result.source_status, 'evidence_ids': list(result.evidence_ids),
                        'source_episode_ids': sorted(sources), 'packets': packets, 'packets_status': 'available',
                        'packet_fingerprints': [hashlib.sha256(packet.encode()).hexdigest() for packet in result.evidence_packets],
                        'bounded': True, 'complete': False, 'verified_accuracy': False},
    )
