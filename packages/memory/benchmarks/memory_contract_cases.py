"""Hand-authored, synthetic diagnostic cases. Labels never enter model requests."""
from __future__ import annotations

from dataclasses import dataclass

from scone_memory.experimental.memory_contracts import ContractRequest, Evidence, Status


@dataclass(frozen=True)
class Case:
    name: str
    request: ContractRequest
    expected: tuple[Status, ...]


def _case(name: str, question: str, claim: str, sources: tuple[tuple[str, str, str], ...],
          labels: tuple[Status, ...]) -> Case:
    return Case(name, ContractRequest('synthetic:nova:read-all:2026-09-23:v1', question, claim,
                                    tuple(Evidence(*source) for source in sources)), labels)


def cases() -> tuple[Case, ...]:
    return (
        _case('independent-support', 'Who owns Nova deployment?', 'Maya owns Nova deployment.', (
            ('minutes', 'a', 'Decision for Nova: Maya owns deployment.'),
            ('roster', 'b', 'Nova deployment responsibility: Maya.'),
            ('changelog', 'c', 'Nova has a new landing page.'),
        ), ('insufficient', 'supported', 'supported', 'supported', 'insufficient', 'supported', 'supported', 'supported')),
        _case('copied-support', 'Who owns Nova deployment?', 'Maya owns Nova deployment.', (
            ('original', 'a', 'Decision for Nova: Maya owns deployment.'),
            ('copy', 'a', 'A copy of the decision: Maya owns Nova deployment.'),
            ('unrelated', 'b', 'Nova has a new landing page.'),
        ), ('insufficient', 'supported', 'insufficient', 'supported')),
        _case('two-hop', 'Who owns Nova deployment?', 'Maya owns Nova deployment.', (
            ('policy', 'a', 'For Nova, the release captain is responsible for deployment.'),
            ('roster', 'b', 'Maya is the release captain of Nova.'),
            ('unrelated', 'c', 'Nova has a new landing page.'),
        ), ('insufficient', 'insufficient', 'insufficient', 'supported', 'insufficient', 'insufficient', 'insufficient', 'supported')),
        _case('conflicting-current-records', 'Who owns Nova deployment?', 'Maya owns Nova deployment.', (
            ('record-one', 'a', 'Nova deployment currently belongs to Maya.'),
            ('record-two', 'b', 'Nova deployment currently belongs to Leo, not Maya.'),
            ('unrelated', 'c', 'Nova has a new landing page.'),
        ), ('insufficient', 'supported', 'refuted', 'conflict', 'insufficient', 'supported', 'refuted', 'conflict')),
        _case('complete-compound-claim', 'What stores does Nova use?', 'Nova uses SQLite for records and local disk for blobs.', (
            ('records', 'a', 'Nova uses SQLite for its records.'),
            ('blobs', 'b', 'Nova stores blobs on local disk.'),
            ('unrelated', 'c', 'Nova uses Python for its API.'),
        ), ('insufficient', 'insufficient', 'insufficient', 'supported', 'insufficient', 'insufficient', 'insufficient', 'supported')),
        _case('absence-is-not-negation', 'Does Nova use Pinecone?', 'Nova does not use Pinecone.', (
            ('vector', 'a', 'Nova stores vectors in Qdrant.'),
            ('decision', 'b', 'Pinecone was removed from Nova and is not used in Nova.'),
            ('unrelated', 'c', 'Nova uses Python for its API.'),
        ), ('insufficient', 'insufficient', 'supported', 'supported', 'insufficient', 'insufficient', 'supported', 'supported')),
        _case('entity-scope', 'Who owns Nova deployment?', 'Maya owns Nova deployment.', (
            ('other-project', 'a', 'Maya owns deployment for the Delta project.'),
            ('nova', 'b', 'Leo is the sole owner of Nova deployment.'),
            ('unrelated', 'c', 'Nova uses Python for its API.'),
        ), ('insufficient', 'insufficient', 'refuted', 'refuted', 'insufficient', 'insufficient', 'refuted', 'refuted')),
        _case('proposal-is-not-decision', 'Who owns Nova deployment?', 'Maya owns Nova deployment.', (
            ('proposal', 'a', 'Proposal, not approved: perhaps Maya should own Nova deployment.'),
            ('question', 'b', 'Should Maya own Nova deployment? No decision has been made.'),
            ('pointer', 'c', 'The deployment owner is listed in an unread file named owners.md.'),
        ), ('insufficient',) * 8),
    )
