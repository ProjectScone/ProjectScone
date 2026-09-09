"""Select complete quote sets for explicit, question-bound fact/path plans.

The existing answer controller owns source validation and quote rendering.
This selector matches recorded triples, not quote entailment or inferred intent.
"""
from __future__ import annotations

from itertools import combinations
import json

from ..retrieval.adaptive import EvidenceCandidate
from ..retrieval.structured_evidence import EvidenceRequirement, StructuredEvidenceAssessor
from .evidence_answer import EvidenceCard, EvidenceClaim, EvidenceSelection


def _cards(cards: tuple[EvidenceCard, ...]) -> tuple[EvidenceCard, ...]:
    if type(cards) is not tuple or len(cards) > 24 or any(not isinstance(card, EvidenceCard) for card in cards):
        raise ValueError('provide at most 24 evidence cards')
    frozen = tuple(EvidenceCard.model_validate(dict(vars(card)), strict=True) for card in cards)
    payload = json.dumps([card.model_dump(mode='json') for card in frozen],
                         ensure_ascii=False, allow_nan=False, separators=(',', ':'))
    if len({card.id for card in frozen}) != len(frozen) or len(payload.encode()) > 128000:
        raise ValueError('card identities or payload exceed limits')
    return frozen


def _candidates(cards: tuple[EvidenceCard, ...]) -> tuple[EvidenceCandidate, ...]:
    claims: dict[int, EvidenceClaim] = {}
    for card in cards:
        for claim in card.claims:
            if claim.fact_id in claims and claims[claim.fact_id] != claim:
                raise ValueError('inconsistent claim identity across cards')
            claims[claim.fact_id] = claim
    # Selection metadata only: public output remains the controller's original
    # quote cards. A text passage is never promoted into a structured witness.
    return tuple(EvidenceCandidate(id=f'fact:{claim.fact_id}', episode_id=claim.source_episode_id,
        text=json.dumps([claim.subject, claim.predicate, claim.object], ensure_ascii=False),
        subject=claim.subject, predicate=claim.predicate, object=claim.object) for claim in claims.values())


def _cover(cards: tuple[EvidenceCard, ...], required: frozenset[str]) -> tuple[str, ...]:
    claim_ids = {card.id: frozenset(f'fact:{claim.fact_id}' for claim in card.claims) for card in cards}
    relevant = tuple(card for card in cards if required.intersection(claim_ids[card.id]))
    # At most 24 choose 1 + 24 choose 2 + 24 choose 3 = 2324 combinations.
    # Prefer fewer whole cards, then fewer quote bytes, then input order.
    for count in range(1, 4):
        best: tuple[EvidenceCard, ...] = ()
        best_bytes: int | None = None
        for group in combinations(relevant, count):
            if not required.issubset(identifier for card in group for identifier in claim_ids[card.id]):
                continue
            size = sum(len(card.text.encode()) for card in group) + 2 * (count - 1)
            if best_bytes is None or size < best_bytes:
                best, best_bytes = group, size
        if best:
            return tuple(card.id for card in best)
    return ()


class StructuredEvidenceSelector:
    """Model-free EvidenceSelector for an application-authored question plan.

    All requirements and competing recorded values must fit in at most three
    whole cards. Missing witnesses, exhausted path work, or insufficient card
    capacity produce no selection. An atomic selection cannot be partially
    rendered when the answer byte budget is smaller than its combined quotes.

    Only offered cards are examined. This does not establish global absence,
    source truth, synonym equivalence, or correct natural-language intent.
    """

    def __init__(self, question: str, requirements: tuple[EvidenceRequirement, ...], *, max_work: int = 256) -> None:
        self._assessor = StructuredEvidenceAssessor(question, requirements, max_work=max_work)

    async def select(self, question: str, cards: tuple[EvidenceCard, ...]) -> EvidenceSelection:
        frozen = _cards(cards)
        decision = await self._assessor.assess(question, _candidates(frozen))
        if decision.status != 'sufficient':
            return EvidenceSelection(atomic=True)
        return EvidenceSelection(card_ids=_cover(frozen, frozenset(decision.selected_ids)), atomic=True)
