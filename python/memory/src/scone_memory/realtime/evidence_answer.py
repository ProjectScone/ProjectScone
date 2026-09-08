"""Bounded extractive answers from retained source quotes, never model prose.

Path cards preserve recorded statement order, not inferred causality or endpoint
claims. Connected contradictions travel together. The caller owns the selector;
prepared evidence is checked before and after its single optional selection call.
"""
from __future__ import annotations

import asyncio
from dataclasses import dataclass
import json
import math
import time
from typing import Annotated, Literal, Protocol

from pydantic import BaseModel, ConfigDict, Field, model_validator

from ..core.models import FactOrigin, LinkKind
from ..retrieval.evidence_graph import MAX_CHUNKS, MAX_FACTS, MAX_LINKS, MAX_SOURCES
from .answer_review import EvidenceId
from .context import _PATH_GUIDANCE, _PREFIX
from .review_evidence import PreparedReviewEvidence, _id, _mapping, _paths, _records, _unique_object

CardId = Annotated[str, Field(pattern=r"^card:[1-9][0-9]*$", max_length=64)]
RecordId = Annotated[int, Field(gt=0, lt=2**63)]
_FAILURES = frozenset({"invalid_evidence", "invalid_selection", "selection_provider_failed", "selection_timeout",
    "stale_evidence", "source_validation_failed", "source_validation_timeout"})
_ABSTENTION = "I could not find supporting memory for that question."


class EvidenceAnswerError(ValueError):
    """Content-free failure: no answer is safe to return from this operation."""
    def __init__(self, reason: str) -> None:
        super().__init__("evidence answer unavailable")
        self.reason = reason if type(reason) is str and reason in _FAILURES else "invalid_selection"


class EvidenceCard(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)
    id: CardId
    kind: Literal["passage", "claim", "path"]
    text: str = Field(min_length=1, max_length=128000)
    evidence_ids: tuple[EvidenceId, ...] = Field(min_length=1, max_length=MAX_CHUNKS+MAX_FACTS+MAX_LINKS)

    @model_validator(mode="after")
    def unique_ids(self) -> EvidenceCard:
        if len(set(self.evidence_ids)) != len(self.evidence_ids):
            raise ValueError("card evidence IDs must be unique")
        return self


class EvidenceSelection(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)
    card_ids: tuple[CardId, ...] = Field(default=(), max_length=3)

    @model_validator(mode="after")
    def unique_ids(self) -> EvidenceSelection:
        if len(set(self.card_ids)) != len(self.card_ids):
            raise ValueError("selected card IDs must be unique")
        return self


class EvidenceSelector(Protocol):
    async def select(self, question: str, cards: tuple[EvidenceCard, ...]) -> EvidenceSelection: ...


@dataclass(frozen=True)
class EvidenceCards:
    cards: tuple[EvidenceCard, ...]
    omitted_count: int
    truncated: bool
    deduplicated_card_count: int = 0


@dataclass(frozen=True)
class ConstructedEvidenceAnswer:
    answer: str
    receipt: dict[str, object]


class _Source(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    chunk_id: RecordId
    episode_id: RecordId
    text: str = Field(min_length=1)
    source: str | None
    created_at: str
    project: object = None
    role: object = None


class _Claim(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    fact_id: RecordId
    subject: str
    predicate: str
    object: str
    origin: FactOrigin
    status: Literal["active"]
    valid_from: str
    valid_until: str | None
    confidence: float = Field(ge=0, le=1, allow_inf_nan=False)
    source_episode_id: RecordId
    quote: str = Field(min_length=1)


class _Relation(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    link_id: RecordId
    from_fact: RecordId
    to_fact: RecordId
    kind: LinkKind
    source_episode_id: RecordId
    quote: str = Field(min_length=1)


@dataclass(frozen=True)
class _Material:
    sources: tuple[_Source, ...]
    claims: dict[int, _Claim]
    relations: dict[int, _Relation]
    paths: tuple[dict[str, object], ...]
    ids: tuple[str, ...]


def _constant(value: str) -> object:
    raise ValueError("nonfinite JSON is invalid")


def _parse(evidence: str) -> _Material:
    if type(evidence) is not str or len(evidence) > 128000 or len(evidence.encode("utf-8")) > 128000:
        raise ValueError("bounded source evidence required")
    prefix, separator, serialized = evidence.partition("\n")
    if not separator or prefix not in (_PREFIX.rstrip("\n"), _PREFIX.rstrip("\n")+" "+_PATH_GUIDANCE.rstrip("\n")):
        raise ValueError("invalid source block prefix")
    packet = _mapping(json.loads(serialized, object_pairs_hook=_unique_object, parse_constant=_constant))
    if (set(packet) - {"schema_version", "coverage", "sources", "claims", "relations", "paths"}
            or type(packet.get("schema_version")) is not int or packet["schema_version"] != 1):
        raise ValueError("invalid source packet schema")
    _mapping(packet.get("coverage"))
    sources = tuple(_Source.model_validate(record, strict=True) for record in _records(packet.get("sources"),MAX_CHUNKS))
    claim_rows = _records(packet.get("claims", []),MAX_FACTS)
    relation_rows = _records(packet.get("relations", []),MAX_LINKS)
    claim_values = tuple(_Claim.model_validate(record,strict=True) for record in claim_rows)
    relation_values = tuple(_Relation.model_validate(record,strict=True) for record in relation_rows)
    source_ids = {source.episode_id for source in sources}
    source_ids.update(claim.source_episode_id for claim in claim_values)
    source_ids.update(relation.source_episode_id for relation in relation_values)
    if len(source_ids) > MAX_SOURCES:
        raise ValueError("too many source episodes")
    claims = {claim.fact_id: claim for claim in claim_values}
    relations = {relation.link_id: relation for relation in relation_values}
    ids = tuple([*(f"chunk:{source.chunk_id}" for source in sources), *(f"fact:{claim.fact_id}" for claim in claim_values),
                 *(f"link:{relation.link_id}" for relation in relation_values)])
    if len(set(ids)) != len(ids):
        raise ValueError("duplicate evidence IDs")
    if any(relation.from_fact not in claims or relation.to_fact not in claims for relation in relation_values):
        raise ValueError("relation cites unavailable claims")
    paths = _records(packet.get("paths", []),16)
    _paths(paths, {claim.fact_id: row for claim,row in zip(claim_values,claim_rows)},
           {relation.link_id: row for relation,row in zip(relation_values,relation_rows)})
    return _Material(sources,claims,relations,tuple(paths),ids)


def _limits(max_cards: int, max_bytes: int) -> None:
    if type(max_cards) is not int or not 1 <= max_cards <= 24:
        raise ValueError("max_cards must be an integer from 1 to 24")
    if type(max_bytes) is not int or not 2 <= max_bytes <= 128000:
        raise ValueError("max_bytes must be an integer from 2 to 128000")


def _payload(cards: tuple[EvidenceCard, ...]) -> int:
    return len(json.dumps([card.model_dump(mode="json") for card in cards],ensure_ascii=False,
                          separators=(",",":"),allow_nan=False).encode("utf-8"))


def _conflicts(material: _Material, ids: set[int]) -> tuple[set[int], set[int]]:
    ids = ids.copy()
    links: set[int] = set()
    for _ in range(len(material.claims)+1):
        previous = len(ids)
        for link in material.relations.values():
            if link.kind == "contradicts" and (link.from_fact in ids or link.to_fact in ids):
                ids.update((link.from_fact,link.to_fact))
                links.add(link.link_id)
        if len(ids) == previous:
            break
    return ids,links


def _statement(claim: _Claim) -> str:
    return f"Recorded statement [fact:{claim.fact_id}; episode:{claim.source_episode_id}]\n{claim.quote}"


def _link(relation: _Relation, direction: str | None = None) -> str:
    traversal = f"; traversal {direction}" if direction else ""
    return (f"Stored relation [link:{relation.link_id}; fact:{relation.from_fact} -> fact:{relation.to_fact}; "
            f"{relation.kind}{traversal}; episode:{relation.source_episode_id}]\n{relation.quote}")


def _cards(material: _Material, max_cards: int, max_bytes: int) -> EvidenceCards:
    proposals: list[EvidenceCard] = []
    covered: set[int] = set()
    protected_episodes: set[int] = set()
    standalone_quotes: dict[str, tuple[int, str]] = {}
    passage_quotes: dict[str, tuple[int, str]] = {}

    def add(kind: Literal["path","claim","passage"], parts: list[str], ids: list[str]) -> None:
        text = "\n\n".join(parts)
        # Oversized whole cards still consume their stable proposal ID, but can
        # never be offered. No quote clipping or constituent escape is allowed.
        proposals.append(EvidenceCard(id=f"card:{len(proposals)+1}",kind=kind,text=text,
                                      evidence_ids=tuple(dict.fromkeys(ids))))

    def competing(ids: set[int], links: set[int], ordered: list[int], parts: list[str], evidence_ids: list[str]) -> None:
        extra = [claim for claim in material.claims.values() if claim.fact_id in ids and claim.fact_id not in ordered]
        if links:
            parts.append("Competing recorded statements")
        for claim in extra:
            parts.append(_statement(claim))
            evidence_ids.append(f"fact:{claim.fact_id}")
        for link in material.relations.values():
            if link.link_id in links:
                parts.append(_link(link))
                evidence_ids.append(f"link:{link.link_id}")
        if links:
            protected_episodes.update(material.claims[identifier].source_episode_id for identifier in ids)
            protected_episodes.update(material.relations[identifier].source_episode_id for identifier in links)

    for path in material.paths:
        raw_ids = path["fact_ids"]
        if not isinstance(raw_ids,list):
            raise ValueError("invalid path IDs")
        ordered = [_id(identifier) for identifier in raw_ids]
        ids,links = _conflicts(material,set(ordered))
        parts = ["Recorded statements in path order"]
        evidence_ids = [f"fact:{identifier}" for identifier in ordered]
        steps = _records(path["steps"],6)
        for index,identifier in enumerate(ordered):
            parts.append(_statement(material.claims[identifier]))
            if index < len(steps) and "link_id" in steps[index]:
                link = material.relations[_id(steps[index]["link_id"])]
                parts.append(_link(link,str(steps[index]["direction"])))
                evidence_ids.append(f"link:{link.link_id}")
                protected_episodes.add(link.source_episode_id)
        competing(ids,links,ordered,parts,evidence_ids)
        add("path",parts,evidence_ids)
        covered.update(ids)
        protected_episodes.update(material.claims[identifier].source_episode_id for identifier in ids)
    for claim in material.claims.values():
        if claim.fact_id in covered:
            continue
        ids,links = _conflicts(material,{claim.fact_id})
        parts = [_statement(claim)]
        evidence_ids = [f"fact:{claim.fact_id}"]
        competing(ids,links,[claim.fact_id],parts,evidence_ids)
        add("claim",parts,evidence_ids)
        if not links:
            standalone_quotes[proposals[-1].id] = (claim.source_episode_id, claim.quote)
        covered.update(ids)
    for source in material.sources:
        if source.episode_id not in protected_episodes:
            add("passage",[f"Recorded passage [chunk:{source.chunk_id}; episode:{source.episode_id}]\n{source.text}"],
                [f"chunk:{source.chunk_id}"])
            passage_quotes[proposals[-1].id] = (source.episode_id, source.text)
    retained: list[EvidenceCard] = []
    omitted = 0
    deduplicated = 0
    offered_quotes: set[tuple[int, str]] = set()
    for card in proposals:
        # Deduplicate only against admitted standalone claims. An oversized
        # claim must not hide a smaller whole passage carrying the same quote.
        if passage_quotes.get(card.id) in offered_quotes:
            deduplicated += 1
            continue
        if len(retained) >= max_cards or _payload((*retained,card)) > max_bytes:
            omitted += 1
        else:
            retained.append(card)
            if card.id in standalone_quotes:
                offered_quotes.add(standalone_quotes[card.id])
    return EvidenceCards(tuple(retained),omitted,omitted>0,deduplicated)


def build_evidence_cards(evidence: str, *, max_cards: int = 24, max_bytes: int = 16000) -> EvidenceCards:
    """Parse quote cards without verifying source retention or answer accuracy.

    Packet caps match the canonical evidence graph (24 chunks, 16 claims,
    48 relations), with at most 16 paths. Only the controller checks sources.
    Exact same-episode passages duplicate an offered standalone claim only;
    differing claims, passage records, and Unicode/whitespace remain distinct.
    Deduplicated cards are counted separately from budget omissions.
    """
    _limits(max_cards,max_bytes)
    try:
        return _cards(_parse(evidence),max_cards,max_bytes)
    except Exception:
        raise EvidenceAnswerError("invalid_evidence") from None


async def construct_evidence_answer(
    selector: EvidenceSelector, question: str, material: PreparedReviewEvidence, *, timeout_s: float = 20,
    max_cards: int = 24, max_evidence_bytes: int = 16000, max_answer_bytes: int = 16000,
    deadline: float | None = None,
) -> ConstructedEvidenceAnswer:
    """Select at most three whole cards within one shared cooperative deadline.

    Optional absolute monotonic ``deadline`` can only shorten the configured
    budget. Source failures and selector errors return no answer. Cancellation
    propagates even if a caller-owned adapter catches it and returns a value.
    """
    _limits(max_cards,max_evidence_bytes)
    if type(max_answer_bytes) is not int or not 64 <= max_answer_bytes <= 128000:
        raise ValueError("max_answer_bytes must be an integer from 64 to 128000")
    if type(timeout_s) not in (float,int) or not math.isfinite(timeout_s) or not 1 <= timeout_s <= 180:
        raise ValueError("timeout_s must be finite from 1 to 180")
    if deadline is not None and (type(deadline) not in (float,int) or not math.isfinite(deadline)):
        raise ValueError("deadline must be a finite monotonic timestamp")
    hard_deadline = min(time.monotonic()+timeout_s,deadline) if deadline is not None else time.monotonic()+timeout_s
    task = asyncio.current_task()
    stage = "selection"

    def check() -> None:
        if task is not None and task.cancelling():
            raise asyncio.CancelledError()
        if time.monotonic() >= hard_deadline:
            raise TimeoutError()

    try:
        check()
        if type(question) is not str or not question.strip() or len(question) > 8000 or len(question.encode("utf-8")) > 8000:
            raise EvidenceAnswerError("invalid_evidence")
        if not isinstance(material,PreparedReviewEvidence):
            raise EvidenceAnswerError("invalid_evidence")
        try:
            parsed = _parse(material.evidence)
            if (type(material.evidence_ids) is not tuple or any(type(identifier) is not str for identifier in material.evidence_ids)
                    or material.evidence_ids != parsed.ids):
                raise ValueError("prepared evidence IDs mismatch")
            cards = _cards(parsed,max_cards,max_evidence_bytes)
            validator = material._validator
            if not callable(validator):
                raise ValueError("source validator required")
        except Exception:
            raise EvidenceAnswerError("invalid_evidence") from None
        async with asyncio.timeout(max(0,hard_deadline-time.monotonic())):
            stage = "source"
            try:
                valid = await validator()
                check()
            except (asyncio.CancelledError,TimeoutError):
                raise
            except Exception:
                raise EvidenceAnswerError("source_validation_failed") from None
            if type(valid) is not bool:
                raise EvidenceAnswerError("source_validation_failed")
            if not valid:
                raise EvidenceAnswerError("stale_evidence")
            stage = "selection"
            try:
                raw = (await selector.select(question,tuple(card.model_copy(deep=True) for card in cards.cards))
                       if cards.cards else EvidenceSelection())
                check()
            except (asyncio.CancelledError,TimeoutError):
                raise
            except EvidenceAnswerError as error:
                reason = getattr(error,"reason",None)
                raise EvidenceAnswerError(reason if type(reason) is str and reason in
                    {"invalid_selection","selection_provider_failed","selection_timeout"} else "invalid_selection") from None
            except Exception:
                raise EvidenceAnswerError("selection_provider_failed") from None
            try:
                if not isinstance(raw,EvidenceSelection):
                    raise ValueError("selection required")
                selection = EvidenceSelection.model_validate(dict(vars(raw)),strict=True)
                by_id = {card.id: card for card in cards.cards}
                if not set(selection.card_ids).issubset(by_id):
                    raise ValueError("unknown card")
            except Exception:
                raise EvidenceAnswerError("invalid_selection") from None
            stage = "source"
            try:
                valid = await validator()
                check()
            except (asyncio.CancelledError,TimeoutError):
                raise
            except Exception:
                raise EvidenceAnswerError("source_validation_failed") from None
            if type(valid) is not bool:
                raise EvidenceAnswerError("source_validation_failed")
            if not valid:
                raise EvidenceAnswerError("stale_evidence")
            delivered: list[EvidenceCard] = []
            omitted = cards.omitted_count
            for identifier in selection.card_ids:
                card = by_id[identifier]
                if len("\n\n".join(c.text for c in (*delivered,card)).encode("utf-8")) > max_answer_bytes:
                    omitted += 1
                else:
                    delivered.append(card)
            answer = "\n\n".join(card.text for card in delivered) if delivered else _ABSTENTION
            check()
            return ConstructedEvidenceAnswer(answer,dict(status="selected" if delivered else "no_selection",
                selected_card_ids=[card.id for card in delivered],
                evidence_ids=list(dict.fromkeys(identifier for card in delivered for identifier in card.evidence_ids)),
                card_count=len(cards.cards),omitted_card_count=omitted,
                deduplicated_card_count=cards.deduplicated_card_count,
                verified_accuracy=False,source_status="retained",mode="extractive"))
    except asyncio.CancelledError:
        raise
    except TimeoutError:
        raise EvidenceAnswerError("source_validation_timeout" if stage == "source" else "selection_timeout") from None
