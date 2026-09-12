"""Derive claims that follow from the claims a space already holds.

A ``Deriver`` runs over the active claims of a space, grouped by subject
and by the subjects one hop away through shared names ("mark works_at
acme" and "acme based_in lisbon" meet), and asks the model for claims
that follow from those together, each with the ids of the premises it
used. Output is validated the way extraction output is; every inference
is stored as a proposal of origin ``inferred`` whose ``derived_from``
links are its provenance. There is no quote: no episode says it. The
same inference again is restated, not stored twice, and a group whose
membership has not changed since the last pass is not sent again.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Mapping, Optional, Sequence

from ..core.models import Fact
from ..core.validation import entity_key
from ..memory.engine import MemoryEngine, derivation_groups
from ..memory.identity import join_match
from ..providers.llm import ChatError, ChatModel
from .distill import DistillError, _find_array

DERIVE_PROMPT = (
    "You are given claims about one subject and its neighbours, each with an id. "
    "Return a JSON array of claims that follow from two or more of them together, "
    "or from one of them plus a general rule you state. Each entry is an object "
    '{"subject": string, "predicate": string, "object": string, "premises": [ids], '
    '"rule": optional string, "confidence": number 0..1}. Never restate a given claim. '
    "Return [] when nothing follows."
)

#: Why an entry of the model's reply did not become a proposal, and why a
#: whole group produced none. The last two are about the call rather than
#: about an entry: a model that answered unusably, and one that did not
#: answer. A real corpus contains something a model will not discuss, and
#: one group it will not touch must cost that group and nothing else.
REJECTIONS = ("malformed", "unknown_premise", "too_few_premises", "restates_premise",
              "unreadable_reply", "unreachable")


@dataclass(frozen=True)
class Derived:
    subject: str
    predicate: str
    object: str
    premises: tuple[int, ...]
    rule: Optional[str]
    confidence: float


@dataclass(frozen=True)
class RejectedDerivation:
    reason: str
    entry: object


@dataclass
class DeriveOutcome:
    space: str
    groups: int = 0
    sent: int = 0
    proposed: list[Fact] = field(default_factory=list)
    restated: int = 0
    rejected: list[RejectedDerivation] = field(default_factory=list)
    latency_ms: float = 0.0

    def as_payload(self) -> dict:
        reasons: dict[str, int] = {}
        for r in self.rejected:
            reasons[r.reason] = reasons.get(r.reason, 0) + 1
        return {
            "groups": self.groups, "sent": self.sent, "proposed": len(self.proposed), "restated": self.restated,
            "rejected": len(self.rejected), "rejected_reasons": dict(sorted(reasons.items())),
            "latency_ms": round(self.latency_ms, 1),
        }


def _text(value: object) -> Optional[str]:
    if not isinstance(value, str):
        return None
    cleaned = " ".join(value.split())
    return cleaned or None


def _same(a: str, b: str) -> bool:
    """One name, by entity identity: for subjects and predicates."""
    return entity_key(a) == entity_key(b)


def _same_object(a: str, b: str) -> bool:
    """One object: the same name by the join rule, or exactly the same text.
    A value whose case can change its meaning ('3 MB', '3 mb') is not
    restated by a different spelling."""
    return a.strip() == b.strip() or join_match(a, b) is not None


def parse_derivations(text: str, group: Mapping[int, Fact]) -> tuple[list[Derived], list[RejectedDerivation]]:
    """The model's reply as validated inferences over ``group``: strict
    entries, every premise in the group, two premises or one plus a
    rule, and nothing that restates a premise. Raises ``DistillError``
    when no JSON array can be found at all."""
    array = _find_array(text)
    if array is None:
        raise DistillError(f"model did not return a JSON array; got: {text.strip()[:120]!r}")
    accepted: list[Derived] = []
    rejected: list[RejectedDerivation] = []
    for entry in array:
        if not isinstance(entry, dict):
            rejected.append(RejectedDerivation("malformed", entry))
            continue
        subject, predicate, obj = _text(entry.get("subject")), _text(entry.get("predicate")), _text(entry.get("object"))
        premises = entry.get("premises")
        if not (subject and predicate and obj) or not isinstance(premises, list) \
                or not all(isinstance(p, int) and not isinstance(p, bool) for p in premises):
            rejected.append(RejectedDerivation("malformed", entry))
            continue
        ids = tuple(sorted(set(premises)))
        if any(p not in group for p in ids):
            rejected.append(RejectedDerivation("unknown_premise", entry))
            continue
        rule = _text(entry.get("rule"))
        if len(ids) < 2 and rule is None:
            rejected.append(RejectedDerivation("too_few_premises", entry))
            continue
        if any(_same(f.subject, subject) and _same(f.predicate, predicate) and _same_object(f.object, obj)
               for f in group.values()):
            rejected.append(RejectedDerivation("restates_premise", entry))
            continue
        raw = entry.get("confidence", 0.5)
        confidence = float(raw) if isinstance(raw, (int, float)) and not isinstance(raw, bool) else 0.5
        accepted.append(Derived(subject, predicate, obj, ids, rule, max(0.0, min(1.0, confidence))))
    return accepted, rejected


class Deriver:
    def __init__(self, engine: MemoryEngine, chat: ChatModel, *, prompt: str = DERIVE_PROMPT, max_group: int = 24) -> None:
        self.engine = engine
        self.chat = chat
        self.prompt = prompt
        #: Claims per group sent to the model; a larger group is cut to its
        #: strongest claims so a prompt stays bounded.
        self.max_group = max_group

    async def derive(self, space: str, *, limit_groups: int = 50) -> DeriveOutcome:
        """One pass over the space: every group not yet seen at its current
        membership goes to the model once; validated inferences become
        proposals, the same inference again is restated. Records a
        ``derive`` event and returns the outcome."""
        started = time.perf_counter()
        engine = self.engine
        active = await engine.facts(space)
        groups = derivation_groups(active)
        outcome = DeriveOutcome(space=space, groups=len(groups))
        for group in groups:
            key = (space, frozenset(f.fact_id for f in group))
            if key in engine._derive_seen:
                continue
            if outcome.sent >= limit_groups:
                break
            sent = sorted(group, key=lambda f: (-f.confidence, f.fact_id))[: self.max_group]
            by_id = {f.fact_id: f for f in sent}
            text = "\n".join(f"{f.fact_id}: {f.subject} {f.predicate} {f.object}" for f in sent)
            outcome.sent += 1
            try:
                reply = await self.chat.complete(self.prompt, text)
            except ChatError as exc:
                # No answer arrived, so nothing about this group is settled.
                # It is left unseen and asked again next pass.
                outcome.rejected.append(RejectedDerivation("unreachable", str(exc)[:200]))
                continue
            try:
                derived, rejected = parse_derivations(reply, by_id)
            except DistillError as exc:
                # The model answered and the answer was unusable, a refusal
                # most often. That is settled: asking again gets the same
                # answer, so the group is marked seen and the pass goes on.
                outcome.rejected.append(RejectedDerivation("unreadable_reply", str(exc)[:200]))
                engine._derive_seen.add(key)
                continue
            outcome.rejected.extend(rejected)
            for d in derived:
                if await self._already_held(space, d):
                    outcome.restated += 1
                    continue
                fact = await engine.assert_fact(
                    space, d.subject, d.predicate, d.object, confidence=d.confidence,
                    origin="inferred", proposed=True, derived_from=list(d.premises),
                )
                outcome.proposed.append(fact)
            engine._derive_seen.add(key)
        outcome.latency_ms = (time.perf_counter() - started) * 1000
        await engine._emit(space, "derive", outcome.as_payload())
        unanswered = sum(1 for r in outcome.rejected if r.reason in ("unreadable_reply", "unreachable"))
        if outcome.sent and unanswered == outcome.sent:
            # Nothing follows and nothing could be read are different
            # results, and reporting the second as the first is how a pass
            # that achieved nothing is filed as one that found nothing.
            raise DistillError(
                f"no group could be read in {space}: {outcome.sent} sent, none answered usably")
        return outcome

    async def _already_held(self, space: str, d: Derived) -> bool:
        """The same inference from the same premises is already a claim
        (proposed or active): a restatement, not a new fact."""
        # Stored subjects and predicates are keys; the model writes names as
        # people do, so look them up the way the ledger stored them.
        for fact in await self.engine.documents.facts_for(space, entity_key(d.subject), entity_key(d.predicate)):
            if fact.status not in ("proposed", "active") or not _same_object(fact.object, d.object):
                continue
            links = await self.engine.documents.fact_links(space, fact.fact_id)
            premises = {l.to_fact for l in links if l.kind == "derived_from" and l.from_fact == fact.fact_id}
            if premises == set(d.premises):
                return True
        return False
