"""A community's report: what the space records about it, in sentences that each quote a record.

The overview digests a community from the graph itself -- size, kinds,
central entities, a few cited facts. A reader who wants prose about a
community, or about the whole space, has had to write it. This writes
it with a model and keeps the overview's discipline: every sentence of a
report cites a passage, and every passage is either a recorded claim
(``fact:<id>``: the claim as the ledger holds it) or a quote from the
episode behind it that this code re-read and found there (``quote:<id>``).
A sentence that quotes nothing is not shown. The report names the
projection digest and revision it was written under, and ``current()``
says whether the graph has moved since: a report is never served as
current across a revision. A space report is the second level: a
synthesis over the community reports' own sentences, cited by community,
two levels and no more until a third is measured.

Nothing here is verified beyond the quotes: ``verified_accuracy`` is
false on every record, as it is on the synthesis this rests on.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import TYPE_CHECKING, Optional

from ..core.errors import InvalidInput
from ..providers.llm import ChatModel
from ..retrieval.synthesis import Passage, Synthesis, SynthesisLimits, synthesize_passages
from .analysis import cached_analysis
from .read import load_projection

if TYPE_CHECKING:  # pragma: no cover - typing only
    from ..memory.engine import MemoryEngine
    from .project import EntityProjection

#: Facts one report may draw on; more is refused, not cut.
MAX_FACTS_EACH = 64
DEFAULT_FACTS_EACH = 16
#: Community reports one space report may fold; more is refused, not cut.
MAX_REPORTS = 50
DEFAULT_REPORTS = 12


@dataclass(frozen=True)
class CommunityReport:
    space: str
    community_id: str
    label: str
    members: tuple[str, ...]
    projection_digest: str
    revision: int
    question: str
    synthesis: Synthesis
    facts_cited: tuple[int, ...]
    facts_total: int
    #: Facts whose quote no longer verifies against its episode, left out.
    stale_evidence: int

    @property
    def status(self) -> str:
        return self.synthesis.status

    def text(self) -> str:
        return self.synthesis.text()

    def current(self, projection: "EntityProjection") -> bool:
        """Whether the graph is still the one this report was written under."""
        return projection.digest == self.projection_digest and projection.revision == self.revision

    def record(self) -> dict[str, object]:
        return {"schema_version": 1, "space": self.space, "community_id": self.community_id, "label": self.label,
                "members": list(self.members), "projection_digest": self.projection_digest, "revision": self.revision,
                "question": self.question, "status": self.status, "text": self.text(),
                "synthesis": self.synthesis.record(), "facts_cited": list(self.facts_cited),
                "facts_total": self.facts_total, "stale_evidence": self.stale_evidence, "verified_accuracy": False}


@dataclass(frozen=True)
class SpaceReport:
    space: str
    projection_digest: str
    revision: int
    reports: tuple[CommunityReport, ...]
    synthesis: Synthesis
    communities_total: int

    def text(self) -> str:
        return self.synthesis.text()

    def record(self) -> dict[str, object]:
        return {"schema_version": 1, "space": self.space, "projection_digest": self.projection_digest,
                "revision": self.revision, "status": self.synthesis.status, "text": self.text(),
                "synthesis": self.synthesis.record(), "reports": [report.record() for report in self.reports],
                "communities_total": self.communities_total,
                "communities_reported": len(self.reports), "verified_accuracy": False}


def _check(facts_each: int, max_reports: int) -> None:
    if type(facts_each) is not int or not 1 <= facts_each <= MAX_FACTS_EACH:
        raise InvalidInput(f"facts_each must be from 1 to {MAX_FACTS_EACH}")
    if type(max_reports) is not int or not 1 <= max_reports <= MAX_REPORTS:
        raise InvalidInput(f"max_reports must be from 1 to {MAX_REPORTS}")


async def _evidence(engine: "MemoryEngine", space: str, projection: "EntityProjection", community: object,
                    facts_each: int) -> tuple[list[Passage], list[int], int, int]:
    """The community's claims and their verified quotes as passages, central relations first."""
    members = set(community.members)  # type: ignore[attr-defined]
    central = set(community.top_entities[:5])  # type: ignore[attr-defined]
    entities = {entity.entity_id: entity for entity in projection.entities}
    inside = [relation for relation in projection.relations
              if relation.subject_id in members and relation.object_id in members]
    inside.sort(key=lambda r: (-(r.subject_id in central or r.object_id in central), -r.support.quoted,
                               -len(r.fact_ids), r.relation_id))
    total = sum(len(relation.fact_ids) for relation in inside)
    passages: list[Passage] = []
    cited: list[int] = []
    stale = 0
    for relation in inside[:facts_each]:
        for fact_id in sorted(relation.fact_ids, reverse=True):
            fact = await engine.documents.get_fact(space, fact_id)
            if fact is None:
                stale += 1
                continue
            claim = f"{entities[relation.subject_id].label} {relation.predicate} {entities[relation.object_id].label}"
            passages.append(Passage(f"fact:{fact_id}", claim, "ledger", fact.valid_from))
            cited.append(fact_id)
            if fact.quote and fact.source_episode_id is not None:
                episode = await engine.documents.get_episode(space, fact.source_episode_id)
                if episode is not None and fact.quote in episode.content:
                    passages.append(Passage(f"quote:{fact_id}", fact.quote, episode.source, fact.valid_from))
                else:
                    stale += 1
            break  # one fact per relation; how many stand behind it is in facts_total
    return passages, cited, total, stale


async def community_report(engine: "MemoryEngine", model: ChatModel, space: str, community_id: str, *,
                           facts_each: int = DEFAULT_FACTS_EACH, limits: Optional[SynthesisLimits] = None,
                           status: str = "current", as_of: Optional[str] = None) -> CommunityReport:
    """What the space records about one community, in cited sentences."""
    _check(facts_each, DEFAULT_REPORTS)
    when = as_of if as_of is not None else engine.clock()
    projection, _ = await load_projection(engine, space, mode=status, as_of=when)  # type: ignore[arg-type]
    analysis = cached_analysis(projection)
    community = next((c for c in analysis.communities if c.community_id == community_id), None)
    if community is None:
        raise InvalidInput(f"no community {community_id!r} in the graph of {space!r} at this revision")
    entities = {entity.entity_id: entity for entity in projection.entities}
    named = ", ".join(entities[member].label for member in community.top_entities[:5] if member in entities)
    question = f"What does this record about {named or community.label}?"
    passages, cited, total, stale = await _evidence(engine, space, projection, community, facts_each)
    bound = limits or SynthesisLimits(max_passages=max(1, min(50, len(passages))))
    made = await synthesize_passages(model, question, passages, limits=bound)
    return CommunityReport(space, community.community_id, community.label, tuple(community.members),
                           projection.digest, projection.revision, question, made, tuple(cited), total, stale)


async def space_report(engine: "MemoryEngine", model: ChatModel, space: str, *, max_reports: int = DEFAULT_REPORTS,
                       facts_each: int = DEFAULT_FACTS_EACH, limits: Optional[SynthesisLimits] = None,
                       status: str = "current", as_of: Optional[str] = None) -> SpaceReport:
    """The space in two levels: a report per community, largest first, then one over the reports."""
    _check(facts_each, max_reports)
    when = as_of if as_of is not None else engine.clock()
    projection, _ = await load_projection(engine, space, mode=status, as_of=when)  # type: ignore[arg-type]
    analysis = cached_analysis(projection)
    ordered = sorted(analysis.communities, key=lambda c: (-len(c.members), c.community_id))
    if len(ordered) > max_reports:
        raise InvalidInput(f"{len(ordered)} communities over the bound of {max_reports}; refused rather than cut")
    reports = tuple([await community_report(engine, model, space, c.community_id, facts_each=facts_each,
                                            limits=limits, status=status, as_of=when) for c in ordered])
    upper = [Passage(f"report:{report.community_id}", report.text(), report.label, when)
             for report in reports if report.text()]
    question = f"What does this space record, across its {len(reports)} communities?"
    made = await synthesize_passages(model, question, upper,
                                     limits=limits or SynthesisLimits(max_passages=max(1, min(50, len(upper)))))
    return SpaceReport(space, projection.digest, projection.revision, reports, made, len(analysis.communities))


class ReportCache:
    """Community reports kept until the graph moves; a stale one is rebuilt, never served."""

    def __init__(self, limit: int = 256) -> None:
        self._kept: dict[tuple[str, str], CommunityReport] = {}
        self._limit = limit

    async def get(self, engine: "MemoryEngine", model: ChatModel, space: str, community_id: str, **options: object
                  ) -> tuple[CommunityReport, bool]:
        """The current report and whether it came from the cache."""
        projection, _ = await load_projection(engine, space, mode=str(options.get("status", "current")))  # type: ignore[arg-type]
        kept = self._kept.get((space, community_id))
        if kept is not None and kept.current(projection):
            return kept, True
        fresh = await community_report(engine, model, space, community_id, **options)  # type: ignore[arg-type]
        if len(self._kept) >= self._limit:
            self._kept.pop(next(iter(self._kept)))
        self._kept[(space, community_id)] = fresh
        return fresh, False
