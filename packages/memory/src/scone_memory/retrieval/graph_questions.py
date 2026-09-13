"""Questions the evidence graph is placed to ask, with the evidence named.

`graph_analysis` finds communities, hubs and bridges over the recorded
concept topology; the evidence graph itself carries typed links between
claims and chunk-to-concept mentions. All of that was reported as ids and
counts and left to the reader. Five shapes in it are each an open
question a person would not think to ask, and each can say exactly which
edge or node made it ask:

- **contradiction** -- a stored `contradicts` link between two claims:
  which side holds?
- **replacement** -- a `superseded_by` edge: what changed, and when?
- **lone_bridge** -- a cut edge between two communities the analysis
  found: is that really the only link?
- **isolated** -- a concept no recorded relation reaches, beside concepts
  that are linked: unread, or unrelated?
- **co_mention** -- two concepts that recur together in passages with no
  claim in this evidence relating them: what is the relationship? The
  ledger may hold one that was not recalled; the question is about what
  the graph shows.

Each has its other half. A `supports` link and a recorded relation ask
nothing: that is what the graph is for, and a generator that always asks
is not asking. A cut edge inside a tree is not a lone bridge: a
query-scoped graph is nearly always a forest, and there every internal
edge is one. When no concept is linked at all, every concept is
unlinked, and asking about each is noise; the analysis's `method`
already says so.

Nothing here is produced by a model and nothing is asserted: each is a
question, and its `why` says what prompted it. The result is bounded,
counts what it cut, cuts the weakest shape first, and carries the
analysis's own status beside it so a missing question is never read as a
missing shape.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Callable
from typing import Literal

from pydantic import BaseModel, Field, JsonValue

from .evidence_graph import EvidenceEdge, EvidenceNode, QueryEvidenceGraph
from .graph_analysis import GraphAnalysisResult

#: Questions returned at most. The rest are counted, never silently lost.
MAX_QUESTIONS = 12

#: Passages named on a co-mention question. `chunk_count` is always whole.
MAX_CHUNKS_NAMED = 3

#: Characters of a label quoted in a question. The analysis never copies text; a
#: question has to, to be readable, and this is how much. Evidence ids stay exact.
MAX_LABEL_CHARS = 80

QuestionType = Literal["contradiction", "replacement", "lone_bridge", "isolated", "co_mention"]

class SuggestedQuestion(BaseModel):
    type: QuestionType
    question: str
    why: str
    evidence: dict[str, JsonValue]


class Questions(BaseModel):
    questions: list[SuggestedQuestion] = Field(default_factory=list)
    omitted: int = 0
    truncated: bool = False
    analysis_status: Literal["complete", "partial", "unavailable"] = "complete"
    analysis_method: Literal["label_propagation", "components", "unlinked"] = "unlinked"
    reason: str | None = None


def questions_for(graph: QueryEvidenceGraph, analysis: GraphAnalysisResult) -> Questions:
    """What the graph is placed to ask, from what it holds and what the analysis found."""
    if analysis.status == "unavailable":
        reasons = analysis.coverage.reasons
        return Questions(analysis_status="unavailable", analysis_method=analysis.method,
                         reason=reasons[0] if reasons else "unavailable")
    nodes = {node.id: node for node in graph.nodes}

    def name(identifier: str) -> str:
        node = nodes.get(identifier)
        label = node.label if node is not None else identifier
        return label if len(label) <= MAX_LABEL_CHARS else label[:MAX_LABEL_CHARS - 1] + "\u2026"

    # Strongest shape first, and the bound cuts from the end: a co-mention --
    # the weakest, and the one that can multiply -- never survives at a
    # contradiction's expense. The order of this list is that rule.
    found = [*_typed_links(graph.edges, nodes, name), *_lone_bridges(analysis, name), *_isolated(analysis, name)]
    # Co-mentions can be quadratic in the names one passage holds, so only as
    # many are built as there is room for; the rest are counted, not made.
    pairs, uncounted = _co_mentions(graph, name, room=max(0, MAX_QUESTIONS - len(found)))
    found.extend(pairs)
    kept = found[:MAX_QUESTIONS]
    omitted = len(found) - len(kept) + uncounted
    return Questions(questions=kept, omitted=omitted, truncated=omitted > 0,
                     analysis_status=analysis.status, analysis_method=analysis.method)


def _typed_links(edges: list[EvidenceEdge], nodes: dict[str, EvidenceNode],
                 name: Callable[[str], str]) -> list[SuggestedQuestion]:
    out: list[SuggestedQuestion] = []
    for edge in sorted(edges, key=lambda edge: (edge.source, edge.target)):
        if edge.kind == "contradicts":
            raw = edge.data.get("provenance_status")
            status = raw if isinstance(raw, str) else "unknown"
            why = "a stored `contradicts` link joins them"
            if status != "retained":
                why += f"; the link's own evidence is {status}"
            out.append(SuggestedQuestion(
                type="contradiction",
                question=f"`{name(edge.source)}` and `{name(edge.target)}` contradict each other -- which holds?",
                why=why,
                evidence={"edge_kind": "contradicts", "source": edge.source, "target": edge.target,
                          "provenance_status": status}))
        elif edge.kind == "superseded_by":
            newer = nodes.get(edge.target)
            since = newer.ts if newer is not None else None
            why = "a `superseded_by` edge joins them"
            if since is not None:
                why += f"; the newer claim is valid from {since}"
            out.append(SuggestedQuestion(
                type="replacement",
                question=f"`{name(edge.source)}` was superseded by `{name(edge.target)}` -- what changed, and when?",
                why=why,
                evidence={"edge_kind": "superseded_by", "source": edge.source, "target": edge.target, "since": since}))
    return out


def _lone_bridges(analysis: GraphAnalysisResult, name: Callable[[str], str]) -> list[SuggestedQuestion]:
    # A query-scoped graph is nearly always a forest, where every internal
    # edge is a cut edge; asking about each would fire on almost every
    # query (23 of 24 on this package's source). A lone bridge is a cut
    # edge that joins two communities the analysis found. That rule also
    # excludes a leaf's only edge on its own: a leaf always joins its
    # neighbour's community, so its edge is never cross-community.
    return [SuggestedQuestion(
                type="lone_bridge",
                question=f"Is `{name(bridge.source)} -> {name(bridge.target)}` really the only link between its two sides?",
                why="removing this one recorded relation disconnects the two communities it joins",
                evidence={"source": bridge.source, "target": bridge.target, "relation_count": bridge.relation_count})
            for bridge in analysis.bridges if bridge.cut_edge and bridge.cross_community]


def _isolated(analysis: GraphAnalysisResult, name: Callable[[str], str]) -> list[SuggestedQuestion]:
    if analysis.method == "unlinked":
        return []
    linked = sum(len(community.node_ids) for community in analysis.communities if not community.isolated)
    out: list[SuggestedQuestion] = []
    for community in analysis.communities:
        if not community.isolated or not community.node_ids:
            continue
        shown = ", ".join(f"`{name(identifier)}`" for identifier in community.node_ids[:3])
        more = f" and {len(community.node_ids) - 3} more" if len(community.node_ids) > 3 else ""
        out.append(SuggestedQuestion(
            type="isolated",
            question=f"Nothing recorded links {shown}{more} to the {linked} linked concepts -- unread, or unrelated?",
            why="no recorded relation reaches these concepts, while others in the same evidence are linked",
            evidence={"node_ids": list(community.node_ids), "linked_concepts": linked}))
    return out


def _co_mentions(graph: QueryEvidenceGraph, name: Callable[[str], str], room: int) -> tuple[list[SuggestedQuestion], int]:
    """At most ``room`` questions, strongest first, and how many more there were."""
    concepts = {node.id for node in graph.nodes if node.kind == "concept"}
    stated = {frozenset((edge.source, edge.target)) for edge in graph.edges
              if edge.kind == "relation" and edge.source in concepts and edge.target in concepts}
    by_chunk: dict[str, set[str]] = defaultdict(set)
    for edge in graph.edges:
        if edge.kind == "mentions" and edge.target in concepts:
            by_chunk[edge.source].add(edge.target)
    pairs: dict[tuple[str, str], list[str]] = defaultdict(list)
    for chunk_id in sorted(by_chunk):
        named = sorted(by_chunk[chunk_id])
        for index, left in enumerate(named):
            for right in named[index + 1:]:
                if frozenset((left, right)) not in stated:
                    pairs[(left, right)].append(chunk_id)
    out: list[SuggestedQuestion] = []
    ranked = sorted(pairs.items(), key=lambda item: (-len(item[1]), item[0]))
    for (left, right), chunks in ranked[:room]:
        count = len(chunks)
        out.append(SuggestedQuestion(
            type="co_mention",
            question=(f"`{name(left)}` and `{name(right)}` appear together in {count} passage{'s' if count != 1 else ''}, "
                      "and no claim in this evidence relates them -- what is the relationship?"),
            why="the same retained text mentions both; in this evidence only that literal mention ties them",
            evidence={"concepts": [left, right], "chunks": [*chunks[:MAX_CHUNKS_NAMED]], "chunk_count": count}))
    return out, max(0, len(ranked) - room)
