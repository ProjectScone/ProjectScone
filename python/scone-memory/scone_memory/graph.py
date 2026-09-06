"""A graph of what is recorded, and nothing else.

Nodes are sessions, turns (agent events), tool calls, episodes, chunks,
claims, recalls and judgements. An edge exists only when a stored field
says so: an episode id on an agent event, a source_episode_id on a
claim, a "superseded by fact N" reason, the items list of a recall
event. Similarity between chunks is never an edge; retrieval results
are edges labelled as retrieval evidence with their lane and rank.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from .ports import Event


@dataclass
class Node:
    id: str
    kind: str
    label: str
    ts: Optional[str] = None
    data: dict = field(default_factory=dict)


@dataclass
class Edge:
    source: str
    target: str
    kind: str
    label: Optional[str] = None
    data: dict = field(default_factory=dict)


@dataclass
class Graph:
    nodes: dict[str, Node] = field(default_factory=dict)
    edges: list[Edge] = field(default_factory=list)
    truncated: bool = False

    def add(self, node: Node) -> None:
        self.nodes.setdefault(node.id, node)

    def link(self, source: str, target: str, kind: str, label: Optional[str] = None, **data) -> None:
        if source in self.nodes and target in self.nodes:
            self.edges.append(Edge(source, target, kind, label, data))

    def as_dict(self) -> dict:
        return {
            "nodes": [n.__dict__ for n in self.nodes.values()],
            "edges": [e.__dict__ for e in self.edges],
            "truncated": self.truncated,
            "counts": {kind: sum(1 for n in self.nodes.values() if n.kind == kind) for kind in sorted({n.kind for n in self.nodes.values()})},
        }


def add_agent_event(g: Graph, e: Event) -> None:
    p = e.payload
    session = f"session:{p.get('agent')}:{p.get('session_id')}"
    g.add(Node(session, "session", f"{p.get('agent', 'agent')} session {str(p.get('session_id'))[:8]}", None,
               {"agent": p.get("agent"), "project": p.get("project"), "session_id": p.get("session_id"),
                "provenance": "connector-reported"}))
    kind = str(p.get("event"))
    if kind in ("tool_use", "tool_result"):
        node = Node(f"tool:{e.event_id}", "tool_call", str(p.get("tool_name") or "tool"), e.ts,
                    {"event": kind, "ok": p.get("ok"), "duration_ms": p.get("duration_ms"), "text": p.get("text")})
        g.add(node)
        g.link(session, node.id, "invoked")
    else:
        node = Node(f"turn:{e.event_id}", "turn", kind.replace("_", " "), e.ts,
                    {"event": kind, "text": p.get("text"), "agent": p.get("agent")})
        g.add(node)
        g.link(session, node.id, "has")
    if p.get("episode_id") is not None:
        g.link(node.id, f"episode:{int(p['episode_id'])}", "captured_as")


def add_recall_event(g: Graph, e: Event) -> None:
    p = e.payload
    rid = f"recall:{e.event_id}"
    g.add(Node(rid, "recall", "recall" if p.get("query_hashed") else str(p.get("query", "recall"))[:60], e.ts,
               {"query_hashed": p.get("query_hashed"), "items": len(p.get("items") or []), "latency_ms": (p.get("latency_ms") or {}).get("total")}))
    for item in p.get("items") or []:
        cid = f"chunk:{item['chunk_id']}"
        lanes = item.get("lanes") or {}
        label = "retrieval evidence: " + ", ".join(f"{lane} rank {rank}" for lane, rank in sorted(lanes.items()))
        g.link(rid, cid, "returned", label, similarity=item.get("similarity"), score=item.get("score"), lanes=lanes)


def add_feedback_event(g: Graph, e: Event) -> None:
    p = e.payload
    fid = f"feedback:{e.event_id}"
    g.add(Node(fid, "feedback", "useful" if p.get("useful") else "not useful", e.ts, {"note": p.get("note")}))
    g.link(fid, f"recall:{p.get('recall_event_id')}", "judged")
    g.link(fid, f"chunk:{p.get('chunk_id')}", "judged")


def add_claim_edges(g: Graph) -> None:
    """Edges among claims and to their source episodes, from stored fields."""
    for node in list(g.nodes.values()):
        if node.kind != "claim":
            continue
        src = node.data.get("source_episode_id")
        if src is not None:
            g.link(f"episode:{src}", node.id, "source_of")
        by = node.data.get("superseded_by")
        if by is not None:
            g.link(node.id, f"claim:{int(by)}", "superseded_by")
