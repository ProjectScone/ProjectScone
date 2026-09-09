"""Build recorded activity graphs from document and event storage ports.

The public engine validates space and clamps the event/provenance window first.
Graph assembly performs reads only; storage references are captured at dispatch.
The window is not a total node or fact-scan limit.
"""
from __future__ import annotations

from typing import Optional, Sequence, TypedDict, cast

from ..core.ports import DocumentStore, EventLog
from . import graph as G


class _RecallEventItem(TypedDict):
    """The chunk identity read from an engine-authored recall event."""

    chunk_id: int


async def build_activity_graph(
    documents: DocumentStore,
    events: EventLog | None,
    space: str,
    session_id: Optional[str] = None,
    episode_id: Optional[int] = None,
    since: Optional[str] = None,
    limit: int = 400,
) -> G.Graph:
    """Hydrate stored sources and typed edges without inferring relationships."""
    g = G.Graph()
    if events is None:
        return g
    focused = session_id is not None or episode_id is not None
    # A focused graph must reach the events that mention its subject even
    # when they are older than the window, and must leave out agent
    # events that merely happened nearby. A session focus keeps that
    # session's events; an episode focus keeps the agent events linked to
    # that episode. Unfocused: the newest window, as is.
    window = await events.query(space, since=since, limit=limit)
    g.truncated = len(window) >= limit
    if focused:
        scan = await events.query(space, kind="agent", since=since, limit=2000)
        g.truncated = g.truncated or len(scan) >= 2000
        if session_id is not None:
            agent_events = [e for e in scan if e.payload.get("session_id") == session_id]
        else:
            agent_events = [e for e in scan if e.payload.get("episode_id") == episode_id]
    else:
        agent_events = [e for e in window if e.kind == "agent"]
    recall_events = [e for e in window if e.kind == "recall" and "error" not in e.payload]
    feedback_events = [e for e in window if e.kind == "feedback"]

    episode_ids: set[int] = set()
    if episode_id is not None:
        episode_ids.add(episode_id)
    for e in agent_events:
        if e.payload.get("episode_id") is not None:
            episode_ids.add(int(cast(int, e.payload["episode_id"])))
    chunk_ids: set[int] = set()
    for e in recall_events:
        for item in cast(Sequence[_RecallEventItem], e.payload.get("items") or []):
            chunk_ids.add(int(item["chunk_id"]))
    chunks = await documents.get_chunks(space, sorted(chunk_ids)) if chunk_ids else []
    if focused:
        keep = {c.chunk_id for c in chunks if c.episode_id in episode_ids}
        recall_events = [e for e in recall_events if any(int(i["chunk_id"]) in keep for i in cast(Sequence[_RecallEventItem], e.payload.get("items") or []))]
        chunks = [c for c in chunks if c.chunk_id in keep]
    for c in chunks:
        episode_ids.add(c.episode_id)

    def draw_episode(ep) -> None:
        g.add(G.Node(f"episode:{ep.episode_id}", "episode", ep.content[:80], ep.created_at,
                     {"kind": ep.kind, "source": ep.source, "tags": list(ep.tags),
                      "metadata": dict(ep.metadata), "content": ep.content}))

    for eid in sorted(episode_ids):
        ep = await documents.get_episode(space, eid)
        if ep is None:
            continue
        draw_episode(ep)
    for c in chunks:
        if f"episode:{c.episode_id}" in g.nodes:
            g.add(G.Node(f"chunk:{c.chunk_id}", "chunk", c.text[:80], c.created_at, {"ordinal": c.ordinal, "start": c.start, "end": c.end, "text": c.text}))
            g.link(f"episode:{c.episode_id}", f"chunk:{c.chunk_id}", "chunked_into")
    facts = await documents.list_facts(space, include_closed=True)
    wanted = [f for f in facts if (f.source_episode_id in episode_ids) or not focused]
    # A claim's source is part of the claim. Drawing the claim while
    # leaving out the episode it came from turns provenance into a
    # silent absence, which reads as "no evidence" rather than "not in
    # this snapshot". Sources the window never mentioned are hydrated
    # here, bounded the same way the window is, and whatever does not
    # fit is counted so the snapshot can say what it left out. An
    # episode that was forgotten has no node and no edge, which is the
    # truth about it rather than an omission.
    elsewhere = list(dict.fromkeys(
        f.source_episode_id for f in wanted
        if f.source_episode_id is not None and f"episode:{f.source_episode_id}" not in g.nodes))
    for eid in sorted(elsewhere)[:limit]:
        ep = await documents.get_episode(space, eid)
        if ep is None:
            # The claim names a source that is gone. That is not a
            # budget: it is evidence that was forgotten, and saying so
            # separately is the difference between "out of view" and
            # "no longer exists".
            g.provenance_missing += 1
        else:
            draw_episode(ep)
    g.provenance_omitted = max(0, len(elsewhere) - limit)
    for f in wanted:
        g.add(G.Node(f"claim:{f.fact_id}", "claim", f"{f.subject} {f.predicate.replace('_', ' ')} {f.object}", f.valid_from, {
            "status": f.status, "origin": f.origin, "confidence": f.confidence, "valid_from": f.valid_from,
            "valid_until": f.valid_until, "closed_reason": f.closed_reason, "excluded_reason": f.excluded_reason,
            "source_episode_id": f.source_episode_id, "superseded_by": f.superseded_by,
        }))
    # The typed relations among the claims drawn, each once.
    drawn = set()
    for f in wanted:
        for link in await documents.fact_links(space, f.fact_id):
            ends = (f"claim:{link.from_fact}", f"claim:{link.to_fact}")
            if link.link_id not in drawn and ends[0] in g.nodes and ends[1] in g.nodes:
                drawn.add(link.link_id)
                g.link(ends[0], ends[1], link.kind, source_episode_id=link.source_episode_id)
    if not focused:
        captured = {f"episode:{e.payload.get('episode_id')}" for e in agent_events}
        missing_capture = {n.id for n in g.nodes.values() if n.kind == "episode"} - captured
        if missing_capture:
            # Hydrating an old source without its capture event leaves a
            # real session path broken. Match stored IDs only, within the
            # same space/time scope, and keep this second read bounded.
            captures = await events.query(space, kind="agent", since=since, limit=2000)
            matching = [e for e in captures if f"episode:{e.payload.get('episode_id')}" in missing_capture]
            agent_events.extend(matching[:limit])
            g.truncated = g.truncated or len(captures) >= 2000 or len(matching) > limit
    for e in agent_events:
        G.add_agent_event(g, e)
    for e in recall_events:
        G.add_recall_event(g, e)
        for fid in cast(Sequence[int], e.payload.get("fact_ids") or []):
            g.link(f"recall:{e.event_id}", f"claim:{int(fid)}", "held")
    for e in feedback_events:
        if f"recall:{e.payload.get('recall_event_id')}" in g.nodes:
            G.add_feedback_event(g, e)
    G.add_claim_edges(g)
    return g
