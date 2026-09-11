"""Where each entity goes when the graph is drawn.

A drawing needs positions, and a force simulation gives different ones on
every run and costs seconds on a large graph. This lays the graph out by
construction instead:

- **By community.** Each community is laid out around its most central
  member, in rings by how many steps away each member is, every ring
  wide enough that no two of its members touch and far enough out that
  no two rings do. A member is placed near the member that reached it.
- **Packed.** Community boxes go in rows, largest first; entities with no
  relation to another share a box of their own.
- **Bounded.** The ``max_nodes`` most connected entities and the
  ``max_edges`` best supported relations between them are drawn, and
  what was left out is counted for the drawing to say.

The same graph always draws the same way, and nothing overlaps: not two
entities, not two communities.
"""

from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import dataclass
import math
from typing import Callable

from .analysis import cached_analysis
from .project import Entity, EntityProjection, Relation

MAX_NODES = 200
MAX_EDGES = 600
#: Room between rings, between members of one ring, and inside a box, and
#: above a box's members for its title and below them for their names.
_GAP = 24.0
_SPACING = 6.0
_PAD = 24.0
_TITLE_ROOM, _NAME_ROOM = 28.0, 18.0
_MIN_RADIUS, _MAX_RADIUS = 6.0, 22.0
UNLINKED = "unlinked"


@dataclass(frozen=True)
class Placed:
    entity_id: str
    label: str
    kind: str | None
    group_id: str
    x: float
    y: float
    radius: float


@dataclass(frozen=True)
class Box:
    group_id: str
    label: str
    x: float
    y: float
    width: float
    height: float
    #: Which of the drawing's colours; the box of unlinked entities has the last.
    colour: int


@dataclass(frozen=True)
class Drawing:
    nodes: tuple[Placed, ...]
    groups: tuple[Box, ...]
    edges: tuple[Relation, ...]
    #: Entities and relations of the projection not drawn.
    left_out: tuple[int, int]
    width: float
    height: float


def _degrees(projection: EntityProjection) -> dict[str, int]:
    degree = {entity.entity_id: 0 for entity in projection.entities}
    for relation in projection.relations:
        degree[relation.subject_id] += 1
        if relation.object_id != relation.subject_id:
            degree[relation.object_id] += 1
    return degree


def _rings(members: list[Entity], centre: str, neighbours: dict[str, list[str]],
           radius: dict[str, float]) -> dict[str, tuple[float, float]]:
    """Offsets from the centre: breadth first from ``centre``, one ring per
    step, members a ring cannot reach on the ring past the last."""
    depth, parent = {centre: 0}, {centre: centre}
    queue = deque([centre])
    while queue:
        at = queue.popleft()
        for near in neighbours[at]:
            if near not in depth:
                depth[near], parent[near] = depth[at] + 1, at
                queue.append(near)
    outer = max(depth.values()) + 1
    rings: dict[int, list[str]] = defaultdict(list)
    for member in members:
        rings[depth.get(member.entity_id, outer)].append(member.entity_id)
    placed: dict[str, tuple[float, float]] = {centre: (0.0, 0.0)}
    angle: dict[str, float] = {centre: -math.pi / 2}
    previous_radius, previous_widest = 0.0, radius[centre]
    for level in sorted(level for level in rings if level > 0):
        # Near the member that reached each one, then as the ranking has it.
        ring = sorted(rings[level], key=lambda entity_id: angle.get(parent.get(entity_id, ""), math.pi * 2))
        widest = max(radius[entity_id] for entity_id in ring)
        across = 2 * widest + _SPACING
        needed = across / (2 * math.sin(math.pi / len(ring))) if len(ring) > 1 else 0.0
        ring_radius = max(previous_radius + previous_widest + widest + _GAP, needed)
        for index, entity_id in enumerate(ring):
            theta = -math.pi / 2 + 2 * math.pi * index / len(ring)
            angle[entity_id] = theta
            placed[entity_id] = (ring_radius * math.cos(theta), ring_radius * math.sin(theta))
        previous_radius, previous_widest = ring_radius, widest
    return placed


def layout_projection(projection: EntityProjection, *, max_nodes: int = MAX_NODES, max_edges: int = MAX_EDGES,
                      resolution: float = 1.0, radii: tuple[float, float] = (_MIN_RADIUS, _MAX_RADIUS),
                      room: Callable[[Entity, float], float] | None = None) -> Drawing:
    """Positions for the ``max_nodes`` most connected entities, by community.
    An entity's radius grows with its relations, within ``radii``. ``room``
    says how far from its centre an entity's drawing reaches, its name
    included, given its radius; rings and boxes are spaced by that, so a
    name never runs into another entity or out of its box."""
    degree = _degrees(projection)
    ranked = sorted(projection.entities, key=lambda e: (-degree[e.entity_id], e.label.casefold(), e.entity_id))
    shown = ranked[:max_nodes]
    ids = {entity.entity_id for entity in shown}
    between = [r for r in projection.relations if r.subject_id in ids and r.object_id in ids]
    edges = sorted(between, key=lambda r: (-len(r.fact_ids), r.subject_id, r.predicate, r.object_id))[:max_edges]
    left_out = (len(projection.entities) - len(shown), len(projection.relations) - len(edges))
    if not shown:
        return Drawing((), (), (), left_out, 2 * _PAD, 2 * _PAD)
    busiest = max(degree[entity.entity_id] for entity in shown) or 1
    smallest, largest = radii
    radius = {entity.entity_id: smallest + (largest - smallest) * math.sqrt(degree[entity.entity_id] / busiest)
              for entity in shown}
    reach = {entity.entity_id: room(entity, radius[entity.entity_id]) if room else radius[entity.entity_id]
             for entity in shown}
    neighbours: dict[str, list[str]] = defaultdict(list)
    for relation in between:
        if relation.subject_id != relation.object_id:
            neighbours[relation.subject_id].append(relation.object_id)
            neighbours[relation.object_id].append(relation.subject_id)
    order = {entity.entity_id: index for index, entity in enumerate(ranked)}
    for entity_id in neighbours:
        neighbours[entity_id] = sorted(set(neighbours[entity_id]), key=order.__getitem__)

    groups: list[tuple[str, str, list[Entity], str]] = []
    placed_ids: set[str] = set()
    for community in cached_analysis(projection, resolution).communities:
        members = [entity for entity in shown if entity.entity_id in set(community.members)]
        if members:
            centre = next((member for member in community.top_entities if member in ids), members[0].entity_id)
            groups.append((community.community_id, community.label, members, centre))
            placed_ids.update(member.entity_id for member in members)
    loose = [entity for entity in shown if entity.entity_id not in placed_ids]
    groups.sort(key=lambda group: (-len(group[2]), group[0]))
    if loose:
        groups.append((UNLINKED, "no relation", loose, loose[0].entity_id))

    # Each group laid out around its centre, boxed as tightly as its
    # members allow, then the boxes packed in rows.
    laid = []
    for group_id, label, members, centre in groups:
        offsets = _rings(members, centre, neighbours if group_id != UNLINKED else defaultdict(list), reach)
        left = min(offsets[m.entity_id][0] - reach[m.entity_id] for m in members)
        right = max(offsets[m.entity_id][0] + reach[m.entity_id] for m in members)
        top = min(offsets[m.entity_id][1] - reach[m.entity_id] for m in members)
        bottom = max(offsets[m.entity_id][1] + reach[m.entity_id] for m in members)
        size = (right - left + 2 * _PAD, bottom - top + 2 * _PAD + _TITLE_ROOM + _NAME_ROOM)
        laid.append((group_id, label, members, offsets, (left, top), size))
    row_limit = max(max(size[0] for *_, size in laid), math.sqrt(sum(w * h for *_, (w, h) in laid)) * 1.3)
    nodes: list[Placed] = []
    boxes: list[Box] = []
    x = y = row_height = width = 0.0
    for index, (group_id, label, members, offsets, (left, top), (box_width, box_height)) in enumerate(laid):
        if x > 0 and x + box_width > row_limit:
            x, y, row_height = 0.0, y + row_height + _GAP, 0.0
        colour = -1 if group_id == UNLINKED else index
        boxes.append(Box(group_id, label, x, y, box_width, box_height, colour))
        origin_x, origin_y = x + _PAD - left, y + _PAD + _TITLE_ROOM - top
        for member in members:
            dx, dy = offsets[member.entity_id]
            nodes.append(Placed(member.entity_id, member.label, member.kind, group_id, origin_x + dx, origin_y + dy,
                                radius[member.entity_id]))
        x += box_width + _GAP
        width, row_height = max(width, x - _GAP), max(row_height, box_height)
    return Drawing(tuple(nodes), tuple(boxes), tuple(edges), left_out, width, y + row_height)
