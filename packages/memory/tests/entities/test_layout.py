"""Where each entity goes when the graph is drawn.

A drawing needs positions. Each community is laid out around its most
central member, in rings by how many steps away each member is, and the
communities are packed in rows, largest first. It is computed, not
simulated, so the same graph always draws the same way, and nothing
overlaps: not two entities, not two communities.
"""

from __future__ import annotations

import math

import pytest

from scone_memory.core.models import Fact
from scone_memory.entities.layout import layout_projection
from scone_memory.entities.project import project_entities

DAY = "2024-01-01T00:00:00.000Z"


def graph(*triples: tuple[str, str, str]):
    return project_entities("alpha", [
        Fact(fact_id=index, space="alpha", subject=subject, predicate=predicate, object=obj, valid_from=DAY)
        for index, (subject, predicate, obj) in enumerate(triples, start=1)], revision=1)


def three_groups():
    triples = [(f"worker {n}", "works_at", "Acme Robotics") for n in range(6)]
    triples += [(f"resident {n}", "lives_in", "Porto") for n in range(4)]
    triples += [("hana kim", "studies_at", "Quantum Lab"), ("ivan oz", "studies_at", "Quantum Lab"),
                ("hana kim", "knows", "ivan oz")]
    return graph(*triples)


def test_the_same_graph_is_always_laid_out_the_same_way():
    first, second = layout_projection(three_groups()), layout_projection(three_groups())
    assert first == second and len(first.nodes) == 15


def test_no_two_entities_overlap():
    drawn = layout_projection(three_groups())
    for left in drawn.nodes:
        for right in drawn.nodes:
            if left.entity_id < right.entity_id:
                assert math.dist((left.x, left.y), (right.x, right.y)) >= left.radius + right.radius, (left, right)


def test_each_community_is_a_box_holding_its_members_and_no_two_boxes_meet():
    drawn = layout_projection(three_groups())
    boxes = {box.group_id: box for box in drawn.groups}
    assert len(boxes) == 3
    for node in drawn.nodes:
        box = boxes[node.group_id]
        assert box.x <= node.x - node.radius and node.x + node.radius <= box.x + box.width
        assert box.y <= node.y - node.radius and node.y + node.radius <= box.y + box.height
    ordered = sorted(boxes.values(), key=lambda box: (box.x, box.y))
    for index, left in enumerate(ordered):
        for right in ordered[index + 1:]:
            apart = (left.x + left.width + 24 <= right.x or right.x + right.width + 24 <= left.x
                     or left.y + left.height + 24 <= right.y or right.y + right.height + 24 <= left.y)
            assert apart, (left, right)
    assert all(0 <= box.x and box.x + box.width <= drawn.width and 0 <= box.y and box.y + box.height <= drawn.height
               for box in boxes.values())


def test_the_busiest_entity_is_drawn_largest_with_its_neighbours_in_a_ring_around_it():
    drawn = layout_projection(three_groups())
    acme = next(node for node in drawn.nodes if node.label == "Acme Robotics")
    assert acme.radius == max(node.radius for node in drawn.nodes)
    workers = [node for node in drawn.nodes if node.label.startswith("worker")]
    distances = {round(math.dist((acme.x, acme.y), (node.x, node.y)), 6) for node in workers}
    assert len(distances) == 1, "one step from the centre, one ring"


def test_each_box_is_as_small_as_its_members_allow():
    drawn = layout_projection(three_groups())
    for box in drawn.groups:
        members = [node for node in drawn.nodes if node.group_id == box.group_id]
        assert min(node.x - node.radius for node in members) - box.x == pytest.approx(24.0)
        assert box.x + box.width - max(node.x + node.radius for node in members) == pytest.approx(24.0)


def test_the_most_connected_are_drawn_and_the_rest_counted():
    drawn = layout_projection(three_groups(), max_nodes=5, max_edges=3)
    assert len(drawn.nodes) == 5 and drawn.left_out == (10, 10)
    assert len(drawn.edges) == 3 and {node.label for node in drawn.nodes} >= {"Acme Robotics", "Porto"}
    shown = {node.entity_id for node in drawn.nodes}
    assert all(edge.subject_id in shown and edge.object_id in shown for edge in drawn.edges)


def test_an_entity_with_no_relation_sits_apart():
    drawn = layout_projection(graph(("alice chen", "works_at", "Acme"), ("zed", "joined_on", "May 2021")))
    zed = next(node for node in drawn.nodes if node.label == "zed")
    assert zed.group_id == "unlinked" and any(box.label == "no relation" for box in drawn.groups)


def test_an_empty_graph_is_an_empty_drawing():
    drawn = layout_projection(graph())
    assert drawn.nodes == () and drawn.groups == () and drawn.width > 0 and drawn.height > 0


def test_many_communities_wrap_into_rows():
    triples = [(f"member {group} {n}", "belongs_to", f"club {group}") for group in range(9) for n in range(3)]
    drawn = layout_projection(graph(*triples))
    assert len(drawn.groups) == 9 and len({box.y for box in drawn.groups}) >= 2
    assert drawn.width <= 2 * drawn.height


def test_the_best_supported_relations_are_drawn_first():
    projection = project_entities("alpha", [
        Fact(fact_id=1, space="alpha", subject="alice chen", predicate="works_with", object="Bob",
             valid_from="2020-01-01T00:00:00.000Z", valid_until="2021-01-01T00:00:00.000Z", status="closed"),
        Fact(fact_id=2, space="alpha", subject="alice chen", predicate="knows", object="Carol", valid_from=DAY),
        Fact(fact_id=3, space="alpha", subject="alice chen", predicate="works_with", object="Bob", valid_from=DAY)],
        revision=1)
    [edge] = layout_projection(projection, max_edges=1).edges
    assert edge.predicate == "works_with" and edge.fact_ids == (1, 3)


def test_a_crowded_ring_widens_so_its_members_never_touch():
    drawn = layout_projection(graph(*[(f"worker {n}", "works_at", "Acme Robotics") for n in range(30)]))
    for left in drawn.nodes:
        for right in drawn.nodes:
            if left.entity_id < right.entity_id:
                assert math.dist((left.x, left.y), (right.x, right.y)) >= left.radius + right.radius


def test_each_entity_takes_the_room_its_caller_says_it_needs():
    """A drawing that writes a name under each circle passes the room the
    name needs; rings and boxes are spaced by that room, not the circle."""
    triples = [(f"worker {n}", "works_at", "Acme Robotics") for n in range(20)]
    drawn = layout_projection(graph(*triples), room=lambda entity, radius: radius + 60)
    for left in drawn.nodes:
        for right in drawn.nodes:
            if left.entity_id < right.entity_id:
                assert math.dist((left.x, left.y), (right.x, right.y)) >= left.radius + right.radius + 120
    [box] = drawn.groups
    assert all(box.x <= node.x - node.radius - 60 and node.x + node.radius + 60 <= box.x + box.width
               for node in drawn.nodes)
