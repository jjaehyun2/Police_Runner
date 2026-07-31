"""Focused offline tests for the injected-network OSM pursuit environment.

Covers capture-wins-ties priority, bbox-crossing escape, max-step timeout,
zero-time virtual-split traversal, dead-end stay logging and deterministic
seeded placement.  All fixtures are built in-process; no external OSM calls.
"""
from __future__ import annotations

import math

import pytest

from pursuit_evasion_rl.osm_demo.coarsening import coarsen_raw_graph, prepare_model_network
from pursuit_evasion_rl.osm_demo.environment import (
    FUGITIVE_ID,
    STAY_ACTION,
    OSMRoadPursuitEnv,
)
from pursuit_evasion_rl.osm_demo.models import (
    EpisodeConfig,
    EpisodeOutcome,
    Intersection,
    ModelNetwork,
    RawEdge,
    RawNode,
    RawOSMGraph,
    Segment,
    VehiclePlacement,
)

pytestmark = pytest.mark.offline


# ---------------------------------------------------------------------------
# Fixture builders
# ---------------------------------------------------------------------------
def build_network(intersections: list[dict], segments: list[dict]) -> ModelNetwork:
    coordinates = {item["id"]: (item["x"], item.get("y", 0.0)) for item in intersections}
    seg_objs: list[Segment] = []
    for spec in segments:
        geometry = spec.get("geometry") or (coordinates[spec["start"]], coordinates[spec["end"]])
        length = spec.get("length")
        if length is None:
            length = sum(math.dist(geometry[i], geometry[i + 1]) for i in range(len(geometry) - 1))
        seg_objs.append(
            Segment(
                spec["id"], spec["start"], spec["end"], length, tuple(geometry),
                (), {}, spec.get("virtual", False), spec.get("crosses", False),
            )
        )
    inter_objs: list[Intersection] = []
    for item in intersections:
        outgoing = tuple(sorted(s.id for s in seg_objs if s.start_id == item["id"]))
        incoming = tuple(sorted(s.id for s in seg_objs if s.end_id == item["id"]))
        inter_objs.append(
            Intersection(
                item["id"], coordinates[item["id"]], f"sig-{item['id']}",
                item.get("boundary"), item.get("virtual", False), outgoing, incoming,
            )
        )
    return ModelNetwork(tuple(inter_objs), tuple(seg_objs))


def chain_network(count: int, *, boundary_last: bool = False, crosses_last: bool = False) -> ModelNetwork:
    intersections = [{"id": i, "x": float(i * 100), "y": 0.0} for i in range(count)]
    if boundary_last:
        intersections[-1]["boundary"] = "bbox"
    segments = [{"id": i, "start": i, "end": i + 1} for i in range(count - 1)]
    if crosses_last and segments:
        segments[-1]["crosses"] = True
    return build_network(intersections, segments)


def config(**overrides) -> EpisodeConfig:
    defaults = dict(
        dt_s=0.1,
        police_speed_mps=10.0,
        fugitive_speed_mps=10.0,
        capture_radius_m=5.0,
        max_steps=50,
    )
    defaults.update(overrides)
    return EpisodeConfig(**defaults)


def parked(intersection_id: int) -> VehiclePlacement:
    return VehiclePlacement(intersection_id=intersection_id)


def on_segment(segment_id: int, progress: float) -> VehiclePlacement:
    return VehiclePlacement(segment_id=segment_id, progress=progress)


# ---------------------------------------------------------------------------
# Capture
# ---------------------------------------------------------------------------
def test_metric_capture_when_police_within_radius():
    network = chain_network(8)
    env = OSMRoadPursuitEnv(network, config())
    police = [on_segment(3, 0.52), parked(0), parked(1), parked(2), parked(6), parked(7)]
    env.reset(options={"police": police, "fugitive": on_segment(3, 0.50)})

    actions = {agent: STAY_ACTION for agent in env.agent_ids}
    _, _, terminated, truncated, info = env.step(actions)

    assert env.outcome is EpisodeOutcome.CAPTURE
    assert info["terminal_priority"] == "capture"
    assert all(terminated.values())
    assert not any(truncated.values())


# ---------------------------------------------------------------------------
# Escape and capture-wins-ties
# ---------------------------------------------------------------------------
def test_boundary_crossing_escape_only_through_crosses_boundary_segment():
    network = chain_network(8, boundary_last=True, crosses_last=True)
    env = OSMRoadPursuitEnv(network, config(dt_s=1.0, fugitive_speed_mps=100.0, police_speed_mps=1.0))
    police = [parked(0), parked(1), parked(2), parked(3), parked(4), parked(5)]
    env.reset(options={"police": police, "fugitive": on_segment(6, 0.50)})

    _, _, terminated, truncated, info = env.step({agent: STAY_ACTION for agent in env.agent_ids})

    assert env.outcome is EpisodeOutcome.ESCAPE
    assert info["terminal_priority"] == "escape"
    assert any("boundary_crossed" in event for event in info["events"])
    assert all(terminated.values())


def test_capture_wins_tie_with_escape_same_step():
    network = chain_network(8, boundary_last=True, crosses_last=True)
    env = OSMRoadPursuitEnv(network, config(dt_s=1.0, fugitive_speed_mps=100.0, police_speed_mps=1.0))
    # police_0 waits on the boundary node the fugitive is about to reach.
    police = [parked(7), parked(0), parked(1), parked(2), parked(3), parked(4)]
    env.reset(options={"police": police, "fugitive": on_segment(6, 0.50)})

    _, _, terminated, truncated, info = env.step({agent: STAY_ACTION for agent in env.agent_ids})

    assert env.outcome is EpisodeOutcome.CAPTURE
    assert info["terminal_priority"] == "capture_over_escape"
    assert any("boundary_crossed" in event for event in info["events"])


# ---------------------------------------------------------------------------
# Timeout
# ---------------------------------------------------------------------------
def test_timeout_when_no_capture_or_escape_by_max_steps():
    network = chain_network(8)
    env = OSMRoadPursuitEnv(network, config(max_steps=3, fugitive_speed_mps=10.0))
    police = [parked(0), parked(1), parked(2), parked(5), parked(6), parked(7)]
    env.reset(options={"police": police, "fugitive": on_segment(3, 0.50)})

    outcome = None
    for _ in range(3):
        _, _, terminated, truncated, info = env.step({agent: STAY_ACTION for agent in env.agent_ids})
        outcome = env.outcome

    assert outcome is EpisodeOutcome.TIMEOUT
    assert info["terminal_priority"] == "timeout"
    assert all(truncated.values())


# ---------------------------------------------------------------------------
# Dead-end stay
# ---------------------------------------------------------------------------
def test_dead_end_logs_stay_and_does_not_move():
    network = chain_network(8)
    env = OSMRoadPursuitEnv(network, config())
    police = [parked(7), parked(0), parked(1), parked(2), parked(3), parked(4)]
    env.reset(options={"police": police, "fugitive": on_segment(5, 0.30)})

    # Even a movement action at a dead end resolves to a logged stay.
    actions = {agent: STAY_ACTION for agent in env.agent_ids}
    actions["police_0"] = 0
    observations, _, _, _, info = env.step(actions)

    assert "police_0:stay_no_action" in info["events"]
    # The dead-end node (id 7) is at x = 700.0 and the vehicle stays there.
    assert tuple(observations["police_0"]["position_xy"]) == (700.0, 0.0)


# ---------------------------------------------------------------------------
# Zero-time virtual split traversal
# ---------------------------------------------------------------------------
def _degree_six_network() -> ModelNetwork:
    center = RawNode("center", 36.35, 127.40, 0.0, 0.0, {})
    leaves = []
    edges = []
    for index in range(6):
        angle = math.radians(index * 60)
        x, y = 200.0 * math.cos(angle), 200.0 * math.sin(angle)
        leaf_id = f"leaf-{index}"
        leaves.append(RawNode(leaf_id, 36.35, 127.40, x, y, {}))
        edges.append(
            RawEdge("center", leaf_id, f"edge-{index}", 200.0, ((0.0, 0.0), (x, y)),
                    (f"way-{index}",), True, "residential", {})
        )
    graph = RawOSMGraph(nodes=(center, *leaves), edges=tuple(edges))
    return prepare_model_network(coarsen_raw_graph(graph)).network


def test_virtual_split_traversal_is_resolved_in_zero_simulated_time():
    network = _degree_six_network()
    physical = [item for item in network.intersections if not item.virtual]
    center = next(item for item in physical if item.outgoing_segment_ids and not item.incoming_segment_ids)
    leaves = [item for item in physical if not item.outgoing_segment_ids]
    assert len(leaves) == 6  # exactly enough distinct drivable positions for six police

    env = OSMRoadPursuitEnv(network, config(capture_radius_m=1.0))
    police = [parked(leaf.id) for leaf in leaves]
    env.reset(options={"police": police, "fugitive": parked(center.id)})

    # The continuation slot is last (index 4); follow it, then take a physical
    # exit hosted on the virtual node - all within a single step.
    actions = {agent: STAY_ACTION for agent in env.agent_ids}
    actions[FUGITIVE_ID] = [4, 0]
    _, _, _, _, info = env.step(actions)

    assert any("virtual_hop" in event for event in info["events"])
    assert any(event.startswith(f"{FUGITIVE_ID}:depart") for event in info["events"])
    assert env.outcome is None
    # Exactly one dt elapsed even though several routing microsteps ran.
    assert env.episode_state().simulated_s == pytest.approx(0.1)


# ---------------------------------------------------------------------------
# Deterministic seeded placement
# ---------------------------------------------------------------------------
def test_default_placement_is_deterministic_and_distinct():
    network = chain_network(8)
    first = OSMRoadPursuitEnv(network, config())
    second = OSMRoadPursuitEnv(network, config())
    first.reset(seed=1234)
    second.reset(seed=1234)

    first_state = first.episode_state()
    second_state = second.episode_state()

    first_positions = [p.identity for p in first_state.police] + [first_state.fugitive.identity]
    second_positions = [p.identity for p in second_state.police] + [second_state.fugitive.identity]

    assert first_positions == second_positions  # reproducible
    assert len(set(first_positions)) == len(first_positions)  # distinct, no duplicate fallback
