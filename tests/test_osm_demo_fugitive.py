"""Focused offline tests for the directed-network heuristic fugitive.

Covers escape-seeking when no police are visible, separation-maximizing flight
when police are visible, dead-end stay, seeded determinism of both the raw
policy and a full driven episode, and injected-RNG reproducibility.  Every
fixture is built in-process; no external OSM calls are made.

Requirements: 9.1-9.3, 14.1
"""
from __future__ import annotations

import math

import numpy as np
import pytest

from pursuit_evasion_rl.osm_demo.coarsening import ordered_outgoing_segment_ids
from pursuit_evasion_rl.osm_demo.environment import STAY_ACTION, OSMRoadPursuitEnv
from pursuit_evasion_rl.osm_demo.fugitive import FugitiveObservation, OSMHeuristicFugitive
from pursuit_evasion_rl.osm_demo.models import (
    EpisodeConfig,
    EpisodeOutcome,
    Intersection,
    ModelNetwork,
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


def config(**overrides) -> EpisodeConfig:
    defaults = dict(
        dt_s=1.0,
        police_speed_mps=1.0,
        fugitive_speed_mps=100.0,
        capture_radius_m=5.0,
        max_steps=50,
    )
    defaults.update(overrides)
    return EpisodeConfig(**defaults)


def parked(intersection_id: int) -> VehiclePlacement:
    return VehiclePlacement(intersection_id=intersection_id)


def escape_branch_network() -> ModelNetwork:
    """Node 0 forks toward an escape chain (via node 1) and a dead end (node 5)."""
    intersections = [
        {"id": 0, "x": 0.0, "y": 0.0},
        {"id": 1, "x": 100.0, "y": 0.0},
        {"id": 2, "x": 200.0, "y": 0.0},
        {"id": 3, "x": 300.0, "y": 0.0},
        {"id": 4, "x": 400.0, "y": 0.0, "boundary": "bbox"},
        {"id": 5, "x": 0.0, "y": -100.0},  # dead end, no route to the boundary
    ]
    segments = [
        {"id": 0, "start": 0, "end": 1},
        {"id": 1, "start": 1, "end": 2},
        {"id": 2, "start": 2, "end": 3},
        {"id": 3, "start": 3, "end": 4, "crosses": True},
        {"id": 4, "start": 0, "end": 5},
    ]
    return build_network(intersections, segments)


def driven_episode_network() -> ModelNetwork:
    """Escape line 0->..->4(boundary) plus a far-south police holding chain."""
    intersections = [
        {"id": 0, "x": 0.0, "y": 0.0},
        {"id": 1, "x": 100.0, "y": 0.0},
        {"id": 2, "x": 200.0, "y": 0.0},
        {"id": 3, "x": 300.0, "y": 0.0},
        {"id": 4, "x": 400.0, "y": 0.0, "boundary": "bbox"},
        # Police holding chain, all far (>= 1000 m) from the escape route.
        {"id": 5, "x": 0.0, "y": -1000.0},
        {"id": 6, "x": 0.0, "y": -1100.0},
        {"id": 7, "x": 0.0, "y": -1200.0},
        {"id": 8, "x": 0.0, "y": -1300.0},
        {"id": 9, "x": 0.0, "y": -1400.0},
    ]
    segments = [
        {"id": 0, "start": 0, "end": 1},
        {"id": 1, "start": 1, "end": 2},
        {"id": 2, "start": 2, "end": 3},
        {"id": 3, "start": 3, "end": 4, "crosses": True},
        {"id": 4, "start": 0, "end": 5},
        {"id": 5, "start": 5, "end": 6},
        {"id": 6, "start": 6, "end": 7},
        {"id": 7, "start": 7, "end": 8},
        {"id": 8, "start": 8, "end": 9},
    ]
    return build_network(intersections, segments)


def flee_fork_network() -> ModelNetwork:
    """Node 0 forks east (node 1) and west (node 2); neither is an escape."""
    intersections = [
        {"id": 0, "x": 0.0, "y": 0.0},
        {"id": 1, "x": 100.0, "y": 0.0},
        {"id": 2, "x": -100.0, "y": 0.0},
    ]
    segments = [
        {"id": 0, "start": 0, "end": 1},
        {"id": 1, "start": 0, "end": 2},
    ]
    return build_network(intersections, segments)


# ---------------------------------------------------------------------------
# Escape-seeking (no visible police)
# ---------------------------------------------------------------------------
def test_seeks_escape_when_no_police_visible():
    network = escape_branch_network()
    fugitive = OSMHeuristicFugitive(vision_range_m=10.0, seed=0)
    ordered = ordered_outgoing_segment_ids(network, 0)
    # Police are far away (parked at the dead end), so none are visible.
    observation = FugitiveObservation(
        intersection_id=0,
        ordered_segment_ids=ordered,
        police_xy=((0.0, -100.0),) * 6,
    )

    action = fugitive.act(network, observation)

    # The chosen slot must be the one heading toward the escape chain (segment 0),
    # not the dead-end branch (segment 4) whose escape distance is infinite.
    assert ordered[action] == 0


def test_dead_end_returns_stay():
    network = escape_branch_network()
    fugitive = OSMHeuristicFugitive(seed=0)
    # Node 5 is a dead end with no outgoing segments.
    observation = FugitiveObservation(intersection_id=5, ordered_segment_ids=(), police_xy=())

    assert fugitive.act(network, observation) == STAY_ACTION


# ---------------------------------------------------------------------------
# Separation-maximizing flight (visible police)
# ---------------------------------------------------------------------------
def test_flees_to_maximize_separation_from_visible_police():
    network = flee_fork_network()
    fugitive = OSMHeuristicFugitive(vision_range_m=150.0, seed=0)
    ordered = ordered_outgoing_segment_ids(network, 0)
    # A single visible officer sits on the east branch destination (node 1).
    observation = FugitiveObservation(
        intersection_id=0,
        ordered_segment_ids=ordered,
        police_xy=((100.0, 0.0),) * 6,
    )

    action = fugitive.act(network, observation)

    # The fugitive should flee west toward node 2 (segment 1), away from police.
    assert ordered[action] == 1


def test_ignores_police_outside_vision_range_and_seeks_escape():
    network = escape_branch_network()
    # Police at node 5 direction is 100m away; vision range 10m keeps it hidden.
    fugitive = OSMHeuristicFugitive(vision_range_m=10.0, seed=0)
    ordered = ordered_outgoing_segment_ids(network, 0)
    observation = FugitiveObservation(
        intersection_id=0,
        ordered_segment_ids=ordered,
        police_xy=((0.0, -100.0),) * 6,
    )

    # Same escape choice as when there are no police at all.
    assert ordered[fugitive.act(network, observation)] == 0


# ---------------------------------------------------------------------------
# Deterministic tie-breaking by action order
# ---------------------------------------------------------------------------
def test_ties_broken_by_lowest_action_index():
    # Symmetric fork: node 0 -> node 1 (north) and node 0 -> node 2 (south),
    # equidistant from a single police officer at the origin. With no escape and
    # equal separation, the deterministic lowest-index slot must be chosen.
    intersections = [
        {"id": 0, "x": 0.0, "y": 0.0},
        {"id": 1, "x": 0.0, "y": 100.0},
        {"id": 2, "x": 0.0, "y": -100.0},
    ]
    segments = [
        {"id": 0, "start": 0, "end": 1},
        {"id": 1, "start": 0, "end": 2},
    ]
    network = build_network(intersections, segments)
    fugitive = OSMHeuristicFugitive(vision_range_m=50.0, seed=7)
    ordered = ordered_outgoing_segment_ids(network, 0)
    observation = FugitiveObservation(
        intersection_id=0,
        ordered_segment_ids=ordered,
        police_xy=((0.0, 0.0),) * 6,
    )

    # Equal min-separation destinations -> first action slot wins deterministically.
    assert fugitive.act(network, observation) == 0


# ---------------------------------------------------------------------------
# Seeded determinism of random exploration
# ---------------------------------------------------------------------------
def test_seeded_random_exploration_is_reproducible():
    network = escape_branch_network()
    ordered = ordered_outgoing_segment_ids(network, 0)
    observation = FugitiveObservation(
        intersection_id=0,
        ordered_segment_ids=ordered,
        police_xy=((0.0, -100.0),) * 6,
    )

    def run(seed: int) -> list[int]:
        fugitive = OSMHeuristicFugitive(vision_range_m=10.0, random_rate=0.7, seed=seed)
        return [fugitive.act(network, observation) for _ in range(25)]

    first = run(2024)
    second = run(2024)
    different = run(99)

    assert first == second  # same seed -> identical decisions
    # Random exploration actually fires (not always the greedy escape slot).
    assert any(ordered[action] != 0 for action in first)
    # A different seed generally yields a different exploration trace.
    assert first != different


def test_injected_generator_matches_equivalent_seed():
    network = escape_branch_network()
    ordered = ordered_outgoing_segment_ids(network, 0)
    observation = FugitiveObservation(
        intersection_id=0,
        ordered_segment_ids=ordered,
        police_xy=((0.0, -100.0),) * 6,
    )

    injected = OSMHeuristicFugitive(
        vision_range_m=10.0, random_rate=0.7, rng=np.random.default_rng(555)
    )
    seeded = OSMHeuristicFugitive(vision_range_m=10.0, random_rate=0.7, seed=555)

    injected_trace = [injected.act(network, observation) for _ in range(20)]
    seeded_trace = [seeded.act(network, observation) for _ in range(20)]

    assert injected_trace == seeded_trace


# ---------------------------------------------------------------------------
# Full driven episode determinism through the environment
# ---------------------------------------------------------------------------
def _run_episode(seed: int) -> tuple[EpisodeOutcome, tuple[str, ...]]:
    network = driven_episode_network()
    env = OSMRoadPursuitEnv(network, config())
    fugitive = OSMHeuristicFugitive(vision_range_m=10.0, seed=seed)
    # Keep police parked far from the escape route so the fugitive can run.
    police = [
        parked(5), parked(6), parked(7), parked(8), parked(9),
        VehiclePlacement(segment_id=5, progress=0.5),  # mid holding-chain, far south
    ]
    observations, _ = env.reset(options={"police": police, "fugitive": parked(0)})

    events: list[str] = []
    for _ in range(env.config.max_steps):
        police_xy = [tuple(observations[pid]["position_xy"]) for pid in env.police_ids]
        actions = {agent: STAY_ACTION for agent in env.agent_ids}
        actions[env.fugitive_id] = fugitive.env_provider(network, police_xy)
        observations, _, terminated, truncated, info = env.step(actions)
        events.extend(info["events"])
        if any(terminated.values()) or any(truncated.values()):
            break
    return env.outcome, tuple(events)


def test_driven_episode_reaches_escape_and_is_reproducible():
    first_outcome, first_events = _run_episode(11)
    second_outcome, second_events = _run_episode(11)

    assert first_outcome is EpisodeOutcome.ESCAPE
    assert first_outcome == second_outcome
    assert first_events == second_events  # deterministic replay
    assert any("boundary_crossed" in event for event in first_events)


# ---------------------------------------------------------------------------
# Constructor validation
# ---------------------------------------------------------------------------
def test_invalid_random_rate_is_rejected():
    from pursuit_evasion_rl.osm_demo.models import DomainValidationError

    with pytest.raises(DomainValidationError):
        OSMHeuristicFugitive(random_rate=1.5)
    with pytest.raises(DomainValidationError):
        OSMHeuristicFugitive(vision_range_m=-1.0)
