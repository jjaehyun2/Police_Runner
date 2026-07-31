"""Task 4.1 hand-computed hybrid SMDP and road-adapter regressions."""
from __future__ import annotations

from dataclasses import FrozenInstanceError
import math

import pytest

from pursuit_evasion_rl.osm_demo.environment import OSMRoadPursuitEnv, STAY_ACTION
from pursuit_evasion_rl.osm_demo.models import (
    EpisodeConfig,
    Intersection,
    ModelNetwork,
    Segment,
    VehiclePlacement,
)
from pursuit_evasion_rl.research.errors import ResearchValidationError
from pursuit_evasion_rl.research.smdp import (
    AsyncDecisionTransitionBuffer,
    DecisionEpoch,
    point_at_arc_progress,
    validate_road_arc_transition,
)

pytestmark = pytest.mark.offline


def _epoch(officer: int, context: float = 0.0) -> DecisionEpoch:
    return DecisionEpoch.create(
        actor_obs=(officer / 10.0, 1.0),
        action=officer % 2,
        action_mask=(True, True, False),
        old_log_prob=-0.25 - officer / 100.0,
        critic_context=(context, float(officer), 1.0),
    )


def _all_epochs(context: float = 0.0) -> dict[int, DecisionEpoch]:
    return {officer: _epoch(officer, context) for officer in range(6)}


def _contexts(value: float) -> tuple[tuple[float, ...], ...]:
    return tuple((value, float(officer), 1.0) for officer in range(6))


def test_hand_computed_asynchronous_rewards_durations_and_bootstrap():
    buffer = AsyncDecisionTransitionBuffer(
        gamma=0.5, max_virtual_hops=3, actor_obs_dim=2,
        critic_context_dim=3, action_count=3,
    )
    buffer.start_decisions(_all_epochs())
    buffer.record_physical_step((1, 10, 0, 0, 0, 0))

    closed_0 = buffer.start_decision(0, _epoch(0, context=1.0))
    assert closed_0 is not None
    assert closed_0.discounted_reward == pytest.approx(1.0)
    assert closed_0.duration_steps == 1
    assert closed_0.bootstrap_discount == pytest.approx(0.5)
    assert closed_0.bootstrap_target(8.0) == pytest.approx(5.0)

    buffer.record_physical_step((2, 20, 0, 0, 0, 0))
    closed_1 = buffer.start_decision(1, _epoch(1, context=2.0))
    assert closed_1 is not None
    assert closed_1.discounted_reward == pytest.approx(10 + 0.5 * 20)
    assert closed_1.duration_steps == 2
    assert closed_1.bootstrap_discount == pytest.approx(0.25)
    assert closed_1.next_critic_context == _epoch(1, 2.0).critic_context

    buffer.record_physical_step((4, 40, 3, 3, 3, 3))
    terminal = buffer.close_terminal(_contexts(9.0))

    assert len(terminal) == 6
    assert {item.officer_id for item in terminal} == set(range(6))
    assert all(item.terminal for item in terminal)
    assert all(item.bootstrap_target(999.0) == item.discounted_reward for item in terminal)
    # Officer 0's replacement spans rewards 2 then 4; officer 2 never re-decided.
    assert terminal[0].discounted_reward == pytest.approx(2 + 0.5 * 4)
    assert terminal[0].duration_steps == 2
    assert terminal[2].discounted_reward == pytest.approx(0 + 0.5 * 0 + 0.25 * 3)
    assert terminal[2].duration_steps == 3
    assert all(item is None for item in buffer.pending)

def test_virtual_hops_are_bounded_and_do_not_add_time_or_discount():
    buffer = AsyncDecisionTransitionBuffer(gamma=0.9, max_virtual_hops=2)
    buffer.start_decisions(_all_epochs())
    buffer.record_physical_step((1, 1, 1, 1, 1, 1), virtual_hops=(2, 0, 1, 0, 0, 0))
    assert all(item is not None and item.duration_steps == 1 for item in buffer.pending)
    assert all(item is not None and item.discounted_reward == 1.0 for item in buffer.pending)

    snapshot = buffer.pending
    with pytest.raises(ResearchValidationError) as raised:
        buffer.record_physical_step((2, 2, 2, 2, 2, 2), virtual_hops=(3, 0, 0, 0, 0, 0))
    assert raised.value.code == "VIRTUAL_HOP_BOUND_EXCEEDED"
    assert buffer.pending == snapshot  # invalid batch is fail-closed and atomic


def test_transition_and_terminal_invariants_fail_closed():
    epoch = _epoch(0)
    with pytest.raises(FrozenInstanceError):
        epoch.action = 2
    with pytest.raises(ResearchValidationError) as invalid_action:
        DecisionEpoch.create(
            actor_obs=(0.0,), action=1, action_mask=(True, False),
            old_log_prob=0.0, critic_context=(0.0,),
        )
    assert invalid_action.value.code == "SELECTED_INVALID_ACTION"
    with pytest.raises(ResearchValidationError) as nonfinite:
        DecisionEpoch.create(
            actor_obs=(math.nan,), action=0, action_mask=(True,),
            old_log_prob=0.0, critic_context=(0.0,),
        )
    assert nonfinite.value.code == "NON_FINITE_SMDP_VALUE"

    buffer = AsyncDecisionTransitionBuffer(gamma=0.9, max_virtual_hops=0)
    buffer.start_decision(0, epoch)
    before = buffer.pending
    with pytest.raises(ResearchValidationError) as incomplete:
        buffer.close_terminal(_contexts(1.0))
    assert incomplete.value.code == "INCOMPLETE_TERMINAL_CLOSURE"
    assert buffer.pending == before

    full = AsyncDecisionTransitionBuffer(gamma=0.9, max_virtual_hops=0)
    full.start_decisions(_all_epochs())
    full.record_physical_step((0, 0, 0, 0, 0, 0))
    before = full.pending
    bad_contexts = list(_contexts(1.0))
    bad_contexts[5] = (1.0,)
    with pytest.raises(ResearchValidationError) as dimension:
        full.close_terminal(bad_contexts)
    assert dimension.value.code == "SMDP_DIMENSION_MISMATCH"
    assert full.pending == before


def _network() -> ModelNetwork:
    coordinates = ((0.0, 0.0), (5.0, 5.0), (15.0, 5.0), (25.0, 5.0),
                   (35.0, 5.0), (45.0, 5.0), (55.0, 5.0), (65.0, 5.0))
    geometries = [((0.0, 0.0), (0.0, 5.0), (5.0, 5.0))]
    geometries.extend((coordinates[index], coordinates[index + 1]) for index in range(1, 7))
    segments = tuple(
        Segment(index, index, index + 1,
                sum(math.dist(a, b) for a, b in zip(geometry, geometry[1:])),
                tuple(geometry))
        for index, geometry in enumerate(geometries)
    )
    intersections = tuple(
        Intersection(
            index, coordinates[index], f"node-{index}",
            outgoing_segment_ids=tuple(segment.id for segment in segments if segment.start_id == index),
            incoming_segment_ids=tuple(segment.id for segment in segments if segment.end_id == index),
        )
        for index in range(8)
    )
    return ModelNetwork(intersections, segments)

def _config(*, dt_s: float, speed: float) -> EpisodeConfig:
    return EpisodeConfig(
        dt_s=dt_s, police_speed_mps=speed, fugitive_speed_mps=1.0,
        capture_radius_m=0.01, max_steps=20,
    )


def _placements(progress: float) -> tuple[list[VehiclePlacement], VehiclePlacement]:
    police = [VehiclePlacement(segment_id=0, progress=progress)]
    police.extend(VehiclePlacement(intersection_id=index) for index in range(1, 6))
    return police, VehiclePlacement(intersection_id=7)


def test_environment_uses_directed_polyline_arc_length():
    network = _network()
    env = OSMRoadPursuitEnv(network, _config(dt_s=1.0, speed=5.0))
    police, fugitive = _placements(0.0)
    observations, _ = env.reset(options={"police": police, "fugitive": fugitive})
    before = env.episode_state().police[0]
    assert tuple(observations["police_0"]["position_xy"]) == (0.0, 0.0)

    actions = {agent: STAY_ACTION for agent in env.agent_ids}
    observations, _, _, _, _ = env.step(actions)
    after = env.episode_state().police[0]

    # Half of the 10 m L-shaped arc is its corner, not the straight chord midpoint.
    assert tuple(observations["police_0"]["position_xy"]) == pytest.approx((0.0, 5.0))
    assert point_at_arc_progress(network.segments[0].geometry_xy, 0.5) == pytest.approx((0.0, 5.0))
    validate_road_arc_transition(
        network, before, after, distance_budget_m=5.0, max_virtual_hops=0
    )


def test_environment_changes_physical_segment_only_at_directed_intersection():
    network = _network()
    env = OSMRoadPursuitEnv(network, _config(dt_s=0.2, speed=10.0))
    police, fugitive = _placements(0.9)
    env.reset(options={"police": police, "fugitive": fugitive})
    before = env.episode_state().police[0]

    actions = {agent: STAY_ACTION for agent in env.agent_ids}
    actions["police_0"] = 0
    env.step(actions)
    after = env.episode_state().police[0]

    assert after.segment_id == 1
    assert after.progress == 0.0
    validate_road_arc_transition(
        network, before, after, distance_budget_m=2.0, max_virtual_hops=0
    )
    with pytest.raises(ResearchValidationError) as discontinuity:
        validate_road_arc_transition(
            network, before, VehiclePlacement(segment_id=2, progress=0.1),
            distance_budget_m=2.0, max_virtual_hops=0,
        )
    assert discontinuity.value.code == "DISCONTINUOUS_SEGMENT_TRANSITION"


def _virtual_network() -> ModelNetwork:
    segments = (
        Segment(0, 0, 1, 0.0, ((0.0, 0.0), (0.0, 0.0)), virtual=True),
        Segment(1, 1, 2, 10.0, ((0.0, 0.0), (10.0, 0.0))),
    )
    intersections = (
        Intersection(0, (0.0, 0.0), "physical-start", outgoing_segment_ids=(0,)),
        Intersection(1, (0.0, 0.0), "virtual", virtual=True,
                     outgoing_segment_ids=(1,), incoming_segment_ids=(0,)),
        Intersection(2, (10.0, 0.0), "physical-end", incoming_segment_ids=(1,)),
    )
    return ModelNetwork(intersections, segments)


def test_road_validator_accepts_only_bounded_zero_time_virtual_departure():
    network = _virtual_network()
    before = VehiclePlacement(intersection_id=0)
    after = VehiclePlacement(segment_id=1, progress=0.0)
    validate_road_arc_transition(
        network, before, after, distance_budget_m=0.0, max_virtual_hops=1
    )
    with pytest.raises(ResearchValidationError) as bounded:
        validate_road_arc_transition(
            network, before, after, distance_budget_m=0.0, max_virtual_hops=0
        )
    assert bounded.value.code == "DIRECTION_VIOLATION"
