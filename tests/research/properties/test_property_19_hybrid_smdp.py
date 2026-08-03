"""Property 19 coverage for hybrid SMDP road-arc transition and discount invariants."""

from __future__ import annotations

from hypothesis import assume, given, settings, strategies as st
import pytest

from pursuit_evasion_rl.osm_demo.models import Intersection, ModelNetwork, Segment, VehiclePlacement
from pursuit_evasion_rl.research.errors import ResearchValidationError
from pursuit_evasion_rl.research.smdp import (
    OFFICER_COUNT,
    AsyncDecisionTransitionBuffer,
    DecisionEpoch,
    validate_road_arc_transition,
)

# **Property 19: Road-arc transitions obey the hybrid SMDP**
# **Validates: Requirements 13.7, 19.4**

pytestmark = pytest.mark.offline

_PBT_SETTINGS = settings(max_examples=100, deadline=None)
_LENGTH = 1000.0

_finite_floats = lambda **kw: st.floats(allow_nan=False, allow_infinity=False, **kw)  # noqa: E731


def _linear_network() -> ModelNetwork:
    """A single directed physical segment 0 -> 1 of length ``_LENGTH``."""
    intersections = (
        Intersection(0, (0.0, 0.0), "start", outgoing_segment_ids=(0,)),
        Intersection(1, (_LENGTH, 0.0), "end", incoming_segment_ids=(0,)),
    )
    segments = (Segment(0, 0, 1, _LENGTH, ((0.0, 0.0), (_LENGTH, 0.0))),)
    return ModelNetwork(intersections, segments)


def _branching_network() -> ModelNetwork:
    """Intersection 1 with a directed continuation (segment 1) and an unconnected decoy (segment 2)."""
    intersections = (
        Intersection(0, (0.0, 0.0), "start", outgoing_segment_ids=(0,)),
        Intersection(1, (_LENGTH, 0.0), "mid", incoming_segment_ids=(0,), outgoing_segment_ids=(1,)),
        Intersection(2, (1500.0, 0.0), "continuation-end", incoming_segment_ids=(1,)),
        Intersection(3, (2000.0, 0.0), "decoy-start", outgoing_segment_ids=(2,)),
        Intersection(4, (2500.0, 0.0), "decoy-end", incoming_segment_ids=(2,)),
    )
    segments = (
        Segment(0, 0, 1, _LENGTH, ((0.0, 0.0), (_LENGTH, 0.0))),
        Segment(1, 1, 2, 500.0, ((_LENGTH, 0.0), (1500.0, 0.0))),
        Segment(2, 3, 4, 500.0, ((2000.0, 0.0), (2500.0, 0.0))),
    )
    return ModelNetwork(intersections, segments)


# ---------------------------------------------------------------------------
# Invariant 1/4: movement that stays within the directed distance budget and
# the remaining segment arc length is always accepted.
# ---------------------------------------------------------------------------
@_PBT_SETTINGS
@given(budget=_finite_floats(min_value=1.0, max_value=500.0))
def test_movement_within_budget_matches_directed_arc_travel(budget: float) -> None:
    network = _linear_network()
    before = VehiclePlacement(segment_id=0, progress=0.0)
    after = VehiclePlacement(segment_id=0, progress=budget / _LENGTH)
    validate_road_arc_transition(network, before, after, distance_budget_m=budget, max_virtual_hops=0)


# ---------------------------------------------------------------------------
# Invariant 1/4: movement exceeding the directed distance budget (excess
# movement) is rejected, regardless of how large the excess is.
# ---------------------------------------------------------------------------
@_PBT_SETTINGS
@given(
    budget=_finite_floats(min_value=1.0, max_value=500.0),
    excess=_finite_floats(min_value=1.0, max_value=400.0),
)
def test_excess_movement_beyond_the_directed_budget_is_rejected(budget: float, excess: float) -> None:
    network = _linear_network()
    before = VehiclePlacement(segment_id=0, progress=0.0)
    travel = budget + excess  # strictly more than the directed arc allows; still < _LENGTH by construction
    after = VehiclePlacement(segment_id=0, progress=travel / _LENGTH)
    with pytest.raises(ResearchValidationError) as excinfo:
        validate_road_arc_transition(network, before, after, distance_budget_m=budget, max_virtual_hops=0)
    assert excinfo.value.code == "ROAD_ARC_DISTANCE_MISMATCH"


# ---------------------------------------------------------------------------
# Invariant 2/4: a directed transition may only continue into the segment
# that is actually reachable from the resolved intersection; jumping to an
# unconnected segment (an intermediate/mid-segment change) is rejected.
# ---------------------------------------------------------------------------
@_PBT_SETTINGS
@given(budget=_finite_floats(min_value=0.0, max_value=1000.0))
def test_jumping_to_an_unconnected_segment_is_rejected(budget: float) -> None:
    network = _branching_network()
    before = VehiclePlacement(intersection_id=1)
    after = VehiclePlacement(segment_id=2, progress=0.0)  # decoy segment 3 -> 4, unrelated to intersection 1
    with pytest.raises(ResearchValidationError) as excinfo:
        validate_road_arc_transition(network, before, after, distance_budget_m=budget, max_virtual_hops=0)
    assert excinfo.value.code == "DIRECTION_VIOLATION"


@_PBT_SETTINGS
@given(budget=_finite_floats(min_value=0.0, max_value=1000.0))
def test_the_directed_continuation_from_the_same_intersection_is_always_accepted(budget: float) -> None:
    network = _branching_network()
    before = VehiclePlacement(intersection_id=1)
    after = VehiclePlacement(segment_id=1, progress=0.0)  # the real directed continuation, 1 -> 2
    validate_road_arc_transition(network, before, after, distance_budget_m=budget, max_virtual_hops=0)


# ---------------------------------------------------------------------------
# Invariant 3/4: a physical step always advances every pending officer's
# duration by exactly one, no matter how many zero-time virtual hops (within
# the configured bound) occurred inside that step; exceeding the bound is
# rejected and leaves pending state untouched (atomic, fail-closed).
# ---------------------------------------------------------------------------
_hop_bounds = st.integers(min_value=0, max_value=6)
_rewards = st.lists(_finite_floats(min_value=-10.0, max_value=10.0), min_size=OFFICER_COUNT, max_size=OFFICER_COUNT)


def _epoch(officer: int) -> DecisionEpoch:
    return DecisionEpoch.create(
        actor_obs=(float(officer),), action=0, action_mask=(True,), old_log_prob=-0.1, critic_context=(0.0,),
    )


@_PBT_SETTINGS
@given(
    max_hops=_hop_bounds,
    hop_counts=st.data(),
    rewards=_rewards,
)
def test_virtual_hops_within_bound_never_change_the_one_step_duration_advance(
    max_hops: int, hop_counts: st.DataObject, rewards: list[float]
) -> None:
    counts = tuple(
        hop_counts.draw(st.integers(min_value=0, max_value=max_hops), label=f"hops_{officer}")
        for officer in range(OFFICER_COUNT)
    )
    buffer = AsyncDecisionTransitionBuffer(gamma=0.9, max_virtual_hops=max_hops)
    buffer.start_decisions({officer: _epoch(officer) for officer in range(OFFICER_COUNT)})
    buffer.record_physical_step(rewards, virtual_hops=counts)
    assert all(item is not None and item.duration_steps == 1 for item in buffer.pending)
    assert all(item.discounted_reward == pytest.approx(rewards[index]) for index, item in enumerate(buffer.pending))


@_PBT_SETTINGS
@given(max_hops=st.integers(min_value=0, max_value=5), excess=st.integers(min_value=1, max_value=5))
def test_a_virtual_hop_count_over_the_bound_is_rejected_and_state_is_unchanged(max_hops: int, excess: int) -> None:
    buffer = AsyncDecisionTransitionBuffer(gamma=0.9, max_virtual_hops=max_hops)
    buffer.start_decisions({officer: _epoch(officer) for officer in range(OFFICER_COUNT)})
    before = buffer.pending
    hops = [0] * OFFICER_COUNT
    hops[0] = max_hops + excess
    with pytest.raises(ResearchValidationError) as excinfo:
        buffer.record_physical_step([0.0] * OFFICER_COUNT, virtual_hops=hops)
    assert excinfo.value.code == "VIRTUAL_HOP_BOUND_EXCEEDED"
    assert buffer.pending == before


# ---------------------------------------------------------------------------
# Invariant 4/4: the bootstrap discount is exactly gamma ** duration_steps,
# and accumulated reward is exactly the gamma-weighted sum of per-step
# rewards, independent of how many (bounded) virtual hops occurred per step.
# ---------------------------------------------------------------------------
@_PBT_SETTINGS
@given(
    gamma=_finite_floats(min_value=0.0, max_value=1.0),
    step_rewards=st.lists(_finite_floats(min_value=-5.0, max_value=5.0), min_size=1, max_size=8),
    next_value=_finite_floats(min_value=-5.0, max_value=5.0),
    terminal=st.booleans(),
)
def test_bootstrap_discount_is_exactly_gamma_to_the_duration_and_reward_is_the_discounted_sum(
    gamma: float, step_rewards: list[float], next_value: float, terminal: bool
) -> None:
    buffer = AsyncDecisionTransitionBuffer(gamma=gamma, max_virtual_hops=2)
    buffer.start_decisions({officer: _epoch(officer) for officer in range(OFFICER_COUNT)})
    for index, reward in enumerate(step_rewards):
        # Virtual hops vary per step but must never contribute to duration or discount.
        hops = [index % 3] * OFFICER_COUNT
        buffer.record_physical_step([reward] * OFFICER_COUNT, virtual_hops=hops)

    if terminal:
        transitions = buffer.close_terminal(tuple((0.0,) for _ in range(OFFICER_COUNT)))
    else:
        transitions = buffer.start_decisions({0: _epoch(0)})

    transition = transitions[0]
    n = len(step_rewards)
    expected_reward = sum(gamma**index * reward for index, reward in enumerate(step_rewards))
    expected_discount = gamma**n

    assert transition.duration_steps == n
    assert transition.discounted_reward == pytest.approx(expected_reward, abs=1e-6)
    assert transition.bootstrap_discount == pytest.approx(expected_discount, abs=1e-9)
    if terminal:
        assert transition.bootstrap_target(next_value) == pytest.approx(expected_reward, abs=1e-6)
    else:
        assert transition.bootstrap_target(next_value) == pytest.approx(
            expected_reward + expected_discount * next_value, abs=1e-6
        )
