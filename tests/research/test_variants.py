"""Task 5.6 unit tests for the variant modules, covering the gaps left by the
per-module suites.

``test_observation_migration``/``test_observation_capacity``/``test_reward_variants``/
``test_placement_stabilization``/``test_condition_factory_and_budget`` already pin
each module in isolation.  This file adds what none of them assert: the literal
28D field order, multi-step reward returns, placement RNG-stream isolation, the
remaining ``EncirclementParameters`` constraints, FLOP-level budget fairness, and
one-factor control across *every* ablation axis rather than just the observation
axis.
"""
from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from pursuit_evasion_rl.osm_demo.models import (
    DomainValidationError,
    Intersection,
    ModelNetwork,
    POLICE_COUNT,
    Segment,
    VehiclePlacement,
)
from pursuit_evasion_rl.osm_demo.observations import OSM_FIELD_ORDER, OSM_FIELD_SIZES
from pursuit_evasion_rl.research.errors import ResearchValidationError
from pursuit_evasion_rl.research.maps.registry import MapScenario
from pursuit_evasion_rl.research.policies.baselines import EncirclementParameters
from pursuit_evasion_rl.research.variants.factory import (
    AXIS_BASELINE,
    AXIS_OBSERVATION,
    AXIS_PLACEMENT,
    AXIS_REWARD,
    AXIS_STABILIZATION,
    baseline_conditions,
    default_condition,
    full_condition_matrix,
    observation_ablation_conditions,
    placement_ablation_conditions,
    reward_ablation_conditions,
    stabilization_ablation_conditions,
    validate_condition_matrix,
)
from pursuit_evasion_rl.research.variants.observations import (
    AUGMENTED_FIELD_ORDER,
    AUGMENTED_NORMALIZATION,
    OBSERVATION_21D_DIM,
    OBSERVATION_28D_DIM,
    Observation28DAdapter,
    actor_capacity,
    find_capacity_matched_hidden_width,
)
from pursuit_evasion_rl.research.variants.placement import (
    PlacementCurriculumConfig,
    PlacementStyle,
    generate_placement,
    placement_rng,
)
from pursuit_evasion_rl.research.variants.rewards import (
    AUDITED_REWARD_COMPONENTS,
    compute_component_trace,
    total_rewards,
)

pytestmark = pytest.mark.offline

_RING_MIN = 100.0
_RING_MAX = 300.0
_AUDITED_HIDDEN_DIMS = (128, 128)
_AUDITED_ACTION_DIM = 6


def _grid_network(size: int = 5, spacing: float = 100.0) -> ModelNetwork:
    positions = {
        row * size + column: (column * spacing, row * spacing)
        for row in range(size)
        for column in range(size)
    }
    segments: list[Segment] = []
    for row in range(size):
        for column in range(size):
            node = row * size + column
            neighbours = []
            if column + 1 < size:
                neighbours.append(node + 1)
            if row + 1 < size:
                neighbours.append(node + size)
            for neighbour in neighbours:
                for start, end in ((node, neighbour), (neighbour, node)):
                    segments.append(
                        Segment(
                            id=len(segments),
                            start_id=start,
                            end_id=end,
                            length_m=spacing,
                            geometry_xy=(positions[start], positions[end]),
                        )
                    )
    intersections = tuple(
        Intersection(
            id=node,
            position_xy=positions[node],
            source_signature=f"variants-grid-{node}",
            outgoing_segment_ids=tuple(item.id for item in segments if item.start_id == node),
            incoming_segment_ids=tuple(item.id for item in segments if item.end_id == node),
        )
        for node in range(size * size)
    )
    return ModelNetwork(intersections=intersections, segments=tuple(segments))


def _placement_config(style: PlacementStyle, seed: int) -> PlacementCurriculumConfig:
    return PlacementCurriculumConfig(
        style=style, ring_min_m=_RING_MIN, ring_max_m=_RING_MAX, placement_seed=seed
    )


# ---------------------------------------------------------------------------
# 28D observation field contract
# ---------------------------------------------------------------------------


def test_augmented_field_order_is_the_documented_seven_field_sequence():
    assert AUGMENTED_FIELD_ORDER == (
        "distance_self_to_fugitive",
        "bearing_self_to_fugitive_sine",
        "bearing_self_to_fugitive_shifted_cosine",
        "team_minimum_distance_to_fugitive",
        "team_mean_distance_to_fugitive",
        "angular_coverage",
        "near_officer_fraction",
    )
    # Every added field must carry its own published normalization rule.
    assert len(AUGMENTED_NORMALIZATION) == len(AUGMENTED_FIELD_ORDER) == 7
    # The 28D variant appends to the 21D order; it never reorders or drops a base field.
    assert OBSERVATION_28D_DIM == OBSERVATION_21D_DIM + len(AUGMENTED_FIELD_ORDER)
    assert len(set(AUGMENTED_FIELD_ORDER)) == len(AUGMENTED_FIELD_ORDER)
    # ``distance_self_to_fugitive`` is the only name shared with the 21D order:
    # the base field is a directed road distance, the augmented one is euclidean.
    assert set(AUGMENTED_FIELD_ORDER) & set(OSM_FIELD_ORDER) == {"distance_self_to_fugitive"}


def test_28d_observation_has_the_declared_shape_dtype_and_unit_range():
    network = _grid_network()
    adapter = Observation28DAdapter(network, clip_distance_m=500.0, near_radius_m=200.0)
    police = tuple(VehiclePlacement(intersection_id=node) for node in (0, 1, 5, 6, 12, 24))
    fugitive = VehiclePlacement(intersection_id=13)

    road_field = OSM_FIELD_ORDER.index("distance_self_to_fugitive")
    road_index = sum(OSM_FIELD_SIZES[:road_field])
    euclidean_index = OBSERVATION_21D_DIM + AUGMENTED_FIELD_ORDER.index("distance_self_to_fugitive")
    metrics_differ = False

    for officer in range(POLICE_COUNT):
        vector = adapter.observe(
            police_index=officer,
            police=police,
            fugitive=fugitive,
            step=3,
            max_steps=30,
        )
        assert vector.shape == (OBSERVATION_28D_DIM,)
        assert vector.dtype == np.float32
        assert np.all(np.isfinite(vector))
        assert np.all(vector >= 0.0) and np.all(vector <= 1.0)
        if vector[road_index] != vector[euclidean_index]:
            metrics_differ = True

    # The shared field name really does carry two different metrics.
    assert metrics_differ


# ---------------------------------------------------------------------------
# Reward sequences
# ---------------------------------------------------------------------------

# A three-step episode: both officers close, officer 0 then retreats, then capture.
_SEQUENCE = (
    ((200.0, 300.0), (150.0, 250.0), False),
    ((150.0, 250.0), (200.0, 200.0), False),
    ((200.0, 200.0), (0.0, 150.0), True),
)
# Hand computed with team=1.0, own=0.6, time=0.02, capture=20.0, regress=1.6, scale=50 m.
_EXPECTED_STEP_TOTALS = (
    (1.58, 1.58),
    (-1.98, -0.42),
    (26.38, 24.58),
)


def test_hand_computed_multi_step_return_matches_the_summed_component_trace():
    returns = [0.0, 0.0]
    capture_credit = [0.0, 0.0]
    for (old, new, captured), expected in zip(_SEQUENCE, _EXPECTED_STEP_TOTALS):
        step_totals = total_rewards(AUDITED_REWARD_COMPONENTS, old, new, captured=captured)
        assert step_totals == pytest.approx(expected)
        trace = compute_component_trace(AUDITED_REWARD_COMPONENTS, old, new, captured=captured)
        for officer in range(2):
            assert trace[officer].total == pytest.approx(step_totals[officer])
            returns[officer] += step_totals[officer]
            capture_credit[officer] += trace[officer].capture

    assert returns == pytest.approx([25.98, 25.74])
    # The capture bonus is credited exactly once over the whole episode.
    assert capture_credit == pytest.approx(
        [AUDITED_REWARD_COMPONENTS.capture_bonus] * 2
    )


def test_retreat_step_is_the_only_step_where_the_regress_multiplier_applies():
    multipliers = []
    for old, new, captured in _SEQUENCE:
        trace = compute_component_trace(AUDITED_REWARD_COMPONENTS, old, new, captured=captured)
        delta = (old[0] - new[0]) / AUDITED_REWARD_COMPONENTS.distance_scale_m
        applied = trace[0].own / (AUDITED_REWARD_COMPONENTS.own_coefficient * delta)
        multipliers.append(round(applied, 6))
    assert multipliers == [1.0, pytest.approx(AUDITED_REWARD_COMPONENTS.regress_multiplier), 1.0]


# ---------------------------------------------------------------------------
# Placement RNG stream isolation
# ---------------------------------------------------------------------------


def test_placement_rng_is_seed_addressed_and_reproducible():
    first = placement_rng(_placement_config(PlacementStyle.RING, seed=7))
    second = placement_rng(_placement_config(PlacementStyle.RING, seed=7))
    assert np.array_equal(first.random(16), second.random(16))

    other = placement_rng(_placement_config(PlacementStyle.RING, seed=8))
    fresh = placement_rng(_placement_config(PlacementStyle.RING, seed=7))
    assert not np.array_equal(fresh.random(16), other.random(16))
    assert (
        _placement_config(PlacementStyle.RING, seed=7).rng_stream
        != _placement_config(PlacementStyle.RING, seed=8).rng_stream
    )


def test_placement_draw_is_isolated_from_global_and_sibling_generators():
    network = _grid_network()
    config = _placement_config(PlacementStyle.MIXED, seed=11)

    np.random.seed(1234)
    baseline = generate_placement(network, config)

    # A different global seed, plus consumption from the global stream.
    np.random.seed(99)
    np.random.random(64)
    after_global_drift = generate_placement(network, config)
    assert after_global_drift.police == baseline.police
    assert after_global_drift.fugitive == baseline.fugitive

    # A sibling generator on the same seed must not advance the placement stream.
    sibling = placement_rng(config)
    sibling.random(128)
    after_sibling_drift = generate_placement(network, config)
    assert after_sibling_drift.police == baseline.police
    assert after_sibling_drift.officer_components == baseline.officer_components


# ---------------------------------------------------------------------------
# Encirclement baseline parameters
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "overrides, message",
    [
        ({"direct_pursuit_count": POLICE_COUNT + 1}, "exceeds police count"),
        ({"direct_pursuit_count": 0}, "positive integer"),
        ({"velocity_history_size": True}, "positive integer"),
        ({"fugitive_speed_mps": 0.0}, "positive and finite"),
        ({"reverse_penalty_m": float("inf")}, "positive and finite"),
        ({"fan_degrees": (0.0, float("nan"))}, "finite values"),
        ({"cordon_degrees": ()}, "finite values"),
    ],
)
def test_encirclement_parameters_reject_invalid_parameter_sets(overrides, message):
    with pytest.raises(DomainValidationError) as excinfo:
        EncirclementParameters(**overrides)
    assert excinfo.value.code == "INVALID_BASELINE_PARAMETER"
    assert message in str(excinfo.value)


def test_encirclement_default_parameter_set_is_internally_consistent():
    parameters = EncirclementParameters()
    assert parameters.direct_pursuit_count <= POLICE_COUNT
    assert parameters.cordon_ring_min_m < parameters.close_pursuit_m < parameters.cordon_ring_max_m
    assert parameters.police_speed_mps > parameters.fugitive_speed_mps


# ---------------------------------------------------------------------------
# Capacity / budget fairness
# ---------------------------------------------------------------------------


def test_capacity_matched_width_equalizes_flops_not_just_parameters():
    comparison, widths = find_capacity_matched_hidden_width(
        base_observation_dim=OBSERVATION_21D_DIM,
        base_hidden_dims=_AUDITED_HIDDEN_DIMS,
        target_observation_dim=OBSERVATION_28D_DIM,
        action_dim=_AUDITED_ACTION_DIM,
        num_officers=POLICE_COUNT,
    )
    assert comparison.is_capacity_matched
    assert len(widths) == len(_AUDITED_HIDDEN_DIMS)

    base = actor_capacity(
        OBSERVATION_21D_DIM,
        action_dim=_AUDITED_ACTION_DIM,
        num_officers=POLICE_COUNT,
        hidden_dims=_AUDITED_HIDDEN_DIMS,
    )
    matched = actor_capacity(
        OBSERVATION_28D_DIM,
        action_dim=_AUDITED_ACTION_DIM,
        num_officers=POLICE_COUNT,
        hidden_dims=widths,
    )
    # Requirement 9.11 is a compute-budget claim, so the forward FLOP estimate
    # must land inside the same 1% band the parameter count is matched to.
    flop_difference = abs(matched.flop_estimate - base.flop_estimate) / base.flop_estimate
    assert flop_difference <= 0.01


# ---------------------------------------------------------------------------
# One-factor control across every ablation axis
# ---------------------------------------------------------------------------

_UNOWNED_DRIFT_FIELDS = {
    AXIS_OBSERVATION: ("policy", "reward", "placement", "uturn", "hysteresis"),
    AXIS_REWARD: ("policy", "observation", "placement", "uturn", "hysteresis"),
    AXIS_PLACEMENT: ("policy", "observation", "reward", "uturn", "hysteresis"),
    AXIS_STABILIZATION: ("policy", "observation", "reward", "placement"),
    AXIS_BASELINE: ("observation", "reward", "placement", "uturn", "hysteresis"),
}
_AXIS_BUILDERS = {
    AXIS_OBSERVATION: observation_ablation_conditions,
    AXIS_REWARD: reward_ablation_conditions,
    AXIS_PLACEMENT: placement_ablation_conditions,
    AXIS_STABILIZATION: stabilization_ablation_conditions,
    AXIS_BASELINE: baseline_conditions,
}


@pytest.mark.parametrize(
    "axis, field_name",
    [(axis, field) for axis, fields in _UNOWNED_DRIFT_FIELDS.items() for field in fields],
)
def test_drifting_any_field_the_axis_does_not_own_fails_matrix_validation(axis, field_name):
    scenario = MapScenario.INTERIOR_CONTAINED
    default = default_condition(scenario)
    arm = _AXIS_BUILDERS[axis](scenario)[0]
    # content_hash=None forces recomputation rather than tripping the stale-hash guard.
    drifted_condition = replace(
        arm.condition, content_hash=None, **{field_name: "drifted-value"}
    )
    specs = (default, replace(arm, condition=drifted_condition))

    with pytest.raises(ResearchValidationError) as excinfo:
        validate_condition_matrix(specs)
    assert excinfo.value.code == "UNDECLARED_FACTOR_DRIFT"


def test_matrix_without_its_scenario_default_condition_is_rejected():
    scenario = MapScenario.INTERIOR_CONTAINED
    specs = tuple(
        spec
        for spec in full_condition_matrix((scenario,))
        if spec.condition.scenario is not scenario or spec.axis != "default"
    )
    assert specs  # the non-default arms survive; only the shared control is gone

    with pytest.raises(ResearchValidationError) as excinfo:
        validate_condition_matrix(specs)
    assert excinfo.value.code == "MISSING_DEFAULT_CONDITION"


def test_empty_condition_matrix_is_rejected():
    with pytest.raises(ResearchValidationError) as excinfo:
        validate_condition_matrix(())
    assert excinfo.value.code == "EMPTY_CONDITION_MATRIX"
