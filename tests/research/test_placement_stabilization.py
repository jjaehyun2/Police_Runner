"""Task 5.3 regressions for placement curriculum and the 2x2 stabilization factorial."""
from __future__ import annotations

import numpy as np
import pytest

from pursuit_evasion_rl.osm_demo.models import (
    POLICE_COUNT,
    DomainValidationError,
    Intersection,
    ModelNetwork,
    Segment,
)
from pursuit_evasion_rl.research.variants.placement import (
    PlacementCurriculumConfig,
    PlacementStyle,
    all_placement_conditions,
    generate_placement,
    placement_protocol,
    validate_placement_matrix,
)
from pursuit_evasion_rl.research.variants.stabilization import (
    FACTOR_FIELDS,
    HysteresisController,
    StabilizationCondition,
    StabilizationResult,
    all_stabilization_arms,
    is_reversal,
    single_factor_diff,
    toggle_factor,
    u_turn_penalties,
    validate_stabilization_matrix,
)

pytestmark = pytest.mark.offline

_GRID_SIZE = 5
_SPACING = 100.0
_RING_MIN = 100.0
_RING_MAX = 300.0


def _grid_network(size: int = _GRID_SIZE, spacing: float = _SPACING) -> ModelNetwork:
    """A bidirectional square grid: every edge is ``spacing`` metres long."""
    positions = {
        row * size + col: (col * spacing, row * spacing)
        for row in range(size)
        for col in range(size)
    }
    segments: list[Segment] = []
    for row in range(size):
        for col in range(size):
            node = row * size + col
            neighbours = []
            if col + 1 < size:
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
            source_signature=f"grid-{node}",
            outgoing_segment_ids=tuple(item.id for item in segments if item.start_id == node),
            incoming_segment_ids=tuple(item.id for item in segments if item.end_id == node),
        )
        for node in range(size * size)
    )
    return ModelNetwork(intersections=intersections, segments=tuple(segments))


def _config(style: PlacementStyle, seed: int = 0) -> PlacementCurriculumConfig:
    return PlacementCurriculumConfig(
        style=style, ring_min_m=_RING_MIN, ring_max_m=_RING_MAX, placement_seed=seed
    )


# ---------------------------------------------------------------------------
# Placement reproducibility and declared support
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("style", list(PlacementStyle))
def test_same_seed_reproduces_identical_placement(style: PlacementStyle) -> None:
    network = _grid_network()
    config = _config(style, seed=11)

    first = generate_placement(network, config)
    second = generate_placement(network, config)
    explicit = generate_placement(network, config, rng=np.random.default_rng(config.placement_seed))

    assert first.police == second.police == explicit.police
    assert first.fugitive == second.fugitive == explicit.fugitive
    assert first.officer_components == second.officer_components
    assert first.config_hash == second.config_hash == explicit.config_hash
    assert len(first.police) == POLICE_COUNT
    assert len({placement.intersection_id for placement in first.police}) == POLICE_COUNT
    assert first.fugitive.intersection_id not in {p.intersection_id for p in first.police}
    assert first.rng_stream.endswith("seed=11")


def test_distinct_seeds_produce_distinct_placements() -> None:
    network = _grid_network()
    draws = {
        generate_placement(network, _config(PlacementStyle.GLOBAL, seed=seed)).police
        for seed in range(8)
    }
    assert len(draws) > 1


def test_ring_respects_feasible_band_and_global_does_not() -> None:
    network = _grid_network()
    ring_distances: list[float] = []
    global_distances: list[float] = []
    for seed in range(20):
        ring_distances.extend(
            generate_placement(network, _config(PlacementStyle.RING, seed=seed)).officer_road_distances_m
        )
        global_distances.extend(
            generate_placement(network, _config(PlacementStyle.GLOBAL, seed=seed)).officer_road_distances_m
        )

    assert ring_distances and global_distances
    assert all(_RING_MIN <= distance <= _RING_MAX for distance in ring_distances)
    # The global support is unconstrained, so it must reach outside the band.
    assert any(distance > _RING_MAX or distance < _RING_MIN for distance in global_distances)


def test_mixed_draws_from_both_declared_components() -> None:
    network = _grid_network()
    seen: set[str] = set()
    for seed in range(20):
        draw = generate_placement(network, _config(PlacementStyle.MIXED, seed=seed))
        seen.update(draw.officer_components)
        for component, distance in zip(draw.officer_components, draw.officer_road_distances_m):
            if component == PlacementStyle.RING.value:
                assert _RING_MIN <= distance <= _RING_MAX
    assert seen == {PlacementStyle.GLOBAL.value, PlacementStyle.RING.value}


def test_mixture_probability_of_one_places_every_officer_on_the_ring() -> None:
    network = _grid_network()
    config = PlacementCurriculumConfig(
        style=PlacementStyle.MIXED,
        ring_min_m=_RING_MIN,
        ring_max_m=_RING_MAX,
        mixture_ring_probability=1.0,
        placement_seed=3,
    )
    draw = generate_placement(network, config)
    assert set(draw.officer_components) == {PlacementStyle.RING.value}
    assert all(_RING_MIN <= distance <= _RING_MAX for distance in draw.officer_road_distances_m)


def test_placement_protocol_records_distribution_support_and_stream() -> None:
    config = _config(PlacementStyle.MIXED, seed=5)
    protocol = placement_protocol(config)
    assert protocol.style is PlacementStyle.MIXED
    assert protocol.support and protocol.feasibility_rule
    assert protocol.rng_stream == config.rng_stream == "research.placement#seed=5"
    assert any("Bernoulli(p=0.5)" in item for item in protocol.distribution)
    # The mixture probability is inside the hashed Condition, so changing it moves the hash.
    other = placement_protocol(PlacementCurriculumConfig(
        style=PlacementStyle.MIXED, ring_min_m=_RING_MIN, ring_max_m=_RING_MAX,
        mixture_ring_probability=0.25, placement_seed=5,
    ))
    assert protocol.config_hash != other.config_hash


@pytest.mark.parametrize(
    "ring_min, ring_max",
    [(300.0, 100.0), (200.0, 200.0), (-10.0, 300.0), (100.0, -1.0)],
)
def test_infeasible_radius_is_rejected(ring_min: float, ring_max: float) -> None:
    with pytest.raises(DomainValidationError) as excinfo:
        PlacementCurriculumConfig(style=PlacementStyle.RING, ring_min_m=ring_min, ring_max_m=ring_max)
    assert excinfo.value.code == "INVALID_FEASIBLE_RADIUS"


def test_unsatisfiable_ring_band_fails_the_draw() -> None:
    network = _grid_network()
    config = PlacementCurriculumConfig(
        style=PlacementStyle.RING, ring_min_m=1.0, ring_max_m=50.0, placement_seed=0
    )
    with pytest.raises(DomainValidationError) as excinfo:
        generate_placement(network, config)
    assert excinfo.value.code == "PLACEMENT_INFEASIBLE"


def test_placement_matrix_has_exactly_three_single_factor_arms() -> None:
    conditions = all_placement_conditions()
    assert len(conditions) == 3
    assert [condition.style for condition in conditions] == list(PlacementStyle)
    assert len({condition.config_hash for condition in conditions}) == 3
    validate_placement_matrix(conditions)

    with pytest.raises(DomainValidationError) as missing:
        validate_placement_matrix(conditions[:2])
    assert missing.value.code == "INVALID_PLACEMENT_MATRIX"

    drifted = (conditions[0], conditions[1], PlacementCurriculumConfig(
        style=PlacementStyle.MIXED, ring_max_m=999.0
    ))
    with pytest.raises(DomainValidationError) as drift:
        validate_placement_matrix(drifted)
    assert drift.value.code == "UNDECLARED_FACTOR_DRIFT"


def test_placement_requires_an_explicit_generator() -> None:
    network = _grid_network()
    with pytest.raises(DomainValidationError) as excinfo:
        generate_placement(network, _config(PlacementStyle.GLOBAL), rng=np.random.RandomState(0))
    assert excinfo.value.code == "INVALID_PLACEMENT_RNG"


# ---------------------------------------------------------------------------
# Stabilization factorial
# ---------------------------------------------------------------------------


def test_exactly_four_factorial_arms_with_distinct_identities() -> None:
    arms = all_stabilization_arms()
    assert len(arms) == 4
    assert {(arm.u_turn_suppression, arm.hysteresis) for arm in arms} == {
        (False, False), (False, True), (True, False), (True, True)
    }
    assert len({arm.condition_id for arm in arms}) == 4
    assert len({arm.condition_hash for arm in arms}) == 4
    validate_stabilization_matrix(arms)


def test_incomplete_or_duplicated_factorial_is_rejected() -> None:
    arms = all_stabilization_arms()
    with pytest.raises(DomainValidationError) as missing:
        validate_stabilization_matrix(arms[:3])
    assert missing.value.code == "INVALID_STABILIZATION_MATRIX"

    with pytest.raises(DomainValidationError) as duplicate:
        validate_stabilization_matrix((arms[0], arms[0], arms[1], arms[2]))
    assert duplicate.value.code == "INVALID_STABILIZATION_MATRIX"

    drifted = (*arms[:3], StabilizationCondition(
        u_turn_suppression=True, hysteresis=True, u_turn_penalty=7.5
    ))
    with pytest.raises(DomainValidationError) as drift:
        validate_stabilization_matrix(drifted)
    assert drift.value.code == "UNDECLARED_FACTOR_DRIFT"


@pytest.mark.parametrize("factor", FACTOR_FIELDS)
def test_single_factor_toggle_changes_only_that_factor(factor: str) -> None:
    base = StabilizationCondition()
    toggled = toggle_factor(base, factor)

    assert single_factor_diff(base, toggled) == factor
    assert getattr(toggled, factor) is not getattr(base, factor)
    other = next(name for name in FACTOR_FIELDS if name != factor)
    assert getattr(toggled, other) == getattr(base, other)
    assert toggled.condition_hash != base.condition_hash


def test_single_factor_diff_rejects_no_diff_and_undeclared_drift() -> None:
    base = StabilizationCondition()
    with pytest.raises(DomainValidationError) as identical:
        single_factor_diff(base, base)
    assert identical.value.code == "INVALID_SINGLE_FACTOR_DIFF"

    with pytest.raises(DomainValidationError) as both:
        single_factor_diff(base, StabilizationCondition(u_turn_suppression=True, hysteresis=True))
    assert both.value.code == "INVALID_SINGLE_FACTOR_DIFF"

    with pytest.raises(DomainValidationError) as drift:
        single_factor_diff(base, StabilizationCondition(u_turn_suppression=True, hysteresis_margin=9.0))
    assert drift.value.code == "UNDECLARED_FACTOR_DRIFT"


# ---------------------------------------------------------------------------
# Soft U-turn suppression
# ---------------------------------------------------------------------------


def test_u_turn_penalty_is_soft_and_only_hits_the_reversing_candidate() -> None:
    candidates = (4, 7, 9)
    on = StabilizationCondition(u_turn_suppression=True)
    off = StabilizationCondition(u_turn_suppression=False)

    penalties = u_turn_penalties(on, candidates, previous_intersection_id=7)
    assert penalties == (0.0, -on.u_turn_penalty, 0.0)
    # Soft, not a mask: the reversing candidate still wins when it leads by more
    # than the penalty.
    scores = (1.0, 5.0, 2.0)
    combined = tuple(score + penalty for score, penalty in zip(scores, penalties))
    assert max(range(3), key=lambda index: combined[index]) == 1

    assert u_turn_penalties(off, candidates, previous_intersection_id=7) == (0.0, 0.0, 0.0)
    assert u_turn_penalties(on, candidates, previous_intersection_id=None) == (0.0, 0.0, 0.0)


def test_reversal_rule_needs_a_previous_decision_intersection() -> None:
    assert is_reversal(previous_intersection_id=7, candidate_end_intersection_id=7)
    assert not is_reversal(previous_intersection_id=7, candidate_end_intersection_id=8)
    assert not is_reversal(previous_intersection_id=None, candidate_end_intersection_id=7)


# ---------------------------------------------------------------------------
# Hysteresis hold / transition boundaries
# ---------------------------------------------------------------------------


def test_hysteresis_holds_for_k_minus_one_then_switches_on_the_kth() -> None:
    condition = StabilizationCondition(hysteresis=True, hysteresis_min_hold_k=3, hysteresis_margin=0.5)
    controller = HysteresisController(condition, initial_target=7)

    for _ in range(condition.hysteresis_min_hold_k - 1):
        # Margin is exceeded the whole time, but the minimum hold is not met.
        assert controller.step(9, 10.0, 5.0) is False
        assert controller.state == 7
        assert controller.switch_count == 0

    assert controller.step(9, 10.0, 5.0) is True
    assert controller.state == 9
    assert controller.switch_count == 1
    assert controller.steps_held == 0
    assert controller.snapshot().target == 9
    assert controller.snapshot().min_hold_k == 3


def test_hysteresis_requires_the_margin_at_the_switch_decision() -> None:
    condition = StabilizationCondition(hysteresis=True, hysteresis_min_hold_k=3, hysteresis_margin=0.5)
    controller = HysteresisController(condition, initial_target=7)

    assert controller.step(9, 10.0, 5.0) is False
    assert controller.step(9, 10.0, 5.0) is False
    # Dwell is now satisfied, but the lead collapsed to 0.2 <= margin.
    assert controller.step(9, 5.2, 5.0) is False
    assert controller.state == 7
    # The lead returns: the hold requirement stays satisfied, so it switches.
    assert controller.step(9, 10.0, 5.0) is True
    assert controller.state == 9


def test_hysteresis_never_switches_while_the_margin_stays_unmet() -> None:
    condition = StabilizationCondition(hysteresis=True, hysteresis_min_hold_k=3, hysteresis_margin=0.5)
    controller = HysteresisController(condition, initial_target=7)

    for _ in range(10):
        assert controller.step(9, 5.1, 5.0) is False
    assert controller.state == 7
    assert controller.switch_count == 0
    assert controller.steps_held == 10


def test_hysteresis_off_switches_immediately_on_any_strict_improvement() -> None:
    controller = HysteresisController(StabilizationCondition(hysteresis=False), initial_target=7)
    assert controller.step(9, 5.001, 5.0) is True
    assert controller.state == 9
    # An equal-scoring competitor is not an improvement, so nothing moves.
    assert controller.step(7, 5.0, 5.0) is False
    assert controller.state == 9


def test_hysteresis_ignores_a_candidate_equal_to_the_held_target() -> None:
    condition = StabilizationCondition(hysteresis=True, hysteresis_min_hold_k=1, hysteresis_margin=0.0)
    controller = HysteresisController(condition, initial_target=7)
    assert controller.step(7, 99.0, 5.0) is False
    assert controller.state == 7
    assert controller.switch_count == 0


# ---------------------------------------------------------------------------
# Result schema
# ---------------------------------------------------------------------------


def _result(**overrides) -> StabilizationResult:
    condition = StabilizationCondition(u_turn_suppression=True, hysteresis=True)
    payload = {
        "condition_id": condition.condition_id,
        "condition": condition,
        "capture_metric_ref": "metrics/capture",
        "containment_metric_ref": "metrics/containment",
        "anti_oscillation_metric_ref": "metrics/anti_oscillation",
        "physical_plausibility_metric_ref": "metrics/plausibility",
    }
    payload.update(overrides)
    return StabilizationResult(**payload)


def test_result_requires_all_four_trade_off_metric_references() -> None:
    assert _result().config_hash
    for name in (
        "capture_metric_ref",
        "containment_metric_ref",
        "anti_oscillation_metric_ref",
        "physical_plausibility_metric_ref",
    ):
        with pytest.raises(DomainValidationError) as excinfo:
            _result(**{name: "   "})
        assert excinfo.value.code == "MISSING_TRADE_OFF_METRIC_REFERENCE"


def test_capture_gain_with_a_regression_must_record_its_limitation() -> None:
    with pytest.raises(DomainValidationError) as excinfo:
        _result(capture_improved=True, stability_degraded=True)
    assert excinfo.value.code == "UNREPORTED_TRADE_OFF"

    with pytest.raises(DomainValidationError):
        _result(capture_improved=True, plausibility_degraded=True)

    flagged = _result(capture_improved=True, stability_degraded=True, limitation_ref="limits/oscillation")
    assert flagged.limitation_ref == "limits/oscillation"
    # A capture gain without any regression needs no limitation reference.
    assert _result(capture_improved=True).limitation_ref is None
