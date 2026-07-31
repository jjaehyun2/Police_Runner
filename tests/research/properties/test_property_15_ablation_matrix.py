"""Property 15 coverage for ablation-matrix completeness and one-factor control."""

from __future__ import annotations

from dataclasses import replace
from functools import lru_cache

from hypothesis import given, settings, strategies as st
import pytest

from pursuit_evasion_rl.osm_demo.models import DomainValidationError
from pursuit_evasion_rl.research.domain import MapScenario
from pursuit_evasion_rl.research.errors import ResearchValidationError
from pursuit_evasion_rl.research.variants.factory import (
    AXIS_BASELINE,
    AXIS_DEFAULT,
    AXIS_OBSERVATION,
    AXIS_PLACEMENT,
    AXIS_REWARD,
    AXIS_STABILIZATION,
    ConditionSpec,
    baseline_conditions,
    default_condition,
    full_condition_matrix,
    observation_ablation_conditions,
    placement_ablation_conditions,
    reward_ablation_conditions,
    stabilization_ablation_conditions,
    validate_condition_matrix,
)
from pursuit_evasion_rl.research.variants.placement import (
    PLACEMENT_RNG_STREAM,
    PlacementCurriculumConfig,
    PlacementStyle,
    all_placement_conditions,
    placement_protocol,
    validate_placement_matrix,
)
from pursuit_evasion_rl.research.variants.stabilization import (
    STABILIZATION_RNG_STREAM,
    StabilizationCondition,
    all_stabilization_arms,
    single_factor_diff,
    validate_stabilization_matrix,
)

# **Property 15: Ablation matrices are complete and one-factor controlled**
# **Validates: Requirements 9.1, 10.1-10.5**

pytestmark = [pytest.mark.property, pytest.mark.offline]

_PBT_SETTINGS = settings(max_examples=100, deadline=None)

# The axis -> owned-field contract, restated here so the test pins the intended
# ownership rather than whatever the factory happens to implement.
_CONTROLLED_FIELDS = ("policy", "observation", "reward", "placement", "uturn", "hysteresis")
_AXIS_OWNED_FIELDS: dict[str, tuple[str, ...]] = {
    AXIS_OBSERVATION: ("observation",),
    AXIS_REWARD: ("reward",),
    AXIS_PLACEMENT: ("placement",),
    AXIS_STABILIZATION: ("uturn", "hysteresis"),
    AXIS_BASELINE: ("policy",),
}

_SCENARIO_SETS = st.sampled_from(
    (
        (MapScenario.INTERIOR_CONTAINED,),
        (MapScenario.BOUNDARY_ESCAPE,),
        (MapScenario.INTERIOR_CONTAINED, MapScenario.BOUNDARY_ESCAPE),
    )
)

# Never collides with a real value: every Condition field the matrix controls
# holds a content hash, a policy id, or an "on:"/"off:" stabilization label.
_DRIFT_VALUES = st.integers(min_value=0, max_value=10**6).map(lambda n: f"drift-{n}")


@lru_cache(maxsize=None)
def _matrix(scenarios: tuple[MapScenario, ...]) -> tuple[ConditionSpec, ...]:
    """The real, unmutated matrix; deterministic, so it is built once per scenario set."""
    return full_condition_matrix(scenarios)


def _arm_keys(specs: tuple[ConditionSpec, ...]) -> set[tuple[MapScenario, str, str]]:
    return {(spec.condition.scenario, spec.axis, spec.arm) for spec in specs}


def _expected_arm_keys(scenarios: tuple[MapScenario, ...]) -> set[tuple[MapScenario, str, str]]:
    """The required arm inventory, rebuilt from the per-axis builders directly."""
    keys: set[tuple[MapScenario, str, str]] = set()
    for scenario in scenarios:
        specs = (
            default_condition(scenario),
            *observation_ablation_conditions(scenario),
            *reward_ablation_conditions(scenario),
            *placement_ablation_conditions(scenario),
            *stabilization_ablation_conditions(scenario),
            *baseline_conditions(scenario),
        )
        for spec in specs:
            keys.add((scenario, spec.axis, spec.arm))
    return keys


def _drift(spec: ConditionSpec, field_name: str, value: str) -> ConditionSpec:
    """Change one field the spec's axis does not own.

    ``content_hash=None`` forces recomputation; carrying the original hash over
    onto changed content trips the immutability guard instead of the matrix gate.
    """
    condition = replace(spec.condition, **{field_name: value}, content_hash=None)
    return replace(spec, condition=condition)


# ---------------------------------------------------------------------------
# Factory-level condition matrix
# ---------------------------------------------------------------------------


@_PBT_SETTINGS
@given(scenarios=_SCENARIO_SETS, data=st.data())
def test_unmutated_matrix_is_complete_and_always_validates(
    scenarios: tuple[MapScenario, ...], data: st.DataObject
) -> None:
    # Row order is presentation, not content: a valid matrix stays valid under
    # any permutation, so shuffling must never manufacture a violation.
    specs = tuple(data.draw(st.permutations(_matrix(scenarios))))

    validate_condition_matrix(specs)
    assert _arm_keys(specs) == _expected_arm_keys(scenarios)
    # Arm identity is the completeness unit, so it must be collision-free.
    assert len(_arm_keys(specs)) == len(specs)
    assert len({spec.condition.condition_id for spec in specs}) == len(specs)
    for scenario in scenarios:
        assert sum(1 for spec in specs if spec.axis == AXIS_DEFAULT and spec.condition.scenario == scenario) == 1


@_PBT_SETTINGS
@given(scenarios=_SCENARIO_SETS, data=st.data())
def test_dropping_any_arm_leaves_the_matrix_incomplete(
    scenarios: tuple[MapScenario, ...], data: st.DataObject
) -> None:
    specs = tuple(data.draw(st.permutations(_matrix(scenarios))))
    index = data.draw(st.integers(min_value=0, max_value=len(specs) - 1))
    dropped = specs[index]
    mutated = tuple(spec for position, spec in enumerate(specs) if position != index)

    missing = _expected_arm_keys(scenarios) - _arm_keys(mutated)
    assert missing == {(dropped.condition.scenario, dropped.axis, dropped.arm)}

    # validate_condition_matrix is a per-row single-factor gate, so it only sees a
    # drop when the dropped row is the scenario default every other row is
    # controlled against; other drops are caught by the inventory gate above.
    if dropped.axis == AXIS_DEFAULT:
        with pytest.raises(ResearchValidationError) as excinfo:
            validate_condition_matrix(mutated)
        assert excinfo.value.code == "MISSING_DEFAULT_CONDITION"
    else:
        validate_condition_matrix(mutated)


@_PBT_SETTINGS
@given(scenarios=_SCENARIO_SETS, data=st.data())
def test_duplicating_any_arm_is_rejected(
    scenarios: tuple[MapScenario, ...], data: st.DataObject
) -> None:
    specs = _matrix(scenarios)
    index = data.draw(st.integers(min_value=0, max_value=len(specs) - 1))
    position = data.draw(st.integers(min_value=0, max_value=len(specs)))
    mutated = list(specs)
    mutated.insert(position, specs[index])

    with pytest.raises(ResearchValidationError) as excinfo:
        validate_condition_matrix(tuple(mutated))
    assert excinfo.value.code == "DUPLICATE_CONDITION_ID"
    assert specs[index].condition.condition_id in excinfo.value.actual


@_PBT_SETTINGS
@given(scenarios=_SCENARIO_SETS, value=_DRIFT_VALUES, data=st.data())
def test_undeclared_factor_drift_on_any_unowned_field_is_rejected(
    scenarios: tuple[MapScenario, ...], value: str, data: st.DataObject
) -> None:
    specs = _matrix(scenarios)
    ablation_indices = [index for index, spec in enumerate(specs) if spec.axis != AXIS_DEFAULT]
    index = data.draw(st.sampled_from(ablation_indices))
    spec = specs[index]
    unowned = tuple(
        name for name in _CONTROLLED_FIELDS if name not in _AXIS_OWNED_FIELDS[spec.axis]
    )
    field_name = data.draw(st.sampled_from(unowned))
    mutated = tuple(
        _drift(spec, field_name, value) if position == index else other
        for position, other in enumerate(specs)
    )

    with pytest.raises(ResearchValidationError) as excinfo:
        validate_condition_matrix(mutated)
    assert excinfo.value.code == "UNDECLARED_FACTOR_DRIFT"
    assert excinfo.value.path == field_name


def test_every_arm_carries_its_axis_arm_and_condition_provenance() -> None:
    # Deterministic sweep: the matrix is fixed, so every arm is checked once
    # rather than sampled.  The randomized provenance coverage lives in the
    # placement/stabilization tests below, whose configs are generated.
    for spec in full_condition_matrix():
        condition = spec.condition
        assert dict(condition.budget) == {"axis": spec.axis, "arm": spec.arm}
        for name in _CONTROLLED_FIELDS:
            assert getattr(condition, name).strip()
        assert condition.evader.strip()
        assert condition.content_hash and condition.verify_hash()
        # The stabilization provenance rides on the "<level>:<condition_hash>" labels.
        for name in ("uturn", "hysteresis"):
            level, _, condition_hash = getattr(condition, name).partition(":")
            assert level in {"on", "off"}
            assert condition_hash.strip()


def test_an_empty_condition_matrix_is_rejected() -> None:
    with pytest.raises(ResearchValidationError) as excinfo:
        validate_condition_matrix(())
    assert excinfo.value.code == "EMPTY_CONDITION_MATRIX"


# ---------------------------------------------------------------------------
# Raw placement arm set (Requirement 10.1-10.3)
# ---------------------------------------------------------------------------


@st.composite
def placement_bases(draw: st.DrawFn) -> PlacementCurriculumConfig:
    """A randomized but valid placement base the three arms are derived from."""
    ring_min = draw(st.floats(min_value=0.0, max_value=500.0, allow_nan=False, allow_infinity=False))
    span = draw(st.floats(min_value=1.0, max_value=500.0, allow_nan=False, allow_infinity=False))
    return PlacementCurriculumConfig(
        style=PlacementStyle.GLOBAL,
        ring_min_m=ring_min,
        ring_max_m=ring_min + span,
        mixture_ring_probability=draw(
            st.floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False)
        ),
        placement_seed=draw(st.integers(min_value=0, max_value=10**6)),
    )


def _placement_drift(
    config: PlacementCurriculumConfig, field_name: str, delta: int
) -> PlacementCurriculumConfig:
    """Perturb one non-style field into a still-valid but different value."""
    if field_name == "ring_max_m":
        return replace(config, ring_max_m=config.ring_max_m + float(delta))
    if field_name == "placement_seed":
        return replace(config, placement_seed=config.placement_seed + delta)
    probability = 0.25 if config.mixture_ring_probability != 0.25 else 0.75
    return replace(config, mixture_ring_probability=probability)


@_PBT_SETTINGS
@given(base=placement_bases())
def test_unmutated_placement_arms_always_validate(base: PlacementCurriculumConfig) -> None:
    conditions = all_placement_conditions(base)

    validate_placement_matrix(conditions)
    assert [condition.style for condition in conditions] == list(PlacementStyle)
    assert len({condition.config_hash for condition in conditions}) == len(PlacementStyle)


@_PBT_SETTINGS
@given(base=placement_bases(), data=st.data())
def test_missing_or_duplicated_placement_arm_is_rejected(
    base: PlacementCurriculumConfig, data: st.DataObject
) -> None:
    conditions = all_placement_conditions(base)
    index = data.draw(st.integers(min_value=0, max_value=len(conditions) - 1))
    mutation = data.draw(st.sampled_from(("drop", "duplicate")))
    if mutation == "drop":
        mutated = tuple(item for position, item in enumerate(conditions) if position != index)
    else:
        mutated = (*conditions, conditions[index])

    with pytest.raises(DomainValidationError) as excinfo:
        validate_placement_matrix(mutated)
    assert excinfo.value.code == "INVALID_PLACEMENT_MATRIX"


@_PBT_SETTINGS
@given(
    base=placement_bases(),
    delta=st.integers(min_value=1, max_value=1000),
    data=st.data(),
)
def test_placement_arms_drifting_outside_style_are_rejected(
    base: PlacementCurriculumConfig, delta: int, data: st.DataObject
) -> None:
    conditions = all_placement_conditions(base)
    index = data.draw(st.integers(min_value=0, max_value=len(conditions) - 1))
    field_name = data.draw(
        st.sampled_from(("ring_max_m", "placement_seed", "mixture_ring_probability"))
    )
    drifted = _placement_drift(conditions[index], field_name, delta)
    assert drifted != conditions[index]
    mutated = tuple(
        drifted if position == index else item for position, item in enumerate(conditions)
    )

    with pytest.raises(DomainValidationError) as excinfo:
        validate_placement_matrix(mutated)
    assert excinfo.value.code == "UNDECLARED_FACTOR_DRIFT"


@_PBT_SETTINGS
@given(base=placement_bases())
def test_every_placement_arm_records_its_stream_and_config_hash(
    base: PlacementCurriculumConfig,
) -> None:
    for config in all_placement_conditions(base):
        assert config.rng_stream == f"{PLACEMENT_RNG_STREAM}#seed={config.placement_seed}"
        assert config.config_hash.strip() and config.config_hash == replace(config).config_hash
        protocol = placement_protocol(config)
        assert protocol.rng_stream == config.rng_stream
        assert protocol.distribution and all(item.strip() for item in protocol.distribution)
        assert protocol.support.strip() and protocol.feasibility_rule.strip()
        assert protocol.config_hash.strip()


# ---------------------------------------------------------------------------
# Raw stabilization 2x2 factorial (Requirement 10.4-10.5)
# ---------------------------------------------------------------------------


stabilization_bases = st.builds(
    StabilizationCondition,
    u_turn_penalty=st.floats(min_value=0.0, max_value=10.0, allow_nan=False, allow_infinity=False),
    hysteresis_min_hold_k=st.integers(min_value=1, max_value=12),
    hysteresis_margin=st.floats(min_value=0.0, max_value=10.0, allow_nan=False, allow_infinity=False),
)

_STABILIZATION_DRIFT_FIELDS = ("u_turn_penalty", "hysteresis_min_hold_k", "hysteresis_margin", "rng_stream")


def _stabilization_drift(
    arm: StabilizationCondition, field_name: str, delta: int
) -> StabilizationCondition:
    if field_name == "hysteresis_min_hold_k":
        return replace(arm, hysteresis_min_hold_k=arm.hysteresis_min_hold_k + delta)
    if field_name == "rng_stream":
        return replace(arm, rng_stream=f"tampered-{delta}")
    return replace(arm, **{field_name: getattr(arm, field_name) + float(delta)})


@_PBT_SETTINGS
@given(base=stabilization_bases)
def test_unmutated_stabilization_factorial_always_validates(base: StabilizationCondition) -> None:
    arms = all_stabilization_arms(base)

    validate_stabilization_matrix(arms)
    assert {(arm.u_turn_suppression, arm.hysteresis) for arm in arms} == {
        (False, False), (False, True), (True, False), (True, True)
    }
    assert len({arm.condition_id for arm in arms}) == 4
    assert len({arm.condition_hash for arm in arms}) == 4


@_PBT_SETTINGS
@given(base=stabilization_bases, data=st.data())
def test_missing_or_duplicated_stabilization_arm_is_rejected(
    base: StabilizationCondition, data: st.DataObject
) -> None:
    arms = all_stabilization_arms(base)
    index = data.draw(st.integers(min_value=0, max_value=len(arms) - 1))
    mutation = data.draw(st.sampled_from(("drop", "duplicate")))
    if mutation == "drop":
        mutated = tuple(arm for position, arm in enumerate(arms) if position != index)
    else:
        # Replace another arm with a copy, so the count stays 4 and only the
        # off/on level coverage is broken.
        victim = data.draw(st.sampled_from([position for position in range(len(arms)) if position != index]))
        mutated = tuple(arms[index] if position == victim else arm for position, arm in enumerate(arms))

    with pytest.raises(DomainValidationError) as excinfo:
        validate_stabilization_matrix(mutated)
    assert excinfo.value.code == "INVALID_STABILIZATION_MATRIX"


@_PBT_SETTINGS
@given(base=stabilization_bases, delta=st.integers(min_value=1, max_value=100), data=st.data())
def test_stabilization_arms_drifting_outside_their_two_factors_are_rejected(
    base: StabilizationCondition, delta: int, data: st.DataObject
) -> None:
    arms = all_stabilization_arms(base)
    index = data.draw(st.integers(min_value=0, max_value=len(arms) - 1))
    field_name = data.draw(st.sampled_from(_STABILIZATION_DRIFT_FIELDS))
    drifted = _stabilization_drift(arms[index], field_name, delta)
    assert drifted != arms[index]
    mutated = tuple(drifted if position == index else arm for position, arm in enumerate(arms))

    with pytest.raises(DomainValidationError) as excinfo:
        validate_stabilization_matrix(mutated)
    assert excinfo.value.code == "UNDECLARED_FACTOR_DRIFT"


# arms are ordered (off,off), (off,on), (on,off), (on,on): these pairs move one factor.
_SINGLE_FACTOR_PAIRS = st.sampled_from(
    ((0, 1, "hysteresis"), (2, 3, "hysteresis"), (0, 2, "u_turn_suppression"), (1, 3, "u_turn_suppression"))
)


@_PBT_SETTINGS
@given(base=stabilization_bases, pair=_SINGLE_FACTOR_PAIRS, delta=st.integers(min_value=1, max_value=100), data=st.data())
def test_single_factor_diff_names_the_factor_and_rejects_drift(
    base: StabilizationCondition, pair: tuple[int, int, str], delta: int, data: st.DataObject
) -> None:
    arms = all_stabilization_arms(base)
    left_index, right_index, factor = pair
    left, right = arms[left_index], arms[right_index]

    assert single_factor_diff(left, right) == factor
    assert single_factor_diff(right, left) == factor

    field_name = data.draw(st.sampled_from(_STABILIZATION_DRIFT_FIELDS))
    with pytest.raises(DomainValidationError) as excinfo:
        single_factor_diff(left, _stabilization_drift(right, field_name, delta))
    assert excinfo.value.code == "UNDECLARED_FACTOR_DRIFT"


@_PBT_SETTINGS
@given(base=stabilization_bases)
def test_every_stabilization_arm_records_its_stream_and_condition_hash(
    base: StabilizationCondition,
) -> None:
    for arm in all_stabilization_arms(base):
        assert arm.rng_stream == STABILIZATION_RNG_STREAM
        assert arm.condition_hash.strip()
        assert arm.condition_id.startswith("uturn_")
        assert arm.reversal_rule.strip() and arm.u_turn_method.strip()
        assert arm.hysteresis_hold_target.strip()
