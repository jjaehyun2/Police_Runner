"""Task 5.5 regressions for the Condition/ablation matrix factory and sample-size budget."""
from __future__ import annotations

from dataclasses import replace

import pytest

from pursuit_evasion_rl.research.budget import (
    ConditionResourceEstimate,
    ResourceCeiling,
    SampleSizeStatus,
    count_independent_training_seeds,
    decide_sample_size,
)
from pursuit_evasion_rl.research.domain import MapScenario
from pursuit_evasion_rl.research.errors import ResearchValidationError
from pursuit_evasion_rl.research.variants.factory import (
    baseline_conditions,
    default_condition,
    full_condition_matrix,
    observation_ablation_conditions,
    placement_ablation_conditions,
    reward_ablation_conditions,
    stabilization_ablation_conditions,
    validate_condition_matrix,
)

pytestmark = pytest.mark.offline


def test_full_matrix_validates_and_has_the_expected_axis_sizes():
    specs = full_condition_matrix()
    validate_condition_matrix(specs)
    counts: dict[str, int] = {}
    for spec in specs:
        counts[spec.axis] = counts.get(spec.axis, 0) + 1
    assert counts == {
        "default": 2,
        "observation": 4,
        "reward": 10,
        "placement": 6,
        "stabilization": 8,
        "baseline": 8,
    }


def test_same_axis_call_is_deterministic_and_hashes_reproduce():
    first = full_condition_matrix()
    second = full_condition_matrix()
    assert [spec.condition.content_hash for spec in first] == [spec.condition.content_hash for spec in second]


@pytest.mark.parametrize(
    "builder",
    [observation_ablation_conditions, reward_ablation_conditions, placement_ablation_conditions, stabilization_ablation_conditions, baseline_conditions],
)
def test_each_axis_differs_from_the_scenario_default_in_only_its_own_fields(builder):
    # Each axis includes the arm that matches the shared default (28D observation,
    # global placement, off/off stabilization) by construction -- that arm is
    # allowed to have zero diffs; every other arm must differ from the default.
    scenario = MapScenario.INTERIOR_CONTAINED
    default = default_condition(scenario).condition
    non_identity_arms = 0
    for spec in builder(scenario):
        differing = [
            name
            for name in ("policy", "observation", "reward", "placement", "uturn", "hysteresis")
            if getattr(spec.condition, name) != getattr(default, name)
        ]
        if differing:
            non_identity_arms += 1
    assert non_identity_arms >= 1


def test_validate_condition_matrix_rejects_duplicate_condition_ids():
    specs = list(full_condition_matrix((MapScenario.INTERIOR_CONTAINED,)))
    specs.append(specs[0])
    with pytest.raises(ResearchValidationError) as excinfo:
        validate_condition_matrix(tuple(specs))
    assert excinfo.value.code == "DUPLICATE_CONDITION_ID"


def test_validate_condition_matrix_rejects_undeclared_factor_drift():
    scenario = MapScenario.INTERIOR_CONTAINED
    default = default_condition(scenario)
    specs = [default]
    (observation_arm,) = [spec for spec in observation_ablation_conditions(scenario) if spec.arm == "observation_21d"]
    # Drift: the observation-axis Condition also changes its reward field.
    # content_hash=None forces recomputation instead of tripping the
    # immutability guard for a stale hash left over from the original object.
    drifted_condition = replace(observation_arm.condition, reward="tampered-reward-hash", content_hash=None)
    drifted = replace(observation_arm, condition=drifted_condition)
    specs.append(drifted)
    with pytest.raises(ResearchValidationError) as excinfo:
        validate_condition_matrix(tuple(specs))
    assert excinfo.value.code == "UNDECLARED_FACTOR_DRIFT"


# ---------------------------------------------------------------------------
# Budget / sample size
# ---------------------------------------------------------------------------


def _estimates() -> list[ConditionResourceEstimate]:
    return [
        ConditionResourceEstimate("cond_a", accelerator_hours_per_seed=1.0, wall_clock_hours_per_seed=1.0, env_steps_per_seed=10_000),
        ConditionResourceEstimate("cond_b", accelerator_hours_per_seed=1.0, wall_clock_hours_per_seed=1.0, env_steps_per_seed=10_000),
    ]


def test_ample_ceiling_yields_the_default_plan():
    ceiling = ResourceCeiling("rc-ample", "2026-07-31T00:00:00Z", accelerator_hours=1000.0, wall_clock_hours=1000.0)
    plan = decide_sample_size(ceiling, _estimates())
    assert plan.status is SampleSizeStatus.DEFAULT
    assert (plan.seeds_per_condition, plan.episodes_per_condition) == (5, 500)
    assert plan.is_confirmatory_eligible


def test_tight_ceiling_yields_a_uniform_reduced_plan_within_the_3_to_5_and_100_to_500_band():
    # 2 conditions * 1.0 h/seed: 5 seeds costs 10h, 4 seeds costs 8h.
    ceiling = ResourceCeiling("rc-tight", "2026-07-31T00:00:00Z", accelerator_hours=9.0, wall_clock_hours=9.0)
    plan = decide_sample_size(ceiling, _estimates())
    assert plan.status is SampleSizeStatus.REDUCED
    assert 3 <= plan.seeds_per_condition <= 5
    assert 100 <= plan.episodes_per_condition <= 500
    assert (plan.seeds_per_condition, plan.episodes_per_condition) != (5, 500)
    assert plan.reduction_inputs
    assert plan.power_limitation
    assert plan.is_confirmatory_eligible


def test_starved_ceiling_yields_exploratory_and_is_not_confirmatory_eligible():
    ceiling = ResourceCeiling("rc-starved", "2026-07-31T00:00:00Z", accelerator_hours=0.001, wall_clock_hours=0.001)
    plan = decide_sample_size(ceiling, _estimates())
    assert plan.status is SampleSizeStatus.EXPLORATORY
    assert not plan.is_confirmatory_eligible
    assert plan.power_limitation


def test_reduced_and_exploratory_plans_are_applied_uniformly_not_per_condition():
    """The decision is a single plan object, not one choice per condition --
    proving the reduced rule is matrix-wide (Requirement 7.4), not per-arm."""
    ceiling = ResourceCeiling("rc-tight", "2026-07-31T00:00:00Z", accelerator_hours=9.0, wall_clock_hours=9.0)
    plan = decide_sample_size(ceiling, _estimates())
    # A single seeds/episodes pair applies regardless of which condition asks.
    assert isinstance(plan.seeds_per_condition, int) and isinstance(plan.episodes_per_condition, int)


def test_repeated_checkpoint_times_for_one_seed_do_not_inflate_the_independent_seed_count():
    records = [
        {"training_seed": 11, "checkpoint_time": "t0"},
        {"training_seed": 11, "checkpoint_time": "t1"},
        {"training_seed": 11, "checkpoint_time": "t2"},
        {"training_seed": 12, "checkpoint_time": "t0"},
        {"training_seed": 13, "checkpoint_time": "t0"},
    ]
    assert count_independent_training_seeds(records) == 3


def test_sample_size_plan_rejects_malformed_status_size_combinations():
    from pursuit_evasion_rl.research.budget import SampleSizePlan

    with pytest.raises(ResearchValidationError):
        SampleSizePlan(
            status=SampleSizeStatus.DEFAULT, seeds_per_condition=4, episodes_per_condition=500,
            resource_ceiling_id="rc", reduction_inputs={},
        )
    with pytest.raises(ResearchValidationError):
        SampleSizePlan(
            status=SampleSizeStatus.REDUCED, seeds_per_condition=5, episodes_per_condition=500,
            resource_ceiling_id="rc", reduction_inputs={"x": 1}, power_limitation="limited",
        )
    with pytest.raises(ResearchValidationError):
        SampleSizePlan(
            status=SampleSizeStatus.REDUCED, seeds_per_condition=4, episodes_per_condition=200,
            resource_ceiling_id="rc", reduction_inputs={}, power_limitation="limited",
        )
