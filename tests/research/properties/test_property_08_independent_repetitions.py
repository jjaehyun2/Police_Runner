"""Property 8 coverage for pre-registered sample sizes and independent-repetition accounting."""

from __future__ import annotations

from hypothesis import given, settings, strategies as st
import pytest

from pursuit_evasion_rl.research.budget import (
    DEFAULT_EPISODES,
    DEFAULT_SEEDS,
    REDUCED_MIN_EPISODES,
    REDUCED_MIN_SEEDS,
    ConditionResourceEstimate,
    ResourceCeiling,
    SampleSizeStatus,
    count_independent_training_seeds,
    decide_sample_size,
    group_seed_replicates,
)

# **Property 8: Independent-repetition accounting cannot be inflated by checkpoints**
# **Validates: Requirements 7.1-7.8**

pytestmark = [pytest.mark.property, pytest.mark.offline]

_PBT_SETTINGS = settings(max_examples=100, deadline=None)

_costs = st.floats(min_value=0.01, max_value=50.0, allow_nan=False, allow_infinity=False)
# Sweeps the ceiling from "cannot afford 3 seeds" through the 3/4 reduced band to
# "affords the full 5 seeds", so every example lands somewhere on the boundary.
_scales = st.floats(min_value=0.5, max_value=7.0, allow_nan=False, allow_infinity=False)


@st.composite
def _condition_estimates(draw: st.DrawFn) -> tuple[ConditionResourceEstimate, ...]:
    count = draw(st.integers(min_value=1, max_value=4))
    return tuple(
        ConditionResourceEstimate(
            condition_id=f"condition_{index}",
            accelerator_hours_per_seed=draw(_costs),
            wall_clock_hours_per_seed=draw(_costs),
            env_steps_per_seed=draw(st.integers(min_value=1, max_value=1_000_000)),
        )
        for index in range(count)
    )


def _ceiling(
    estimates: tuple[ConditionResourceEstimate, ...],
    accelerator_scale: float,
    wall_clock_scale: float,
    *,
    ceiling_id: str = "measured-ceiling",
) -> ResourceCeiling:
    """A ceiling expressed as a multiple of the matrix's own per-seed cost."""
    return ResourceCeiling(
        resource_ceiling_id=ceiling_id,
        measured_at_utc="2026-01-01T00:00:00Z",
        accelerator_hours=sum(item.accelerator_hours_per_seed for item in estimates) * accelerator_scale,
        wall_clock_hours=sum(item.wall_clock_hours_per_seed for item in estimates) * wall_clock_scale,
    )


def _largest_affordable_seeds(
    ceiling: ResourceCeiling, estimates: tuple[ConditionResourceEstimate, ...]
) -> int | None:
    """Reference implementation of the affordability search, computed independently."""
    accelerator_unit = sum(item.accelerator_hours_per_seed for item in estimates)
    wall_clock_unit = sum(item.wall_clock_hours_per_seed for item in estimates)
    for seeds in range(DEFAULT_SEEDS, REDUCED_MIN_SEEDS - 1, -1):
        if (
            accelerator_unit * seeds <= ceiling.accelerator_hours
            and wall_clock_unit * seeds <= ceiling.wall_clock_hours
        ):
            return seeds
    return None


# ---------------------------------------------------------------------------
# Invariant 1/5: the plan always lands in exactly one of DEFAULT / REDUCED /
# EXPLORATORY, and which one is fixed by the documented 5-seed and 3-seed
# affordability boundary alone -- never by any observed result.
# ---------------------------------------------------------------------------
@_PBT_SETTINGS
@given(
    estimates=_condition_estimates(),
    accelerator_scale=_scales,
    wall_clock_scale=_scales,
)
def test_the_sample_size_status_is_fixed_by_the_affordability_boundary(
    estimates: tuple[ConditionResourceEstimate, ...],
    accelerator_scale: float,
    wall_clock_scale: float,
) -> None:
    ceiling = _ceiling(estimates, accelerator_scale, wall_clock_scale)
    plan = decide_sample_size(ceiling, estimates)
    affordable = _largest_affordable_seeds(ceiling, estimates)

    if affordable == DEFAULT_SEEDS:
        assert plan.status is SampleSizeStatus.DEFAULT
        assert (plan.seeds_per_condition, plan.episodes_per_condition) == (DEFAULT_SEEDS, DEFAULT_EPISODES)
        assert plan.power_limitation is None
    elif affordable is not None:
        assert plan.status is SampleSizeStatus.REDUCED
        assert plan.seeds_per_condition == affordable
        assert REDUCED_MIN_SEEDS <= plan.seeds_per_condition < DEFAULT_SEEDS
        assert REDUCED_MIN_EPISODES <= plan.episodes_per_condition <= DEFAULT_EPISODES
        # A reduction is never silent: what forced it is recorded on the plan.
        assert plan.reduction_inputs and plan.power_limitation
    else:
        assert plan.status is SampleSizeStatus.EXPLORATORY
        assert (
            plan.seeds_per_condition < REDUCED_MIN_SEEDS
            or plan.episodes_per_condition < REDUCED_MIN_EPISODES
        )
        assert plan.reduction_inputs and plan.power_limitation

    assert plan.resource_ceiling_id == ceiling.resource_ceiling_id
    assert plan.is_confirmatory_eligible is (plan.status is not SampleSizeStatus.EXPLORATORY)


# ---------------------------------------------------------------------------
# Invariant 2/5: the decision is monotone in the ceiling -- a strictly larger
# Resource_Ceiling can never buy fewer Training_Seeds.
# ---------------------------------------------------------------------------
@_PBT_SETTINGS
@given(
    estimates=_condition_estimates(),
    lower_scale=_scales,
    extra=st.floats(min_value=0.0, max_value=5.0, allow_nan=False, allow_infinity=False),
)
def test_a_larger_ceiling_never_reduces_the_planned_seed_count(
    estimates: tuple[ConditionResourceEstimate, ...], lower_scale: float, extra: float
) -> None:
    small = decide_sample_size(_ceiling(estimates, lower_scale, lower_scale), estimates)
    large = decide_sample_size(_ceiling(estimates, lower_scale + extra, lower_scale + extra), estimates)
    assert large.seeds_per_condition >= small.seeds_per_condition


# ---------------------------------------------------------------------------
# Invariant 3/5: the decision is a pure function of the ceiling and the cost
# estimates, so re-deciding with the same inputs yields the identical plan hash.
# ---------------------------------------------------------------------------
@_PBT_SETTINGS
@given(estimates=_condition_estimates(), accelerator_scale=_scales, wall_clock_scale=_scales)
def test_the_same_inputs_always_produce_the_same_plan_hash(
    estimates: tuple[ConditionResourceEstimate, ...],
    accelerator_scale: float,
    wall_clock_scale: float,
) -> None:
    ceiling = _ceiling(estimates, accelerator_scale, wall_clock_scale)
    first = decide_sample_size(ceiling, estimates)
    second = decide_sample_size(ceiling, estimates)
    assert first.plan_hash == second.plan_hash


# ---------------------------------------------------------------------------
# Invariant 4/5 (the headline property): repeating a Checkpoint_Time for a
# Training_Seed can never increase the independent-repetition count.
# ---------------------------------------------------------------------------
@st.composite
def _seed_records(draw: st.DrawFn) -> list[dict[str, object]]:
    seeds = draw(st.lists(st.integers(min_value=0, max_value=6), min_size=1, max_size=12))
    return [
        {
            "training_seed": seed,
            "checkpoint_time": draw(st.sampled_from(("step_100", "step_200", "step_300", "final"))),
        }
        for seed in seeds
    ]


@_PBT_SETTINGS
@given(records=_seed_records(), duplication=st.data())
def test_duplicating_checkpoint_times_never_inflates_the_independent_seed_count(
    records: list[dict[str, object]], duplication: st.DataObject
) -> None:
    baseline = count_independent_training_seeds(records)
    duplicates = [
        dict(records[index])
        for index in duplication.draw(
            st.lists(st.integers(min_value=0, max_value=len(records) - 1), max_size=20),
            label="duplicated_records",
        )
    ]
    inflated = list(records) + duplicates

    inflated_count = count_independent_training_seeds(inflated)
    assert inflated_count <= baseline
    assert inflated_count == baseline
    assert baseline == len({record["training_seed"] for record in records})


# ---------------------------------------------------------------------------
# Invariant 5/5: grouping preserves the exact seed set and loses no measurement,
# so the between-seed variance denominator survives every checkpoint repeat.
# ---------------------------------------------------------------------------
@_PBT_SETTINGS
@given(records=_seed_records(), duplication=st.data())
def test_grouping_preserves_the_seed_set_and_every_checkpoint_measurement(
    records: list[dict[str, object]], duplication: st.DataObject
) -> None:
    duplicates = [
        dict(records[index])
        for index in duplication.draw(
            st.lists(st.integers(min_value=0, max_value=len(records) - 1), max_size=20),
            label="duplicated_records",
        )
    ]
    inflated = list(records) + duplicates
    replicates = group_seed_replicates(inflated)

    seeds = [replicate.training_seed for replicate in replicates]
    assert seeds == sorted(seeds)  # deterministic order
    assert len(seeds) == len(set(seeds))  # every seed is one replicate, exactly once
    assert set(seeds) == {record["training_seed"] for record in records}
    # No checkpoint measurement is discarded; they are folded into their seed.
    assert sum(len(replicate.checkpoint_times) for replicate in replicates) == len(inflated)
    assert len(replicates) == count_independent_training_seeds(inflated)
