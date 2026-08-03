"""Property 10 coverage: confidence intervals and paired tests preserve their registered domain."""

from __future__ import annotations

import math

from hypothesis import given, settings, strategies as st
import numpy as np
import pytest

from pursuit_evasion_rl.research.errors import ResearchValidationError
from pursuit_evasion_rl.research.statistics.paired import (
    BootstrapPlan,
    hierarchical_paired_bootstrap,
    wilson_interval,
)

# **Property 10: Confidence intervals and paired tests preserve their registered domain**
# **Validates: Requirements 8.2-8.4**

pytestmark = pytest.mark.offline

_PBT_SETTINGS = settings(max_examples=100, deadline=None)


@_PBT_SETTINGS
@given(trials=st.integers(min_value=1, max_value=5000), fraction=st.floats(min_value=0.0, max_value=1.0, allow_nan=False))
def test_wilson_interval_always_stays_bounded_in_the_unit_interval(trials: int, fraction: float) -> None:
    successes = int(round(fraction * trials))
    interval = wilson_interval(successes, trials)
    assert 0.0 <= interval.lower <= interval.upper <= 1.0
    assert interval.confidence_level == pytest.approx(0.95)


@_PBT_SETTINGS
@given(
    trials=st.integers(min_value=1, max_value=2000),
    fraction=st.floats(min_value=0.0, max_value=1.0, allow_nan=False),
    confidence=st.floats(min_value=0.5, max_value=0.999, allow_nan=False),
)
def test_a_wider_confidence_level_never_produces_a_narrower_interval(trials: int, fraction: float, confidence: float) -> None:
    successes = int(round(fraction * trials))
    narrow = wilson_interval(successes, trials, confidence_level=confidence)
    wide = wilson_interval(successes, trials, confidence_level=min(0.999, confidence + 0.0499))
    assert wide.width >= narrow.width - 1e-9


@_PBT_SETTINGS
@given(
    seed_count=st.integers(min_value=1, max_value=6),
    rows_per_seed=st.integers(min_value=1, max_value=8),
    rng_seed=st.integers(min_value=0, max_value=10_000),
)
def test_hierarchical_bootstrap_never_mixes_values_across_seed_groups(
    seed_count: int, rows_per_seed: int, rng_seed: int
) -> None:
    # Each seed's rows are literally that seed's own unique marker value, in
    # both paired columns, so any within-seed resample can only ever contain
    # that one marker; cross-seed mixing would show up as a resampled group
    # with more than one distinct value.
    groups = [np.full((rows_per_seed, 2), float(marker)) for marker in range(seed_count)]
    seen_more_than_one_value: list[bool] = []

    def statistic(resampled) -> float:
        for group in resampled:
            unique = np.unique(group)
            seen_more_than_one_value.append(len(unique) != 1)
        return float(np.mean([group[0, 0] for group in resampled]))

    plan = BootstrapPlan(n_bootstrap=30, rationale="small replicate count keeps the property test fast")
    interval, replicates = hierarchical_paired_bootstrap(groups, statistic, plan=plan)

    assert not any(seen_more_than_one_value)
    assert len(replicates) == 30
    assert all(math.isfinite(value) for value in replicates)
    assert interval.lower <= interval.upper
    # Every replicate's statistic is a mean of markers 0..seed_count-1, so it
    # can never leave that range.
    assert all(0.0 - 1e-9 <= value <= float(seed_count - 1) + 1e-9 for value in replicates)


@_PBT_SETTINGS
@given(seed_count=st.integers(min_value=1, max_value=5))
def test_bootstrap_rejects_an_empty_seed_group(seed_count: int) -> None:
    groups = [np.ones((1, 2)) for _ in range(seed_count)] + [np.empty((0, 2))]
    with pytest.raises(ResearchValidationError) as excinfo:
        hierarchical_paired_bootstrap(groups, lambda resampled: 0.0, plan=BootstrapPlan(n_bootstrap=5, rationale="fast"))
    assert excinfo.value.code == "EMPTY_SEED_GROUP"
