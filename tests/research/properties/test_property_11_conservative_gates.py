"""Property 11 coverage: multiple-comparison and practical-effect gates are conservative."""

from __future__ import annotations

from hypothesis import given, settings, strategies as st
import pytest

from pursuit_evasion_rl.research.statistics.paired import (
    ConfidenceInterval,
    EffectDirection,
    PracticalThreshold,
    benjamini_hochberg_adjusted_p_values,
    evaluate_practical_gate,
    holm_adjusted_p_values,
)

# **Property 11: Multiple-comparison and practical-effect gates are conservative**
# **Validates: Requirements 8.5-8.6**

pytestmark = pytest.mark.offline

_PBT_SETTINGS = settings(max_examples=100, deadline=None)
_p_values = st.lists(st.floats(min_value=0.0, max_value=1.0, allow_nan=False), min_size=1, max_size=10)


@_PBT_SETTINGS
@given(raw=_p_values)
def test_holm_adjusted_values_are_never_below_the_raw_p_value(raw: list[float]) -> None:
    adjusted = holm_adjusted_p_values(raw)
    assert len(adjusted) == len(raw)
    for original, corrected in zip(raw, adjusted):
        assert corrected >= original - 1e-12
        assert 0.0 <= corrected <= 1.0


@_PBT_SETTINGS
@given(raw=_p_values)
def test_benjamini_hochberg_adjusted_values_are_never_below_the_raw_p_value(raw: list[float]) -> None:
    adjusted = benjamini_hochberg_adjusted_p_values(raw)
    for original, corrected in zip(raw, adjusted):
        assert corrected >= original - 1e-12
        assert 0.0 <= corrected <= 1.0


@_PBT_SETTINGS
@given(raw=_p_values)
def test_holm_is_at_least_as_conservative_as_benjamini_hochberg(raw: list[float]) -> None:
    holm = holm_adjusted_p_values(raw)
    bh = benjamini_hochberg_adjusted_p_values(raw)
    assert all(h >= b - 1e-12 for h, b in zip(holm, bh))


@_PBT_SETTINGS
@given(raw=_p_values)
def test_holm_adjustment_is_monotone_in_sorted_raw_order(raw: list[float]) -> None:
    adjusted = holm_adjusted_p_values(raw)
    ordered = sorted(zip(raw, adjusted), key=lambda pair: pair[0])
    adjusted_in_raw_order = [item[1] for item in ordered]
    assert adjusted_in_raw_order == sorted(adjusted_in_raw_order)


@_PBT_SETTINGS
@given(
    base=st.floats(min_value=0.01, max_value=10.0, allow_nan=False),
    half_width=st.floats(min_value=0.01, max_value=5.0, allow_nan=False),
    extra_needed=st.floats(min_value=0.01, max_value=20.0, allow_nan=False),
    direction=st.sampled_from(list(EffectDirection)),
)
def test_a_significant_but_sub_threshold_effect_never_passes_superiority(
    base: float, half_width: float, extra_needed: float, direction: EffectDirection
) -> None:
    # Force statistical significance in the favourable direction with oriented
    # effect exactly ``base``, then set the practical threshold strictly above
    # ``base`` so practical significance must fail regardless of direction.
    if direction is EffectDirection.GREATER_IS_BETTER:
        effect_size = base
        interval = ConfidenceInterval(lower=base, upper=base + half_width, confidence_level=0.95, method="test")
    else:
        effect_size = -base
        interval = ConfidenceInterval(lower=-(base + half_width), upper=-base, confidence_level=0.95, method="test")

    threshold = PracticalThreshold(
        metric="capture_rate", unit="probability", minimum_effect=base + extra_needed, direction=direction,
    )
    decision = evaluate_practical_gate(effect_size, interval, threshold)
    assert decision.statistically_significant
    assert not decision.practically_significant
    assert not decision.superiority


@_PBT_SETTINGS
@given(
    lower_margin=st.floats(min_value=0.01, max_value=10.0, allow_nan=False),
    extra=st.floats(min_value=0.01, max_value=10.0, allow_nan=False),
    minimum_effect=st.floats(min_value=0.001, max_value=5.0, allow_nan=False),
    direction=st.sampled_from(list(EffectDirection)),
)
def test_significant_and_above_threshold_always_passes_superiority(
    lower_margin: float, extra: float, minimum_effect: float, direction: EffectDirection
) -> None:
    oriented_effect = lower_margin + extra + minimum_effect
    if direction is EffectDirection.GREATER_IS_BETTER:
        interval = ConfidenceInterval(lower=lower_margin, upper=lower_margin + extra + 1.0, confidence_level=0.95, method="test")
        effect = oriented_effect
    else:
        interval = ConfidenceInterval(lower=-(lower_margin + extra + 1.0), upper=-lower_margin, confidence_level=0.95, method="test")
        effect = -oriented_effect

    threshold = PracticalThreshold(metric="capture_rate", unit="probability", minimum_effect=minimum_effect, direction=direction)
    decision = evaluate_practical_gate(effect, interval, threshold)
    assert decision.statistically_significant
    assert decision.practically_significant
    assert decision.superiority
