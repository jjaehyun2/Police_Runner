"""Property 14 coverage for reward-variant component decomposition."""

from __future__ import annotations

from dataclasses import replace
from string import ascii_lowercase, digits

from hypothesis import given, settings, strategies as st
import pytest

from pursuit_evasion_rl.osm_demo.models import DomainValidationError
from pursuit_evasion_rl.research.variants.rewards import (
    AUDITED_REWARD_COMPONENTS,
    ComponentTrace,
    RewardAblationResult,
    RewardComponent,
    RewardComponentSet,
    asymmetric_retreat_components,
    compute_component_trace,
    leave_one_component_out,
    symmetric_retreat_components,
    total_rewards,
)

# **Property 14: Reward variants differ only in their declared component**
# **Validates: Requirements 9.3-9.5**

pytestmark = [pytest.mark.property, pytest.mark.offline]

_PBT_SETTINGS = settings(max_examples=100, deadline=None)
_TOL = 1e-6
_MAX_DISTANCE_M = 1.0e4

# Component -> the ComponentTrace field it contributes, and the coefficient
# leave_one_component_out zeroes to remove it.
_TRACE_FIELD = {
    RewardComponent.TEAM: "team",
    RewardComponent.OWN: "own",
    RewardComponent.TIME: "time",
    RewardComponent.CAPTURE: "capture",
}
_COEFFICIENT_FIELD = {
    RewardComponent.TEAM: "team_coefficient",
    RewardComponent.OWN: "own_coefficient",
    RewardComponent.TIME: "time_penalty",
    RewardComponent.CAPTURE: "capture_bonus",
}
_AUDITED_COEFFICIENTS = (
    "team_coefficient",
    "own_coefficient",
    "time_penalty",
    "capture_bonus",
    "regress_multiplier",
    "distance_scale_m",
)

_distance = st.floats(min_value=0.0, max_value=_MAX_DISTANCE_M, allow_nan=False, allow_infinity=False)
_components = st.sampled_from(tuple(RewardComponent))
_ref_suffix = st.text(alphabet=ascii_lowercase + digits + "_", min_size=1, max_size=16)


@st.composite
def _officer_move(draw: st.DrawFn) -> tuple[float, float]:
    """One officer's (before, after) distance, biased to cover advance/retreat/hold."""
    old = draw(_distance)
    mode = draw(st.sampled_from(("advance", "retreat", "hold", "unconstrained")))
    if mode == "hold":
        return old, old
    if mode == "advance":
        return old, draw(st.floats(min_value=0.0, max_value=old, allow_nan=False, allow_infinity=False))
    if mode == "retreat":
        return old, draw(
            st.floats(min_value=old, max_value=_MAX_DISTANCE_M, allow_nan=False, allow_infinity=False)
        )
    return old, draw(_distance)


@st.composite
def _step(draw: st.DrawFn) -> tuple[tuple[float, ...], tuple[float, ...], bool]:
    """One physical step: matched per-officer distance tuples plus a capture flag."""
    moves = draw(st.lists(_officer_move(), min_size=2, max_size=6))
    return tuple(m[0] for m in moves), tuple(m[1] for m in moves), draw(st.booleans())


_event_sequence = st.lists(_step(), min_size=1, max_size=4)


def _explicit_sum(trace: ComponentTrace) -> float:
    return trace.team + trace.own + trace.time + trace.capture


# ---------------------------------------------------------------------------
# Invariant 1/4: the component decomposition and the scalar reward can never
# disagree -- every officer's total is exactly the sum of its four components,
# at every step of an event sequence.
# ---------------------------------------------------------------------------
@_PBT_SETTINGS
@given(events=_event_sequence)
def test_every_officer_total_equals_the_sum_of_its_four_components(
    events: list[tuple[tuple[float, ...], tuple[float, ...], bool]],
) -> None:
    for old, new, captured in events:
        traces = compute_component_trace(AUDITED_REWARD_COMPONENTS, old, new, captured=captured)
        totals = total_rewards(AUDITED_REWARD_COMPONENTS, old, new, captured=captured)
        assert len(traces) == len(totals) == len(old)
        for trace, total in zip(traces, totals):
            assert total == pytest.approx(_explicit_sum(trace), abs=_TOL)


# ---------------------------------------------------------------------------
# Invariant 2/4: removing exactly one component removes exactly that
# component's contribution -- the reward delta between the audited set and the
# leave-one-out set equals the excluded component's traced value, per officer.
# ---------------------------------------------------------------------------
@_PBT_SETTINGS
@given(step=_step(), excluded=_components)
def test_leave_one_out_delta_equals_the_excluded_components_traced_contribution(
    step: tuple[tuple[float, ...], tuple[float, ...], bool], excluded: RewardComponent
) -> None:
    old, new, captured = step
    full_trace = compute_component_trace(AUDITED_REWARD_COMPONENTS, old, new, captured=captured)
    full = total_rewards(AUDITED_REWARD_COMPONENTS, old, new, captured=captured)
    ablated_set = leave_one_component_out(excluded)
    ablated = total_rewards(ablated_set, old, new, captured=captured)

    # The variant differs from the audited set in exactly one declared coefficient.
    changed = _changed_coefficients(ablated_set)
    assert changed <= {_COEFFICIENT_FIELD[excluded]}

    for trace, before, after in zip(full_trace, full, ablated):
        contribution = getattr(trace, _TRACE_FIELD[excluded])
        assert before - after == pytest.approx(contribution, abs=_TOL)


# ---------------------------------------------------------------------------
# Invariant 3/4: the two retreat arms differ only in regress_multiplier, and
# that difference is observable only for officers that actually retreated.
# ---------------------------------------------------------------------------
@_PBT_SETTINGS
@given(step=_step())
def test_retreat_arms_differ_only_for_officers_whose_distance_increased(
    step: tuple[tuple[float, ...], tuple[float, ...], bool],
) -> None:
    old, new, captured = step
    symmetric = symmetric_retreat_components()
    asymmetric = asymmetric_retreat_components()
    assert _changed_coefficients(symmetric, base=asymmetric) == {"regress_multiplier"}

    symmetric_trace = compute_component_trace(symmetric, old, new, captured=captured)
    asymmetric_trace = compute_component_trace(asymmetric, old, new, captured=captured)
    symmetric_totals = total_rewards(symmetric, old, new, captured=captured)
    asymmetric_totals = total_rewards(asymmetric, old, new, captured=captured)

    ratio = asymmetric.regress_multiplier / symmetric.regress_multiplier
    for index, (before, after) in enumerate(zip(old, new)):
        sym, asym = symmetric_trace[index], asymmetric_trace[index]
        # Non-own components are arm-independent by construction.
        assert (sym.team, sym.time, sym.capture) == (asym.team, asym.time, asym.capture)
        if after > before:
            # Retreat: the asymmetric arm scales the (negative) own term by the multiplier.
            assert asym.own == pytest.approx(ratio * sym.own, abs=_TOL)
            if after - before > 1.0:  # a retreat large enough that the scaling must be visible
                assert asym.own < sym.own
        else:
            # Advance or exact hold: identical own term, identical total.
            assert asym.own == sym.own
            assert asymmetric_totals[index] == symmetric_totals[index]


# ---------------------------------------------------------------------------
# Invariant 4/4: a result row must declare every component it changed and must
# carry both its capture and stability metric references.
# ---------------------------------------------------------------------------
def _changed_coefficients(
    variant: RewardComponentSet, base: RewardComponentSet = AUDITED_REWARD_COMPONENTS
) -> set[str]:
    return {name for name in _AUDITED_COEFFICIENTS if getattr(variant, name) != getattr(base, name)}


def _undeclared_coefficient_changes(result: RewardAblationResult) -> set[str]:
    """Coefficients a row changed without declaring them via ``excluded_component``."""
    declared = {_COEFFICIENT_FIELD[result.excluded_component]} if result.excluded_component else set()
    return _changed_coefficients(result.components) - declared


def _ablation_row(
    excluded: RewardComponent | None,
    trace: tuple[ComponentTrace, ...],
    suffix: str,
    *,
    components: RewardComponentSet | None = None,
    capture_ref: str | None = None,
    stability_ref: str | None = None,
) -> RewardAblationResult:
    declared = components if components is not None else (
        AUDITED_REWARD_COMPONENTS if excluded is None else leave_one_component_out(excluded)
    )
    label = "full" if excluded is None else excluded.value
    return RewardAblationResult(
        condition_id=f"reward_{label}_{suffix}",
        components=declared,
        excluded_component=excluded,
        capture_metric_ref=f"metrics/capture_rate.json#{suffix}" if capture_ref is None else capture_ref,
        stability_metric_ref=(
            f"metrics/anti_oscillation.json#{suffix}" if stability_ref is None else stability_ref
        ),
        component_trace=trace,
    )


@_PBT_SETTINGS
@given(
    step=_step(),
    excluded=st.one_of(st.none(), _components),
    suffix=_ref_suffix,
    drifted=_components,
    drift=st.floats(min_value=0.1, max_value=5.0, allow_nan=False, allow_infinity=False),
)
def test_undeclared_component_drift_is_detectable_while_declared_ablations_are_clean(
    step: tuple[tuple[float, ...], tuple[float, ...], bool],
    excluded: RewardComponent | None,
    suffix: str,
    drifted: RewardComponent,
    drift: float,
) -> None:
    old, new, captured = step
    base = AUDITED_REWARD_COMPONENTS if excluded is None else leave_one_component_out(excluded)
    trace = compute_component_trace(base, old, new, captured=captured)

    clean = _ablation_row(excluded, trace, suffix)
    assert _undeclared_coefficient_changes(clean) == set()

    drifted_field = _COEFFICIENT_FIELD[drifted]
    drifted_components = replace(base, **{drifted_field: getattr(base, drifted_field) + drift})
    row = _ablation_row(excluded, trace, suffix, components=drifted_components)
    if drifted is excluded:
        # The change is on the component the row already declares -- still auditable.
        assert _undeclared_coefficient_changes(row) == set()
    else:
        assert _undeclared_coefficient_changes(row) == {drifted_field}


@_PBT_SETTINGS
@given(
    step=_step(),
    excluded=st.one_of(st.none(), _components),
    suffix=_ref_suffix,
    missing=st.sampled_from(("capture", "stability")),
    blank=st.sampled_from(("", " ", "\t", "\n  ")),
)
def test_a_row_missing_its_capture_or_stability_metric_reference_is_rejected(
    step: tuple[tuple[float, ...], tuple[float, ...], bool],
    excluded: RewardComponent | None,
    suffix: str,
    missing: str,
    blank: str,
) -> None:
    old, new, captured = step
    base = AUDITED_REWARD_COMPONENTS if excluded is None else leave_one_component_out(excluded)
    trace = compute_component_trace(base, old, new, captured=captured)
    assert _ablation_row(excluded, trace, suffix).config_hash  # the same inputs build a valid row

    blanked = {"capture_ref": blank} if missing == "capture" else {"stability_ref": blank}
    with pytest.raises(DomainValidationError) as excinfo:
        _ablation_row(excluded, trace, suffix, **blanked)
    assert excinfo.value.code == (
        "MISSING_CAPTURE_METRIC_REFERENCE" if missing == "capture" else "MISSING_STABILITY_METRIC_REFERENCE"
    )
