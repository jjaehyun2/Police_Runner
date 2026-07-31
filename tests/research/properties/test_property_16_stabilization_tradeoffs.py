"""Property 16 coverage for stabilization result rows that cannot hide a trade-off."""

from __future__ import annotations

import string

from hypothesis import given, settings, strategies as st
import pytest

from pursuit_evasion_rl.osm_demo.models import DomainValidationError
from pursuit_evasion_rl.research.variants.stabilization import (
    StabilizationCondition,
    StabilizationResult,
    all_stabilization_arms,
)

# **Property 16: Stabilization trade-offs cannot be hidden**
# **Validates: Requirements 10.6-10.7**

pytestmark = [pytest.mark.property, pytest.mark.offline]

_PBT_SETTINGS = settings(max_examples=100, deadline=None)

# Requirement 10.9: capture is never reportable without the behavioral and
# physical evidence in the same row.
_METRIC_FIELDS = (
    "capture_metric_ref",
    "containment_metric_ref",
    "anti_oscillation_metric_ref",
    "physical_plausibility_metric_ref",
)

_REFERENCES = st.text(
    alphabet=string.ascii_letters + string.digits + "_-./#", min_size=1, max_size=24
)
_BLANKS = st.sampled_from(("", " ", "   ", "\t", "\n", " \t\n "))

_CONDITIONS = st.one_of(
    st.sampled_from(all_stabilization_arms()),
    st.builds(
        StabilizationCondition,
        u_turn_suppression=st.booleans(),
        hysteresis=st.booleans(),
        u_turn_penalty=st.floats(min_value=0.0, max_value=10.0, allow_nan=False, allow_infinity=False),
        hysteresis_min_hold_k=st.integers(min_value=1, max_value=12),
        hysteresis_margin=st.floats(min_value=0.0, max_value=10.0, allow_nan=False, allow_infinity=False),
    ),
)

# capture_improved with at least one degradation is the reportable conflict.
_CONFLICTING_FLAGS = st.sampled_from(((True, True, False), (True, False, True), (True, True, True)))
_NON_CONFLICTING_FLAGS = st.sampled_from(
    ((False, False, False), (False, True, False), (False, False, True), (False, True, True), (True, False, False))
)


def _row(**overrides) -> StabilizationResult:
    fields = {
        "condition_id": "uturn_on__hyst_on",
        "condition": StabilizationCondition(u_turn_suppression=True, hysteresis=True),
        "capture_metric_ref": "metric/capture",
        "containment_metric_ref": "metric/containment",
        "anti_oscillation_metric_ref": "metric/anti_oscillation",
        "physical_plausibility_metric_ref": "metric/physical_plausibility",
    }
    fields.update(overrides)
    return StabilizationResult(**fields)


@_PBT_SETTINGS
@given(
    condition_id=_REFERENCES,
    condition=_CONDITIONS,
    references=st.lists(_REFERENCES, min_size=4, max_size=4),
    blank=_BLANKS,
    blanked_field=st.sampled_from(_METRIC_FIELDS),
    flags=st.tuples(st.booleans(), st.booleans(), st.booleans()),
    limitation=_REFERENCES,
)
def test_a_blank_metric_reference_in_any_position_is_rejected(
    condition_id: str,
    condition: StabilizationCondition,
    references: list[str],
    blank: str,
    blanked_field: str,
    flags: tuple[bool, bool, bool],
    limitation: str,
) -> None:
    metrics = dict(zip(_METRIC_FIELDS, references))
    complete = _row(
        condition_id=condition_id,
        condition=condition,
        capture_improved=flags[0],
        stability_degraded=flags[1],
        plausibility_degraded=flags[2],
        limitation_ref=limitation,
        **metrics,
    )
    assert complete.condition_id == condition_id
    assert complete.config_hash.strip()

    metrics[blanked_field] = blank
    with pytest.raises(DomainValidationError) as excinfo:
        _row(
            condition_id=condition_id,
            condition=condition,
            capture_improved=flags[0],
            stability_degraded=flags[1],
            plausibility_degraded=flags[2],
            limitation_ref=limitation,
            **metrics,
        )
    assert excinfo.value.code == "MISSING_TRADE_OFF_METRIC_REFERENCE"
    assert excinfo.value.path == blanked_field


@_PBT_SETTINGS
@given(condition_id=_REFERENCES, condition=_CONDITIONS, blank=_BLANKS)
def test_a_blank_condition_id_is_a_required_field_failure_not_a_metric_failure(
    condition_id: str, condition: StabilizationCondition, blank: str
) -> None:
    assert _row(condition_id=condition_id, condition=condition).condition is condition

    with pytest.raises(DomainValidationError) as excinfo:
        _row(condition_id=blank, condition=condition)
    assert excinfo.value.code == "MISSING_REQUIRED_FIELD"
    assert excinfo.value.path == "condition_id"


@_PBT_SETTINGS
@given(
    condition_id=_REFERENCES,
    condition=_CONDITIONS,
    references=st.lists(_REFERENCES, min_size=4, max_size=4),
    flags=_CONFLICTING_FLAGS,
    limitation=_REFERENCES,
    missing_limitation=st.one_of(st.none(), _BLANKS),
)
def test_capture_gain_beside_a_regression_must_record_its_limitation(
    condition_id: str,
    condition: StabilizationCondition,
    references: list[str],
    flags: tuple[bool, bool, bool],
    limitation: str,
    missing_limitation: str | None,
) -> None:
    metrics = dict(zip(_METRIC_FIELDS, references))
    capture_improved, stability_degraded, plausibility_degraded = flags

    with pytest.raises(DomainValidationError) as excinfo:
        _row(
            condition_id=condition_id,
            condition=condition,
            capture_improved=capture_improved,
            stability_degraded=stability_degraded,
            plausibility_degraded=plausibility_degraded,
            limitation_ref=missing_limitation,
            **metrics,
        )
    assert excinfo.value.code == "UNREPORTED_TRADE_OFF"
    assert excinfo.value.path == "limitation_ref"

    reported = _row(
        condition_id=condition_id,
        condition=condition,
        capture_improved=capture_improved,
        stability_degraded=stability_degraded,
        plausibility_degraded=plausibility_degraded,
        limitation_ref=limitation,
        **metrics,
    )
    assert reported.limitation_ref == limitation
    assert reported.capture_improved and (reported.stability_degraded or reported.plausibility_degraded)


@_PBT_SETTINGS
@given(
    condition_id=_REFERENCES,
    condition=_CONDITIONS,
    references=st.lists(_REFERENCES, min_size=4, max_size=4),
    flags=_NON_CONFLICTING_FLAGS,
)
def test_without_a_conflict_the_limitation_reference_stays_optional(
    condition_id: str,
    condition: StabilizationCondition,
    references: list[str],
    flags: tuple[bool, bool, bool],
) -> None:
    capture_improved, stability_degraded, plausibility_degraded = flags

    row = _row(
        condition_id=condition_id,
        condition=condition,
        capture_improved=capture_improved,
        stability_degraded=stability_degraded,
        plausibility_degraded=plausibility_degraded,
        limitation_ref=None,
        **dict(zip(_METRIC_FIELDS, references)),
    )

    assert row.limitation_ref is None
    assert not (row.capture_improved and (row.stability_degraded or row.plausibility_degraded))


@_PBT_SETTINGS
@given(
    condition_id=_REFERENCES,
    condition=_CONDITIONS,
    references=st.lists(_REFERENCES, min_size=4, max_size=4),
    flags=st.tuples(st.booleans(), st.booleans(), st.booleans()),
    limitation=st.one_of(st.none(), _BLANKS, _REFERENCES),
)
def test_the_trade_off_gate_fires_exactly_when_a_conflict_goes_unreported(
    condition_id: str,
    condition: StabilizationCondition,
    references: list[str],
    flags: tuple[bool, bool, bool],
    limitation: str | None,
) -> None:
    capture_improved, stability_degraded, plausibility_degraded = flags
    unreported = capture_improved and (stability_degraded or plausibility_degraded) and not (
        limitation or ""
    ).strip()

    def build() -> StabilizationResult:
        return _row(
            condition_id=condition_id,
            condition=condition,
            capture_improved=capture_improved,
            stability_degraded=stability_degraded,
            plausibility_degraded=plausibility_degraded,
            limitation_ref=limitation,
            **dict(zip(_METRIC_FIELDS, references)),
        )

    if unreported:
        with pytest.raises(DomainValidationError) as excinfo:
            build()
        assert excinfo.value.code == "UNREPORTED_TRADE_OFF"
    else:
        assert build().condition_id == condition_id
