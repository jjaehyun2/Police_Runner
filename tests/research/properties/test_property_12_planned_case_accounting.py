"""Property 12 coverage: planned-case accounting is conserved under failures."""

from __future__ import annotations

from hypothesis import given, settings, strategies as st
import pytest

from pursuit_evasion_rl.research.errors import ResearchValidationError
from pursuit_evasion_rl.research.metrics.physical import (
    FailureLedgerRecord,
    PhysicalPlausibilityReport,
    ViolationCategory,
    account_intention_to_evaluate,
)
from pursuit_evasion_rl.research.statistics.paired import CaseStatus, PairedCase, account_cases

# **Property 12: Planned-case accounting is conserved under failures**
# **Validates: Requirements 8.7, 15.8, 19.4**

pytestmark = pytest.mark.offline

_PBT_SETTINGS = settings(max_examples=100, deadline=None)

_statuses = st.sampled_from(list(CaseStatus))


@st.composite
def paired_case_lists(draw: st.DrawFn) -> list[PairedCase]:
    n = draw(st.integers(min_value=1, max_value=30))
    cases = []
    for index in range(n):
        status = draw(_statuses)
        if status is CaseStatus.VALID:
            cases.append(
                PairedCase(
                    case_id=f"case-{index}", training_seed=index % 5, status=status,
                    outcome_a=draw(st.booleans()), outcome_b=draw(st.booleans()),
                )
            )
        else:
            cases.append(
                PairedCase(case_id=f"case-{index}", training_seed=index % 5, status=status, reason="synthetic reason")
            )
    return cases


@_PBT_SETTINGS
@given(cases=paired_case_lists())
def test_statistics_case_accounting_always_conserves_the_planned_total(cases: list[PairedCase]) -> None:
    accounting = account_cases(cases)
    assert accounting.planned == len(cases)
    assert accounting.valid + accounting.missing + accounting.failed + accounting.interrupted == accounting.planned
    # Every non-valid case contributes exactly one reason count, and the total
    # of all reason counts equals the number of unusable cases -- no case's
    # exclusion reason silently disappears.
    assert accounting.unusable == sum(accounting.reasons.values())


@_PBT_SETTINGS
@given(cases=paired_case_lists(), extra_status=_statuses)
def test_appending_one_more_case_always_increments_the_planned_total_by_exactly_one(
    cases: list[PairedCase], extra_status: CaseStatus
) -> None:
    before = account_cases(cases)
    if extra_status is CaseStatus.VALID:
        extra = PairedCase(case_id="extra-case", training_seed=0, status=extra_status, outcome_a=True, outcome_b=False)
    else:
        extra = PairedCase(case_id="extra-case", training_seed=0, status=extra_status, reason="added afterward")
    after = account_cases([*cases, extra])
    assert after.planned == before.planned + 1
    assert (after.valid - before.valid) + (after.missing - before.missing) + (after.failed - before.failed) + (
        after.interrupted - before.interrupted
    ) == 1


# ---------------------------------------------------------------------------
# The physical-plausibility layer's parallel conservation discipline
# ---------------------------------------------------------------------------


def _valid_report(episode_id: str) -> PhysicalPlausibilityReport:
    counts = {category: 0 for category in ViolationCategory}
    return PhysicalPlausibilityReport(
        episode_id=episode_id, transition_count=1, counts=counts, violations=(), failure=None,
    )


def _failed_report(episode_id: str) -> PhysicalPlausibilityReport:
    from pursuit_evasion_rl.research.metrics.physical import PhysicalViolation

    violation = PhysicalViolation(
        category=ViolationCategory.TELEPORT, step_index=0, officer_id=0, code="TELEPORT", message="synthetic",
        before=None, after=None,  # type: ignore[arg-type]
    )
    failure = FailureLedgerRecord(episode_id=episode_id, counts=(("teleport", 1),), violations=(violation,))
    counts = {category: 0 for category in ViolationCategory}
    counts[ViolationCategory.TELEPORT] = 1
    return PhysicalPlausibilityReport(
        episode_id=episode_id, transition_count=1, counts=counts, violations=(violation,), failure=failure,
    )


@st.composite
def episode_outcomes(draw: st.DrawFn) -> tuple[list[str], list[str]]:
    n = draw(st.integers(min_value=1, max_value=20))
    planned = [f"episode-{i}" for i in range(n)]
    reported_indices = draw(st.lists(st.integers(min_value=0, max_value=n - 1), unique=True, max_size=n))
    reported = sorted(reported_indices)
    return planned, [planned[i] for i in reported]


@_PBT_SETTINGS
@given(data=episode_outcomes(), failure_indices=st.data())
def test_intention_to_evaluate_accounting_always_conserves_the_planned_population(data, failure_indices) -> None:
    planned, reported = data
    reports = []
    for episode_id in reported:
        is_failure = failure_indices.draw(st.booleans(), label=f"fail_{episode_id}")
        reports.append(_failed_report(episode_id) if is_failure else _valid_report(episode_id))

    accounting = account_intention_to_evaluate(reports, planned_episode_ids=planned)
    assert accounting.planned_count == len(planned)
    assert accounting.valid_count + accounting.failed_count + accounting.missing_count == accounting.planned_count
    assert accounting.conserved
    # Every reported episode lands in exactly one of valid/failed, never both,
    # and every unreported planned episode surfaces as missing.
    assert set(accounting.valid_episode_ids) | {item.episode_id for item in accounting.failed_records} == set(reported)
    assert set(accounting.missing_episode_ids) == set(planned) - set(reported)


@_PBT_SETTINGS
@given(planned=st.lists(st.integers(min_value=0, max_value=1000).map(lambda i: f"episode-{i}"), min_size=2, max_size=10, unique=True))
def test_a_duplicate_report_for_the_same_episode_is_rejected_not_silently_merged(planned: list[str]) -> None:
    reports = [_valid_report(planned[0]), _valid_report(planned[0])]
    with pytest.raises(ResearchValidationError) as excinfo:
        account_intention_to_evaluate(reports, planned_episode_ids=planned)
    assert excinfo.value.code == "DUPLICATE_EPISODE_REPORT"
