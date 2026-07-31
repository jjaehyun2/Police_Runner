"""Task 7.4 unit tests for :mod:`pursuit_evasion_rl.research.statistics.paired`.

The fixed-data checks deliberately avoid re-running the implementation to build
their own expectations.  Each reference is reached by a *different* route:

* the Wilson bounds are recovered by inverting the score equation
  ``|p_hat - p| = z * sqrt(p(1-p)/n)`` -- once by bisection and once through the
  quadratic formula -- which is the interval's definition rather than the
  centre-plus-half-width algebra the module evaluates;
* the exact McNemar p-value is recovered by enumerating all ``2**m`` discordant
  sign sequences, rather than by summing binomial coefficients;
* the standard-normal quantiles are compared against published constants.

SciPy is installed in this development environment but is *not* a declared
dependency in ``pyproject.toml``, so it is only ever used as an optional extra
cross-check guarded by ``importorskip``.
"""

from __future__ import annotations

from itertools import product
import math

import numpy as np
import pytest

from pursuit_evasion_rl.research.errors import ResearchValidationError
from pursuit_evasion_rl.research.statistics import (
    BootstrapPlan,
    CaseStatus,
    ConfidenceInterval,
    CorrectionMethod,
    EffectDirection,
    MissingDataPolicy,
    PairedCase,
    PracticalThreshold,
    account_cases,
    apply_family_correction,
    benjamini_hochberg_adjusted_p_values,
    compare_paired_binary,
    compare_paired_continuous,
    evaluate_practical_gate,
    exact_mcnemar_p_value,
    factorial_2x2_contrasts,
    hierarchical_paired_bootstrap,
    holm_adjusted_p_values,
    normal_quantile,
    paired_table_from_outcomes,
    restricted_mean,
    wilson_interval,
)

pytestmark = pytest.mark.offline

# Published two-sided normal quantile for 95% coverage (z_{0.975}).
Z_975 = 1.959963984540054


# ---------------------------------------------------------------------------
# Independent reference implementations, used only by these tests
# ---------------------------------------------------------------------------


def _score_equation_roots_by_bisection(successes: int, trials: int, z: float) -> tuple[float, float]:
    """Invert the score test by bisection: the p where ``|p_hat - p| = z*se(p)``."""
    p_hat = successes / trials

    def discrepancy(p: float) -> float:
        return (p_hat - p) ** 2 - z * z * p * (1.0 - p) / trials

    def root(low: float, high: float) -> float:
        for _ in range(200):
            middle = 0.5 * (low + high)
            if discrepancy(low) * discrepancy(middle) <= 0.0:
                high = middle
            else:
                low = middle
        return 0.5 * (low + high)

    return root(0.0, p_hat) if p_hat > 0.0 else 0.0, root(p_hat, 1.0) if p_hat < 1.0 else 1.0


def _score_equation_roots_by_quadratic(successes: int, trials: int, z: float) -> tuple[float, float]:
    """The same roots from ``(1 + z^2/n) p^2 - (2 p_hat + z^2/n) p + p_hat^2 = 0``."""
    p_hat = successes / trials
    a = 1.0 + z * z / trials
    b = -(2.0 * p_hat + z * z / trials)
    c = p_hat * p_hat
    discriminant = math.sqrt(b * b - 4.0 * a * c)
    return (-b - discriminant) / (2.0 * a), (-b + discriminant) / (2.0 * a)


def _mcnemar_by_enumeration(n10: int, n01: int) -> float:
    """Two-sided exact p by enumerating every discordant sign sequence."""
    m = n10 + n01
    if m == 0:
        return 1.0
    extreme = min(n10, n01)
    outcomes = list(product((0, 1), repeat=m))
    tail = sum(1 for outcome in outcomes if sum(outcome) <= extreme)
    return min(1.0, 2.0 * tail / len(outcomes))


def _binary_cases(
    per_seed_pairs: dict[int, list[tuple[bool, bool]]], prefix: str = "case"
) -> list[PairedCase]:
    return [
        PairedCase(case_id=f"{prefix}-{seed}-{index}", training_seed=seed, outcome_a=a, outcome_b=b)
        for seed, pairs in per_seed_pairs.items()
        for index, (a, b) in enumerate(pairs)
    ]


def _risk_difference(groups) -> float:
    pooled = np.concatenate([np.asarray(group, dtype=float) for group in groups])
    return float(pooled[:, 0].mean() - pooled[:, 1].mean())


# ---------------------------------------------------------------------------
# Fixed synthetic data vs independent references
# ---------------------------------------------------------------------------


def test_normal_quantile_matches_published_constants() -> None:
    assert normal_quantile(0.975) == pytest.approx(Z_975, abs=1e-12)
    assert normal_quantile(0.95) == pytest.approx(1.6448536269514722, abs=1e-12)
    assert normal_quantile(0.995) == pytest.approx(2.5758293035489004, abs=1e-12)
    assert normal_quantile(0.5) == pytest.approx(0.0, abs=1e-12)


@pytest.mark.parametrize("successes,trials", [(40, 100), (2, 10), (7, 13), (95, 100)])
def test_wilson_interval_matches_score_equation_inversion(successes: int, trials: int) -> None:
    interval = wilson_interval(successes, trials)
    bisection = _score_equation_roots_by_bisection(successes, trials, Z_975)
    quadratic = _score_equation_roots_by_quadratic(successes, trials, Z_975)
    assert interval.lower == pytest.approx(bisection[0], abs=1e-10)
    assert interval.upper == pytest.approx(bisection[1], abs=1e-10)
    assert interval.lower == pytest.approx(quadratic[0], abs=1e-12)
    assert interval.upper == pytest.approx(quadratic[1], abs=1e-12)


def test_wilson_interval_fixed_reference_values() -> None:
    """40/100 successes: the score-equation roots are 0.30940 and 0.49800."""
    interval = wilson_interval(40, 100)
    assert interval.lower == pytest.approx(0.309401, abs=5e-7)
    assert interval.upper == pytest.approx(0.497997, abs=5e-7)
    assert 0.0 <= interval.lower < 0.40 < interval.upper <= 1.0


@pytest.mark.parametrize("successes,trials", [(0, 20), (20, 20), (1, 1), (0, 1)])
def test_wilson_interval_stays_inside_the_unit_interval(successes: int, trials: int) -> None:
    interval = wilson_interval(successes, trials)
    assert 0.0 <= interval.lower <= interval.upper <= 1.0
    assert interval.method == "wilson_score"


def test_wilson_interval_rejects_impossible_counts() -> None:
    with pytest.raises(ResearchValidationError):
        wilson_interval(5, 4)
    with pytest.raises(ResearchValidationError):
        wilson_interval(0, 0)


def test_exact_mcnemar_matches_explicit_enumeration() -> None:
    """n01=3, n10=9: 2 * P(X <= 3) with X ~ Binomial(12, 1/2) = 598/4096."""
    p_value = exact_mcnemar_p_value(n10=9, n01=3)
    assert p_value == pytest.approx(_mcnemar_by_enumeration(9, 3), abs=1e-15)
    assert p_value == pytest.approx(598 / 4096, abs=1e-15)
    assert p_value == pytest.approx(0.146, abs=5e-4)


@pytest.mark.parametrize("n10,n01", [(0, 0), (1, 0), (5, 5), (7, 2), (2, 7), (10, 1)])
def test_exact_mcnemar_edges_and_symmetry(n10: int, n01: int) -> None:
    p_value = exact_mcnemar_p_value(n10=n10, n01=n01)
    assert 0.0 <= p_value <= 1.0
    assert p_value == pytest.approx(_mcnemar_by_enumeration(n10, n01), abs=1e-15)
    assert p_value == pytest.approx(exact_mcnemar_p_value(n10=n01, n01=n10), abs=1e-15)
    if n10 == n01:
        assert p_value == pytest.approx(1.0)


def test_optional_scipy_cross_check() -> None:
    """Extra confirmation against SciPy, which is not a declared dependency."""
    stats = pytest.importorskip("scipy.stats")
    lower, upper = stats.binomtest(40, 100).proportion_ci(method="wilson")
    interval = wilson_interval(40, 100)
    assert interval.lower == pytest.approx(lower, abs=1e-12)
    assert interval.upper == pytest.approx(upper, abs=1e-12)
    assert exact_mcnemar_p_value(n10=9, n01=3) == pytest.approx(
        stats.binomtest(3, 12, 0.5).pvalue, abs=1e-15
    )


def test_paired_table_framing_is_a_over_b() -> None:
    pairs = [(True, True)] * 20 + [(True, False)] * 9 + [(False, True)] * 3 + [(False, False)] * 8
    table = paired_table_from_outcomes(pairs)
    assert (table.n11, table.n10, table.n01, table.n00) == (20, 9, 3, 8)
    assert table.n_pairs == 40 and table.n_discordant == 12
    assert table.proportion_a == pytest.approx(29 / 40)
    assert table.proportion_b == pytest.approx(23 / 40)
    # Positive risk difference means policy A succeeded more often.
    assert table.risk_difference == pytest.approx((9 - 3) / 40)
    assert table.risk_difference == pytest.approx(table.proportion_a - table.proportion_b)
    assert table.odds_ratio == pytest.approx(9 / 3)
    assert not table.has_empty_discordant_cell


def test_paired_odds_ratio_stays_positive_and_finite_with_an_empty_cell() -> None:
    table = paired_table_from_outcomes([(True, False)] * 6 + [(True, True)] * 4)
    assert table.has_empty_discordant_cell
    assert table.odds_ratio == pytest.approx(6.5 / 0.5)
    assert 0.0 < table.odds_ratio < math.inf


def test_restricted_mean_censors_at_the_horizon() -> None:
    # min(t, 60) over [10, 50, 90, never] = (10 + 50 + 60 + 60) / 4.
    assert restricted_mean([10.0, 50.0, 90.0, None], horizon=60.0) == pytest.approx(45.0)
    with pytest.raises(ResearchValidationError):
        restricted_mean([1.0], horizon=0.0)


# ---------------------------------------------------------------------------
# Multiplicity corrections
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "raw",
    [
        (0.001, 0.008, 0.039, 0.041, 0.042, 0.6),
        (0.01, 0.02, 0.03, 0.04),
        (0.5,),
        (0.2, 0.2, 0.2, 0.2, 0.2),
        (0.0, 1.0, 0.049),
    ],
)
def test_holm_and_bh_are_monotone_and_conservative(raw: tuple[float, ...]) -> None:
    holm = holm_adjusted_p_values(raw)
    bh = benjamini_hochberg_adjusted_p_values(raw)
    assert len(holm) == len(bh) == len(raw)
    for index, value in enumerate(raw):
        assert holm[index] >= value - 1e-12, "Holm must never shrink a raw p-value"
        assert bh[index] >= value - 1e-12, "BH must never shrink a raw p-value"
        assert holm[index] >= bh[index] - 1e-12, "Holm must be at least as conservative as BH"
        assert 0.0 <= holm[index] <= 1.0 and 0.0 <= bh[index] <= 1.0
    # Rank order is preserved, i.e. results are returned in the caller's order.
    order = sorted(range(len(raw)), key=lambda index: raw[index])
    assert [holm[index] for index in order] == sorted(holm)
    assert [bh[index] for index in order] == sorted(bh)


def test_holm_matches_the_hand_computed_step_down_sequence() -> None:
    # Sorted raw p: 0.01, 0.02, 0.04 with m = 3.
    # Step-down multipliers 3, 2, 1 give 0.03, 0.04, 0.04 after the running max.
    assert holm_adjusted_p_values((0.04, 0.01, 0.02)) == pytest.approx((0.04, 0.03, 0.04))
    # BH multipliers m/rank = 3/1, 3/2, 3/3 give 0.03, 0.03, 0.04 after the running min.
    assert benjamini_hochberg_adjusted_p_values((0.04, 0.01, 0.02)) == pytest.approx(
        (0.04, 0.03, 0.03)
    )


def test_correction_rejects_an_empty_or_invalid_family() -> None:
    with pytest.raises(ResearchValidationError):
        holm_adjusted_p_values(())
    with pytest.raises(ResearchValidationError):
        benjamini_hochberg_adjusted_p_values((0.5, 1.5))


# ---------------------------------------------------------------------------
# Hierarchical bootstrap structure and calibration
# ---------------------------------------------------------------------------


def test_bootstrap_never_mixes_pairs_across_training_seeds() -> None:
    """Each seed's rows carry its own tag; a resampled group must stay single-tagged."""
    groups = [np.full((25, 2), float(seed)) for seed in (1.0, 2.0, 3.0, 4.0)]
    seen: list[float] = []

    def statistic(resampled) -> float:
        for group in resampled:
            tags = set(np.asarray(group).ravel().tolist())
            assert len(tags) == 1, "a resampled seed group mixed pairs from different seeds"
            seen.append(tags.pop())
        return float(np.concatenate(resampled).mean())

    interval, replicates = hierarchical_paired_bootstrap(
        groups, statistic, plan=BootstrapPlan(n_bootstrap=50, rng_seed=11, rationale="structural check")
    )
    assert len(replicates) == 50
    assert set(seen) == {1.0, 2.0, 3.0, 4.0}, "every seed should be drawn at least once"
    assert 1.0 <= interval.lower <= interval.upper <= 4.0


def test_bootstrap_plan_requires_a_rationale_when_it_leaves_the_default() -> None:
    assert BootstrapPlan().n_bootstrap == 10_000
    assert BootstrapPlan(n_bootstrap=2_000, rationale="runtime ceiling").n_bootstrap == 2_000
    with pytest.raises(ResearchValidationError):
        BootstrapPlan(n_bootstrap=2_000)
    with pytest.raises(ResearchValidationError):
        BootstrapPlan(n_bootstrap=0)


def test_null_simulation_single_cluster_coverage_is_near_nominal() -> None:
    """With one Training_Seed the design degenerates to a plain paired bootstrap.

    That is the case where a percentile interval should be *calibrated* rather
    than merely conservative, so the tolerance band is tight: 150 datasets give
    a coverage standard error near 0.016, and the band [0.88, 0.995] is roughly
    +/- 4 standard errors around the nominal 0.95.
    """
    rng = np.random.default_rng(2024)
    covered = 0
    widths = []
    datasets = 150
    for index in range(datasets):
        groups = [rng.random((200, 2)) < 0.55]
        interval, _ = hierarchical_paired_bootstrap(
            groups,
            _risk_difference,
            plan=BootstrapPlan(n_bootstrap=150, rng_seed=index, rationale="calibration simulation"),
        )
        covered += interval.lower <= 0.0 <= interval.upper
        widths.append(interval.width)
    coverage = covered / datasets
    assert 0.88 <= coverage <= 0.995, f"single-cluster coverage {coverage} left the calibration band"
    # Analytic width for a difference of two independent Bernoulli(0.55) means.
    analytic = 2.0 * Z_975 * math.sqrt(2.0 * 0.55 * 0.45 / 200)
    assert np.mean(widths) == pytest.approx(analytic, rel=0.15)


def test_null_simulation_hierarchical_coverage_is_at_least_nominal_and_not_vacuous() -> None:
    """Seed-outer resampling must not under-cover, and must not be trivially wide.

    Five homogeneous seeds carry no real between-seed variance, so the outer
    layer adds spurious cluster variability and the interval is expected to be
    conservative; the assertion is therefore one-sided (>= nominal minus slack).
    The width bound rules out a degenerate implementation that always returns
    the whole [-1, 1] domain.
    """
    rng = np.random.default_rng(99)
    covered = 0
    widths = []
    datasets = 80
    for index in range(datasets):
        groups = [rng.random((40, 2)) < 0.55 for _ in range(5)]
        interval, _ = hierarchical_paired_bootstrap(
            groups,
            _risk_difference,
            plan=BootstrapPlan(n_bootstrap=120, rng_seed=index, rationale="calibration simulation"),
        )
        covered += interval.lower <= 0.0 <= interval.upper
        widths.append(interval.width)
    coverage = covered / datasets
    assert coverage >= 0.90, f"hierarchical coverage {coverage} fell below nominal"
    assert np.mean(widths) < 0.5, "the hierarchical interval is vacuously wide"
    assert all(-1.0 <= width and width <= 2.0 for width in widths)


def test_known_effect_simulation_recovers_direction_and_significance() -> None:
    """Policy A succeeds with probability 0.7, policy B with 0.5, on shared cases.

    Expected discordant rates are 0.7*0.5 = 0.35 favouring A and 0.3*0.5 = 0.15
    favouring B, so the true risk difference is +0.20 and the true conditional
    odds ratio is 0.35/0.15 = 2.33.
    """
    rng = np.random.default_rng(4242)
    cases = []
    for seed in range(5):
        for index in range(100):
            cases.append(
                PairedCase(
                    case_id=f"case-{seed}-{index}",
                    training_seed=seed,
                    outcome_a=bool(rng.random() < 0.7),
                    outcome_b=bool(rng.random() < 0.5),
                )
            )
    result = compare_paired_binary(
        cases,
        comparison_id="known-effect",
        metric="capture_success",
        plan=BootstrapPlan(n_bootstrap=400, rng_seed=5, rationale="unit-test runtime ceiling"),
    )
    primary = result.primary
    assert primary.n_pairs == 500 and primary.n_seeds == 5
    assert primary.risk_difference == pytest.approx(0.20, abs=0.06)
    assert primary.risk_difference_interval.lower > 0.0, "the CI must exclude the null"
    assert primary.odds_ratio == pytest.approx(7 / 3, rel=0.35)
    assert primary.odds_ratio_interval.lower > 1.0
    assert primary.p_value_raw < 1e-6, "a 0.20 risk difference over 500 pairs must be detected"
    assert len(primary.seed_rows) == 5
    assert all(row.effect_size > 0.0 for row in primary.seed_rows)
    assert primary.between_seed_sd is not None and 0.0 < primary.between_seed_sd < 0.2
    # The mirror comparison must reverse the sign and reproduce the same p-value.
    mirrored = compare_paired_binary(
        [
            PairedCase(
                case_id=case.case_id,
                training_seed=case.training_seed,
                outcome_a=case.outcome_b,
                outcome_b=case.outcome_a,
            )
            for case in cases
        ],
        comparison_id="known-effect-mirrored",
        metric="capture_success",
        plan=BootstrapPlan(n_bootstrap=400, rng_seed=5, rationale="unit-test runtime ceiling"),
    )
    assert mirrored.primary.risk_difference == pytest.approx(-primary.risk_difference)
    assert mirrored.primary.p_value_raw == pytest.approx(primary.p_value_raw)
    assert mirrored.primary.odds_ratio == pytest.approx(1.0 / primary.odds_ratio)


def test_null_effect_comparison_does_not_claim_a_direction() -> None:
    rng = np.random.default_rng(7)
    cases = [
        PairedCase(
            case_id=f"case-{seed}-{index}",
            training_seed=seed,
            outcome_a=bool(rng.random() < 0.6),
            outcome_b=bool(rng.random() < 0.6),
        )
        for seed in range(5)
        for index in range(100)
    ]
    result = compare_paired_binary(
        cases,
        comparison_id="null",
        metric="capture_success",
        plan=BootstrapPlan(n_bootstrap=400, rng_seed=3, rationale="unit-test runtime ceiling"),
    )
    assert result.primary.risk_difference_interval.excludes(0.0) is False
    assert result.primary.p_value_raw > 0.05


def test_bootstrap_is_reproducible_for_a_fixed_rng_seed() -> None:
    cases = _binary_cases({0: [(True, False)] * 10 + [(False, True)] * 4, 1: [(True, True)] * 14})
    plan = BootstrapPlan(n_bootstrap=300, rng_seed=17, rationale="unit-test runtime ceiling")
    first = compare_paired_binary(cases, comparison_id="repro", metric="capture_success", plan=plan)
    second = compare_paired_binary(cases, comparison_id="repro", metric="capture_success", plan=plan)
    assert first.result_hash == second.result_hash


# ---------------------------------------------------------------------------
# Case accounting (Requirement 8.8)
# ---------------------------------------------------------------------------


def _mixed_status_cases() -> list[PairedCase]:
    cases = _binary_cases(
        {
            0: [(True, False)] * 6 + [(False, True)] * 2 + [(True, True)] * 4,
            1: [(True, False)] * 5 + [(False, True)] * 3 + [(False, False)] * 4,
        }
    )
    cases += [
        PairedCase(case_id="miss-0", training_seed=0, status=CaseStatus.MISSING, reason="evaluation_never_scheduled"),
        PairedCase(case_id="miss-1", training_seed=1, status=CaseStatus.MISSING, reason="evaluation_never_scheduled"),
        PairedCase(case_id="fail-0", training_seed=0, status=CaseStatus.FAILED, reason="simulator_crash"),
        PairedCase(case_id="int-0", training_seed=1, status=CaseStatus.INTERRUPTED, reason="wall_clock_ceiling"),
        PairedCase(case_id="int-1", training_seed=1, status=CaseStatus.INTERRUPTED, reason="wall_clock_ceiling"),
    ]
    return cases


def test_case_accounting_conserves_the_planned_total() -> None:
    accounting = account_cases(_mixed_status_cases())
    assert accounting.planned == 29
    assert accounting.valid == 24
    assert (accounting.missing, accounting.failed, accounting.interrupted) == (2, 1, 2)
    assert accounting.valid + accounting.unusable == accounting.planned
    assert accounting.reasons == {
        "evaluation_never_scheduled": 2,
        "simulator_crash": 1,
        "wall_clock_ceiling": 2,
    }


def test_primary_and_complete_case_analyses_are_both_reported_and_distinct() -> None:
    result = compare_paired_binary(
        _mixed_status_cases(),
        comparison_id="accounting",
        metric="capture_success",
        plan=BootstrapPlan(n_bootstrap=300, rng_seed=1, rationale="unit-test runtime ceiling"),
    )
    primary, complete = result.primary, result.complete_case
    assert complete is not None and result.complete_case_note is None
    assert primary.label == "primary" and complete.label == "complete_case_sensitivity"
    assert primary.missing_data_policy is MissingDataPolicy.PRE_REGISTERED_WORST_CASE
    assert complete.missing_data_policy is MissingDataPolicy.COMPLETE_CASE

    # Nothing is dropped from the primary number: it keeps every planned pair.
    assert primary.n_pairs == result.accounting.planned == 29
    assert complete.n_pairs == result.accounting.valid == 24
    assert primary.n_pairs - complete.n_pairs == result.accounting.unusable

    # The five unusable pairs enter the primary analysis as concordant failures,
    # which dilutes the risk difference without changing the discordant counts.
    assert primary.table.n00 == complete.table.n00 + 5
    assert (primary.table.n10, primary.table.n01) == (complete.table.n10, complete.table.n01)
    assert abs(primary.risk_difference) < abs(complete.risk_difference)
    assert primary.p_value_raw == pytest.approx(complete.p_value_raw)


def test_a_case_may_not_carry_both_a_failure_status_and_an_outcome() -> None:
    with pytest.raises(ResearchValidationError):
        PairedCase(case_id="bad", training_seed=0, status=CaseStatus.FAILED, reason="crash", outcome_a=True)
    with pytest.raises(ResearchValidationError):
        PairedCase(case_id="bad", training_seed=0, status=CaseStatus.FAILED)
    with pytest.raises(ResearchValidationError):
        PairedCase(case_id="bad", training_seed=0, outcome_a=True)


def test_repeated_checkpoints_of_one_seed_do_not_inflate_the_seed_count() -> None:
    cases = [
        PairedCase(
            case_id=f"case-{checkpoint}-{index}",
            training_seed=0,
            outcome_a=True,
            outcome_b=index % 2 == 0,
            checkpoint_time=checkpoint,
        )
        for checkpoint in ("step_100k", "step_200k", "final")
        for index in range(10)
    ]
    result = compare_paired_binary(
        cases,
        comparison_id="checkpoints",
        metric="capture_success",
        plan=BootstrapPlan(n_bootstrap=100, rng_seed=2, rationale="unit-test runtime ceiling"),
    )
    assert result.primary.n_seeds == 1
    assert result.primary.n_pairs == 30
    assert result.primary.between_seed_sd is None


# ---------------------------------------------------------------------------
# Practical threshold gate (Requirement 8.7)
# ---------------------------------------------------------------------------


def _threshold(minimum: float = 0.05, margin: float | None = 0.02) -> PracticalThreshold:
    return PracticalThreshold(
        metric="capture_success",
        unit="proportion_of_episode_cases",
        minimum_effect=minimum,
        non_inferiority_margin=margin,
    )


def test_a_significant_but_practically_tiny_effect_fails_the_superiority_gate() -> None:
    interval = ConfidenceInterval(lower=0.004, upper=0.016, confidence_level=0.95, method="test")
    decision = evaluate_practical_gate(0.01, interval, _threshold(minimum=0.05))
    assert decision.statistically_significant is True
    assert decision.practically_significant is False
    assert decision.superiority is False
    assert any("practical threshold" in reason for reason in decision.reasons)
    # It may still clear a non-inferiority margin, which is a weaker claim.
    assert decision.non_inferiority is True


def test_a_significant_and_practically_large_effect_passes_the_superiority_gate() -> None:
    interval = ConfidenceInterval(lower=0.06, upper=0.20, confidence_level=0.95, method="test")
    decision = evaluate_practical_gate(0.13, interval, _threshold(minimum=0.05))
    assert decision.statistically_significant and decision.practically_significant
    assert decision.superiority is True and decision.non_inferiority is True


def test_a_large_but_non_significant_effect_fails_the_superiority_gate() -> None:
    interval = ConfidenceInterval(lower=-0.03, upper=0.31, confidence_level=0.95, method="test")
    decision = evaluate_practical_gate(0.14, interval, _threshold(minimum=0.05))
    assert decision.practically_significant is True
    assert decision.statistically_significant is False
    assert decision.superiority is False
    assert decision.non_inferiority is False


def test_the_gate_honours_a_less_is_better_metric() -> None:
    interval = ConfidenceInterval(lower=-40.0, upper=-12.0, confidence_level=0.95, method="test")
    threshold = PracticalThreshold(
        metric="time_to_capture",
        unit="s",
        minimum_effect=10.0,
        direction=EffectDirection.LESS_IS_BETTER,
    )
    decision = evaluate_practical_gate(-25.0, interval, threshold)
    assert decision.superiority is True
    # The same interval read as greater-is-better is a significant *regression*.
    reversed_decision = evaluate_practical_gate(-25.0, interval, _threshold(minimum=10.0, margin=None))
    assert reversed_decision.statistically_significant is True
    assert reversed_decision.superiority is False


def test_the_comparison_attaches_the_gate_to_the_primary_effect() -> None:
    cases = _binary_cases({seed: [(True, False)] * 30 + [(False, True)] * 5 for seed in range(4)})
    result = compare_paired_binary(
        cases,
        comparison_id="gated",
        metric="capture_success",
        plan=BootstrapPlan(n_bootstrap=300, rng_seed=8, rationale="unit-test runtime ceiling"),
        threshold=_threshold(minimum=0.05),
    )
    assert result.gate is not None and result.gate.superiority is True
    strict = compare_paired_binary(
        cases,
        comparison_id="gated-strict",
        metric="capture_success",
        plan=BootstrapPlan(n_bootstrap=300, rng_seed=8, rationale="unit-test runtime ceiling"),
        threshold=_threshold(minimum=0.95),
    )
    assert strict.gate is not None and strict.gate.superiority is False


# ---------------------------------------------------------------------------
# Family correction over comparisons, continuous effects, and the 2x2 contrast
# ---------------------------------------------------------------------------


def test_family_correction_keeps_each_adjusted_p_with_its_own_comparison() -> None:
    results = []
    for index, (wins_a, wins_b) in enumerate([(30, 4), (18, 12), (11, 9)]):
        cases = _binary_cases(
            {seed: [(True, False)] * wins_a + [(False, True)] * wins_b for seed in range(3)},
            prefix=f"family{index}",
        )
        results.append(
            compare_paired_binary(
                cases,
                comparison_id=f"comparison-{index}",
                metric="capture_success",
                plan=BootstrapPlan(n_bootstrap=100, rng_seed=index, rationale="unit-test runtime ceiling"),
            )
        )
    corrected = apply_family_correction(results, CorrectionMethod.HOLM)
    assert [item.comparison_id for item in corrected] == [item.comparison_id for item in results]
    for original, adjusted in zip(results, corrected):
        assert adjusted.primary.adjusted_p_value >= original.primary.p_value_raw - 1e-12
        assert adjusted.primary.p_value_raw == original.primary.p_value_raw
    assert results[0].primary.adjusted_p_value is None, "the input results stay untouched"


def test_continuous_comparison_reports_restricted_means_and_both_analyses() -> None:
    rng = np.random.default_rng(31)
    cases = []
    for seed in range(4):
        for index in range(50):
            cases.append(
                PairedCase(
                    case_id=f"rmst-{seed}-{index}",
                    training_seed=seed,
                    outcome_a=float(rng.normal(120.0, 20.0)),
                    outcome_b=float(rng.normal(150.0, 20.0)),
                )
            )
    cases.append(
        PairedCase(case_id="rmst-lost", training_seed=0, status=CaseStatus.INTERRUPTED, reason="wall_clock_ceiling")
    )
    result = compare_paired_continuous(
        cases,
        comparison_id="rmst",
        metric="restricted_mean_time_to_capture",
        unit="s",
        horizon=300.0,
        plan=BootstrapPlan(n_bootstrap=300, rng_seed=6, rationale="unit-test runtime ceiling"),
    )
    assert result.primary.n_pairs == 201 and result.complete_case.n_pairs == 200
    assert result.primary.effect_size == pytest.approx(-30.0, abs=8.0)
    assert result.primary.effect_interval.upper < 0.0
    assert result.primary.p_value_raw < 0.05
    # The interrupted case enters both arms at the horizon, so it cannot shift
    # the direction but it does move the point estimate slightly.
    assert result.primary.effect_size != result.complete_case.effect_size
    assert result.primary.n_seeds == 4 and len(result.primary.seed_rows) == 4


def test_factorial_2x2_contrasts_recover_known_main_effects_and_interaction() -> None:
    """Cell means 0, 2, 3, 10 give main effects 4.5 and 5.5 and an interaction of 5."""
    rng = np.random.default_rng(1234)
    truth = {(False, False): 0.0, (True, False): 2.0, (False, True): 3.0, (True, True): 10.0}
    cells = {
        levels: [rng.normal(mean, 0.1, size=40).tolist() for _ in range(5)]
        for levels, mean in truth.items()
    }
    result = factorial_2x2_contrasts(
        cells,
        metric="capture_success_rate",
        unit="proportion_of_episode_cases",
        plan=BootstrapPlan(n_bootstrap=300, rng_seed=21, rationale="unit-test runtime ceiling"),
    )
    assert result.main_effect_a.name == "u_turn_suppression_main_effect"
    assert result.main_effect_b.name == "hysteresis_main_effect"
    assert result.interaction.name == "interaction"
    assert result.main_effect_a.effect_size == pytest.approx(4.5, abs=0.05)
    assert result.main_effect_b.effect_size == pytest.approx(5.5, abs=0.05)
    assert result.interaction.effect_size == pytest.approx(5.0, abs=0.05)
    for estimate, expected in (
        (result.main_effect_a, 4.5),
        (result.main_effect_b, 5.5),
        (result.interaction, 5.0),
    ):
        assert estimate.interval.lower <= expected <= estimate.interval.upper
        assert estimate.interval.width > 0.0
    assert result.n_seeds == 5 and result.n_pairs_per_cell == 200


def test_factorial_contrast_rejects_a_broken_matrix() -> None:
    cells = {
        (False, False): [[1.0, 2.0]],
        (True, False): [[1.0, 2.0]],
        (False, True): [[1.0, 2.0]],
    }
    with pytest.raises(ResearchValidationError):
        factorial_2x2_contrasts(cells, metric="m", unit="u")
    cells[(True, True)] = [[1.0, 2.0, 3.0]]
    with pytest.raises(ResearchValidationError):
        factorial_2x2_contrasts(cells, metric="m", unit="u")
