"""Focused offline tests for outcome, confidence and latency metrics.

Covers exactly-one-outcome accounting, count conservation, rates, configured
binomial confidence intervals, episode-length distribution, the preliminary
flag, per-episode measurements and ``perf_counter_ns`` inference-latency timing
with explicit warmups, device metadata and ``-1/not_measured`` rows outside
policy inference (design section 9, "Metrics, latency and claim evidence";
Requirements 10.1-10.9, 13.8).

All networks and records are built in process from committed offline fixtures
and deterministic runners; no external OSM access is performed.
"""

from __future__ import annotations

import random

import pytest

from pursuit_evasion_rl.osm_demo.coarsening import coarsen_raw_graph, prepare_model_network
from pursuit_evasion_rl.osm_demo.fixtures import daejeon
from pursuit_evasion_rl.osm_demo.models import (
    DomainValidationError,
    EpisodeOutcome,
    LatencySample,
    MetricsSummary,
)
from pursuit_evasion_rl.osm_demo.metrics import (
    LATENCY_MEASURED,
    LATENCY_NOT_MEASURED,
    NOT_MEASURED_VALUE_MS,
    aggregate_outcomes,
    confidence_interval,
    episode_step_count,
    measure_inference_latency,
    not_measured_latency,
    placement_position,
    summarize_episode,
    summarize_latency_ms,
    summarize_latency_samples,
    summarize_metrics,
    wald_interval,
    wilson_interval,
)
from pursuit_evasion_rl.osm_demo.policies import BaselinePolicePolicy
from pursuit_evasion_rl.osm_demo.runner import run_batch

pytestmark = pytest.mark.offline


# ---------------------------------------------------------------------------
# Fixtures and helpers
# ---------------------------------------------------------------------------
def _grid_network():
    return prepare_model_network(coarsen_raw_graph(daejeon())).network


def _records(count: int = 6, *, run_seed: int = 4242):
    network = _grid_network()
    from pursuit_evasion_rl.osm_demo.models import EpisodeConfig

    config = EpisodeConfig(
        dt_s=1.0,
        police_speed_mps=15.0,
        fugitive_speed_mps=10.0,
        capture_radius_m=20.0,
        max_steps=12,
    )
    policy = BaselinePolicePolicy(network)
    batch = run_batch(network, config, policy, run_id="metrics", run_seed=run_seed, episode_count=count)
    return network, batch, policy.profile


def _outcomes(captures: int, escapes: int, timeouts: int):
    return (
        [EpisodeOutcome.CAPTURE] * captures
        + [EpisodeOutcome.ESCAPE] * escapes
        + [EpisodeOutcome.TIMEOUT] * timeouts
    )


# ---------------------------------------------------------------------------
# Outcome aggregation: exactly one outcome + count conservation (10.2-10.4, 13.8)
# ---------------------------------------------------------------------------
def test_aggregate_outcomes_conserves_counts_and_rates():
    outcomes = _outcomes(3, 2, 1)
    aggregate = aggregate_outcomes(outcomes)

    assert aggregate.total == 6
    assert aggregate.counts == {"capture": 3, "escape": 2, "timeout": 1}
    # Counts always sum to the sequence length (conservation).
    assert sum(aggregate.counts.values()) == len(outcomes)
    # Each rate is its count divided by the total.
    assert aggregate.rates["capture"] == pytest.approx(0.5)
    assert aggregate.rates["escape"] == pytest.approx(2 / 6)
    assert aggregate.rates["timeout"] == pytest.approx(1 / 6)
    assert sum(aggregate.rates.values()) == pytest.approx(1.0)


def test_aggregate_outcomes_empty_sequence_is_all_zero():
    aggregate = aggregate_outcomes([])
    assert aggregate.total == 0
    assert aggregate.counts == {"capture": 0, "escape": 0, "timeout": 0}
    # No division by zero; empty rates are 0.0.
    assert aggregate.rates == {"capture": 0.0, "escape": 0.0, "timeout": 0.0}


def test_aggregate_outcomes_rejects_non_outcome_values():
    with pytest.raises(DomainValidationError) as excinfo:
        aggregate_outcomes(["capture"])  # raw string is not an EpisodeOutcome
    assert excinfo.value.code == "INVALID_OUTCOME"


def test_outcome_conservation_holds_over_many_generated_sequences():
    """Property-style: for any finite outcome sequence, counts sum to length and
    each rate equals count/length (Requirements 10.3, 10.4, 13.8)."""
    rng = random.Random(20240607)
    members = list(EpisodeOutcome)
    for _ in range(500):
        length = rng.randint(0, 40)
        outcomes = [rng.choice(members) for _ in range(length)]
        aggregate = aggregate_outcomes(outcomes)
        assert sum(aggregate.counts.values()) == length
        for key, count in aggregate.counts.items():
            expected = (count / length) if length else 0.0
            assert aggregate.rates[key] == pytest.approx(expected)
        # Exactly one outcome per item: per-outcome counts match a direct tally.
        for outcome in members:
            assert aggregate.counts[outcome.value] == outcomes.count(outcome)


# ---------------------------------------------------------------------------
# Confidence intervals (Requirement 10.7)
# ---------------------------------------------------------------------------
def test_confidence_interval_methods_are_configurable_and_bounded():
    for method in ("wilson", "wald", "normal"):
        low, high = confidence_interval(3, 10, method=method)
        assert 0.0 <= low <= high <= 1.0


def test_wilson_and_wald_bracket_the_point_estimate_in_interior():
    low_w, high_w = wilson_interval(4, 10)
    low_d, high_d = wald_interval(4, 10)
    assert low_w <= 0.4 <= high_w
    assert low_d <= 0.4 <= high_d


def test_confidence_interval_extremes_stay_within_unit_range():
    assert confidence_interval(0, 10) == pytest.approx(wilson_interval(0, 10))
    low, high = confidence_interval(10, 10)
    assert 0.0 <= low <= 1.0 and high == pytest.approx(1.0)
    # Zero total is well defined.
    assert confidence_interval(0, 0) == (0.0, 0.0)


def test_confidence_interval_rejects_unknown_method_and_bad_inputs():
    with pytest.raises(DomainValidationError) as unknown:
        confidence_interval(1, 2, method="student-t")
    assert unknown.value.code == "UNKNOWN_CONFIDENCE_METHOD"

    with pytest.raises(DomainValidationError) as bad:
        confidence_interval(5, 3)  # successes > total
    assert bad.value.code == "INVALID_BINOMIAL_INPUT"


# ---------------------------------------------------------------------------
# Per-episode measurements (Requirement 10.1)
# ---------------------------------------------------------------------------
def test_summarize_episode_records_required_fields():
    network, batch, profile = _records(count=3)
    record = batch[0]

    metrics = summarize_episode(record, network, policy_kind=profile)

    assert metrics.episode_id == record.episode_id
    assert metrics.outcome == record.outcome
    assert metrics.steps == episode_step_count(record)
    assert metrics.simulated_s >= 0.0
    assert metrics.min_separation_m >= 0.0
    assert len(metrics.final_police_positions) == len(record.initial_state.police)
    assert len(metrics.final_fugitive_position) == 2
    assert metrics.policy_kind == profile
    # Minimum separation never exceeds the final-state separation (it is a min
    # over all states, including the terminal one).
    final_state = record.transitions[-1].state if record.transitions else record.initial_state
    fugitive_xy = placement_position(network, final_state.fugitive)
    final_min = min(
        ((px - fugitive_xy[0]) ** 2 + (py - fugitive_xy[1]) ** 2) ** 0.5
        for px, py in (placement_position(network, p) for p in final_state.police)
    )
    assert metrics.min_separation_m <= final_min + 1e-9


def test_episode_step_count_matches_terminal_step():
    _, batch, _ = _records(count=2)
    for record in batch:
        expected = record.transitions[-1].step if record.transitions else 0
        assert episode_step_count(record) == expected


# ---------------------------------------------------------------------------
# Latency: perf_counter_ns timing, warmups, device metadata (10.5, 10.6, 10.9)
# ---------------------------------------------------------------------------
def test_measure_inference_latency_excludes_warmups_and_records_metadata():
    calls = {"count": 0}

    def inference():
        calls["count"] += 1
        # Trivial work standing in for six-action generation.
        return sum(range(64))

    summary = measure_inference_latency(inference, sample_count=5, warmup=2, device="cpu")

    # Warmups + measured samples were both invoked, but only samples are counted.
    assert calls["count"] == 7
    assert summary["status"] == LATENCY_MEASURED
    assert summary["count"] == 5
    assert len(summary["samples_ms"]) == 5
    assert summary["warmup_count"] == 2
    assert summary["device"] == "cpu"
    assert "torch_version" in summary
    # Quantiles are ordered and nonnegative.
    assert summary["mean_ms"] >= 0.0
    assert summary["median_ms"] >= 0.0
    assert summary["max_ms"] >= summary["p95_ms"] >= 0.0


def test_measure_inference_latency_rejects_bad_counts():
    with pytest.raises(DomainValidationError) as bad_samples:
        measure_inference_latency(lambda: None, sample_count=0)
    assert bad_samples.value.code == "INVALID_SAMPLE_COUNT"

    with pytest.raises(DomainValidationError) as bad_warmup:
        measure_inference_latency(lambda: None, sample_count=1, warmup=-1)
    assert bad_warmup.value.code == "INVALID_WARMUP_COUNT"

    # Booleans must not masquerade as ints.
    with pytest.raises(DomainValidationError):
        measure_inference_latency(lambda: None, sample_count=True)


def test_not_measured_latency_stores_minus_one_and_status():
    summary = not_measured_latency(device="cpu", warmup_count=3)
    assert summary["status"] == LATENCY_NOT_MEASURED
    assert summary["count"] == 0
    for key in ("mean_ms", "median_ms", "p95_ms", "max_ms"):
        assert summary[key] == NOT_MEASURED_VALUE_MS
    assert summary["warmup_count"] == 3


def test_summarize_latency_ms_empty_falls_back_to_not_measured():
    summary = summarize_latency_ms([])
    assert summary["status"] == LATENCY_NOT_MEASURED
    assert summary["mean_ms"] == NOT_MEASURED_VALUE_MS


def test_summarize_latency_samples_excludes_not_measured_rows():
    samples = (
        LatencySample(value_ms=2.0, status=LATENCY_MEASURED, device="cpu", warmup=False),
        LatencySample(value_ms=4.0, status=LATENCY_MEASURED, device="cpu", warmup=False),
        LatencySample(value_ms=NOT_MEASURED_VALUE_MS, status=LATENCY_NOT_MEASURED, device="cpu", warmup=False),
    )
    summary = summarize_latency_samples(samples)

    assert summary["status"] == LATENCY_MEASURED
    # The not_measured row is dropped from the quantile computation.
    assert summary["count"] == 2
    assert summary["mean_ms"] == pytest.approx(3.0)
    assert summary["device"] == "cpu"


def test_summarize_latency_samples_all_not_measured_is_not_measured():
    samples = (
        LatencySample(value_ms=NOT_MEASURED_VALUE_MS, status=LATENCY_NOT_MEASURED, device="cpu", warmup=False),
    )
    summary = summarize_latency_samples(samples)
    assert summary["status"] == LATENCY_NOT_MEASURED
    assert summary["count"] == 0


# ---------------------------------------------------------------------------
# Batch summary (Requirements 10.1-10.4, 10.7, 10.8, 10.9)
# ---------------------------------------------------------------------------
def test_summarize_metrics_produces_conserved_summary():
    network, batch, profile = _records(count=6)

    summary = summarize_metrics(
        batch,
        network,
        policy_kind=profile,
        minimum_evaluation_episodes=3,
        confidence_interval_method="wilson",
    )

    assert isinstance(summary, MetricsSummary)
    # Count conservation: outcome counts sum to episode count.
    assert sum(summary.outcomes.values()) == len(batch)
    assert len(summary.episode_lengths) == len(batch)
    assert set(summary.rates) == {"capture", "escape", "timeout"}
    assert sum(summary.rates.values()) == pytest.approx(1.0)
    # A configured confidence interval is present per outcome and bounded.
    for low, high in summary.confidence_intervals.values():
        assert 0.0 <= low <= high <= 1.0
    # Six episodes meet the minimum, so the summary is not preliminary.
    assert summary.preliminary is False
    # No policy latency supplied -> not_measured (Requirement 10.9).
    assert summary.latency["status"] == LATENCY_NOT_MEASURED
    assert summary.latency["confidence_interval_method"] == "wilson"


def test_summarize_metrics_flags_preliminary_below_minimum():
    network, batch, profile = _records(count=2)
    summary = summarize_metrics(
        batch, network, policy_kind=profile, minimum_evaluation_episodes=10
    )
    assert summary.preliminary is True


def test_summarize_metrics_carries_measured_latency():
    network, batch, profile = _records(count=4)
    latency = measure_inference_latency(lambda: sum(range(16)), sample_count=3, warmup=1, device="cpu")

    summary = summarize_metrics(
        batch,
        network,
        policy_kind=profile,
        minimum_evaluation_episodes=1,
        latency=latency,
    )
    assert summary.latency["status"] == LATENCY_MEASURED
    assert summary.latency["count"] == 3


def test_summarize_metrics_rejects_nonpositive_minimum():
    network, batch, profile = _records(count=2)
    with pytest.raises(DomainValidationError) as excinfo:
        summarize_metrics(batch, network, policy_kind=profile, minimum_evaluation_episodes=0)
    assert excinfo.value.code == "INVALID_MINIMUM_EPISODES"
