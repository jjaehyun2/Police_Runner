"""Outcome, confidence-interval and inference-latency metrics for OSM episodes.

This module aggregates deterministic :class:`~pursuit_evasion_rl.osm_demo.models.EpisodeRecord`
outputs from :mod:`pursuit_evasion_rl.osm_demo.runner` into a
:class:`~pursuit_evasion_rl.osm_demo.models.MetricsSummary` (design section 9,
"Metrics, latency and claim evidence"; Requirements 10.1-10.9).

Design guarantees implemented here:

* **Exactly one outcome per episode and count conservation** (Requirements 10.3,
  10.4).  Every :class:`EpisodeRecord` carries exactly one
  :class:`EpisodeOutcome`; :func:`aggregate_outcomes` counts each of ``capture``,
  ``escape`` and ``timeout`` so the three counts always sum to the number of
  episodes, and each rate is its count divided by that total.
* **Configured binomial confidence intervals** (Requirement 10.7).  Rates are
  proportions of a binomial process, so :func:`confidence_interval` computes a
  configurable interval; the default ``"wilson"`` score interval behaves well for
  the small-sample, extreme-rate regime typical of a demo.
* **Per-episode measurements** (Requirement 10.1).  :func:`summarize_episode`
  records the outcome, step count, simulated seconds, minimum metric
  police-fugitive separation, terminal positions, policy kind and input hashes.
* **Inference latency** (Requirements 10.5, 10.6, 10.9).  :func:`measure_inference_latency`
  times generation of all six police actions with :func:`time.perf_counter_ns`,
  runs explicit warm-up calls that are excluded from the sample, and records raw
  milliseconds, count, mean, median, p95, max, device, torch version and warm-up
  count.  Non-policy / no-inference rows store ``-1`` with status
  ``not_measured`` and are excluded from the latency quantiles.
* **Preliminary flag** (Requirement 10.8).  A summary built from fewer episodes
  than the configured minimum is flagged ``preliminary``.

The module is fully offline: it consumes already-computed records and a callable
for latency, and performs no external OSM access.
"""

from __future__ import annotations

import math
import statistics
import time
from dataclasses import dataclass
from typing import Any, Callable, Mapping, Sequence

from .models import (
    DomainValidationError,
    EpisodeOutcome,
    EpisodeRecord,
    EpisodeState,
    LatencySample,
    MetricsSummary,
    ModelNetwork,
    VehiclePlacement,
)

# 95% two-sided normal quantile used for binomial confidence intervals.
Z_95 = 1.959963984540054

# Latency measurement statuses (Requirement 10.9).
LATENCY_MEASURED = "measured"
LATENCY_NOT_MEASURED = "not_measured"
NOT_MEASURED_VALUE_MS = -1.0

InferenceCallable = Callable[[], Any]


# ---------------------------------------------------------------------------
# Geometry helpers (metric separation, Requirement 10.1)
# ---------------------------------------------------------------------------
def _geometry_length(geometry: Sequence[tuple[float, float]]) -> float:
    return sum(math.dist(geometry[i], geometry[i + 1]) for i in range(len(geometry) - 1))


def _point_at_fraction(
    geometry: Sequence[tuple[float, float]], fraction: float
) -> tuple[float, float]:
    """Interpolate a point at ``fraction`` of a polyline's arc length.

    This mirrors the environment's movement model so metric separation computed
    from a persisted record matches the geometry the episode actually traversed.
    """
    fraction = min(1.0, max(0.0, fraction))
    total = _geometry_length(geometry)
    if total <= 0.0:
        return (float(geometry[0][0]), float(geometry[0][1]))
    target = fraction * total
    walked = 0.0
    for first, second in zip(geometry, geometry[1:]):
        segment_len = math.dist(first, second)
        if segment_len <= 0.0:
            continue
        if walked + segment_len >= target:
            local = (target - walked) / segment_len
            return (
                first[0] + local * (second[0] - first[0]),
                first[1] + local * (second[1] - first[1]),
            )
        walked += segment_len
    return (float(geometry[-1][0]), float(geometry[-1][1]))


def placement_position(network: ModelNetwork, placement: VehiclePlacement) -> tuple[float, float]:
    """Return the projected metric ``(x, y)`` position of a vehicle placement."""
    if placement.segment_id is not None:
        segment = next((item for item in network.segments if item.id == placement.segment_id), None)
        if segment is None:
            raise DomainValidationError(
                "INVALID_VEHICLE_POSITION",
                f"Placement references unknown segment {placement.segment_id}",
                actual=placement.segment_id,
            )
        return _point_at_fraction(segment.geometry_xy, placement.progress)
    intersection = next(
        (item for item in network.intersections if item.id == placement.intersection_id), None
    )
    if intersection is None:
        raise DomainValidationError(
            "INVALID_VEHICLE_POSITION",
            f"Placement references unknown intersection {placement.intersection_id}",
            actual=placement.intersection_id,
        )
    return intersection.position_xy


def _state_min_separation(network: ModelNetwork, state: EpisodeState) -> float:
    """Smallest metric police-fugitive distance in a single episode state."""
    fugitive_xy = placement_position(network, state.fugitive)
    return min(
        math.dist(placement_position(network, police), fugitive_xy) for police in state.police
    )


# ---------------------------------------------------------------------------
# Confidence intervals (Requirement 10.7)
# ---------------------------------------------------------------------------
def wilson_interval(successes: int, total: int, *, z: float = Z_95) -> tuple[float, float]:
    """Wilson score interval for a binomial proportion, clamped to ``[0, 1]``."""
    if total <= 0:
        return (0.0, 0.0)
    phat = successes / total
    z2 = z * z
    denom = 1.0 + z2 / total
    center = (phat + z2 / (2.0 * total)) / denom
    margin = (z * math.sqrt(phat * (1.0 - phat) / total + z2 / (4.0 * total * total))) / denom
    return (max(0.0, center - margin), min(1.0, center + margin))


def wald_interval(successes: int, total: int, *, z: float = Z_95) -> tuple[float, float]:
    """Normal-approximation (Wald) interval for a binomial proportion."""
    if total <= 0:
        return (0.0, 0.0)
    phat = successes / total
    margin = z * math.sqrt(phat * (1.0 - phat) / total)
    return (max(0.0, phat - margin), min(1.0, phat + margin))


_CI_METHODS: Mapping[str, Callable[[int, int], tuple[float, float]]] = {
    "wilson": wilson_interval,
    "wald": wald_interval,
    "normal": wald_interval,
}


def confidence_interval(
    successes: int, total: int, *, method: str = "wilson"
) -> tuple[float, float]:
    """Compute a configured binomial confidence interval for ``successes/total``."""
    estimator = _CI_METHODS.get(method)
    if estimator is None:
        raise DomainValidationError(
            "UNKNOWN_CONFIDENCE_METHOD",
            f"Unsupported confidence interval method {method!r}; supported: {sorted(_CI_METHODS)}",
            expected=sorted(_CI_METHODS),
            actual=method,
        )
    if total < 0 or successes < 0 or successes > total:
        raise DomainValidationError(
            "INVALID_BINOMIAL_INPUT",
            "Confidence interval requires 0 <= successes <= total",
            actual=(successes, total),
        )
    return estimator(successes, total)


# ---------------------------------------------------------------------------
# Outcome aggregation (Requirements 10.2, 10.3, 10.4)
# ---------------------------------------------------------------------------
@dataclass(frozen=True, slots=True)
class OutcomeAggregate:
    """Conserved outcome counts and rates for a finite episode sequence."""

    counts: Mapping[str, int]
    rates: Mapping[str, float]
    total: int


def aggregate_outcomes(outcomes: Sequence[EpisodeOutcome]) -> OutcomeAggregate:
    """Count each outcome, conserving membership (Requirements 10.2-10.4).

    Assigns exactly one outcome per item, so the three counts sum to the
    sequence length and each rate is ``count / total`` when the sequence is
    nonempty (``0.0`` for an empty sequence).
    """
    counts = {item.value: 0 for item in EpisodeOutcome}
    for outcome in outcomes:
        if not isinstance(outcome, EpisodeOutcome):
            raise DomainValidationError(
                "INVALID_OUTCOME",
                "Every episode must carry exactly one EpisodeOutcome",
                actual=outcome,
            )
        counts[outcome.value] += 1
    total = len(outcomes)
    if sum(counts.values()) != total:
        # Defensive: the loop above guarantees this, but conservation is a
        # first-class invariant so we assert it explicitly.
        raise DomainValidationError(
            "OUTCOME_COUNT_MISMATCH",
            "Outcome counts must sum to the number of episodes",
            expected=total,
            actual=sum(counts.values()),
        )
    rates = {key: (value / total if total else 0.0) for key, value in counts.items()}
    return OutcomeAggregate(counts=counts, rates=rates, total=total)


# ---------------------------------------------------------------------------
# Per-episode metrics (Requirement 10.1)
# ---------------------------------------------------------------------------
@dataclass(frozen=True, slots=True)
class EpisodeMetrics:
    """Per-episode record fields required by Requirement 10.1."""

    episode_id: str
    outcome: EpisodeOutcome
    steps: int
    simulated_s: float
    min_separation_m: float
    final_police_positions: tuple[tuple[float, float], ...]
    final_fugitive_position: tuple[float, float]
    policy_kind: str
    hashes: Mapping[str, str]


def _episode_states(record: EpisodeRecord) -> tuple[EpisodeState, ...]:
    return (record.initial_state,) + tuple(transition.state for transition in record.transitions)


def episode_step_count(record: EpisodeRecord) -> int:
    """Number of simulated steps: the terminal step, or 0 if none were taken."""
    if not record.transitions:
        return 0
    return int(record.transitions[-1].step)


def summarize_episode(
    record: EpisodeRecord, network: ModelNetwork, *, policy_kind: str
) -> EpisodeMetrics:
    """Extract per-episode outcome, distance and terminal-position metrics."""
    states = _episode_states(record)
    final_state = states[-1]
    min_separation = min(_state_min_separation(network, state) for state in states)
    return EpisodeMetrics(
        episode_id=record.episode_id,
        outcome=record.outcome,
        steps=episode_step_count(record),
        simulated_s=float(final_state.simulated_s),
        min_separation_m=float(min_separation),
        final_police_positions=tuple(
            placement_position(network, police) for police in final_state.police
        ),
        final_fugitive_position=placement_position(network, final_state.fugitive),
        policy_kind=policy_kind,
        hashes=dict(record.hashes),
    )


# ---------------------------------------------------------------------------
# Latency measurement and summaries (Requirements 10.5, 10.6, 10.9)
# ---------------------------------------------------------------------------
def _torch_metadata(device: str | None) -> tuple[str, str]:
    """Return ``(device, torch_version)`` provenance, resilient to a missing torch."""
    try:  # pragma: no cover - torch is a declared dependency, guard is defensive
        import torch

        torch_version = str(torch.__version__)
        resolved_device = device if device is not None else "cpu"
    except Exception:  # pragma: no cover
        torch_version = "unavailable"
        resolved_device = device if device is not None else "cpu"
    return resolved_device, torch_version


def _percentile(sorted_values: Sequence[float], fraction: float) -> float:
    """Linear-interpolation percentile (numpy-compatible ``linear`` method)."""
    if not sorted_values:
        raise DomainValidationError("EMPTY_LATENCY", "Cannot compute a percentile of no samples")
    if len(sorted_values) == 1:
        return float(sorted_values[0])
    rank = fraction * (len(sorted_values) - 1)
    low = math.floor(rank)
    high = math.ceil(rank)
    if low == high:
        return float(sorted_values[low])
    weight = rank - low
    return float(sorted_values[low] * (1.0 - weight) + sorted_values[high] * weight)


def not_measured_latency(*, device: str | None = None, warmup_count: int = 0) -> dict[str, Any]:
    """Latency summary for non-policy / no-inference runs (Requirement 10.9)."""
    resolved_device, torch_version = _torch_metadata(device)
    return {
        "status": LATENCY_NOT_MEASURED,
        "samples_ms": (),
        "count": 0,
        "mean_ms": NOT_MEASURED_VALUE_MS,
        "median_ms": NOT_MEASURED_VALUE_MS,
        "p95_ms": NOT_MEASURED_VALUE_MS,
        "max_ms": NOT_MEASURED_VALUE_MS,
        "device": resolved_device,
        "torch_version": torch_version,
        "warmup_count": int(warmup_count),
    }


def summarize_latency_ms(
    samples_ms: Sequence[float], *, device: str | None = None, warmup_count: int = 0
) -> dict[str, Any]:
    """Summarize measured millisecond samples (Requirement 10.6).

    An empty sample set yields a ``not_measured`` summary so callers can treat
    "no inference happened" and "inference measured zero samples" uniformly.
    """
    measured = [float(value) for value in samples_ms if float(value) >= 0.0]
    if not measured:
        return not_measured_latency(device=device, warmup_count=warmup_count)
    resolved_device, torch_version = _torch_metadata(device)
    ordered = sorted(measured)
    return {
        "status": LATENCY_MEASURED,
        "samples_ms": tuple(measured),
        "count": len(measured),
        "mean_ms": statistics.fmean(measured),
        "median_ms": statistics.median(measured),
        "p95_ms": _percentile(ordered, 0.95),
        "max_ms": ordered[-1],
        "device": resolved_device,
        "torch_version": torch_version,
        "warmup_count": int(warmup_count),
    }


def summarize_latency_samples(
    samples: Sequence[LatencySample], *, device: str | None = None, warmup_count: int = 0
) -> dict[str, Any]:
    """Summarize :class:`LatencySample` rows, excluding ``not_measured`` rows.

    ``-1`` / ``not_measured`` rows are dropped from the quantile computation so a
    mix of policy and non-policy rows never contaminates the measured statistics
    (Requirement 10.9).
    """
    measured = [
        sample.value_ms
        for sample in samples
        if sample.status == LATENCY_MEASURED and sample.value_ms >= 0.0
    ]
    resolved_device = device
    if resolved_device is None:
        measured_devices = {sample.device for sample in samples if sample.status == LATENCY_MEASURED}
        if len(measured_devices) == 1:
            resolved_device = next(iter(measured_devices))
    if warmup_count == 0:
        warmup_count = sum(1 for sample in samples if sample.warmup)
    return summarize_latency_ms(measured, device=resolved_device, warmup_count=warmup_count)


def measure_inference_latency(
    inference: InferenceCallable,
    *,
    sample_count: int,
    warmup: int = 1,
    device: str | None = None,
) -> dict[str, Any]:
    """Time ``sample_count`` calls that each generate all six police actions.

    ``warmup`` explicit calls run first and are excluded from the sample; timing
    uses :func:`time.perf_counter_ns` around the callable only, excluding
    checkpoint / network setup (Requirement 10.5).  Returns the summary described
    by Requirement 10.6.
    """
    if isinstance(sample_count, bool) or not isinstance(sample_count, int) or sample_count <= 0:
        raise DomainValidationError(
            "INVALID_SAMPLE_COUNT", "sample_count must be a positive integer", actual=sample_count
        )
    if isinstance(warmup, bool) or not isinstance(warmup, int) or warmup < 0:
        raise DomainValidationError(
            "INVALID_WARMUP_COUNT", "warmup must be a nonnegative integer", actual=warmup
        )

    for _ in range(warmup):
        inference()

    samples_ms: list[float] = []
    for _ in range(sample_count):
        start = time.perf_counter_ns()
        inference()
        elapsed_ns = time.perf_counter_ns() - start
        samples_ms.append(elapsed_ns / 1_000_000.0)

    return summarize_latency_ms(samples_ms, device=device, warmup_count=warmup)


# ---------------------------------------------------------------------------
# Batch summary (Requirements 10.1-10.4, 10.7, 10.8)
# ---------------------------------------------------------------------------
def summarize_metrics(
    records: Sequence[EpisodeRecord],
    network: ModelNetwork,
    *,
    policy_kind: str,
    minimum_evaluation_episodes: int,
    confidence_interval_method: str = "wilson",
    latency: Mapping[str, Any] | None = None,
) -> MetricsSummary:
    """Aggregate episode records into a :class:`MetricsSummary`.

    Enforces one outcome per episode and count conservation, computes rates and
    configured binomial confidence intervals, records the episode-length
    distribution and flags the summary ``preliminary`` when fewer than
    ``minimum_evaluation_episodes`` episodes are present (Requirements 10.1-10.4,
    10.7, 10.8).  ``latency`` defaults to a ``not_measured`` summary for runs
    without policy inference (Requirement 10.9).
    """
    if isinstance(minimum_evaluation_episodes, bool) or not isinstance(
        minimum_evaluation_episodes, int
    ) or minimum_evaluation_episodes <= 0:
        raise DomainValidationError(
            "INVALID_MINIMUM_EPISODES",
            "minimum_evaluation_episodes must be a positive integer",
            actual=minimum_evaluation_episodes,
        )

    per_episode = tuple(
        summarize_episode(record, network, policy_kind=policy_kind) for record in records
    )
    aggregate = aggregate_outcomes(tuple(item.outcome for item in per_episode))
    confidence_intervals = {
        outcome: confidence_interval(
            aggregate.counts[outcome], aggregate.total, method=confidence_interval_method
        )
        for outcome in aggregate.counts
    }
    episode_lengths = tuple(item.steps for item in per_episode)
    latency_summary = (
        dict(latency) if latency is not None else not_measured_latency()
    )
    latency_summary.setdefault("confidence_interval_method", confidence_interval_method)

    return MetricsSummary(
        outcomes=aggregate.counts,
        rates=aggregate.rates,
        confidence_intervals=confidence_intervals,
        episode_lengths=episode_lengths,
        latency=latency_summary,
        preliminary=aggregate.total < minimum_evaluation_episodes,
    )


__all__ = (
    "Z_95",
    "LATENCY_MEASURED",
    "LATENCY_NOT_MEASURED",
    "NOT_MEASURED_VALUE_MS",
    "EpisodeMetrics",
    "OutcomeAggregate",
    "aggregate_outcomes",
    "confidence_interval",
    "episode_step_count",
    "measure_inference_latency",
    "not_measured_latency",
    "placement_position",
    "summarize_episode",
    "summarize_latency_ms",
    "summarize_latency_samples",
    "summarize_metrics",
    "wald_interval",
    "wilson_interval",
)
