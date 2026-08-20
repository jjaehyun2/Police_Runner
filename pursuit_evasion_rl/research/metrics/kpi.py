"""The three-axis (기술/안전/운영 -- tech/safety/ops) KPI table for the mentoring
feedback item T6.

The deck needs one small table, not a new metric philosophy: this module is a
computation-only aggregator over quantities already produced elsewhere
(:mod:`osm_demo.metrics`, :mod:`research.metrics.behavior`,
:mod:`research.variants.safety`).  It performs no I/O and no simulation of its
own -- every input is a value or sequence the caller already measured.

Axis membership:

* **tech** -- capture rate, mean time-to-capture (captured episodes only),
  containment completion rate, and mean inference latency;
* **safety** -- mean risk-class-road traversal per episode, and the final
  risk-zone fraction of the fugitive's reachable region (a simulation proxy,
  see :mod:`research.variants.safety` and ``research.paper.scope`` -- never a
  measured pedestrian-safety outcome);
* **ops** -- deliberately metric-free.  HITL (human-in-the-loop) acceptance is
  an operational criterion decided at deployment time by people evaluating the
  live system; it cannot be produced or measured inside this simulation, so
  the ops axis carries an explanatory note instead of a fabricated number.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
import statistics
from typing import Any, Mapping, Sequence

from pursuit_evasion_rl.osm_demo.metrics import EpisodeOutcome, aggregate_outcomes
from pursuit_evasion_rl.research.errors import ResearchValidationError
from pursuit_evasion_rl.research.metrics.behavior import containment_angles

KPI_SCHEMA_VERSION = "1.0"

DEFAULT_CONTAINMENT_THRESHOLD = 0.75

OPS_NOTE = (
    "HITL (human-in-the-loop) acceptance is a deployment-time operational "
    "criterion decided by operators evaluating the live system; it has no "
    "simulation-measurable value, so this axis intentionally carries no metric."
)


def _finite_fraction(value: float, name: str) -> float:
    numeric = float(value)
    if not math.isfinite(numeric) or not 0.0 <= numeric <= 1.0:
        raise ResearchValidationError(
            "INVALID_KPI_VALUE", f"{name} must be a finite value in [0, 1]", path=name, actual=value
        )
    return numeric


def _finite_nonnegative(value: float, name: str) -> float:
    numeric = float(value)
    if not math.isfinite(numeric) or numeric < 0.0:
        raise ResearchValidationError(
            "INVALID_KPI_VALUE", f"{name} must be finite and nonnegative", path=name, actual=value
        )
    return numeric


def _optional_finite_nonnegative(value: float | None, name: str) -> float | None:
    if value is None:
        return None
    return _finite_nonnegative(value, name)


@dataclass(frozen=True, slots=True)
class TechKPI:
    """기술 (technical) axis: does the policy find and contain the fugitive."""

    capture_rate: float
    mean_time_to_capture_s: float | None
    containment_completion_rate: float
    mean_inference_latency_ms: float | None
    schema_version: str = KPI_SCHEMA_VERSION

    def __post_init__(self) -> None:
        object.__setattr__(self, "capture_rate", _finite_fraction(self.capture_rate, "capture_rate"))
        object.__setattr__(
            self,
            "containment_completion_rate",
            _finite_fraction(self.containment_completion_rate, "containment_completion_rate"),
        )
        object.__setattr__(
            self,
            "mean_time_to_capture_s",
            _optional_finite_nonnegative(self.mean_time_to_capture_s, "mean_time_to_capture_s"),
        )
        object.__setattr__(
            self,
            "mean_inference_latency_ms",
            _optional_finite_nonnegative(self.mean_inference_latency_ms, "mean_inference_latency_ms"),
        )


@dataclass(frozen=True, slots=True)
class SafetyKPI:
    """안전 (safety) axis: simulation proxies only -- see module docstring."""

    risk_class_traversal_m_per_episode: float
    risk_zone_fraction_final: float | None
    schema_version: str = KPI_SCHEMA_VERSION

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "risk_class_traversal_m_per_episode",
            _finite_nonnegative(self.risk_class_traversal_m_per_episode, "risk_class_traversal_m_per_episode"),
        )
        if self.risk_zone_fraction_final is not None:
            object.__setattr__(
                self,
                "risk_zone_fraction_final",
                _finite_fraction(self.risk_zone_fraction_final, "risk_zone_fraction_final"),
            )


@dataclass(frozen=True, slots=True)
class OpsKPI:
    """운영 (operations) axis: intentionally metric-free, see :data:`OPS_NOTE`."""

    notes: str = OPS_NOTE
    schema_version: str = KPI_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if not isinstance(self.notes, str) or not self.notes.strip():
            raise ResearchValidationError("MISSING_REQUIRED_FIELD", "notes must be a non-empty statement", path="notes")


@dataclass(frozen=True, slots=True)
class KPIReport:
    """The complete 기술/안전/운영 KPI table for one evaluation batch."""

    tech: TechKPI
    safety: SafetyKPI
    ops: OpsKPI
    n_episodes: int
    schema_version: str = KPI_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if not isinstance(self.tech, TechKPI):
            raise ResearchValidationError("INVALID_KPI_AXIS", "tech must be a TechKPI", path="tech")
        if not isinstance(self.safety, SafetyKPI):
            raise ResearchValidationError("INVALID_KPI_AXIS", "safety must be a SafetyKPI", path="safety")
        if not isinstance(self.ops, OpsKPI):
            raise ResearchValidationError("INVALID_KPI_AXIS", "ops must be an OpsKPI", path="ops")
        if isinstance(self.n_episodes, bool) or not isinstance(self.n_episodes, int) or self.n_episodes < 0:
            raise ResearchValidationError(
                "INVALID_KPI_VALUE", "n_episodes must be a nonnegative integer", path="n_episodes", actual=self.n_episodes
            )


@dataclass(frozen=True, slots=True)
class ContainmentCompletion:
    """Per-episode containment-completion verdicts plus the aggregate rate."""

    per_episode: tuple[bool, ...]
    rate: float
    threshold: float


def compute_containment_completion(
    episode_position_traces: Sequence[bool | Sequence[tuple[Sequence[Any], Any]]],
    threshold: float = DEFAULT_CONTAINMENT_THRESHOLD,
) -> ContainmentCompletion:
    """기술 axis containment completion: did the team ever encircle tightly enough.

    Definition: episode ``i`` counts as containment-complete when
    ``containment_angles(police_positions, fugitive_position).angular_coverage``
    (:func:`research.metrics.behavior.containment_angles`) reaches at least
    ``threshold`` at ANY recorded step before the episode terminates -- a
    transient tight encirclement counts even if the formation loosens again
    later, since the KPI asks whether containment was ever achieved, not
    whether it was held.

    Each element of ``episode_position_traces`` is either:

    * a precomputed ``bool`` (the caller already decided completion), or
    * a sequence of ``(police_positions, fugitive_position)`` step states for
      that episode, one pair per recorded step, which this function replays
      through ``containment_angles`` itself.

    Mixing both forms across episodes in the same call is fine; each episode
    is classified independently.
    """
    if not math.isfinite(threshold) or not 0.0 <= threshold <= 1.0:
        raise ResearchValidationError(
            "INVALID_KPI_VALUE", "threshold must be a finite value in [0, 1]", path="threshold", actual=threshold
        )
    completions: list[bool] = []
    for trace in episode_position_traces:
        if isinstance(trace, bool):
            completions.append(trace)
            continue
        reached = False
        for police_positions, fugitive_position in trace:
            coverage = containment_angles(police_positions, fugitive_position).angular_coverage
            if coverage >= threshold:
                reached = True
                break
        completions.append(reached)
    per_episode = tuple(completions)
    rate = (sum(per_episode) / len(per_episode)) if per_episode else 0.0
    return ContainmentCompletion(per_episode=per_episode, rate=rate, threshold=float(threshold))


def _mean(values: Sequence[float], *, name: str) -> float:
    if not values:
        raise ResearchValidationError("MISSING_REQUIRED_FIELD", f"{name} must not be empty", path=name)
    return float(statistics.fmean(values))


def build_kpi_report(
    *,
    outcomes: Sequence[EpisodeOutcome],
    step_counts: Sequence[int],
    dt_s: float,
    risk_traversal_m_per_episode: Sequence[float],
    latency_summary: Mapping[str, Any] | None = None,
    containment_completions: Sequence[bool] | None = None,
    containment_traces: Sequence[Sequence[tuple[Sequence[Any], Any]]] | None = None,
    containment_threshold: float = DEFAULT_CONTAINMENT_THRESHOLD,
    risk_zone_fraction_final: Sequence[float] | None = None,
) -> KPIReport:
    """Assemble the tech/safety/ops KPI table from already-measured per-episode data.

    Exactly one of ``containment_completions`` (precomputed per-episode
    booleans) or ``containment_traces`` (raw per-step position sequences, fed
    through :func:`compute_containment_completion`) must be supplied -- never
    both, never neither, so containment is never silently defaulted.
    ``risk_traversal_m_per_episode`` is required (safety is a mandatory axis);
    ``risk_zone_fraction_final`` and ``latency_summary`` are optional and
    become ``None`` fields rather than fabricated zeros when absent.
    """
    n_episodes = len(outcomes)
    if n_episodes == 0:
        raise ResearchValidationError("MISSING_REQUIRED_FIELD", "outcomes must not be empty", path="outcomes")
    if len(step_counts) != n_episodes:
        raise ResearchValidationError(
            "KPI_LENGTH_MISMATCH",
            "step_counts must carry exactly one entry per episode",
            expected=n_episodes,
            actual=len(step_counts),
            path="step_counts",
        )
    if len(risk_traversal_m_per_episode) != n_episodes:
        raise ResearchValidationError(
            "KPI_LENGTH_MISMATCH",
            "risk_traversal_m_per_episode must carry exactly one entry per episode",
            expected=n_episodes,
            actual=len(risk_traversal_m_per_episode),
            path="risk_traversal_m_per_episode",
        )
    if risk_zone_fraction_final is not None and len(risk_zone_fraction_final) != n_episodes:
        raise ResearchValidationError(
            "KPI_LENGTH_MISMATCH",
            "risk_zone_fraction_final must carry exactly one entry per episode",
            expected=n_episodes,
            actual=len(risk_zone_fraction_final),
            path="risk_zone_fraction_final",
        )
    dt_s = _finite_nonnegative(dt_s, "dt_s")

    supplied_containment = [
        name
        for name, value in (("containment_completions", containment_completions), ("containment_traces", containment_traces))
        if value is not None
    ]
    if len(supplied_containment) != 1:
        raise ResearchValidationError(
            "CONTAINMENT_INPUT_AMBIGUOUS",
            "exactly one of containment_completions or containment_traces must be supplied",
            expected=1,
            actual=supplied_containment,
        )

    aggregate = aggregate_outcomes(list(outcomes))
    capture_rate = aggregate.rates[EpisodeOutcome.CAPTURE.value]

    captured_seconds = [
        step_counts[index] * dt_s
        for index in range(n_episodes)
        if outcomes[index] == EpisodeOutcome.CAPTURE
    ]
    mean_time_to_capture_s = _mean(captured_seconds, name="captured episode durations") if captured_seconds else None

    if containment_completions is not None:
        if len(containment_completions) != n_episodes:
            raise ResearchValidationError(
                "KPI_LENGTH_MISMATCH",
                "containment_completions must carry exactly one entry per episode",
                expected=n_episodes,
                actual=len(containment_completions),
                path="containment_completions",
            )
        containment_completion_rate = (
            sum(1 for value in containment_completions if value) / n_episodes
        )
    else:
        if len(containment_traces) != n_episodes:
            raise ResearchValidationError(
                "KPI_LENGTH_MISMATCH",
                "containment_traces must carry exactly one entry per episode",
                expected=n_episodes,
                actual=len(containment_traces),
                path="containment_traces",
            )
        containment_completion_rate = compute_containment_completion(
            containment_traces, threshold=containment_threshold
        ).rate

    mean_inference_latency_ms = None
    if latency_summary is not None and latency_summary.get("mean_ms") is not None:
        mean_inference_latency_ms = float(latency_summary["mean_ms"])

    tech = TechKPI(
        capture_rate=capture_rate,
        mean_time_to_capture_s=mean_time_to_capture_s,
        containment_completion_rate=containment_completion_rate,
        mean_inference_latency_ms=mean_inference_latency_ms,
    )
    safety = SafetyKPI(
        risk_class_traversal_m_per_episode=_mean(
            list(risk_traversal_m_per_episode), name="risk_traversal_m_per_episode"
        ),
        risk_zone_fraction_final=(
            _mean(list(risk_zone_fraction_final), name="risk_zone_fraction_final")
            if risk_zone_fraction_final is not None
            else None
        ),
    )
    return KPIReport(tech=tech, safety=safety, ops=OpsKPI(), n_episodes=n_episodes)


__all__ = (
    "DEFAULT_CONTAINMENT_THRESHOLD",
    "KPI_SCHEMA_VERSION",
    "OPS_NOTE",
    "ContainmentCompletion",
    "KPIReport",
    "OpsKPI",
    "SafetyKPI",
    "TechKPI",
    "build_kpi_report",
    "compute_containment_completion",
)
