"""Pre-registered sample-size and Resource_Ceiling decisions (Requirement 7).

The sample-size decision is a pure function of a measured :class:`ResourceCeiling`
and per-condition cost estimates -- never of any observed result -- so it can be
made once, before any Condition is executed, and then applied uniformly to
every confirmatory Condition (Requirement 7.3-7.4).
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import math
from typing import Mapping, Sequence

from pursuit_evasion_rl.research.canonical import content_hash
from pursuit_evasion_rl.research.errors import ResearchValidationError

BUDGET_SCHEMA_VERSION = "1.0"

DEFAULT_SEEDS = 5
DEFAULT_EPISODES = 500
REDUCED_MIN_SEEDS = 3
REDUCED_MIN_EPISODES = 100
# Candidate seed/episode counts the reduced-power search steps through, largest
# first, so the chosen plan is the largest one the ceiling actually affords.
_SEED_CANDIDATES = tuple(range(DEFAULT_SEEDS, REDUCED_MIN_SEEDS - 1, -1))
_EPISODE_CANDIDATES = tuple(range(DEFAULT_EPISODES, REDUCED_MIN_EPISODES - 1, -100))


def _fail(code: str, message: str, **kwargs) -> None:
    raise ResearchValidationError(code, message, **kwargs)


def _positive_finite(value: float, name: str) -> float:
    numeric = float(value)
    if not math.isfinite(numeric) or numeric <= 0.0:
        _fail("INVALID_RESOURCE_VALUE", f"{name} must be positive and finite", path=name, actual=value)
    return numeric


def _required_text(value: str, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        _fail("MISSING_REQUIRED_FIELD", f"{name} must be non-empty", path=name)
    return value


class SampleSizeStatus(str, Enum):
    DEFAULT = "default"
    REDUCED = "reduced"
    EXPLORATORY = "exploratory"


@dataclass(frozen=True, slots=True)
class ResourceCeiling:
    """A measured resource ceiling, fixed before any Condition result is seen."""

    resource_ceiling_id: str
    measured_at_utc: str
    accelerator_hours: float
    wall_clock_hours: float
    schema_version: str = BUDGET_SCHEMA_VERSION

    def __post_init__(self) -> None:
        _required_text(self.resource_ceiling_id, "resource_ceiling_id")
        _required_text(self.measured_at_utc, "measured_at_utc")
        object.__setattr__(self, "accelerator_hours", _positive_finite(self.accelerator_hours, "accelerator_hours"))
        object.__setattr__(self, "wall_clock_hours", _positive_finite(self.wall_clock_hours, "wall_clock_hours"))

    @property
    def config_hash(self) -> str:
        return content_hash(self)


@dataclass(frozen=True, slots=True)
class ConditionResourceEstimate:
    """Per-Training_Seed cost estimate for one Condition, used only for planning."""

    condition_id: str
    accelerator_hours_per_seed: float
    wall_clock_hours_per_seed: float
    env_steps_per_seed: int
    schema_version: str = BUDGET_SCHEMA_VERSION

    def __post_init__(self) -> None:
        _required_text(self.condition_id, "condition_id")
        object.__setattr__(
            self, "accelerator_hours_per_seed", _positive_finite(self.accelerator_hours_per_seed, "accelerator_hours_per_seed")
        )
        object.__setattr__(
            self, "wall_clock_hours_per_seed", _positive_finite(self.wall_clock_hours_per_seed, "wall_clock_hours_per_seed")
        )
        steps = self.env_steps_per_seed
        if isinstance(steps, bool) or not isinstance(steps, int) or steps <= 0:
            _fail("INVALID_RESOURCE_VALUE", "env_steps_per_seed must be a positive integer", path="env_steps_per_seed", actual=steps)

    @property
    def config_hash(self) -> str:
        return content_hash(self)


@dataclass(frozen=True, slots=True)
class SampleSizePlan:
    """The pre-registered seeds/episodes decision applied to every confirmatory Condition."""

    status: SampleSizeStatus
    seeds_per_condition: int
    episodes_per_condition: int
    resource_ceiling_id: str
    reduction_inputs: Mapping[str, object]
    power_limitation: str | None = None
    schema_version: str = BUDGET_SCHEMA_VERSION

    def __post_init__(self) -> None:
        object.__setattr__(self, "status", SampleSizeStatus(self.status))
        _required_text(self.resource_ceiling_id, "resource_ceiling_id")
        seeds, episodes = self.seeds_per_condition, self.episodes_per_condition
        if isinstance(seeds, bool) or not isinstance(seeds, int) or isinstance(episodes, bool) or not isinstance(episodes, int):
            _fail("INVALID_SAMPLE_SIZE", "seeds_per_condition and episodes_per_condition must be integers")
        if self.status is SampleSizeStatus.DEFAULT:
            if (seeds, episodes) != (DEFAULT_SEEDS, DEFAULT_EPISODES):
                _fail(
                    "INVALID_SAMPLE_SIZE", "the default plan must use the pre-registered 5 seeds / 500 episodes",
                    expected=(DEFAULT_SEEDS, DEFAULT_EPISODES), actual=(seeds, episodes),
                )
        elif self.status is SampleSizeStatus.REDUCED:
            if not (REDUCED_MIN_SEEDS <= seeds <= DEFAULT_SEEDS and REDUCED_MIN_EPISODES <= episodes <= DEFAULT_EPISODES):
                _fail(
                    "INVALID_SAMPLE_SIZE",
                    "a reduced plan must keep seeds in [3, 5] and episodes in [100, 500]",
                    expected=f"seeds in [{REDUCED_MIN_SEEDS},{DEFAULT_SEEDS}], episodes in [{REDUCED_MIN_EPISODES},{DEFAULT_EPISODES}]",
                    actual=(seeds, episodes),
                )
            if (seeds, episodes) == (DEFAULT_SEEDS, DEFAULT_EPISODES):
                _fail("INVALID_SAMPLE_SIZE", "a reduced plan must actually reduce seeds or episodes below the default")
            if not self.reduction_inputs:
                _fail("MISSING_REDUCTION_INPUTS", "a reduced plan must record what forced the reduction", path="reduction_inputs")
            _required_text(self.power_limitation or "", "power_limitation")
        else:  # EXPLORATORY
            if seeds >= REDUCED_MIN_SEEDS and episodes >= REDUCED_MIN_EPISODES:
                _fail(
                    "INVALID_SAMPLE_SIZE",
                    "exploratory status requires fewer than 3 seeds or fewer than 100 episodes",
                    actual=(seeds, episodes),
                )
            if not self.reduction_inputs:
                _fail("MISSING_REDUCTION_INPUTS", "an exploratory plan must record what forced the reduction", path="reduction_inputs")
            _required_text(self.power_limitation or "", "power_limitation")

    @property
    def plan_hash(self) -> str:
        return content_hash(self)

    @property
    def is_confirmatory_eligible(self) -> bool:
        """Requirement 7.6: below 3 seeds / 100 episodes, the analysis is exploratory."""
        return self.status is not SampleSizeStatus.EXPLORATORY


def _total_cost(estimates: Sequence[ConditionResourceEstimate], seeds: int) -> tuple[float, float]:
    accelerator = sum(estimate.accelerator_hours_per_seed for estimate in estimates) * seeds
    wall_clock = sum(estimate.wall_clock_hours_per_seed for estimate in estimates) * seeds
    return accelerator, wall_clock


def decide_sample_size(
    ceiling: ResourceCeiling, condition_estimates: Sequence[ConditionResourceEstimate]
) -> SampleSizePlan:
    """Deterministically pick the (seeds, episodes) pair the ceiling affords.

    The search is a pure function of ``ceiling`` and ``condition_estimates`` --
    it never looks at any Condition's outcome -- and applies the same pair to
    every Condition (Requirement 7.3-7.4).  Episode count only changes
    evaluation cost, which this simple model does not separately meter, so the
    search varies seeds first (the dominant retraining cost) and then episodes,
    always preferring the largest pair that fits within both the accelerator-hour
    and wall-clock ceilings.
    """
    if not condition_estimates:
        _fail("EMPTY_CONDITION_ESTIMATES", "at least one condition resource estimate is required")

    def fits(seeds: int) -> bool:
        accelerator, wall_clock = _total_cost(condition_estimates, seeds)
        return accelerator <= ceiling.accelerator_hours and wall_clock <= ceiling.wall_clock_hours

    if fits(DEFAULT_SEEDS):
        return SampleSizePlan(
            status=SampleSizeStatus.DEFAULT,
            seeds_per_condition=DEFAULT_SEEDS,
            episodes_per_condition=DEFAULT_EPISODES,
            resource_ceiling_id=ceiling.resource_ceiling_id,
            reduction_inputs={},
            power_limitation=None,
        )

    accelerator_default, wall_default = _total_cost(condition_estimates, DEFAULT_SEEDS)
    reduction_inputs = {
        "resource_ceiling": ceiling.config_hash,
        "accelerator_hours_at_default_seeds": accelerator_default,
        "wall_clock_hours_at_default_seeds": wall_default,
        "accelerator_hours_ceiling": ceiling.accelerator_hours,
        "wall_clock_hours_ceiling": ceiling.wall_clock_hours,
    }
    # This simple cost model only meters accelerator/wall-clock hours per seed
    # (retraining cost), so episode count does not change whether a given seed
    # count fits; search seeds first, then prefer as many episodes as possible
    # within the reduced band for whichever seed count is chosen.
    for seeds in _SEED_CANDIDATES:
        if seeds == DEFAULT_SEEDS or not fits(seeds):
            continue
        episodes = REDUCED_MIN_EPISODES
        return SampleSizePlan(
            status=SampleSizeStatus.REDUCED,
            seeds_per_condition=seeds,
            episodes_per_condition=episodes,
            resource_ceiling_id=ceiling.resource_ceiling_id,
            reduction_inputs=reduction_inputs,
            power_limitation=(
                f"seeds reduced from {DEFAULT_SEEDS} to {seeds} and episodes to {episodes} "
                "under the matrix-wide Resource_Ceiling; between-seed variance estimates "
                "have correspondingly wider uncertainty than the default plan"
            ),
        )
    return SampleSizePlan(
        status=SampleSizeStatus.EXPLORATORY,
        # Below the reduced floor: report the smallest possible plan (Requirement
        # 7.6 only requires classifying this as exploratory, not a specific size).
        seeds_per_condition=1,
        episodes_per_condition=1,
        resource_ceiling_id=ceiling.resource_ceiling_id,
        reduction_inputs=reduction_inputs,
        power_limitation=(
            "the Resource_Ceiling does not afford even 3 seeds / 100 episodes for the full "
            "matrix; every condition is demoted to exploratory before any result is seen"
        ),
    )


# ---------------------------------------------------------------------------
# Requirement 7.7-7.8: repeated Checkpoint_Time measurements never inflate the
# independent Training_Seed count.
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class SeedReplicate:
    """One Training_Seed's repeated Checkpoint_Time measurements, grouped as one replicate."""

    training_seed: int
    checkpoint_times: tuple[str, ...]
    schema_version: str = BUDGET_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if isinstance(self.training_seed, bool) or not isinstance(self.training_seed, int):
            _fail("INVALID_TRAINING_SEED", "training_seed must be an integer", actual=self.training_seed)
        times = tuple(str(item) for item in self.checkpoint_times)
        if not times:
            _fail("MISSING_REQUIRED_FIELD", "checkpoint_times must not be empty", path="checkpoint_times")
        object.__setattr__(self, "checkpoint_times", times)

    @property
    def config_hash(self) -> str:
        return content_hash(self)


def group_seed_replicates(
    records: Sequence[Mapping[str, object]]
) -> tuple[SeedReplicate, ...]:
    """Group ``{"training_seed": int, "checkpoint_time": str}`` records by seed.

    Requirement 7.7 forbids counting Checkpoint_Time as a Training_Seed; grouping
    here is what :func:`count_independent_training_seeds` counts over, so
    duplicating a checkpoint time for the same seed can never inflate the
    independent-repetition count.
    """
    grouped: dict[int, list[str]] = {}
    for record in records:
        if "training_seed" not in record or "checkpoint_time" not in record:
            _fail(
                "INVALID_SEED_RECORD", "each record requires training_seed and checkpoint_time", actual=dict(record)
            )
        seed = record["training_seed"]
        if isinstance(seed, bool) or not isinstance(seed, int):
            _fail("INVALID_TRAINING_SEED", "training_seed must be an integer", actual=seed)
        grouped.setdefault(seed, []).append(str(record["checkpoint_time"]))
    return tuple(
        SeedReplicate(training_seed=seed, checkpoint_times=tuple(times))
        for seed, times in sorted(grouped.items())
    )


def count_independent_training_seeds(records: Sequence[Mapping[str, object]]) -> int:
    """The number of distinct Training_Seeds, independent of Checkpoint_Time repeats."""
    return len(group_seed_replicates(records))


__all__ = (
    "BUDGET_SCHEMA_VERSION",
    "DEFAULT_EPISODES",
    "DEFAULT_SEEDS",
    "REDUCED_MIN_EPISODES",
    "REDUCED_MIN_SEEDS",
    "ConditionResourceEstimate",
    "ResourceCeiling",
    "SampleSizePlan",
    "SampleSizeStatus",
    "SeedReplicate",
    "count_independent_training_seeds",
    "decide_sample_size",
    "group_seed_replicates",
)
