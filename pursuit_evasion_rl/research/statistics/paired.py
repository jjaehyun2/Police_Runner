"""Hierarchical paired statistics for confirmatory policy comparisons.

Scope: Requirements 7.1-7.8 (independent-repetition accounting), 8.2-8.7 (paired
comparison, hierarchical bootstrap CI, two-sided tests, Holm correction,
practical-threshold gating) and 10.6-10.7 (2x2 factorial contrasts).

Everything here is implemented from elementary mathematics -- ``math``,
``fractions`` and ``numpy`` only -- because this project pins its dependency set
in ``pyproject.toml`` and SciPy is not part of it.

Formulas and their sources
--------------------------
* **Wilson score interval.**  Wilson, E. B. (1927), *Probable inference, the law
  of succession, and statistical inference*, JASA 22(158), 209-212.  For ``x``
  successes in ``n`` trials with ``p = x / n`` and normal quantile ``z``::

      centre = (p + z^2 / (2n)) / (1 + z^2 / n)
      half   = z / (1 + z^2 / n) * sqrt(p(1 - p)/n + z^2/(4 n^2))

  The interval is a subset of [0, 1] by construction, which is what keeps the
  registered domain intact for a proportion.
* **Paired 2x2 table.**  Pairs are ``(outcome_a, outcome_b)`` on the *same*
  Episode_Case and the same Training_Seed, so the table is the standard
  McNemar layout::

                       B success   B failure
      A success           n11         n10
      A failure           n01         n00

  ``n10`` are the discordant pairs favouring policy A, ``n01`` those favouring
  policy B.  The **risk difference** is the marginal contrast
  ``p_a - p_b = (n10 - n01) / n``, so a *positive* value always means policy A
  succeeded more often; its unit is a proportion of Episode_Cases.  The
  **paired odds ratio** is the conditional (McNemar) odds ratio ``n10 / n01``,
  which is the conditional maximum-likelihood estimate of the marginal odds
  ratio for a matched-pair design and is a strictly positive quantity.  When a
  discordant cell is empty the Haldane-Anscombe correction (add 0.5 to both
  discordant cells) is applied so the estimate and its CI stay finite and
  positive; the correction is always reported, never silent.
* **Exact McNemar test.**  Conditional on the number of discordant pairs
  ``m = n01 + n10``, the count ``n10`` is Binomial(m, 1/2) under the null.  The
  two-sided exact p-value is ``min(1, 2 * P(X <= min(n01, n10)))`` with
  ``X ~ Binomial(m, 1/2)``, evaluated exactly with integer binomial
  coefficients (Agresti, *Categorical Data Analysis*, 3rd ed., sec. 11.2).
* **Restricted mean.**  ``E[min(T, tau)]`` estimated by the sample mean of
  ``min(t_i, tau)``, with a never-observed event contributing the full horizon
  ``tau``.  This keeps a metric such as time-to-capture finite and comparable
  when some episodes never capture.
* **Hierarchical paired bootstrap.**  Requirement 8.3: Training_Seed is the
  outer cluster and the Episode_Case pair is the inner unit.  Each replicate
  resamples seeds with replacement and then, *within each drawn seed*,
  resamples that seed's own pairs with replacement.  Pairs are never resampled
  across seed boundaries and a pair's two arms always travel together, so the
  matched structure survives every replicate.  The interval is the empirical
  percentile interval of the replicate statistics.
* **Holm and Benjamini-Hochberg.**  Holm, S. (1979), Scand. J. Statist. 6(2),
  65-70 (step-down, family-wise error rate); Benjamini & Hochberg (1995),
  JRSS-B 57(1), 289-300 (step-up, false discovery rate).  Both are returned in
  the caller's original order.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from enum import Enum
from fractions import Fraction
import math
from typing import Callable, Mapping, Sequence

import numpy as np

from pursuit_evasion_rl.research.budget import (
    SampleSizePlan,
    count_independent_training_seeds,
)
from pursuit_evasion_rl.research.canonical import content_hash
from pursuit_evasion_rl.research.errors import ResearchValidationError

STATISTICS_SCHEMA_VERSION = "1.0"

DEFAULT_BOOTSTRAP_SAMPLES = 10_000
DEFAULT_CONFIDENCE_LEVEL = 0.95

PRIMARY_ANALYSIS_LABEL = "primary"
COMPLETE_CASE_ANALYSIS_LABEL = "complete_case_sensitivity"


def _fail(code: str, message: str, **kwargs) -> None:
    raise ResearchValidationError(code, message, **kwargs)


def _finite(value: float, name: str) -> float:
    numeric = float(value)
    if not math.isfinite(numeric):
        _fail("NON_FINITE_VALUE", f"{name} must be finite", path=name, actual=value)
    return numeric


def _required_text(value: str | None, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        _fail("MISSING_REQUIRED_FIELD", f"{name} must be non-empty", path=name)
    return value


def _count(value: int, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        _fail("INVALID_COUNT", f"{name} must be a non-negative integer", path=name, actual=value)
    return value


# ---------------------------------------------------------------------------
# Elementary distribution functions (no SciPy)
# ---------------------------------------------------------------------------


def normal_quantile(probability: float) -> float:
    """Inverse standard-normal CDF, by bisection on ``math.erf``.

    Bisection over [-40, 40] converges to double precision well inside the
    iteration budget, and depends on nothing but the C library's ``erf``.
    """
    p = float(probability)
    if not 0.0 < p < 1.0:
        _fail("INVALID_PROBABILITY", "probability must lie strictly inside (0, 1)", actual=probability)
    low, high = -40.0, 40.0
    for _ in range(200):
        middle = 0.5 * (low + high)
        if 0.5 * (1.0 + math.erf(middle / math.sqrt(2.0))) < p:
            low = middle
        else:
            high = middle
    return 0.5 * (low + high)


def binomial_cdf_half(successes: int, trials: int) -> float:
    """``P(X <= successes)`` for ``X ~ Binomial(trials, 1/2)``, computed exactly.

    The partial sum of binomial coefficients and ``2**trials`` are exact
    integers; :class:`~fractions.Fraction` performs the final correctly-rounded
    division, which stays accurate for counts far larger than any evaluation
    campaign this project plans.
    """
    n = _count(trials, "trials")
    k = _count(successes, "successes")
    if k >= n:
        return 1.0
    numerator = sum(math.comb(n, index) for index in range(k + 1))
    return float(Fraction(numerator, 1 << n))


# ---------------------------------------------------------------------------
# Intervals
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class ConfidenceInterval:
    """A two-sided interval estimate with its construction method recorded."""

    lower: float
    upper: float
    confidence_level: float
    method: str
    schema_version: str = STATISTICS_SCHEMA_VERSION

    def __post_init__(self) -> None:
        object.__setattr__(self, "lower", _finite(self.lower, "lower"))
        object.__setattr__(self, "upper", _finite(self.upper, "upper"))
        level = _finite(self.confidence_level, "confidence_level")
        if not 0.0 < level < 1.0:
            _fail("INVALID_CONFIDENCE_LEVEL", "confidence_level must lie in (0, 1)", actual=level)
        object.__setattr__(self, "confidence_level", level)
        _required_text(self.method, "method")
        if self.lower > self.upper:
            _fail(
                "INVERTED_CONFIDENCE_INTERVAL",
                "the lower bound must not exceed the upper bound",
                expected="lower <= upper",
                actual=(self.lower, self.upper),
            )

    def excludes(self, null_value: float) -> bool:
        """True when ``null_value`` lies strictly outside the interval."""
        return null_value < self.lower or null_value > self.upper

    @property
    def width(self) -> float:
        return self.upper - self.lower


def wilson_interval(
    successes: int, trials: int, *, confidence_level: float = DEFAULT_CONFIDENCE_LEVEL
) -> ConfidenceInterval:
    """Wilson score interval for a single proportion (always inside [0, 1])."""
    n = _count(trials, "trials")
    x = _count(successes, "successes")
    if n == 0:
        _fail("EMPTY_SAMPLE", "a Wilson interval needs at least one trial", path="trials")
    if x > n:
        _fail("INVALID_COUNT", "successes cannot exceed trials", expected=f"<= {n}", actual=x)
    z = normal_quantile(0.5 * (1.0 + confidence_level))
    p = x / n
    denominator = 1.0 + z * z / n
    centre = (p + z * z / (2.0 * n)) / denominator
    half_width = z / denominator * math.sqrt(p * (1.0 - p) / n + z * z / (4.0 * n * n))
    return ConfidenceInterval(
        lower=max(0.0, centre - half_width),
        upper=min(1.0, centre + half_width),
        confidence_level=confidence_level,
        method="wilson_score",
    )


# ---------------------------------------------------------------------------
# The paired 2x2 table
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class PairedTable:
    """The matched-pair 2x2 table for two policies on identical Episode_Cases.

    ``n10`` counts pairs where A succeeded and B failed (discordant, favours A);
    ``n01`` counts the mirror image (favours B).  See the module docstring for
    the full layout and sign convention.
    """

    n11: int
    n10: int
    n01: int
    n00: int
    schema_version: str = STATISTICS_SCHEMA_VERSION

    def __post_init__(self) -> None:
        for name in ("n11", "n10", "n01", "n00"):
            _count(getattr(self, name), name)
        if self.n_pairs == 0:
            _fail("EMPTY_PAIRED_SAMPLE", "a paired table needs at least one pair")

    @property
    def n_pairs(self) -> int:
        return self.n11 + self.n10 + self.n01 + self.n00

    @property
    def n_discordant(self) -> int:
        return self.n10 + self.n01

    @property
    def successes_a(self) -> int:
        return self.n11 + self.n10

    @property
    def successes_b(self) -> int:
        return self.n11 + self.n01

    @property
    def proportion_a(self) -> float:
        return self.successes_a / self.n_pairs

    @property
    def proportion_b(self) -> float:
        return self.successes_b / self.n_pairs

    @property
    def risk_difference(self) -> float:
        """``p_a - p_b``: positive favours policy A."""
        return (self.n10 - self.n01) / self.n_pairs

    @property
    def has_empty_discordant_cell(self) -> bool:
        return self.n10 == 0 or self.n01 == 0

    @property
    def odds_ratio(self) -> float:
        """Conditional (McNemar) odds ratio ``n10 / n01``, Haldane-corrected if needed."""
        n10, n01 = float(self.n10), float(self.n01)
        if self.has_empty_discordant_cell:
            n10, n01 = n10 + 0.5, n01 + 0.5
        return n10 / n01

    @property
    def table_hash(self) -> str:
        return content_hash(self)


def paired_table_from_outcomes(pairs: Sequence[tuple[bool, bool]]) -> PairedTable:
    """Build the 2x2 table from ``(outcome_a, outcome_b)`` pairs."""
    counts = {(True, True): 0, (True, False): 0, (False, True): 0, (False, False): 0}
    for index, pair in enumerate(pairs):
        if len(pair) != 2:
            _fail("INVALID_PAIR", "each pair must hold exactly two outcomes", path=f"pairs[{index}]", actual=pair)
        key = (bool(pair[0]), bool(pair[1]))
        counts[key] += 1
    return PairedTable(
        n11=counts[(True, True)],
        n10=counts[(True, False)],
        n01=counts[(False, True)],
        n00=counts[(False, False)],
    )


def exact_mcnemar_p_value(n10: int, n01: int) -> float:
    """Two-sided exact (binomial) McNemar p-value from the discordant counts.

    Conditional on ``m = n10 + n01`` discordant pairs, ``n10 ~ Binomial(m, 1/2)``
    under the null of no policy difference.  With no discordant pair at all the
    data carry no information about the direction, so the p-value is 1.
    """
    a, b = _count(n10, "n10"), _count(n01, "n01")
    discordant = a + b
    if discordant == 0:
        return 1.0
    return min(1.0, 2.0 * binomial_cdf_half(min(a, b), discordant))


# ---------------------------------------------------------------------------
# Restricted mean
# ---------------------------------------------------------------------------


def restricted_mean(values: Sequence[float | None], horizon: float) -> float:
    """Sample estimate of ``E[min(T, horizon)]``; ``None`` means "never observed"."""
    tau = _finite(horizon, "horizon")
    if tau <= 0.0:
        _fail("INVALID_HORIZON", "horizon must be positive", path="horizon", actual=horizon)
    if not values:
        _fail("EMPTY_SAMPLE", "a restricted mean needs at least one value")
    total = 0.0
    for index, value in enumerate(values):
        if value is None:
            total += tau
            continue
        total += min(_finite(value, f"values[{index}]"), tau)
    return total / len(values)


# ---------------------------------------------------------------------------
# Hierarchical paired bootstrap (Requirement 8.3-8.4)
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class BootstrapPlan:
    """The pre-registered resampling plan for one effect estimate.

    Requirement 8.4 fixes 10,000 replicates as the default and allows any other
    positive integer *provided the reason is recorded*, so a non-default count
    without a ``rationale`` is rejected here rather than quietly accepted.
    """

    n_bootstrap: int = DEFAULT_BOOTSTRAP_SAMPLES
    rng_seed: int = 0
    confidence_level: float = DEFAULT_CONFIDENCE_LEVEL
    rationale: str | None = None
    schema_version: str = STATISTICS_SCHEMA_VERSION

    def __post_init__(self) -> None:
        n = self.n_bootstrap
        if isinstance(n, bool) or not isinstance(n, int) or n <= 0:
            _fail("INVALID_BOOTSTRAP_COUNT", "n_bootstrap must be a positive integer", actual=n)
        if isinstance(self.rng_seed, bool) or not isinstance(self.rng_seed, int):
            _fail("INVALID_RNG_SEED", "rng_seed must be an integer", actual=self.rng_seed)
        level = _finite(self.confidence_level, "confidence_level")
        if not 0.0 < level < 1.0:
            _fail("INVALID_CONFIDENCE_LEVEL", "confidence_level must lie in (0, 1)", actual=level)
        object.__setattr__(self, "confidence_level", level)
        if n != DEFAULT_BOOTSTRAP_SAMPLES:
            _required_text(
                self.rationale,
                "rationale",
            )

    @property
    def plan_hash(self) -> str:
        return content_hash(self)


SeedGroups = Sequence[np.ndarray]
BootstrapStatistic = Callable[[SeedGroups], float]


def _as_seed_groups(seed_groups: Sequence[Sequence[object]]) -> tuple[np.ndarray, ...]:
    if not seed_groups:
        _fail("EMPTY_PAIRED_SAMPLE", "at least one Training_Seed group is required")
    groups: list[np.ndarray] = []
    for index, group in enumerate(seed_groups):
        array = np.asarray(group)
        if array.shape[0] == 0:
            _fail(
                "EMPTY_SEED_GROUP",
                "every Training_Seed group must contain at least one Episode_Case pair",
                path=f"seed_groups[{index}]",
            )
        groups.append(array)
    return tuple(groups)


def hierarchical_paired_bootstrap(
    seed_groups: Sequence[Sequence[object]],
    statistic: BootstrapStatistic,
    *,
    plan: BootstrapPlan = BootstrapPlan(),
) -> tuple[ConfidenceInterval, np.ndarray]:
    """Seed-outer, pair-inner percentile bootstrap of ``statistic``.

    ``seed_groups`` is one entry per Training_Seed, each holding that seed's own
    Episode_Case pairs (rows of an array; for a binary comparison, an ``(n, 2)``
    array of ``(outcome_a, outcome_b)``).  Each replicate draws ``n_seeds`` seeds
    with replacement and then resamples each drawn seed's rows with replacement
    *within that seed only*, so a pair can never migrate to another seed and the
    two arms of a pair are never separated.

    Returns the percentile interval and the raw replicate values, so callers can
    derive a bootstrap p-value from the same replicates instead of resampling
    twice.
    """
    groups = _as_seed_groups(seed_groups)
    rng = np.random.default_rng(plan.rng_seed)
    n_seeds = len(groups)
    replicates = np.empty(plan.n_bootstrap, dtype=float)
    sizes = np.array([len(group) for group in groups])
    for index in range(plan.n_bootstrap):
        drawn_seeds = rng.integers(0, n_seeds, size=n_seeds)
        resampled = [
            groups[seed_index][rng.integers(0, sizes[seed_index], size=sizes[seed_index])]
            for seed_index in drawn_seeds
        ]
        replicates[index] = statistic(resampled)
    if not np.all(np.isfinite(replicates)):
        _fail(
            "NON_FINITE_BOOTSTRAP_REPLICATE",
            "the bootstrap statistic produced a non-finite replicate",
            actual=int(np.count_nonzero(~np.isfinite(replicates))),
        )
    alpha = 1.0 - plan.confidence_level
    lower, upper = np.percentile(replicates, [100.0 * alpha / 2.0, 100.0 * (1.0 - alpha / 2.0)])
    interval = ConfidenceInterval(
        lower=float(lower),
        upper=float(upper),
        confidence_level=plan.confidence_level,
        method="hierarchical_paired_bootstrap_percentile",
    )
    return interval, replicates


def bootstrap_two_sided_p_value(replicates: np.ndarray, null_value: float = 0.0) -> float:
    """Percentile-bootstrap achieved significance level for a two-sided test.

    Twice the smaller tail mass on either side of ``null_value``, capped at 1.
    Used for continuous effects, where no exact conditional test is available;
    binary comparisons use :func:`exact_mcnemar_p_value` instead.
    """
    values = np.asarray(replicates, dtype=float)
    if values.size == 0:
        _fail("EMPTY_SAMPLE", "a bootstrap p-value needs at least one replicate")
    below = float(np.count_nonzero(values <= null_value)) / values.size
    above = float(np.count_nonzero(values >= null_value)) / values.size
    return min(1.0, 2.0 * min(below, above))


# ---------------------------------------------------------------------------
# Multiple-comparison correction (Requirement 8.6)
# ---------------------------------------------------------------------------


def _validated_p_values(p_values: Sequence[float]) -> list[float]:
    if not p_values:
        _fail("EMPTY_TEST_FAMILY", "a correction family must contain at least one p-value")
    validated: list[float] = []
    for index, value in enumerate(p_values):
        numeric = _finite(value, f"p_values[{index}]")
        if not 0.0 <= numeric <= 1.0:
            _fail("INVALID_P_VALUE", "p-values must lie in [0, 1]", path=f"p_values[{index}]", actual=numeric)
        validated.append(numeric)
    return validated


def holm_adjusted_p_values(p_values: Sequence[float]) -> tuple[float, ...]:
    """Holm step-down family-wise adjustment, returned in the input order."""
    values = _validated_p_values(p_values)
    m = len(values)
    order = sorted(range(m), key=lambda index: values[index])
    adjusted = [0.0] * m
    running = 0.0
    for rank, index in enumerate(order):
        candidate = (m - rank) * values[index]
        running = max(running, candidate)
        adjusted[index] = min(1.0, running)
    return tuple(adjusted)


def benjamini_hochberg_adjusted_p_values(p_values: Sequence[float]) -> tuple[float, ...]:
    """Benjamini-Hochberg step-up FDR adjustment, returned in the input order."""
    values = _validated_p_values(p_values)
    m = len(values)
    order = sorted(range(m), key=lambda index: values[index])
    adjusted = [0.0] * m
    running = 1.0
    for rank in range(m - 1, -1, -1):
        index = order[rank]
        candidate = m * values[index] / (rank + 1)
        running = min(running, candidate)
        adjusted[index] = min(1.0, running)
    return tuple(adjusted)


class CorrectionMethod(str, Enum):
    HOLM = "holm"
    BENJAMINI_HOCHBERG = "benjamini_hochberg"


def adjust_p_values(p_values: Sequence[float], method: CorrectionMethod) -> tuple[float, ...]:
    if CorrectionMethod(method) is CorrectionMethod.HOLM:
        return holm_adjusted_p_values(p_values)
    return benjamini_hochberg_adjusted_p_values(p_values)


# ---------------------------------------------------------------------------
# Case accounting (Requirement 8.8)
# ---------------------------------------------------------------------------


class CaseStatus(str, Enum):
    VALID = "valid"
    MISSING = "missing"
    FAILED = "failed"
    INTERRUPTED = "interrupted"


class MissingDataPolicy(str, Enum):
    """How a non-``VALID`` Episode_Case enters an analysis.

    ``PRE_REGISTERED_WORST_CASE`` is the primary analysis: an unusable pair is
    scored as the worst outcome for *both* policies (a failure for a binary
    metric, the full horizon for a restricted-mean metric), which never
    manufactures an advantage for either arm.  ``COMPLETE_CASE`` drops the pair
    and is reported alongside as the sensitivity analysis.
    """

    PRE_REGISTERED_WORST_CASE = "pre_registered_worst_case"
    COMPLETE_CASE = "complete_case"


@dataclass(frozen=True, slots=True)
class PairedCase:
    """One planned Episode_Case evaluated under both policies.

    ``checkpoint_time`` exists so repeated measurements of the same
    Training_Seed collapse into a single independent replicate (Requirement
    7.7-7.8); it never increases the reported seed count.
    """

    case_id: str
    training_seed: int
    status: CaseStatus = CaseStatus.VALID
    outcome_a: bool | float | None = None
    outcome_b: bool | float | None = None
    checkpoint_time: str = "final"
    reason: str | None = None
    schema_version: str = STATISTICS_SCHEMA_VERSION

    def __post_init__(self) -> None:
        _required_text(self.case_id, "case_id")
        _required_text(self.checkpoint_time, "checkpoint_time")
        if isinstance(self.training_seed, bool) or not isinstance(self.training_seed, int):
            _fail("INVALID_TRAINING_SEED", "training_seed must be an integer", actual=self.training_seed)
        object.__setattr__(self, "status", CaseStatus(self.status))
        if self.status is CaseStatus.VALID:
            for name in ("outcome_a", "outcome_b"):
                value = getattr(self, name)
                if value is None:
                    _fail("MISSING_OUTCOME", "a valid case requires both policies' outcomes", path=name)
                if not isinstance(value, bool):
                    _finite(value, name)
        else:
            _required_text(self.reason, "reason")
            for name in ("outcome_a", "outcome_b"):
                if getattr(self, name) is not None:
                    _fail(
                        "UNEXPECTED_OUTCOME",
                        "a missing/failed/interrupted case must not carry an outcome",
                        path=name,
                        actual=getattr(self, name),
                    )


@dataclass(frozen=True, slots=True)
class CaseAccounting:
    """Planned-versus-usable Episode_Case ledger for one comparison.

    The four status counts always sum to ``planned``: no case can leave the
    ledger, which is what makes a silently dropped pair impossible to hide.
    """

    planned: int
    valid: int
    missing: int
    failed: int
    interrupted: int
    reasons: Mapping[str, int] = field(default_factory=dict)
    schema_version: str = STATISTICS_SCHEMA_VERSION

    def __post_init__(self) -> None:
        for name in ("planned", "valid", "missing", "failed", "interrupted"):
            _count(getattr(self, name), name)
        object.__setattr__(self, "reasons", dict(self.reasons))
        if self.valid + self.missing + self.failed + self.interrupted != self.planned:
            _fail(
                "CASE_ACCOUNTING_MISMATCH",
                "valid, missing, failed and interrupted cases must sum to the planned count",
                expected=self.planned,
                actual=self.valid + self.missing + self.failed + self.interrupted,
            )

    @property
    def unusable(self) -> int:
        return self.missing + self.failed + self.interrupted

    @property
    def accounting_hash(self) -> str:
        return content_hash(self)


def account_cases(cases: Sequence[PairedCase]) -> CaseAccounting:
    """Build the ledger; every case is counted exactly once, by status."""
    if not cases:
        _fail("EMPTY_PAIRED_SAMPLE", "at least one planned Episode_Case is required")
    tallies = {status: 0 for status in CaseStatus}
    reasons: dict[str, int] = {}
    for case in cases:
        tallies[case.status] += 1
        if case.status is not CaseStatus.VALID:
            reasons[case.reason] = reasons.get(case.reason, 0) + 1
    return CaseAccounting(
        planned=len(cases),
        valid=tallies[CaseStatus.VALID],
        missing=tallies[CaseStatus.MISSING],
        failed=tallies[CaseStatus.FAILED],
        interrupted=tallies[CaseStatus.INTERRUPTED],
        reasons=reasons,
    )


# ---------------------------------------------------------------------------
# Practical-threshold gating (Requirement 8.7)
# ---------------------------------------------------------------------------


class EffectDirection(str, Enum):
    """Which sign of the effect counts as an improvement for policy A."""

    GREATER_IS_BETTER = "greater_is_better"
    LESS_IS_BETTER = "less_is_better"


@dataclass(frozen=True, slots=True)
class PracticalThreshold:
    """A pre-registered smallest effect worth claiming, in the metric's own unit."""

    metric: str
    unit: str
    minimum_effect: float
    direction: EffectDirection = EffectDirection.GREATER_IS_BETTER
    non_inferiority_margin: float | None = None
    schema_version: str = STATISTICS_SCHEMA_VERSION

    def __post_init__(self) -> None:
        _required_text(self.metric, "metric")
        _required_text(self.unit, "unit")
        object.__setattr__(self, "direction", EffectDirection(self.direction))
        minimum = _finite(self.minimum_effect, "minimum_effect")
        if minimum <= 0.0:
            _fail("INVALID_PRACTICAL_THRESHOLD", "minimum_effect must be positive", actual=minimum)
        object.__setattr__(self, "minimum_effect", minimum)
        if self.non_inferiority_margin is not None:
            margin = _finite(self.non_inferiority_margin, "non_inferiority_margin")
            if margin <= 0.0:
                _fail("INVALID_PRACTICAL_THRESHOLD", "non_inferiority_margin must be positive", actual=margin)
            object.__setattr__(self, "non_inferiority_margin", margin)


@dataclass(frozen=True, slots=True)
class PracticalGateDecision:
    """The joint statistical-and-practical verdict for one comparison."""

    statistically_significant: bool
    practically_significant: bool
    superiority: bool
    non_inferiority: bool
    reasons: tuple[str, ...]
    schema_version: str = STATISTICS_SCHEMA_VERSION


def evaluate_practical_gate(
    effect_size: float,
    interval: ConfidenceInterval,
    threshold: PracticalThreshold,
    *,
    null_value: float = 0.0,
) -> PracticalGateDecision:
    """Superiority needs statistical *and* practical significance, both.

    An effect whose CI excludes the null but whose magnitude falls short of the
    pre-registered ``minimum_effect`` fails the gate: statistical significance
    alone can never carry a superiority claim.  Non-inferiority, when a margin
    is registered, requires the whole interval to stay on the acceptable side of
    that margin.
    """
    effect = _finite(effect_size, "effect_size")
    null = _finite(null_value, "null_value")
    if threshold.direction is EffectDirection.LESS_IS_BETTER:
        oriented_effect = null - effect
        oriented_lower, oriented_upper = null - interval.upper, null - interval.lower
    else:
        oriented_effect = effect - null
        oriented_lower, oriented_upper = interval.lower - null, interval.upper - null

    reasons: list[str] = []
    favourable_significance = oriented_lower > 0.0
    statistically_significant = favourable_significance or oriented_upper < 0.0
    practically_significant = oriented_effect >= threshold.minimum_effect

    if not statistically_significant:
        reasons.append("the confidence interval covers the null value")
    elif not favourable_significance:
        reasons.append("the confidence interval excludes the null in the unfavourable direction")
    if not practically_significant:
        reasons.append(
            f"the effect {oriented_effect:.6g} {threshold.unit} does not reach the pre-registered "
            f"practical threshold {threshold.minimum_effect:.6g} {threshold.unit}"
        )

    superiority = favourable_significance and practically_significant
    non_inferiority = False
    if threshold.non_inferiority_margin is None:
        reasons.append("no non-inferiority margin was pre-registered")
    else:
        non_inferiority = oriented_lower > -threshold.non_inferiority_margin
        if not non_inferiority:
            reasons.append(
                f"the interval's unfavourable bound {oriented_lower:.6g} {threshold.unit} crosses the "
                f"non-inferiority margin -{threshold.non_inferiority_margin:.6g} {threshold.unit}"
            )
    if superiority:
        reasons.append("statistical significance and the practical threshold are both met")
    return PracticalGateDecision(
        statistically_significant=statistically_significant,
        practically_significant=practically_significant,
        superiority=superiority,
        non_inferiority=non_inferiority,
        reasons=tuple(reasons),
    )


# ---------------------------------------------------------------------------
# Per-seed rows and analyses
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class SeedRow:
    """Requirement 7.9: one Training_Seed's own estimate, reported beside the pooled one."""

    training_seed: int
    n_pairs: int
    estimate_a: float
    estimate_b: float
    effect_size: float
    schema_version: str = STATISTICS_SCHEMA_VERSION


def _between_seed_sd(rows: Sequence[SeedRow]) -> float | None:
    """Sample SD (ddof=1) of the per-seed effects; ``None`` below two seeds."""
    if len(rows) < 2:
        return None
    return float(np.std([row.effect_size for row in rows], ddof=1))


def _binary_risk_difference(groups: SeedGroups) -> float:
    pooled = np.concatenate([np.asarray(group, dtype=bool) for group in groups])
    return float(pooled[:, 0].mean() - pooled[:, 1].mean())


def _binary_log_odds_ratio(groups: SeedGroups) -> float:
    pooled = np.concatenate([np.asarray(group, dtype=bool) for group in groups])
    n10 = float(np.count_nonzero(pooled[:, 0] & ~pooled[:, 1]))
    n01 = float(np.count_nonzero(~pooled[:, 0] & pooled[:, 1]))
    if n10 == 0.0 or n01 == 0.0:
        n10, n01 = n10 + 0.5, n01 + 0.5
    return math.log(n10 / n01)


def _mean_difference(groups: SeedGroups) -> float:
    pooled = np.concatenate([np.asarray(group, dtype=float) for group in groups])
    return float(pooled[:, 0].mean() - pooled[:, 1].mean())


@dataclass(frozen=True, slots=True)
class PairedBinaryAnalysis:
    """One analysis arm (primary or complete-case) of a paired binary comparison."""

    label: str
    missing_data_policy: MissingDataPolicy
    table: PairedTable
    n_pairs: int
    n_seeds: int
    proportion_a_interval: ConfidenceInterval
    proportion_b_interval: ConfidenceInterval
    risk_difference: float
    risk_difference_interval: ConfidenceInterval
    odds_ratio: float
    odds_ratio_interval: ConfidenceInterval
    haldane_correction_applied: bool
    p_value_raw: float
    seed_rows: tuple[SeedRow, ...]
    between_seed_sd: float | None
    adjusted_p_value: float | None = None
    schema_version: str = STATISTICS_SCHEMA_VERSION


@dataclass(frozen=True, slots=True)
class PairedContinuousAnalysis:
    """One analysis arm of a paired continuous (or restricted-mean) comparison."""

    label: str
    missing_data_policy: MissingDataPolicy
    n_pairs: int
    n_seeds: int
    estimate_a: float
    estimate_b: float
    effect_size: float
    effect_interval: ConfidenceInterval
    p_value_raw: float
    seed_rows: tuple[SeedRow, ...]
    between_seed_sd: float | None
    adjusted_p_value: float | None = None
    schema_version: str = STATISTICS_SCHEMA_VERSION


@dataclass(frozen=True, slots=True)
class PairedComparisonResult:
    """A complete paired comparison: ledger, primary analysis, sensitivity, gate.

    ``complete_case`` is ``None`` only when no usable pair survives complete-case
    filtering; the reason is then recorded in ``complete_case_note`` rather than
    left implicit.
    """

    comparison_id: str
    metric: str
    unit: str
    accounting: CaseAccounting
    primary: PairedBinaryAnalysis | PairedContinuousAnalysis
    complete_case: PairedBinaryAnalysis | PairedContinuousAnalysis | None
    bootstrap_plan: BootstrapPlan
    complete_case_note: str | None = None
    gate: PracticalGateDecision | None = None
    sample_size_plan_hash: str | None = None
    confirmatory_eligible: bool = True
    schema_version: str = STATISTICS_SCHEMA_VERSION

    @property
    def result_hash(self) -> str:
        return content_hash(self)


def _seed_order(cases: Sequence[PairedCase]) -> tuple[int, ...]:
    return tuple(sorted({case.training_seed for case in cases}))


def _independent_seed_count(cases: Sequence[PairedCase]) -> int:
    """Distinct Training_Seeds, counted through :mod:`budget` so checkpoints cannot inflate it."""
    return count_independent_training_seeds(
        [{"training_seed": case.training_seed, "checkpoint_time": case.checkpoint_time} for case in cases]
    )


def _resolve_plan_gate(
    cases: Sequence[PairedCase], sample_size_plan: SampleSizePlan | None
) -> tuple[str | None, bool]:
    if sample_size_plan is None:
        return None, True
    eligible = sample_size_plan.is_confirmatory_eligible and (
        _independent_seed_count(cases) >= sample_size_plan.seeds_per_condition
    )
    return sample_size_plan.plan_hash, eligible


def _binary_seed_groups(
    cases: Sequence[PairedCase], policy: MissingDataPolicy
) -> tuple[tuple[int, ...], list[np.ndarray]]:
    grouped: dict[int, list[tuple[bool, bool]]] = {seed: [] for seed in _seed_order(cases)}
    for case in cases:
        if case.status is CaseStatus.VALID:
            if not isinstance(case.outcome_a, bool) or not isinstance(case.outcome_b, bool):
                _fail(
                    "INVALID_BINARY_OUTCOME",
                    "a binary comparison requires boolean outcomes",
                    path=case.case_id,
                    actual=(case.outcome_a, case.outcome_b),
                )
            grouped[case.training_seed].append((case.outcome_a, case.outcome_b))
        elif policy is MissingDataPolicy.PRE_REGISTERED_WORST_CASE:
            grouped[case.training_seed].append((False, False))
    seeds = tuple(seed for seed in grouped if grouped[seed])
    groups = [np.asarray(grouped[seed], dtype=bool) for seed in seeds]
    return seeds, groups


def _continuous_seed_groups(
    cases: Sequence[PairedCase], policy: MissingDataPolicy, horizon: float | None
) -> tuple[tuple[int, ...], list[np.ndarray]]:
    grouped: dict[int, list[tuple[float, float]]] = {seed: [] for seed in _seed_order(cases)}
    for case in cases:
        if case.status is CaseStatus.VALID:
            values = (float(case.outcome_a), float(case.outcome_b))
            if horizon is not None:
                values = (min(values[0], horizon), min(values[1], horizon))
            grouped[case.training_seed].append(values)
        elif policy is MissingDataPolicy.PRE_REGISTERED_WORST_CASE:
            if horizon is None:
                _fail(
                    "MISSING_HORIZON",
                    "the worst-case primary analysis of a continuous metric needs a restriction horizon",
                    path="horizon",
                )
            grouped[case.training_seed].append((horizon, horizon))
    seeds = tuple(seed for seed in grouped if grouped[seed])
    groups = [np.asarray(grouped[seed], dtype=float) for seed in seeds]
    return seeds, groups


def _binary_analysis(
    label: str,
    policy: MissingDataPolicy,
    seeds: Sequence[int],
    groups: Sequence[np.ndarray],
    plan: BootstrapPlan,
) -> PairedBinaryAnalysis:
    pooled = np.concatenate(groups)
    table = paired_table_from_outcomes([(bool(row[0]), bool(row[1])) for row in pooled])
    risk_interval, _ = hierarchical_paired_bootstrap(groups, _binary_risk_difference, plan=plan)
    log_or_interval, _ = hierarchical_paired_bootstrap(groups, _binary_log_odds_ratio, plan=plan)
    rows = tuple(
        SeedRow(
            training_seed=seed,
            n_pairs=len(group),
            estimate_a=float(group[:, 0].mean()),
            estimate_b=float(group[:, 1].mean()),
            effect_size=float(group[:, 0].mean() - group[:, 1].mean()),
        )
        for seed, group in zip(seeds, groups)
    )
    return PairedBinaryAnalysis(
        label=label,
        missing_data_policy=policy,
        table=table,
        n_pairs=table.n_pairs,
        n_seeds=len(groups),
        proportion_a_interval=wilson_interval(
            table.successes_a, table.n_pairs, confidence_level=plan.confidence_level
        ),
        proportion_b_interval=wilson_interval(
            table.successes_b, table.n_pairs, confidence_level=plan.confidence_level
        ),
        risk_difference=table.risk_difference,
        risk_difference_interval=risk_interval,
        odds_ratio=table.odds_ratio,
        odds_ratio_interval=ConfidenceInterval(
            lower=math.exp(log_or_interval.lower),
            upper=math.exp(log_or_interval.upper),
            confidence_level=plan.confidence_level,
            method="hierarchical_paired_bootstrap_percentile_log_odds_ratio",
        ),
        haldane_correction_applied=table.has_empty_discordant_cell,
        p_value_raw=exact_mcnemar_p_value(table.n10, table.n01),
        seed_rows=rows,
        between_seed_sd=_between_seed_sd(rows),
    )


def _continuous_analysis(
    label: str,
    policy: MissingDataPolicy,
    seeds: Sequence[int],
    groups: Sequence[np.ndarray],
    plan: BootstrapPlan,
) -> PairedContinuousAnalysis:
    pooled = np.concatenate(groups)
    interval, replicates = hierarchical_paired_bootstrap(groups, _mean_difference, plan=plan)
    rows = tuple(
        SeedRow(
            training_seed=seed,
            n_pairs=len(group),
            estimate_a=float(group[:, 0].mean()),
            estimate_b=float(group[:, 1].mean()),
            effect_size=float(group[:, 0].mean() - group[:, 1].mean()),
        )
        for seed, group in zip(seeds, groups)
    )
    return PairedContinuousAnalysis(
        label=label,
        missing_data_policy=policy,
        n_pairs=int(pooled.shape[0]),
        n_seeds=len(groups),
        estimate_a=float(pooled[:, 0].mean()),
        estimate_b=float(pooled[:, 1].mean()),
        effect_size=float(pooled[:, 0].mean() - pooled[:, 1].mean()),
        effect_interval=interval,
        p_value_raw=bootstrap_two_sided_p_value(replicates),
        seed_rows=rows,
        between_seed_sd=_between_seed_sd(rows),
    )


def compare_paired_binary(
    cases: Sequence[PairedCase],
    *,
    comparison_id: str,
    metric: str,
    unit: str = "proportion_of_episode_cases",
    plan: BootstrapPlan = BootstrapPlan(),
    threshold: PracticalThreshold | None = None,
    sample_size_plan: SampleSizePlan | None = None,
) -> PairedComparisonResult:
    """Full paired binary comparison of policy A against policy B.

    The primary analysis applies the pre-registered worst-case handling of
    unusable Episode_Cases and the complete-case analysis is always computed
    beside it (Requirement 8.8), so no pair ever disappears from the headline
    number without its sensitivity counterpart.
    """
    _required_text(comparison_id, "comparison_id")
    _required_text(metric, "metric")
    _required_text(unit, "unit")
    accounting = account_cases(cases)

    seeds, groups = _binary_seed_groups(cases, MissingDataPolicy.PRE_REGISTERED_WORST_CASE)
    if not groups:
        _fail("EMPTY_PAIRED_SAMPLE", "the primary analysis has no Episode_Case pair")
    primary = _binary_analysis(
        PRIMARY_ANALYSIS_LABEL, MissingDataPolicy.PRE_REGISTERED_WORST_CASE, seeds, groups, plan
    )

    complete_seeds, complete_groups = _binary_seed_groups(cases, MissingDataPolicy.COMPLETE_CASE)
    complete_case: PairedBinaryAnalysis | None = None
    note: str | None = None
    if complete_groups:
        complete_case = _binary_analysis(
            COMPLETE_CASE_ANALYSIS_LABEL, MissingDataPolicy.COMPLETE_CASE, complete_seeds, complete_groups, plan
        )
    else:
        note = "no Episode_Case survived complete-case filtering, so no sensitivity analysis exists"

    gate = None
    if threshold is not None:
        gate = evaluate_practical_gate(
            primary.risk_difference, primary.risk_difference_interval, threshold
        )
    plan_hash, eligible = _resolve_plan_gate(cases, sample_size_plan)
    return PairedComparisonResult(
        comparison_id=comparison_id,
        metric=metric,
        unit=unit,
        accounting=accounting,
        primary=primary,
        complete_case=complete_case,
        bootstrap_plan=plan,
        complete_case_note=note,
        gate=gate,
        sample_size_plan_hash=plan_hash,
        confirmatory_eligible=eligible,
    )


def compare_paired_continuous(
    cases: Sequence[PairedCase],
    *,
    comparison_id: str,
    metric: str,
    unit: str,
    horizon: float | None = None,
    plan: BootstrapPlan = BootstrapPlan(),
    threshold: PracticalThreshold | None = None,
    sample_size_plan: SampleSizePlan | None = None,
) -> PairedComparisonResult:
    """Paired mean (or restricted-mean, when ``horizon`` is given) difference.

    With a ``horizon`` every value is truncated at it before averaging, so the
    effect is the difference of two restricted means ``E[min(T, tau)]`` and an
    unusable Episode_Case enters the primary analysis at the full horizon for
    both arms.
    """
    _required_text(comparison_id, "comparison_id")
    _required_text(metric, "metric")
    _required_text(unit, "unit")
    if horizon is not None:
        horizon = _finite(horizon, "horizon")
        if horizon <= 0.0:
            _fail("INVALID_HORIZON", "horizon must be positive", path="horizon", actual=horizon)
    accounting = account_cases(cases)

    seeds, groups = _continuous_seed_groups(cases, MissingDataPolicy.PRE_REGISTERED_WORST_CASE, horizon)
    if not groups:
        _fail("EMPTY_PAIRED_SAMPLE", "the primary analysis has no Episode_Case pair")
    primary = _continuous_analysis(
        PRIMARY_ANALYSIS_LABEL, MissingDataPolicy.PRE_REGISTERED_WORST_CASE, seeds, groups, plan
    )

    complete_seeds, complete_groups = _continuous_seed_groups(cases, MissingDataPolicy.COMPLETE_CASE, horizon)
    complete_case: PairedContinuousAnalysis | None = None
    note: str | None = None
    if complete_groups:
        complete_case = _continuous_analysis(
            COMPLETE_CASE_ANALYSIS_LABEL, MissingDataPolicy.COMPLETE_CASE, complete_seeds, complete_groups, plan
        )
    else:
        note = "no Episode_Case survived complete-case filtering, so no sensitivity analysis exists"

    gate = None
    if threshold is not None:
        gate = evaluate_practical_gate(primary.effect_size, primary.effect_interval, threshold)
    plan_hash, eligible = _resolve_plan_gate(cases, sample_size_plan)
    return PairedComparisonResult(
        comparison_id=comparison_id,
        metric=metric,
        unit=unit,
        accounting=accounting,
        primary=primary,
        complete_case=complete_case,
        bootstrap_plan=plan,
        complete_case_note=note,
        gate=gate,
        sample_size_plan_hash=plan_hash,
        confirmatory_eligible=eligible,
    )


def apply_family_correction(
    results: Sequence[PairedComparisonResult],
    method: CorrectionMethod = CorrectionMethod.HOLM,
) -> tuple[PairedComparisonResult, ...]:
    """Fill ``adjusted_p_value`` across one RQ family (Requirement 8.6).

    The correction is applied to the *primary* analyses, in the caller's order,
    and the adjusted value is written back to the matching result.  The
    complete-case sensitivity analysis keeps its raw p-value: it is not a
    confirmatory member of the family.
    """
    if not results:
        _fail("EMPTY_TEST_FAMILY", "a correction family must contain at least one comparison")
    adjusted = adjust_p_values([result.primary.p_value_raw for result in results], method)
    return tuple(
        replace(result, primary=replace(result.primary, adjusted_p_value=value))
        for result, value in zip(results, adjusted)
    )


# ---------------------------------------------------------------------------
# 2x2 factorial contrast (Requirement 10.6-10.7)
# ---------------------------------------------------------------------------

FactorLevels = tuple[bool, bool]
FACTORIAL_CELLS: tuple[FactorLevels, ...] = ((False, False), (True, False), (False, True), (True, True))


@dataclass(frozen=True, slots=True)
class ContrastEstimate:
    """One factorial contrast: its effect size, unit and interval."""

    name: str
    metric: str
    unit: str
    effect_size: float
    interval: ConfidenceInterval
    schema_version: str = STATISTICS_SCHEMA_VERSION


@dataclass(frozen=True, slots=True)
class FactorialContrastResult:
    """Both main effects and the interaction for one 2x2 metric."""

    metric: str
    unit: str
    factor_a_name: str
    factor_b_name: str
    cell_means: Mapping[FactorLevels, float]
    n_pairs_per_cell: int
    n_seeds: int
    main_effect_a: ContrastEstimate
    main_effect_b: ContrastEstimate
    interaction: ContrastEstimate
    bootstrap_plan: BootstrapPlan
    schema_version: str = STATISTICS_SCHEMA_VERSION


def _validated_factorial_cells(
    cells: Mapping[FactorLevels, Sequence[Sequence[float]]]
) -> tuple[tuple[np.ndarray, ...], ...]:
    """Return one tuple of per-seed arrays per cell, in :data:`FACTORIAL_CELLS` order.

    Every cell must carry the same seed count and the same number of
    Episode_Cases within each seed, because the four arms are evaluated on
    identical Episode_Cases; that shared structure is what lets one resampling
    draw be applied to all four cells and keep the pairing intact.
    """
    keys = {(bool(key[0]), bool(key[1])) for key in cells}
    if keys != set(FACTORIAL_CELLS):
        _fail(
            "INVALID_FACTORIAL_MATRIX",
            "a 2x2 contrast needs exactly the four off/on x off/on cells",
            expected=sorted(FACTORIAL_CELLS),
            actual=sorted(keys),
        )
    resolved = tuple(
        tuple(np.asarray(group, dtype=float) for group in _as_seed_groups(cells[key]))
        for key in FACTORIAL_CELLS
    )
    shapes = [tuple(len(group) for group in cell) for cell in resolved]
    if len(set(shapes)) != 1:
        _fail(
            "FACTORIAL_PAIRING_MISMATCH",
            "all four cells must share the same Training_Seed and Episode_Case structure",
            actual=shapes,
        )
    return resolved


def _cell_mean(groups: Sequence[np.ndarray]) -> float:
    return float(np.concatenate(groups).mean())


def factorial_2x2_contrasts(
    cells: Mapping[FactorLevels, Sequence[Sequence[float]]],
    *,
    metric: str,
    unit: str,
    factor_a_name: str = "u_turn_suppression",
    factor_b_name: str = "hysteresis",
    plan: BootstrapPlan = BootstrapPlan(),
) -> FactorialContrastResult:
    """Main effects and interaction for the 2x2 stabilization factorial.

    ``cells`` maps ``(factor_a_on, factor_b_on)`` to that arm's per-seed lists of
    per-Episode_Case metric values.  With cell means ``m[a][b]``::

        main effect A  = (m[1][0] + m[1][1]) / 2 - (m[0][0] + m[0][1]) / 2
        main effect B  = (m[0][1] + m[1][1]) / 2 - (m[0][0] + m[1][0]) / 2
        interaction    = (m[1][1] - m[0][1]) - (m[1][0] - m[0][0])

    Intervals come from the same seed-outer, pair-inner bootstrap as the paired
    comparisons; one resampling draw is shared by all four cells so that the
    Episode_Case pairing across arms survives every replicate.
    """
    _required_text(metric, "metric")
    _required_text(unit, "unit")
    _required_text(factor_a_name, "factor_a_name")
    _required_text(factor_b_name, "factor_b_name")
    resolved = _validated_factorial_cells(cells)
    off_off, on_off, off_on, on_on = resolved

    def contrasts(quad: Sequence[Sequence[np.ndarray]]) -> tuple[float, float, float]:
        m00, m10, m01, m11 = (_cell_mean(cell) for cell in quad)
        return (
            0.5 * (m10 + m11) - 0.5 * (m00 + m01),
            0.5 * (m01 + m11) - 0.5 * (m00 + m10),
            (m11 - m01) - (m10 - m00),
        )

    point = contrasts(resolved)
    rng = np.random.default_rng(plan.rng_seed)
    n_seeds = len(off_off)
    sizes = [len(group) for group in off_off]
    replicates = np.empty((plan.n_bootstrap, 3), dtype=float)
    for index in range(plan.n_bootstrap):
        drawn_seeds = rng.integers(0, n_seeds, size=n_seeds)
        inner = [rng.integers(0, sizes[seed_index], size=sizes[seed_index]) for seed_index in drawn_seeds]
        resampled = [
            [cell[seed_index][rows] for seed_index, rows in zip(drawn_seeds, inner)] for cell in resolved
        ]
        replicates[index] = contrasts(resampled)
    if not np.all(np.isfinite(replicates)):
        _fail("NON_FINITE_BOOTSTRAP_REPLICATE", "a factorial contrast replicate was not finite")

    alpha = 1.0 - plan.confidence_level
    bounds = np.percentile(replicates, [100.0 * alpha / 2.0, 100.0 * (1.0 - alpha / 2.0)], axis=0)
    names = (f"{factor_a_name}_main_effect", f"{factor_b_name}_main_effect", "interaction")
    estimates = tuple(
        ContrastEstimate(
            name=names[column],
            metric=metric,
            unit=unit,
            effect_size=point[column],
            interval=ConfidenceInterval(
                lower=float(bounds[0, column]),
                upper=float(bounds[1, column]),
                confidence_level=plan.confidence_level,
                method="hierarchical_paired_bootstrap_percentile",
            ),
        )
        for column in range(3)
    )
    return FactorialContrastResult(
        metric=metric,
        unit=unit,
        factor_a_name=factor_a_name,
        factor_b_name=factor_b_name,
        cell_means={key: _cell_mean(cell) for key, cell in zip(FACTORIAL_CELLS, resolved)},
        n_pairs_per_cell=int(sum(sizes)),
        n_seeds=n_seeds,
        main_effect_a=estimates[0],
        main_effect_b=estimates[1],
        interaction=estimates[2],
        bootstrap_plan=plan,
    )


__all__ = (
    "COMPLETE_CASE_ANALYSIS_LABEL",
    "DEFAULT_BOOTSTRAP_SAMPLES",
    "DEFAULT_CONFIDENCE_LEVEL",
    "FACTORIAL_CELLS",
    "PRIMARY_ANALYSIS_LABEL",
    "STATISTICS_SCHEMA_VERSION",
    "BootstrapPlan",
    "CaseAccounting",
    "CaseStatus",
    "ConfidenceInterval",
    "ContrastEstimate",
    "CorrectionMethod",
    "EffectDirection",
    "FactorialContrastResult",
    "MissingDataPolicy",
    "PairedBinaryAnalysis",
    "PairedCase",
    "PairedComparisonResult",
    "PairedContinuousAnalysis",
    "PairedTable",
    "PracticalGateDecision",
    "PracticalThreshold",
    "SeedRow",
    "account_cases",
    "adjust_p_values",
    "apply_family_correction",
    "benjamini_hochberg_adjusted_p_values",
    "binomial_cdf_half",
    "bootstrap_two_sided_p_value",
    "compare_paired_binary",
    "compare_paired_continuous",
    "evaluate_practical_gate",
    "exact_mcnemar_p_value",
    "factorial_2x2_contrasts",
    "hierarchical_paired_bootstrap",
    "holm_adjusted_p_values",
    "normal_quantile",
    "paired_table_from_outcomes",
    "restricted_mean",
    "wilson_interval",
)
