"""Placement curriculum Conditions: global / ring / mixed (Requirement 10.1-10.3).

Placement draws from a dedicated RNG stream that is never shared with the
environment or policy generators, so a placement Condition is reproducible
from its seed alone.  The declared distribution, support, feasibility rule and
stream identity all live in :class:`PlacementCurriculumConfig` and therefore
enter its ``config_hash``, which is what Requirement 10.2 requires a Condition
hash to record.

The ring band is measured as directed road distance *from* the fugitive to the
officer's start intersection, matching the cordon ring already used by
``pursuit_evasion_rl.research.policies.baselines.EncirclementPolice``.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from enum import Enum
import math

import numpy as np

from pursuit_evasion_rl.osm_demo.models import (
    POLICE_COUNT,
    DomainValidationError,
    Intersection,
    ModelNetwork,
    VehiclePlacement,
)
from pursuit_evasion_rl.research.canonical import content_hash
from pursuit_evasion_rl.research.policies.baselines import _Graph

PLACEMENT_SCHEMA_VERSION = "1.0"

# Requirement 10.2: placement draws use their own named stream, never the
# environment or policy generator.
PLACEMENT_RNG_STREAM = "research.placement"

DEFAULT_RING_MIN_M = 120.0
DEFAULT_RING_MAX_M = 380.0
DEFAULT_MIXTURE_RING_PROBABILITY = 0.5

# Recorded when an officer's start node is not road-reachable from the
# fugitive; canonical JSON forbids non-finite numbers, so no infinity here.
UNREACHABLE_DISTANCE_M = -1.0

_SUPPORT_RULE = (
    "non-virtual intersections with at least one incident segment, excluding "
    "the fugitive node; the fugitive is drawn from non-boundary intersections "
    "that have at least one outgoing segment"
)
_FEASIBILITY_RULE = (
    "ring components additionally require ring_min_m <= directed road distance "
    "from the fugitive <= ring_max_m; a component with fewer remaining "
    "candidates than officers to place is infeasible and fails the draw"
)


class PlacementStyle(str, Enum):
    """The three placement Conditions required by Requirement 10.1."""

    GLOBAL = "global"
    RING = "ring"
    MIXED = "mixed"


@dataclass(frozen=True, slots=True)
class PlacementCurriculumConfig:
    """Immutable placement Condition: rule, distribution and RNG identity.

    ``ring_min_m``/``ring_max_m`` [m] bound the feasible ring band.
    ``mixture_ring_probability`` is the protocol-fixed, never-adapted
    probability that a single officer is drawn from the ring component under
    :attr:`PlacementStyle.MIXED` (Requirement 10.3).  ``placement_seed`` seeds
    the dedicated placement stream.
    """

    style: PlacementStyle = PlacementStyle.GLOBAL
    ring_min_m: float = DEFAULT_RING_MIN_M
    ring_max_m: float = DEFAULT_RING_MAX_M
    mixture_ring_probability: float = DEFAULT_MIXTURE_RING_PROBABILITY
    placement_seed: int = 0
    schema_version: str = PLACEMENT_SCHEMA_VERSION

    def __post_init__(self) -> None:
        try:
            object.__setattr__(self, "style", PlacementStyle(self.style))
        except ValueError as exc:
            raise DomainValidationError(
                "INVALID_PLACEMENT_STYLE",
                "style must be one of the declared placement curriculum styles",
                path="style",
                expected=[item.value for item in PlacementStyle],
                actual=self.style,
            ) from exc
        for name in ("ring_min_m", "ring_max_m"):
            value = float(getattr(self, name))
            if not math.isfinite(value) or value < 0.0:
                raise DomainValidationError(
                    "INVALID_FEASIBLE_RADIUS",
                    f"{name} must be finite and nonnegative",
                    path=name,
                    actual=value,
                )
            object.__setattr__(self, name, value)
        if self.ring_min_m >= self.ring_max_m:
            raise DomainValidationError(
                "INVALID_FEASIBLE_RADIUS",
                "ring_min_m must be strictly below ring_max_m",
                path="ring_min_m",
                expected=f"< {self.ring_max_m}",
                actual=self.ring_min_m,
            )
        probability = float(self.mixture_ring_probability)
        if not math.isfinite(probability) or not 0.0 <= probability <= 1.0:
            raise DomainValidationError(
                "INVALID_MIXTURE_PROBABILITY",
                "mixture_ring_probability must lie in [0, 1]",
                path="mixture_ring_probability",
                actual=self.mixture_ring_probability,
            )
        object.__setattr__(self, "mixture_ring_probability", probability)
        seed = self.placement_seed
        if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
            raise DomainValidationError(
                "INVALID_PLACEMENT_SEED",
                "placement_seed must be a nonnegative integer",
                path="placement_seed",
                actual=seed,
            )

    @property
    def config_hash(self) -> str:
        return content_hash(self)

    @property
    def rng_stream(self) -> str:
        """Fully qualified identity of the dedicated placement stream."""
        return f"{PLACEMENT_RNG_STREAM}#seed={self.placement_seed}"


@dataclass(frozen=True, slots=True)
class PlacementProtocol:
    """The recorded rule/distribution/support/RNG for one placement Condition.

    Requirement 10.2 requires every fixed placement Condition to record these
    four things; this dataclass is what a manifest or Condition hash stores.
    """

    style: PlacementStyle
    distribution: tuple[str, ...]
    support: str
    feasibility_rule: str
    rng_stream: str
    config: PlacementCurriculumConfig
    schema_version: str = PLACEMENT_SCHEMA_VERSION

    @property
    def config_hash(self) -> str:
        return content_hash(self)


def _distribution_description(config: PlacementCurriculumConfig) -> tuple[str, ...]:
    ring = (
        f"uniform without replacement over ring candidates with "
        f"{config.ring_min_m} <= road_distance_from_fugitive_m <= {config.ring_max_m}"
    )
    global_component = "uniform without replacement over all support candidates"
    if config.style is PlacementStyle.GLOBAL:
        return (f"global: {global_component}",)
    if config.style is PlacementStyle.RING:
        return (f"ring: {ring}",)
    return (
        f"mixed: per officer, Bernoulli(p={config.mixture_ring_probability}) selects the "
        "ring component, otherwise the global component",
        f"ring: {ring}",
        f"global: {global_component}",
    )


def placement_protocol(config: PlacementCurriculumConfig) -> PlacementProtocol:
    """Return the auditable protocol record for a placement Condition."""
    return PlacementProtocol(
        style=config.style,
        distribution=_distribution_description(config),
        support=_SUPPORT_RULE,
        feasibility_rule=_FEASIBILITY_RULE,
        rng_stream=config.rng_stream,
        config=config,
    )


def placement_rng(config: PlacementCurriculumConfig) -> np.random.Generator:
    """Build the dedicated placement generator; never reuse an env/policy stream."""
    return np.random.default_rng(config.placement_seed)


@dataclass(frozen=True, slots=True)
class PlacementDraw:
    """One reproducible placement outcome and its provenance.

    ``officer_components`` records which component distribution produced each
    officer, so a mixed draw stays auditable.  ``officer_road_distances_m``
    holds the directed road distance from the fugitive, or
    :data:`UNREACHABLE_DISTANCE_M` when no directed path exists.
    """

    config: PlacementCurriculumConfig
    fugitive: VehiclePlacement
    police: tuple[VehiclePlacement, ...]
    officer_components: tuple[str, ...]
    officer_road_distances_m: tuple[float, ...]
    rng_stream: str
    schema_version: str = PLACEMENT_SCHEMA_VERSION

    def __post_init__(self) -> None:
        object.__setattr__(self, "police", tuple(self.police))
        object.__setattr__(self, "officer_components", tuple(self.officer_components))
        object.__setattr__(self, "officer_road_distances_m", tuple(self.officer_road_distances_m))
        if len(self.police) != POLICE_COUNT:
            raise DomainValidationError(
                "PLACEMENT_FAILED",
                f"A placement draw must place exactly {POLICE_COUNT} officers",
                expected=POLICE_COUNT,
                actual=len(self.police),
            )
        if len(self.officer_components) != POLICE_COUNT or len(self.officer_road_distances_m) != POLICE_COUNT:
            raise DomainValidationError(
                "PLACEMENT_FAILED",
                "Per-officer provenance must cover every placed officer",
                expected=POLICE_COUNT,
            )

    @property
    def config_hash(self) -> str:
        return content_hash(self)


def _drivable(network: ModelNetwork) -> list[Intersection]:
    return sorted(
        (
            item
            for item in network.intersections
            if not item.virtual and (item.outgoing_segment_ids or item.incoming_segment_ids)
        ),
        key=lambda item: item.id,
    )


def _take(rng: np.random.Generator, available: list[int]) -> int:
    """Draw one id uniformly without replacement, mutating ``available``."""
    return available.pop(int(rng.integers(len(available))))


def generate_placement(
    network: ModelNetwork,
    config: PlacementCurriculumConfig,
    *,
    rng: np.random.Generator | None = None,
) -> PlacementDraw:
    """Draw a fugitive and six officers under ``config``'s placement Condition.

    ``rng`` defaults to the dedicated stream seeded by ``config.placement_seed``,
    so the same Condition always reproduces the same placement.
    """
    if not isinstance(network, ModelNetwork):
        raise DomainValidationError("INVALID_NETWORK", "A ModelNetwork instance is required", actual=type(network).__name__)
    if rng is None:
        rng = placement_rng(config)
    elif not isinstance(rng, np.random.Generator):
        raise DomainValidationError(
            "INVALID_PLACEMENT_RNG",
            "rng must be an explicit numpy Generator so the placement stream stays separate",
            path="rng",
            actual=type(rng).__name__,
        )

    drivable = _drivable(network)
    interior_movable = [
        item.id for item in drivable if item.boundary_kind is None and item.outgoing_segment_ids
    ]
    if not interior_movable:
        raise DomainValidationError(
            "PLACEMENT_INFEASIBLE",
            "No non-boundary drivable intersection is available for the fugitive",
        )
    fugitive_id = int(rng.choice(interior_movable))

    graph = _Graph(network)
    distances = graph.dist_from(fugitive_id)
    # An officer needs at least one outgoing segment to ever move again after
    # its first decision; without this filter an officer could be drawn onto
    # a pure sink intersection (incoming_segment_ids only, e.g. a bbox-edge
    # dead end) and sit there for the rest of the episode no matter how good
    # the policy is -- indistinguishable from a trained "give up" decision.
    global_pool = [
        item.id for item in drivable if item.id != fugitive_id and item.outgoing_segment_ids
    ]
    ring_pool = [
        identity
        for identity in global_pool
        if config.ring_min_m <= distances.get(identity, float("inf")) <= config.ring_max_m
    ]

    if config.style is PlacementStyle.GLOBAL:
        components = [PlacementStyle.GLOBAL.value] * POLICE_COUNT
    elif config.style is PlacementStyle.RING:
        components = [PlacementStyle.RING.value] * POLICE_COUNT
    else:
        components = [
            PlacementStyle.RING.value
            if float(rng.random()) < config.mixture_ring_probability
            else PlacementStyle.GLOBAL.value
            for _ in range(POLICE_COUNT)
        ]

    ring_needed = components.count(PlacementStyle.RING.value)
    if len(global_pool) < POLICE_COUNT:
        raise DomainValidationError(
            "PLACEMENT_INFEASIBLE",
            f"Network cannot place {POLICE_COUNT} distinct officers",
            expected=POLICE_COUNT,
            actual=len(global_pool),
        )
    if len(ring_pool) < ring_needed:
        raise DomainValidationError(
            "PLACEMENT_INFEASIBLE",
            f"The ring component cannot place {ring_needed} distinct officers within the feasible band",
            path="ring_min_m",
            expected=ring_needed,
            actual=len(ring_pool),
        )

    # The ring band is a subset of the global support, so the constrained
    # component draws first; otherwise a global pick could starve the ring.
    remaining_global = list(global_pool)
    remaining_ring = list(ring_pool)
    chosen: list[int | None] = [None] * POLICE_COUNT
    for component in (PlacementStyle.RING.value, PlacementStyle.GLOBAL.value):
        pool = remaining_ring if component == PlacementStyle.RING.value else remaining_global
        for officer, officer_component in enumerate(components):
            if officer_component != component:
                continue
            identity = _take(rng, pool)
            for other in (remaining_global, remaining_ring):
                if identity in other:
                    other.remove(identity)
            chosen[officer] = identity

    placed = [int(identity) for identity in chosen if identity is not None]
    return PlacementDraw(
        config=config,
        fugitive=VehiclePlacement(intersection_id=fugitive_id),
        police=tuple(VehiclePlacement(intersection_id=identity) for identity in placed),
        officer_components=tuple(components),
        officer_road_distances_m=tuple(
            float(distances.get(identity, UNREACHABLE_DISTANCE_M)) for identity in placed
        ),
        rng_stream=config.rng_stream,
    )


def all_placement_conditions(
    base: PlacementCurriculumConfig = PlacementCurriculumConfig(),
) -> tuple[PlacementCurriculumConfig, ...]:
    """The exactly-three placement arms of Requirement 10.1, one factor apart."""
    return tuple(replace(base, style=style) for style in PlacementStyle)


def validate_placement_matrix(conditions: tuple[PlacementCurriculumConfig, ...]) -> None:
    """Reject a placement matrix with a missing, duplicate, or drifted arm.

    Requirement 10.1 fixes the arm set, and the three arms must differ in
    ``style`` alone: any other differing field is undeclared-factor drift.
    """
    styles = [condition.style for condition in conditions]
    if len(styles) != len(PlacementStyle) or set(styles) != set(PlacementStyle):
        raise DomainValidationError(
            "INVALID_PLACEMENT_MATRIX",
            "placement matrix must contain exactly the global, ring and mixed arms",
            expected=[item.value for item in PlacementStyle],
            actual=[item.value for item in styles],
        )
    baseline = replace(conditions[0], style=PlacementStyle.GLOBAL)
    for condition in conditions[1:]:
        if replace(condition, style=PlacementStyle.GLOBAL) != baseline:
            raise DomainValidationError(
                "UNDECLARED_FACTOR_DRIFT",
                "placement arms must differ in style alone",
                expected=baseline,
                actual=condition,
            )


__all__ = (
    "DEFAULT_MIXTURE_RING_PROBABILITY",
    "DEFAULT_RING_MAX_M",
    "DEFAULT_RING_MIN_M",
    "PLACEMENT_RNG_STREAM",
    "PLACEMENT_SCHEMA_VERSION",
    "PlacementCurriculumConfig",
    "PlacementDraw",
    "PlacementProtocol",
    "PlacementStyle",
    "UNREACHABLE_DISTANCE_M",
    "all_placement_conditions",
    "generate_placement",
    "placement_protocol",
    "placement_rng",
    "validate_placement_matrix",
)
