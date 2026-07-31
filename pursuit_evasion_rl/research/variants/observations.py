"""Audited 21D/28D observation variants for OSM pursuit research.

The 28D adapter intentionally preserves the legacy ``osm_obs_aug`` feature
order and final clipping behavior.  In particular, bearing sine is calculated
raw and negative values are clipped only after all 28 values are concatenated.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Sequence

import numpy as np

from pursuit_evasion_rl.osm_demo.metrics import placement_position
from pursuit_evasion_rl.osm_demo.models import (
    DomainValidationError,
    ModelNetwork,
    VehiclePlacement,
)
from pursuit_evasion_rl.osm_demo.observations import (
    DEFAULT_CLIP_DISTANCE_M,
    OBSERVATION_DIM,
    OSM_PROFILE_ID,
    OSMTopologyV1Adapter,
    osm_topology_v1_contract,
)
from pursuit_evasion_rl.research.canonical import content_hash

OBSERVATION_21D_DIM = OBSERVATION_DIM
AUG_EXTRA = 7
AUG_OBS_DIM = OBSERVATION_21D_DIM + AUG_EXTRA
OBSERVATION_28D_DIM = AUG_OBS_DIM
OBSERVATION_28D_PROFILE_ID = "osm_topology_augmented_v1"
DEFAULT_NEAR_RADIUS_M = 200.0
DECISION_TIMING = "immediately_before_police_action_selection_same_state_snapshot"

AUGMENTED_FIELD_ORDER: tuple[str, ...] = (
    "distance_self_to_fugitive",
    "bearing_self_to_fugitive_sine",
    "bearing_self_to_fugitive_shifted_cosine",
    "team_minimum_distance_to_fugitive",
    "team_mean_distance_to_fugitive",
    "angular_coverage",
    "near_officer_fraction",
)

AUGMENTED_NORMALIZATION: tuple[str, ...] = (
    "min(distance_self_to_fugitive/clip_distance_m,1)",
    "sin(atan2(fugitive-self)); raw sine then final vector clip to [0,1]",
    "(cos(atan2(fugitive-self))+1)/2",
    "min(team_minimum_distance/clip_distance_m,1)",
    "min(team_mean_distance/clip_distance_m,1)",
    "1-max_circular_bearing_gap/(2*pi)",
    "count(team_distance<=near_radius_m)/police_count",
)


@dataclass(frozen=True, slots=True)
class ObservationVariantContract:
    """Canonical, auditable metadata for an observation adapter."""

    profile_id: str
    base_profile_id: str
    base_dimension: int
    dimension: int
    added_fields: tuple[str, ...]
    calculation_timing: str
    clip_distance_m: float
    near_radius_m: float
    normalization: tuple[str, ...]
    final_vector_clip: tuple[float, float]
    dtype: str
    config_hash: str


def _validate_radius(value: float, name: str) -> float:
    value = float(value)
    if not math.isfinite(value) or value <= 0.0:
        raise DomainValidationError(
            "INVALID_OBSERVATION_DISTANCE",
            f"{name} must be a positive finite number",
            actual=value,
        )
    return value


def _contract_payload(clip_distance_m: float, near_radius_m: float) -> dict[str, object]:
    return {
        "profile_id": OBSERVATION_28D_PROFILE_ID,
        "base_profile_id": OSM_PROFILE_ID,
        "base_dimension": OBSERVATION_21D_DIM,
        "dimension": OBSERVATION_28D_DIM,
        "added_fields": AUGMENTED_FIELD_ORDER,
        "calculation_timing": DECISION_TIMING,
        "clip_distance_m": clip_distance_m,
        "near_radius_m": near_radius_m,
        "normalization": AUGMENTED_NORMALIZATION,
        "final_vector_clip": (0.0, 1.0),
        "dtype": "float32",
    }


def observation_28d_contract(
    clip_distance_m: float = DEFAULT_CLIP_DISTANCE_M,
    near_radius_m: float = DEFAULT_NEAR_RADIUS_M,
) -> ObservationVariantContract:
    """Return the audited 28D contract and its canonical configuration hash."""
    clip = _validate_radius(clip_distance_m, "clip_distance_m")
    near = _validate_radius(near_radius_m, "near_radius_m")
    payload = _contract_payload(clip, near)
    return ObservationVariantContract(
        **payload,
        config_hash=content_hash(payload),
    )


def canonical_observation_config_hash(
    clip_distance_m: float = DEFAULT_CLIP_DISTANCE_M,
    near_radius_m: float = DEFAULT_NEAR_RADIUS_M,
) -> str:
    """Return the canonical SHA-256 identity of all result-affecting 28D semantics."""
    return observation_28d_contract(clip_distance_m, near_radius_m).config_hash


# The research-facing 21D adapter is the audited topology implementation itself.
Observation21DAdapter = OSMTopologyV1Adapter
observation_21d_contract = osm_topology_v1_contract


class Observation28DAdapter:
    """Append the seven audited geometric/team features to the 21D adapter."""

    def __init__(
        self,
        network: ModelNetwork,
        *,
        clip_distance_m: float = DEFAULT_CLIP_DISTANCE_M,
        near_radius_m: float = DEFAULT_NEAR_RADIUS_M,
    ) -> None:
        self.net = network
        self.clip = _validate_radius(clip_distance_m, "clip_distance_m")
        self.near_radius_m = _validate_radius(near_radius_m, "near_radius_m")
        self.adapter = Observation21DAdapter(network, clip_distance_m=self.clip)
        self._contract = observation_28d_contract(self.clip, self.near_radius_m)

    @property
    def contract(self) -> ObservationVariantContract:
        return self._contract

    @property
    def config_hash(self) -> str:
        return self._contract.config_hash

    @property
    def calculation_timing(self) -> str:
        return self._contract.calculation_timing

    def _extra(
        self,
        police_index: int,
        police: Sequence[VehiclePlacement],
        fugitive: VehiclePlacement,
    ) -> np.ndarray:
        # Keep operation ordering identical to the audited root implementation.
        f = placement_position(self.net, fugitive)
        ppos = [placement_position(self.net, placement) for placement in police]
        me = ppos[police_index]
        d_self = min(math.dist(me, f) / self.clip, 1.0)
        angle = math.atan2(f[1] - me[1], f[0] - me[0])
        distances = [math.dist(position, f) for position in ppos]
        team_min = min(min(distances) / self.clip, 1.0)
        team_mean = min((sum(distances) / len(distances)) / self.clip, 1.0)
        bearings = sorted(math.atan2(p[1] - f[1], p[0] - f[0]) for p in ppos)
        gaps = [
            (bearings[(index + 1) % len(bearings)] - bearings[index]) % (2 * math.pi)
            for index in range(len(bearings))
        ]
        coverage = 1.0 - (max(gaps) / (2 * math.pi))
        near_fraction = sum(1 for distance in distances if distance <= self.near_radius_m) / len(distances)
        return np.array(
            [
                d_self,
                math.sin(angle),
                (math.cos(angle) + 1) / 2,
                team_min,
                team_mean,
                coverage,
                near_fraction,
            ],
            dtype=np.float32,
        )

    def observe(
        self,
        *,
        police_index: int,
        police: Sequence[VehiclePlacement],
        fugitive: VehiclePlacement,
        step: int,
        max_steps: int,
        incoming_heading: float | None = None,
    ) -> np.ndarray:
        """Calculate both variants from the same pre-action state snapshot."""
        base = self.adapter.observe(
            police_index=police_index,
            police=police,
            fugitive=fugitive,
            step=step,
            max_steps=max_steps,
            incoming_heading=incoming_heading,
        )
        extra = self._extra(police_index, police, fugitive)
        vector = np.concatenate([base, extra]).astype(np.float32)
        # This final clip is legacy behavior, including clipping raw negative sine.
        return np.clip(vector, 0.0, 1.0)


# Historical public name retained in both package and root compatibility paths.
AugmentedObs = Observation28DAdapter


# ----------------------------------------------------------------------------
# Parameter/FLOP capacity accounting (Requirement 9.10-9.11).
#
# The rule mirrors the exact Linear(+bias)/Tanh stack built by
# ``pursuit_evasion_rl.research.policies.masked_mappo._mlp``: a shared actor
# takes ``observation_dim + num_officers`` (one-hot identity) as input and
# emits ``action_dim`` logits through ``len(hidden_dims)`` Tanh-activated
# hidden layers.  This module intentionally recomputes the same analytic
# formula instead of importing torch so capacity accounting stays a pure,
# fast, torch-free function of dimensions.
# ----------------------------------------------------------------------------

FLOPS_PER_MULTIPLY_ADD = 2  # one multiply + one accumulate per MAC, the standard reporting convention.


def _layer_dims(input_dim: int, output_dim: int, hidden_dims: Sequence[int]) -> tuple[int, ...]:
    if input_dim <= 0 or output_dim <= 0:
        raise DomainValidationError(
            "INVALID_CAPACITY_DIMENSION", "input_dim and output_dim must be positive",
            actual={"input_dim": input_dim, "output_dim": output_dim},
        )
    if not hidden_dims or any((isinstance(w, bool) or not isinstance(w, int) or w <= 0) for w in hidden_dims):
        raise DomainValidationError(
            "INVALID_CAPACITY_DIMENSION", "hidden_dims must be a non-empty sequence of positive integers",
            actual=tuple(hidden_dims),
        )
    return (input_dim, *hidden_dims, output_dim)


def mlp_parameter_count(input_dim: int, output_dim: int, hidden_dims: Sequence[int]) -> int:
    """Exact parameter count of a bias-including Linear/Tanh stack (see module note)."""
    dims = _layer_dims(input_dim, output_dim, hidden_dims)
    return sum((dims[index] + 1) * dims[index + 1] for index in range(len(dims) - 1))


def mlp_flop_estimate(input_dim: int, output_dim: int, hidden_dims: Sequence[int]) -> int:
    """Forward-pass FLOP estimate: ``2 * in * out`` multiply-adds per linear layer.

    Bias adds are one FLOP per output unit and are folded in as ``2 * in * out``
    already dominates them; this keeps the rule identical for every reported
    condition, as Requirement 9.10 requires the same computation for both.
    """
    dims = _layer_dims(input_dim, output_dim, hidden_dims)
    return sum(FLOPS_PER_MULTIPLY_ADD * dims[index] * dims[index + 1] for index in range(len(dims) - 1))


@dataclass(frozen=True, slots=True)
class ActorCapacity:
    """Auditable actor parameter/FLOP accounting for one observation variant."""

    observation_dim: int
    action_dim: int
    num_officers: int
    hidden_dims: tuple[int, ...]
    parameter_count: int
    flop_estimate: int


def actor_capacity(
    observation_dim: int, *, action_dim: int, num_officers: int, hidden_dims: Sequence[int]
) -> ActorCapacity:
    """Capacity of :class:`SharedOfficerActor` for a given observation variant.

    The actor input is ``observation_dim + num_officers`` because officer
    identity is concatenated as a one-hot vector before the first layer.
    """
    if num_officers <= 0:
        raise DomainValidationError("INVALID_CAPACITY_DIMENSION", "num_officers must be positive", actual=num_officers)
    hidden = tuple(hidden_dims)
    input_dim = observation_dim + num_officers
    return ActorCapacity(
        observation_dim=observation_dim,
        action_dim=action_dim,
        num_officers=num_officers,
        hidden_dims=hidden,
        parameter_count=mlp_parameter_count(input_dim, action_dim, hidden),
        flop_estimate=mlp_flop_estimate(input_dim, action_dim, hidden),
    )


@dataclass(frozen=True, slots=True)
class CapacityComparison:
    """Paired actor capacity accounting used to report or enforce Requirement 9.11."""

    base: ActorCapacity
    matched: ActorCapacity
    relative_parameter_difference: float

    @property
    def is_capacity_matched(self) -> bool:
        return self.relative_parameter_difference <= 0.01


def compare_observation_capacities(
    *,
    base_observation_dim: int,
    matched_observation_dim: int,
    action_dim: int,
    num_officers: int,
    hidden_dims: Sequence[int],
) -> CapacityComparison:
    """Report whether two observation variants share the same actor hidden_dims capacity."""
    base = actor_capacity(base_observation_dim, action_dim=action_dim, num_officers=num_officers, hidden_dims=hidden_dims)
    matched = actor_capacity(matched_observation_dim, action_dim=action_dim, num_officers=num_officers, hidden_dims=hidden_dims)
    difference = abs(matched.parameter_count - base.parameter_count) / base.parameter_count
    return CapacityComparison(base=base, matched=matched, relative_parameter_difference=difference)


def find_capacity_matched_hidden_width(
    *,
    base_observation_dim: int,
    base_hidden_dims: Sequence[int],
    target_observation_dim: int,
    action_dim: int,
    num_officers: int,
    max_relative_difference: float = 0.01,
    search_widths: Sequence[int] | None = None,
) -> tuple[CapacityComparison, tuple[int, ...]]:
    """Search a uniform hidden width for ``target_observation_dim`` within Requirement 9.11's 1% budget.

    The search keeps the number of hidden layers fixed at ``len(base_hidden_dims)``
    and varies a single repeated width, mirroring how ``TrainerConfig.hidden_dims``
    is actually configured (e.g. ``(128, 128)``).  It returns the best match found
    and raises if nothing in ``search_widths`` (default: a window around the base
    width) reaches the required tolerance, so silent capacity drift cannot occur.
    """
    base_hidden = tuple(base_hidden_dims)
    base = actor_capacity(base_observation_dim, action_dim=action_dim, num_officers=num_officers, hidden_dims=base_hidden)
    layer_count = len(base_hidden)
    if search_widths is None:
        anchor = max(base_hidden)
        search_widths = tuple(range(max(1, anchor - 32), anchor + 33))
    best: CapacityComparison | None = None
    best_width: int | None = None
    for width in search_widths:
        if isinstance(width, bool) or not isinstance(width, int) or width <= 0:
            raise DomainValidationError("INVALID_CAPACITY_DIMENSION", "search_widths must be positive integers", actual=width)
        candidate_hidden = (width,) * layer_count
        matched = actor_capacity(target_observation_dim, action_dim=action_dim, num_officers=num_officers, hidden_dims=candidate_hidden)
        difference = abs(matched.parameter_count - base.parameter_count) / base.parameter_count
        comparison = CapacityComparison(base=base, matched=matched, relative_parameter_difference=difference)
        if best is None or difference < best.relative_parameter_difference:
            best, best_width = comparison, width
    assert best is not None and best_width is not None
    if best.relative_parameter_difference > max_relative_difference:
        raise DomainValidationError(
            "CAPACITY_MATCH_NOT_FOUND",
            "no searched hidden width reached the required capacity-matched tolerance",
            expected=max_relative_difference,
            actual=best.relative_parameter_difference,
        )
    return best, (best_width,) * layer_count


__all__ = (
    "AUG_EXTRA",
    "AUG_OBS_DIM",
    "AUGMENTED_FIELD_ORDER",
    "AUGMENTED_NORMALIZATION",
    "ActorCapacity",
    "AugmentedObs",
    "CapacityComparison",
    "DECISION_TIMING",
    "DEFAULT_NEAR_RADIUS_M",
    "FLOPS_PER_MULTIPLY_ADD",
    "OBSERVATION_21D_DIM",
    "OBSERVATION_28D_DIM",
    "OBSERVATION_28D_PROFILE_ID",
    "Observation21DAdapter",
    "Observation28DAdapter",
    "ObservationVariantContract",
    "actor_capacity",
    "canonical_observation_config_hash",
    "compare_observation_capacities",
    "find_capacity_matched_hidden_width",
    "mlp_flop_estimate",
    "mlp_parameter_count",
    "observation_21d_contract",
    "observation_28d_contract",
)
