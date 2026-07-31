"""Immutable ``EpisodeCase`` sealing, evaluation strata and frozen policy replay.

Requirement 8.1 fixes the sealed field list of an episode case: map Content_Hash,
data kind, scenario, the six police initial states, the fugitive initial state,
the fugitive RNG stream identity, the environment configuration and the
termination configuration.  :class:`EpisodeCase` carries exactly those fields and
nothing else, so its :attr:`EpisodeCase.case_hash` is the complete contract
identity two policies must share before any paired statement is allowed.

Requirement 8.8 forbids paired marking, paired tests and paired confidence
intervals whenever one sealed field or the seed correspondence differs.
:func:`paired_status` therefore never answers a bare boolean: a non-paired result
carries the *named* fields that diverged (``police_0`` .. ``police_5`` are named
individually), and :func:`require_paired` raises with that list attached.
:func:`compare_cases` is fail-closed about its own completeness -- if it finds no
field mismatch while the two case hashes differ it raises rather than reporting a
pair, which is what makes "paired iff complete contract equality" hold.

Requirement 6.10 separates train-distribution, spatial-holdout and cross-city
zero-shot reporting layers.  :class:`EvaluationStratumLabel` crosses the existing
``MapStratum`` (data kind x scenario) with :class:`EvaluationProvenance`, and
:class:`MapProvenanceIndex` assigns every case to exactly one of them or refuses
(Requirement 5.9-5.10).  Requirement 6.6 keeps held-out and cross-city data out
of selection: :class:`SealedCaseBook` exposes in-region cases through
``tuning_cases``/``tuning_data_view`` and held-out or cross-city cases only
through ``frozen_evaluation_cases``, which demands the hash of the already-frozen
policy.  Episode handles carry their ``SplitScope``, so feeding a held-out case
into ``TuningDataView`` fails with the existing ``TUNING_DATA_LEAKAGE`` gate.

``policies/baselines.py`` keeps its lightweight ``EpisodeCaseRef``; the sealed
case is a strict superset and produces one through :meth:`EpisodeCase.to_ref`.
The map-registry ``EpisodeResult`` stays the per-episode outcome record used for
kind x scenario aggregation -- :func:`replay_to_episode_result` produces one from
a replay rather than replacing it.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import re
from typing import Any, Iterable, Mapping, Protocol, Sequence, runtime_checkable

import numpy as np

from pursuit_evasion_rl.osm_demo.environment import FUGITIVE_ID, OSMRoadPursuitEnv
from pursuit_evasion_rl.osm_demo.metrics import placement_position
from pursuit_evasion_rl.osm_demo.models import (
    EpisodeConfig,
    ModelNetwork,
    POLICE_COUNT,
    VehiclePlacement,
    validate_vehicle_placements,
)

from ..canonical import canonical_json, content_hash
from ..domain import DataKind, EpisodeOutcome, MapScenario, PersistedModel, exactly_one
from ..errors import ResearchValidationError
from ..maps.registry import BOUNDARY_OUTCOMES, INTERIOR_OUTCOMES, EpisodeResult, MapStratum
from ..maps.splits import (
    CrossCityMapRef,
    DataHandle,
    HandleKind,
    SpatialPartition,
    SplitScope,
    TuningDataView,
)
from ..policies.baselines import EpisodeCaseRef, GoalEvader, _Graph
from ..variants.observations import Observation28DAdapter

EVALUATION_SCHEMA_VERSION = "research-paired-evaluation-v1"

# Replay observation and evader settings, held as constants so that the replay of
# a sealed case has no caller-tunable degrees of freedom.  The values mirror the
# ``TrainerConfig`` defaults so a policy trained by ``training/trainer.py`` sees
# the same 28D contract at evaluation time.
REPLAY_CLIP_DISTANCE_M = 2000.0
REPLAY_NEAR_RADIUS_M = 200.0
REPLAY_VISION_RANGE_M = 150.0

_SHA256 = re.compile(r"^[0-9a-f]{64}$")


def _required(value: str, path: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ResearchValidationError("MISSING_REQUIRED_FIELD", f"{path} must be non-empty", path=path)
    return value


def _hash(value: str, path: str) -> str:
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        raise ResearchValidationError(
            "INVALID_CONTENT_HASH", f"{path} must be a lowercase SHA-256", path=path, actual=value
        )
    return value


def scenario_outcome_domain(scenario: MapScenario) -> frozenset[EpisodeOutcome]:
    """Return the registered outcome domain of a scenario (Requirements 5.7-5.8)."""
    scenario = exactly_one(scenario, MapScenario, path="scenario")
    return INTERIOR_OUTCOMES if scenario is MapScenario.INTERIOR_CONTAINED else BOUNDARY_OUTCOMES


def _nonnegative_int(value: Any, path: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ResearchValidationError(
            "INVALID_EPISODE_CASE", f"{path} must be a nonnegative integer", path=path, actual=value
        )
    return value


# ---------------------------------------------------------------------------
# Sealed case contract (Requirement 8.1)
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class FugitiveRNGSpec(PersistedModel):
    """The fugitive's deterministic RNG stream identity.

    Requirement 8.1 allows either the RNG state or its stream hash to be sealed.
    A named stream plus its seed reconstructs the state exactly, and
    :attr:`stream_hash` is the comparable identity used by the pair contract.
    """

    stream_id: str
    seed: int

    def __post_init__(self) -> None:
        _required(self.stream_id, "stream_id")
        _nonnegative_int(self.seed, "seed")
        super(FugitiveRNGSpec, self).__post_init__()

    @property
    def stream_hash(self) -> str:
        return str(self.content_hash)

    def generator(self) -> np.random.Generator:
        return np.random.default_rng(self.seed)


@dataclass(frozen=True, slots=True)
class TerminationConfig(PersistedModel):
    """Episode-end rules that the environment's physics configuration omits.

    ``EpisodeConfig`` carries the capture radius and the environment's own step
    ceiling; it does not carry the admissible outcome domain or the evaluation
    harness's own timeout.  Sealing them separately is what lets Requirement 8.8
    detect a termination-only divergence between two policies.
    """

    outcome_domain: frozenset[EpisodeOutcome]
    timeout_step_limit: int
    capture_precedence_over_escape: bool = True

    def __post_init__(self) -> None:
        domain = frozenset(
            exactly_one(item, EpisodeOutcome, path="outcome_domain") for item in self.outcome_domain
        )
        if not {EpisodeOutcome.CAPTURE, EpisodeOutcome.TIMEOUT} <= domain:
            raise ResearchValidationError(
                "INVALID_TERMINATION_CONFIG",
                "every scenario admits capture and timeout",
                path="outcome_domain",
                actual=sorted(item.value for item in domain),
            )
        if isinstance(self.timeout_step_limit, bool) or not isinstance(self.timeout_step_limit, int):
            raise ResearchValidationError(
                "INVALID_TERMINATION_CONFIG", "timeout_step_limit must be an integer",
                path="timeout_step_limit", actual=self.timeout_step_limit,
            )
        if self.timeout_step_limit <= 0:
            raise ResearchValidationError(
                "INVALID_TERMINATION_CONFIG", "timeout_step_limit must be positive",
                path="timeout_step_limit", actual=self.timeout_step_limit,
            )
        if not self.capture_precedence_over_escape:
            raise ResearchValidationError(
                "INVALID_TERMINATION_CONFIG",
                "Requirement 5.6 fixes capture as the winning outcome over escape",
                path="capture_precedence_over_escape", expected=True, actual=False,
            )
        object.__setattr__(self, "outcome_domain", domain)
        super(TerminationConfig, self).__post_init__()


@dataclass(frozen=True, slots=True)
class EpisodeCase(PersistedModel):
    """The complete, immutable contract two policies must share to be paired.

    ``map_hash`` is the content hash of the ``ModelNetwork`` the episode runs on,
    which is exactly ``RegisteredMap.network_hash``.  :func:`replay_episode`
    re-derives it from the supplied network, so a case cannot be replayed against
    a map it was not sealed for.
    """

    map_hash: str
    data_kind: DataKind
    scenario: MapScenario
    police: tuple[VehiclePlacement, ...]
    fugitive: VehiclePlacement
    fugitive_rng: FugitiveRNGSpec
    environment: EpisodeConfig
    termination: TerminationConfig

    def __post_init__(self) -> None:
        _hash(self.map_hash, "map_hash")
        object.__setattr__(self, "data_kind", exactly_one(self.data_kind, DataKind, path="data_kind"))
        object.__setattr__(self, "scenario", exactly_one(self.scenario, MapScenario, path="scenario"))
        police = tuple(self.police)
        if len(police) != POLICE_COUNT:
            raise ResearchValidationError(
                "INVALID_POLICE_COUNT", f"a case seals exactly {POLICE_COUNT} police initial states",
                path="police", expected=POLICE_COUNT, actual=len(police),
            )
        if any(not isinstance(item, VehiclePlacement) for item in police) or not isinstance(
            self.fugitive, VehiclePlacement
        ):
            raise ResearchValidationError(
                "INVALID_EPISODE_CASE", "police and fugitive states must be VehiclePlacement values",
                path="police",
            )
        if not isinstance(self.fugitive_rng, FugitiveRNGSpec):
            raise ResearchValidationError(
                "INVALID_EPISODE_CASE", "fugitive_rng must be a FugitiveRNGSpec", path="fugitive_rng"
            )
        if not isinstance(self.environment, EpisodeConfig):
            raise ResearchValidationError(
                "INVALID_EPISODE_CASE", "environment must be an EpisodeConfig", path="environment"
            )
        if not isinstance(self.termination, TerminationConfig):
            raise ResearchValidationError(
                "INVALID_EPISODE_CASE", "termination must be a TerminationConfig", path="termination"
            )
        expected_domain = scenario_outcome_domain(self.scenario)
        if self.termination.outcome_domain != expected_domain:
            raise ResearchValidationError(
                "OUTCOME_DOMAIN_MISMATCH", "the scenario fixes the admissible outcome domain",
                path="termination.outcome_domain",
                expected=sorted(item.value for item in expected_domain),
                actual=sorted(item.value for item in self.termination.outcome_domain),
            )
        if self.termination.timeout_step_limit > self.environment.max_steps:
            raise ResearchValidationError(
                "INVALID_TERMINATION_CONFIG",
                "the evaluation timeout cannot exceed the environment step ceiling",
                path="termination.timeout_step_limit",
                expected=f"<= {self.environment.max_steps}",
                actual=self.termination.timeout_step_limit,
            )
        object.__setattr__(self, "police", police)
        super(EpisodeCase, self).__post_init__()

    @property
    def case_hash(self) -> str:
        return str(self.content_hash)

    @property
    def placement_hash(self) -> str:
        return content_hash({"police": self.police, "fugitive": self.fugitive})

    @property
    def map_stratum(self) -> MapStratum:
        return MapStratum(self.data_kind, self.scenario)

    def to_ref(self) -> EpisodeCaseRef:
        """Summarise the sealed case as the baseline registry's generic reference."""
        return EpisodeCaseRef(
            map_hash=self.map_hash,
            scenario=self.scenario,
            placement_hash=self.placement_hash,
            seed=self.fugitive_rng.seed,
        )


def seal_episode_case(
    *,
    network: ModelNetwork,
    data_kind: DataKind,
    scenario: MapScenario,
    police: Sequence[VehiclePlacement],
    fugitive: VehiclePlacement,
    fugitive_rng: FugitiveRNGSpec,
    environment: EpisodeConfig,
    termination: TerminationConfig,
) -> EpisodeCase:
    """Seal a case against a concrete network, validating the full placement contract."""
    if not isinstance(network, ModelNetwork):
        raise ResearchValidationError("INVALID_NETWORK", "network must be a ModelNetwork", path="network")
    validate_vehicle_placements(network, tuple(police), fugitive)
    return EpisodeCase(
        map_hash=content_hash(network),
        data_kind=data_kind,
        scenario=scenario,
        police=tuple(police),
        fugitive=fugitive,
        fugitive_rng=fugitive_rng,
        environment=environment,
        termination=termination,
    )


# ---------------------------------------------------------------------------
# Evaluation strata (Requirements 5.9-5.10, 6.10)
# ---------------------------------------------------------------------------


class EvaluationProvenance(str, Enum):
    """The generalization layer a case belongs to (Requirement 6.10)."""

    IN_REGION = "in_region"
    HELD_OUT = "held_out"
    CROSS_CITY = "cross_city"


_PROVENANCE_BY_SCOPE: Mapping[SplitScope, EvaluationProvenance] = {
    SplitScope.TRAIN: EvaluationProvenance.IN_REGION,
    SplitScope.VALIDATION: EvaluationProvenance.IN_REGION,
    SplitScope.TEST: EvaluationProvenance.HELD_OUT,
    SplitScope.CROSS_CITY: EvaluationProvenance.CROSS_CITY,
}

TUNING_SCOPES = frozenset({SplitScope.TRAIN, SplitScope.VALIDATION})


@dataclass(frozen=True, slots=True)
class EvaluationStratumLabel(PersistedModel):
    """One reporting layer: data kind x scenario x generalization provenance."""

    data_kind: DataKind
    scenario: MapScenario
    provenance: EvaluationProvenance

    def __post_init__(self) -> None:
        object.__setattr__(self, "data_kind", exactly_one(self.data_kind, DataKind, path="data_kind"))
        object.__setattr__(self, "scenario", exactly_one(self.scenario, MapScenario, path="scenario"))
        object.__setattr__(
            self, "provenance", exactly_one(self.provenance, EvaluationProvenance, path="provenance")
        )
        if (
            self.provenance is EvaluationProvenance.CROSS_CITY
            and self.data_kind is not DataKind.ACTUAL_OSM_MAP
        ):
            raise ResearchValidationError(
                "ZERO_SHOT_REQUIRES_ACTUAL_OSM",
                "cross-city zero-shot layers contain only Actual OSM maps",
                path="data_kind", expected=DataKind.ACTUAL_OSM_MAP.value, actual=self.data_kind.value,
            )
        super(EvaluationStratumLabel, self).__post_init__()

    @property
    def stratum_id(self) -> str:
        return f"{self.data_kind.value}/{self.scenario.value}/{self.provenance.value}"

    @property
    def map_stratum(self) -> MapStratum:
        return MapStratum(self.data_kind, self.scenario)


@dataclass(frozen=True, slots=True)
class EvaluationMapRef:
    """Binds the network a case replays on to its split scope."""

    map_hash: str
    scope: SplitScope
    city_id: str

    def __post_init__(self) -> None:
        _hash(self.map_hash, "map_hash")
        _required(self.city_id, "city_id")
        object.__setattr__(self, "scope", exactly_one(self.scope, SplitScope, path="scope"))

    @classmethod
    def from_partition(cls, partition: SpatialPartition, *, map_hash: str) -> "EvaluationMapRef":
        """Reference the evaluation network derived from one split partition."""
        if not isinstance(partition, SpatialPartition):
            raise ResearchValidationError(
                "INVALID_SPLIT_PARTITION", "a SpatialPartition is required", path="partition"
            )
        return cls(map_hash=map_hash, scope=partition.scope, city_id=partition.network.city_id)

    @classmethod
    def from_cross_city(cls, reference: CrossCityMapRef) -> "EvaluationMapRef":
        if not isinstance(reference, CrossCityMapRef):
            raise ResearchValidationError(
                "INVALID_CROSS_CITY_REF", "a CrossCityMapRef is required", path="reference"
            )
        return cls(
            map_hash=reference.registered_map.network_hash,
            scope=SplitScope.CROSS_CITY,
            city_id=reference.city_id,
        )

    @property
    def provenance(self) -> EvaluationProvenance:
        return _PROVENANCE_BY_SCOPE[self.scope]


class MapProvenanceIndex:
    """Resolves a case's map hash to exactly one split scope, or refuses."""

    __slots__ = ("_refs",)

    def __init__(self, references: Iterable[EvaluationMapRef]) -> None:
        refs: dict[str, EvaluationMapRef] = {}
        for reference in references:
            if not isinstance(reference, EvaluationMapRef):
                raise ResearchValidationError(
                    "INVALID_MAP_REFERENCE", "provenance entries must be EvaluationMapRef values"
                )
            existing = refs.setdefault(reference.map_hash, reference)
            if existing != reference:
                raise ResearchValidationError(
                    "AMBIGUOUS_MAP_PROVENANCE",
                    "a map belongs to exactly one split scope",
                    path="map_hash",
                    expected=existing.scope.value,
                    actual=reference.scope.value,
                    details={"map_hash": reference.map_hash},
                )
        self._refs = refs

    def scope_for(self, map_hash: str) -> SplitScope:
        try:
            return self._refs[map_hash].scope
        except KeyError as exc:
            raise ResearchValidationError(
                "UNKNOWN_MAP_PROVENANCE",
                "the case's map has no declared split provenance",
                path="map_hash", actual=map_hash,
            ) from exc

    def provenance_for(self, map_hash: str) -> EvaluationProvenance:
        return _PROVENANCE_BY_SCOPE[self.scope_for(map_hash)]


def assign_stratum(case: EpisodeCase, index: MapProvenanceIndex) -> EvaluationStratumLabel:
    """Assign one case to exactly one data-kind x scenario x provenance layer."""
    return EvaluationStratumLabel(
        data_kind=case.data_kind,
        scenario=case.scenario,
        provenance=index.provenance_for(case.map_hash),
    )


def episode_handle(case: EpisodeCase, index: MapProvenanceIndex) -> DataHandle[SplitScope]:
    """Return the scoped handle for a case; test and cross-city scopes stay visible."""
    return DataHandle(index.scope_for(case.map_hash), HandleKind.EPISODE, case.case_hash)


class SealedCaseBook:
    """Sealed cases partitioned by evaluation stratum, with separate lanes.

    ``tuning_cases`` and ``tuning_data_view`` can only ever return in-region
    cases.  Held-out and cross-city cases are reachable through
    ``frozen_evaluation_cases``, which requires the hash of the policy already
    selected on validation -- a caller holding only a ``TuningDataView`` has no
    such hash and no path to those cases (Requirements 6.6-6.8).
    """

    __slots__ = ("_index", "_by_stratum")

    def __init__(self, *, cases: Iterable[EpisodeCase], provenance: MapProvenanceIndex) -> None:
        if not isinstance(provenance, MapProvenanceIndex):
            raise ResearchValidationError(
                "INVALID_PROVENANCE_INDEX", "a MapProvenanceIndex is required", path="provenance"
            )
        by_stratum: dict[EvaluationStratumLabel, list[EpisodeCase]] = {}
        seen: set[str] = set()
        for case in cases:
            if not isinstance(case, EpisodeCase):
                raise ResearchValidationError(
                    "INVALID_EPISODE_CASE", "a case book holds EpisodeCase values", path="cases"
                )
            if case.case_hash in seen:
                raise ResearchValidationError(
                    "DUPLICATE_EPISODE_CASE",
                    "a case is assigned to its stratum exactly once",
                    path="cases", actual=case.case_hash,
                )
            seen.add(case.case_hash)
            by_stratum.setdefault(assign_stratum(case, provenance), []).append(case)
        self._index = provenance
        self._by_stratum = {label: tuple(items) for label, items in by_stratum.items()}

    @property
    def strata(self) -> tuple[EvaluationStratumLabel, ...]:
        return tuple(sorted(self._by_stratum, key=lambda label: label.stratum_id))

    def cases_for(self, label: EvaluationStratumLabel) -> tuple[EpisodeCase, ...]:
        return self._by_stratum.get(label, ())

    def _by_provenance(self, provenance: EvaluationProvenance) -> tuple[EpisodeCase, ...]:
        return tuple(
            case
            for label in self.strata
            if label.provenance is provenance
            for case in self._by_stratum[label]
        )

    def tuning_cases(self) -> tuple[EpisodeCase, ...]:
        """Train-distribution cases only -- the sole lane a selection path may read."""
        return self._by_provenance(EvaluationProvenance.IN_REGION)

    def tuning_data_view(self) -> TuningDataView:
        """Build the leakage-safe capability object from in-region episode handles."""
        handles = [episode_handle(case, self._index) for case in self.tuning_cases()]
        return TuningDataView(
            train=tuple(handle for handle in handles if handle.scope is SplitScope.TRAIN),
            validation=tuple(handle for handle in handles if handle.scope is SplitScope.VALIDATION),
        )

    def frozen_evaluation_cases(
        self, provenance: EvaluationProvenance, *, frozen_policy_hash: str
    ) -> tuple[EpisodeCase, ...]:
        """Return held-out or cross-city cases for an already-frozen policy."""
        provenance = exactly_one(provenance, EvaluationProvenance, path="provenance")
        if provenance is EvaluationProvenance.IN_REGION:
            raise ResearchValidationError(
                "EVALUATION_LANE_MISMATCH",
                "in-region cases belong to the tuning lane, not the frozen-policy lane",
                path="provenance", expected=[
                    EvaluationProvenance.HELD_OUT.value, EvaluationProvenance.CROSS_CITY.value,
                ],
                actual=provenance.value,
            )
        _hash(frozen_policy_hash, "frozen_policy_hash")
        return self._by_provenance(provenance)


# ---------------------------------------------------------------------------
# Paired contract (Requirement 8.8, Correctness Property 9)
# ---------------------------------------------------------------------------

CASE_CONTRACT_FIELDS: tuple[str, ...] = (
    "map_hash",
    "data_kind",
    "scenario",
    *(f"police_{index}" for index in range(POLICE_COUNT)),
    "fugitive",
    "fugitive_rng",
    "environment",
    "termination",
)

RECORD_CONTRACT_FIELDS: tuple[str, ...] = ("training_seed",) + CASE_CONTRACT_FIELDS


def _contract_values(case: EpisodeCase) -> dict[str, Any]:
    values: dict[str, Any] = {
        "map_hash": case.map_hash,
        "data_kind": case.data_kind,
        "scenario": case.scenario,
        "fugitive": case.fugitive,
        "fugitive_rng": case.fugitive_rng,
        "environment": case.environment,
        "termination": case.termination,
    }
    for index, placement in enumerate(case.police):
        values[f"police_{index}"] = placement
    return values


@dataclass(frozen=True, slots=True)
class CaseFieldMismatch:
    """One named sealed field that differs between two compared records."""

    field: str
    left: str
    right: str

    def __post_init__(self) -> None:
        _required(self.field, "field")


@dataclass(frozen=True, slots=True)
class PairedStatus(PersistedModel):
    """Whether two policy records share the complete case contract."""

    paired: bool
    mismatches: tuple[CaseFieldMismatch, ...] = ()
    case_hash: str | None = None

    def __post_init__(self) -> None:
        mismatches = tuple(self.mismatches)
        if bool(self.paired) is bool(mismatches):
            raise ResearchValidationError(
                "INVALID_PAIRED_STATUS",
                "paired status and recorded mismatches must agree",
                path="paired", expected=not mismatches, actual=self.paired,
            )
        if self.paired:
            _hash(self.case_hash or "", "case_hash")
        elif self.case_hash is not None:
            raise ResearchValidationError(
                "INVALID_PAIRED_STATUS",
                "an unpaired comparison has no shared case identity",
                path="case_hash", expected=None, actual=self.case_hash,
            )
        object.__setattr__(self, "paired", bool(self.paired))
        object.__setattr__(self, "mismatches", mismatches)
        super(PairedStatus, self).__post_init__()

    @property
    def mismatched_fields(self) -> tuple[str, ...]:
        return tuple(item.field for item in self.mismatches)


class PairedContractError(ResearchValidationError):
    """Paired marking, tests and confidence intervals are refused."""

    def __init__(self, status: PairedStatus) -> None:
        super().__init__(
            "PAIRED_CONTRACT_MISMATCH",
            "compared policies do not share the sealed episode case contract",
            path="episode_case",
            expected=[],
            actual=list(status.mismatched_fields),
            details={
                "mismatches": [
                    {"field": item.field, "left": item.left, "right": item.right}
                    for item in status.mismatches
                ]
            },
        )
        self.status = status


def _rendered(value: Any) -> str:
    return canonical_json(value).decode("utf-8")


def compare_cases(left: EpisodeCase, right: EpisodeCase) -> PairedStatus:
    """Compare every sealed field, naming each one that differs.

    Correctness Property 9 requires paired status to be exactly equivalent to
    complete contract equality.  The final guard enforces that: a comparison that
    finds no named difference while the case hashes disagree means this function's
    field list is incomplete, and it fails closed instead of reporting a pair.
    """
    for name, case in (("left", left), ("right", right)):
        if not isinstance(case, EpisodeCase):
            raise ResearchValidationError(
                "INVALID_EPISODE_CASE", f"{name} must be an EpisodeCase", path=name
            )
    left_values, right_values = _contract_values(left), _contract_values(right)
    mismatches = tuple(
        CaseFieldMismatch(
            field=name,
            left=_rendered(left_values[name]),
            right=_rendered(right_values[name]),
        )
        for name in CASE_CONTRACT_FIELDS
        if left_values[name] != right_values[name]
    )
    if not mismatches and left.case_hash != right.case_hash:
        raise ResearchValidationError(
            "INCOMPLETE_CONTRACT_COMPARISON",
            "case hashes differ while every named field compares equal",
            path="case_hash", expected=left.case_hash, actual=right.case_hash,
        )
    if mismatches:
        return PairedStatus(paired=False, mismatches=mismatches)
    return PairedStatus(paired=True, case_hash=left.case_hash)


def paired_status(left: "PolicyEpisodeRecord", right: "PolicyEpisodeRecord") -> PairedStatus:
    """Compare two policies' records, including the training-seed correspondence."""
    for name, record in (("left", left), ("right", right)):
        if not isinstance(record, PolicyEpisodeRecord):
            raise ResearchValidationError(
                "INVALID_EPISODE_RECORD", f"{name} must be a PolicyEpisodeRecord", path=name
            )
    seed_mismatch = (
        ()
        if left.training_seed == right.training_seed
        else (
            CaseFieldMismatch(
                field="training_seed",
                left=_rendered(left.training_seed),
                right=_rendered(right.training_seed),
            ),
        )
    )
    case_status = compare_cases(left.case, right.case)
    mismatches = seed_mismatch + case_status.mismatches
    if mismatches:
        return PairedStatus(paired=False, mismatches=mismatches)
    return PairedStatus(paired=True, case_hash=left.case.case_hash)


def require_paired(left: "PolicyEpisodeRecord", right: "PolicyEpisodeRecord") -> str:
    """Return the shared case hash, or refuse the pair with its mismatch fields."""
    status = paired_status(left, right)
    if not status.paired:
        raise PairedContractError(status)
    return str(status.case_hash)


# ---------------------------------------------------------------------------
# Frozen policy replay (Requirement 8.1 determinism)
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class OfficerObservation:
    """One officer's decision-time snapshot, in the audited 28D field order."""

    officer_id: int
    step: int
    features: tuple[float, ...]
    action_mask: tuple[bool, ...]


@runtime_checkable
class FrozenPolicy(Protocol):
    """A policy under pure inference: no learning, no state mutation across calls.

    ``act`` must be a deterministic function of its observation so that replaying
    one sealed case twice yields bitwise-identical replay hashes.  A trained
    ``ResearchMaskedMAPPO`` satisfies this once wrapped with a fixed sampling rule
    (greedy, or a generator reseeded per call).
    """

    @property
    def policy_id(self) -> str: ...

    def act(self, observation: OfficerObservation) -> int: ...


@dataclass(frozen=True, slots=True)
class ReplayStep:
    """The post-step trace of one physical environment step."""

    step: int
    officer_actions: tuple[int, ...]
    police: tuple[VehiclePlacement, ...]
    fugitive: VehiclePlacement


@dataclass(frozen=True, slots=True)
class EpisodeReplayRecord(PersistedModel):
    """A deterministic replay of one sealed case under one frozen policy."""

    case_hash: str
    policy_id: str
    outcome: EpisodeOutcome
    physical_steps: int
    trajectory: tuple[ReplayStep, ...]

    def __post_init__(self) -> None:
        _hash(self.case_hash, "case_hash")
        _required(self.policy_id, "policy_id")
        object.__setattr__(self, "outcome", exactly_one(self.outcome, EpisodeOutcome, path="outcome"))
        _nonnegative_int(self.physical_steps, "physical_steps")
        object.__setattr__(self, "trajectory", tuple(self.trajectory))
        super(EpisodeReplayRecord, self).__post_init__()

    @property
    def replay_hash(self) -> str:
        return str(self.content_hash)


@dataclass(frozen=True, slots=True)
class PolicyEpisodeRecord(PersistedModel):
    """One policy's result on one sealed case, for paired comparison."""

    policy_id: str
    training_seed: int
    case: EpisodeCase
    replay: EpisodeReplayRecord

    def __post_init__(self) -> None:
        _required(self.policy_id, "policy_id")
        _nonnegative_int(self.training_seed, "training_seed")
        if not isinstance(self.case, EpisodeCase) or not isinstance(self.replay, EpisodeReplayRecord):
            raise ResearchValidationError(
                "INVALID_EPISODE_RECORD", "a record embeds one EpisodeCase and one replay", path="case"
            )
        if self.replay.case_hash != self.case.case_hash:
            raise ResearchValidationError(
                "REPLAY_CASE_MISMATCH", "the replay was produced from a different case",
                path="replay.case_hash", expected=self.case.case_hash, actual=self.replay.case_hash,
            )
        if self.replay.policy_id != self.policy_id:
            raise ResearchValidationError(
                "REPLAY_POLICY_MISMATCH", "the replay was produced by a different policy",
                path="replay.policy_id", expected=self.policy_id, actual=self.replay.policy_id,
            )
        super(PolicyEpisodeRecord, self).__post_init__()


def replay_episode(
    case: EpisodeCase, policy: FrozenPolicy, *, network: ModelNetwork
) -> EpisodeReplayRecord:
    """Re-run one sealed case under a frozen policy, deterministically."""
    if not isinstance(case, EpisodeCase):
        raise ResearchValidationError("INVALID_EPISODE_CASE", "case must be an EpisodeCase", path="case")
    if not isinstance(network, ModelNetwork):
        raise ResearchValidationError("INVALID_NETWORK", "network must be a ModelNetwork", path="network")
    actual_map_hash = content_hash(network)
    if actual_map_hash != case.map_hash:
        raise ResearchValidationError(
            "REPLAY_MAP_MISMATCH", "the supplied network is not the sealed map",
            path="network", expected=case.map_hash, actual=actual_map_hash,
        )
    policy_id = _required(getattr(policy, "policy_id", ""), "policy.policy_id")

    env = OSMRoadPursuitEnv(network, case.environment)
    env.reset(
        seed=case.fugitive_rng.seed,
        options={"police": list(case.police), "fugitive": case.fugitive},
    )
    adapter = Observation28DAdapter(
        network, clip_distance_m=REPLAY_CLIP_DISTANCE_M, near_radius_m=REPLAY_NEAR_RADIUS_M
    )
    evader = GoalEvader(
        network,
        rng=case.fugitive_rng.generator(),
        graph=_Graph(network),
        vision_range_m=REPLAY_VISION_RANGE_M,
    )

    trajectory: list[ReplayStep] = []
    while env.outcome is None and len(trajectory) < case.termination.timeout_step_limit:
        state = env.episode_state()
        masks = env.action_masks()
        actions: dict[str, Any] = {}
        officer_actions: list[int] = []
        for officer_id in range(POLICE_COUNT):
            mask = tuple(bool(value) for value in masks[f"police_{officer_id}"])
            features = tuple(
                float(value)
                for value in adapter.observe(
                    police_index=officer_id,
                    police=state.police,
                    fugitive=state.fugitive,
                    step=state.step,
                    max_steps=case.environment.max_steps,
                    incoming_heading=state.incoming_headings.get(f"police_{officer_id}"),
                )
            )
            action = policy.act(
                OfficerObservation(
                    officer_id=officer_id, step=state.step, features=features, action_mask=mask
                )
            )
            if isinstance(action, bool) or not isinstance(action, int):
                raise ResearchValidationError(
                    "INVALID_REPLAY_ACTION", "a frozen policy returns an integer action",
                    path=f"police_{officer_id}", actual=action,
                )
            if not 0 <= action < len(mask) or not mask[action]:
                raise ResearchValidationError(
                    "ILLEGAL_REPLAY_ACTION", "a frozen policy emitted an action outside its legal mask",
                    path=f"police_{officer_id}", expected=[
                        index for index, legal in enumerate(mask) if legal
                    ], actual=action,
                )
            actions[f"police_{officer_id}"] = action
            officer_actions.append(action)
        actions[FUGITIVE_ID] = evader.env_provider(
            network, tuple(placement_position(network, placement) for placement in state.police)
        )
        env.step(actions)
        after = env.episode_state()
        trajectory.append(
            ReplayStep(
                step=after.step,
                officer_actions=tuple(officer_actions),
                police=after.police,
                fugitive=after.fugitive,
            )
        )

    outcome = EpisodeOutcome(env.outcome.value) if env.outcome is not None else EpisodeOutcome.TIMEOUT
    if outcome not in case.termination.outcome_domain:
        raise ResearchValidationError(
            "OUTCOME_OUTSIDE_SCENARIO_DOMAIN",
            "the replay produced an outcome outside the sealed scenario domain",
            path="outcome",
            expected=sorted(item.value for item in case.termination.outcome_domain),
            actual=outcome.value,
        )
    return EpisodeReplayRecord(
        case_hash=case.case_hash,
        policy_id=policy_id,
        outcome=outcome,
        physical_steps=len(trajectory),
        trajectory=tuple(trajectory),
    )


def replay_to_episode_result(
    case: EpisodeCase, replay: EpisodeReplayRecord, *, episode_id: str, map_id: str
) -> EpisodeResult:
    """Produce the map registry's per-episode outcome record from a replay."""
    if replay.case_hash != case.case_hash:
        raise ResearchValidationError(
            "REPLAY_CASE_MISMATCH", "the replay was produced from a different case",
            path="replay.case_hash", expected=case.case_hash, actual=replay.case_hash,
        )
    return EpisodeResult(
        episode_id=episode_id,
        map_id=map_id,
        map_hash=case.map_hash,
        data_kind=case.data_kind,
        scenario=case.scenario,
        outcome=replay.outcome,
    )


__all__ = (
    "CASE_CONTRACT_FIELDS",
    "EVALUATION_SCHEMA_VERSION",
    "RECORD_CONTRACT_FIELDS",
    "REPLAY_CLIP_DISTANCE_M",
    "REPLAY_NEAR_RADIUS_M",
    "REPLAY_VISION_RANGE_M",
    "TUNING_SCOPES",
    "CaseFieldMismatch",
    "EpisodeCase",
    "EpisodeReplayRecord",
    "EvaluationMapRef",
    "EvaluationProvenance",
    "EvaluationStratumLabel",
    "FrozenPolicy",
    "FugitiveRNGSpec",
    "MapProvenanceIndex",
    "OfficerObservation",
    "PairedContractError",
    "PairedStatus",
    "PolicyEpisodeRecord",
    "ReplayStep",
    "SealedCaseBook",
    "TerminationConfig",
    "assign_stratum",
    "compare_cases",
    "episode_handle",
    "paired_status",
    "replay_episode",
    "replay_to_episode_result",
    "require_paired",
    "scenario_outcome_domain",
    "seal_episode_case",
)
