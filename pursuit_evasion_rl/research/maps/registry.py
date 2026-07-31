"""Typed, fail-closed registry for map provenance and scenario outcomes."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
import re
from types import MappingProxyType
from typing import Any, Iterable, Mapping

from pursuit_evasion_rl.osm_demo.models import ModelNetwork

from ..canonical import content_hash
from ..domain import (
    DataKind,
    EpisodeOutcome,
    MapScenario,
    PersistedModel,
    exactly_one,
)
from ..errors import ResearchValidationError

_SHA256 = re.compile(r"^[0-9a-f]{64}$")
INTERIOR_OUTCOMES = frozenset({EpisodeOutcome.CAPTURE, EpisodeOutcome.TIMEOUT})
BOUNDARY_OUTCOMES = frozenset(
    {EpisodeOutcome.CAPTURE, EpisodeOutcome.ESCAPE, EpisodeOutcome.TIMEOUT}
)


def _required(value: str, path: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ResearchValidationError(
            "MISSING_REQUIRED_FIELD", f"{path} must be non-empty", path=path
        )
    return value


def _sha256(value: str, path: str) -> str:
    _required(value, path)
    if _SHA256.fullmatch(value) is None:
        raise ResearchValidationError(
            "INVALID_CONTENT_HASH", f"{path} must be a lowercase SHA-256", path=path,
            expected="64 lowercase hexadecimal characters", actual=value,
        )
    return value

def _utc(value: str, path: str) -> str:
    _required(value, path)
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise ResearchValidationError(
            "INVALID_UTC_TIMESTAMP", f"{path} must be an ISO-8601 UTC timestamp",
            path=path, actual=value,
        ) from exc
    if parsed.tzinfo is None or parsed.utcoffset() != timezone.utc.utcoffset(parsed):
        raise ResearchValidationError(
            "INVALID_UTC_TIMESTAMP", f"{path} must include the UTC offset",
            path=path, actual=value,
        )
    return value


def _metric_crs(value: str) -> str:
    normalized = _required(value, "metric_crs").replace(" ", "").casefold()
    if any(token in normalized for token in ("epsg:4326", "crs84", "wgs84")):
        raise ResearchValidationError(
            "NON_METRIC_CRS", "metric_crs must be a projected metric CRS",
            path="metric_crs", actual=value,
        )
    return value


def _identifiers(values: Iterable[str], path: str) -> tuple[str, ...]:
    normalized = tuple(str(item).strip() for item in values)
    if not normalized or any(not item for item in normalized):
        raise ResearchValidationError(
            "MISSING_REQUIRED_FIELD", f"{path} must contain non-empty identifiers", path=path
        )
    if len(set(normalized)) != len(normalized):
        raise ResearchValidationError(
            "DUPLICATE_IDENTIFIER", f"{path} must be unique", path=path
        )
    return tuple(sorted(normalized))


@dataclass(frozen=True, slots=True)
class ActualOSMProvenance(PersistedModel):
    query_geometry: Any
    source: str
    acquired_at_utc: str
    raw_content_hash: str
    preprocessing_version: str
    metric_crs: str
    network_content_hash: str
    source_edge_ids: tuple[str, ...]
    query_geometry_hash: str = field(init=False)
    source_edge_hash: str = field(init=False)

    def __post_init__(self) -> None:
        if self.query_geometry is None or self.query_geometry == "" or self.query_geometry == {}:
            raise ResearchValidationError(
                "MISSING_REQUIRED_FIELD", "query_geometry must be recorded", path="query_geometry"
            )
        _required(self.source, "source")
        _utc(self.acquired_at_utc, "acquired_at_utc")
        _sha256(self.raw_content_hash, "raw_content_hash")
        _required(self.preprocessing_version, "preprocessing_version")
        _metric_crs(self.metric_crs)
        _sha256(self.network_content_hash, "network_content_hash")
        edges = _identifiers(self.source_edge_ids, "source_edge_ids")
        object.__setattr__(self, "source_edge_ids", edges)
        object.__setattr__(self, "query_geometry_hash", content_hash(self.query_geometry))
        object.__setattr__(self, "source_edge_hash", content_hash(edges))
        super(ActualOSMProvenance, self).__post_init__()


@dataclass(frozen=True, slots=True)
class SyntheticFixtureProvenance(PersistedModel):
    generator_path: str
    generator_version: str
    parameters: Mapping[str, Any]
    seed: int
    fixture_content_hash: str
    purpose: str

    def __post_init__(self) -> None:
        _required(self.generator_path, "generator_path")
        _required(self.generator_version, "generator_version")
        if not isinstance(self.parameters, Mapping):
            raise ResearchValidationError(
                "INVALID_PROVENANCE", "parameters must be a mapping",
                path="parameters", actual=type(self.parameters).__name__,
            )
        if isinstance(self.seed, bool) or not isinstance(self.seed, int):
            raise ResearchValidationError(
                "INVALID_PROVENANCE", "seed must be an integer", path="seed", actual=self.seed
            )
        _sha256(self.fixture_content_hash, "fixture_content_hash")
        _required(self.purpose, "purpose")
        object.__setattr__(self, "parameters", MappingProxyType(dict(self.parameters)))
        super(SyntheticFixtureProvenance, self).__post_init__()


@dataclass(frozen=True, slots=True)
class MapStratum:
    data_kind: DataKind
    scenario: MapScenario

    def __post_init__(self) -> None:
        object.__setattr__(self, "data_kind", exactly_one(self.data_kind, DataKind, path="data_kind"))
        object.__setattr__(self, "scenario", exactly_one(self.scenario, MapScenario, path="scenario"))


@dataclass(frozen=True, slots=True)
class ActualOSMSpec:
    map_id: str
    scenario: MapScenario
    network: ModelNetwork
    provenance: ActualOSMProvenance
    data_kind: DataKind = DataKind.ACTUAL_OSM_MAP

    def __post_init__(self) -> None:
        _required(self.map_id, "map_id")
        kind = exactly_one(self.data_kind, DataKind, path="data_kind")
        scenario = exactly_one(self.scenario, MapScenario, path="scenario")
        if kind is not DataKind.ACTUAL_OSM_MAP or not isinstance(self.provenance, ActualOSMProvenance):
            raise ResearchValidationError(
                "PROVENANCE_MISMATCH", "Actual OSM maps require ActualOSMProvenance",
                path="provenance", actual=kind.value,
            )
        if not isinstance(self.network, ModelNetwork):
            raise ResearchValidationError("INVALID_NETWORK", "network must be a ModelNetwork", path="network")
        object.__setattr__(self, "data_kind", kind)
        object.__setattr__(self, "scenario", scenario)


@dataclass(frozen=True, slots=True)
class FixtureSpec:
    map_id: str
    scenario: MapScenario
    network: ModelNetwork
    provenance: SyntheticFixtureProvenance
    data_kind: DataKind = DataKind.SYNTHETIC_FIXTURE

    def __post_init__(self) -> None:
        _required(self.map_id, "map_id")
        kind = exactly_one(self.data_kind, DataKind, path="data_kind")
        scenario = exactly_one(self.scenario, MapScenario, path="scenario")
        if kind is not DataKind.SYNTHETIC_FIXTURE or not isinstance(
            self.provenance, SyntheticFixtureProvenance
        ):
            raise ResearchValidationError(
                "PROVENANCE_MISMATCH", "Fixtures require SyntheticFixtureProvenance",
                path="provenance", actual=kind.value,
            )
        if not isinstance(self.network, ModelNetwork):
            raise ResearchValidationError("INVALID_NETWORK", "network must be a ModelNetwork", path="network")
        object.__setattr__(self, "data_kind", kind)
        object.__setattr__(self, "scenario", scenario)


@dataclass(frozen=True, slots=True)
class RegisteredMap(PersistedModel):
    map_id: str
    data_kind: DataKind
    scenario: MapScenario
    network_hash: str
    boundary_arc_ids: tuple[int, ...]
    outcome_domain: frozenset[EpisodeOutcome]
    provenance: ActualOSMProvenance | SyntheticFixtureProvenance

    def __post_init__(self) -> None:
        _required(self.map_id, "map_id")
        object.__setattr__(self, "data_kind", exactly_one(self.data_kind, DataKind, path="data_kind"))
        object.__setattr__(self, "scenario", exactly_one(self.scenario, MapScenario, path="scenario"))
        _sha256(self.network_hash, "network_hash")
        arcs = tuple(self.boundary_arc_ids)
        if len(arcs) != len(set(arcs)):
            raise ResearchValidationError(
                "DUPLICATE_IDENTIFIER", "boundary_arc_ids must be unique", path="boundary_arc_ids"
            )
        object.__setattr__(self, "boundary_arc_ids", arcs)
        expected = _scenario_domain(self.scenario)
        supplied = frozenset(exactly_one(item, EpisodeOutcome, path="outcome_domain") for item in self.outcome_domain)
        if supplied != expected:
            raise ResearchValidationError(
                "OUTCOME_DOMAIN_MISMATCH", "scenario outcome domain is fixed",
                path="outcome_domain", expected=sorted(item.value for item in expected),
                actual=sorted(item.value for item in supplied),
            )
        object.__setattr__(self, "outcome_domain", supplied)
        _validate_boundary_markers(self.scenario, arcs)
        super(RegisteredMap, self).__post_init__()

    @property
    def stratum(self) -> MapStratum:
        return MapStratum(self.data_kind, self.scenario)


@dataclass(frozen=True, slots=True)
class EpisodeResult(PersistedModel):
    episode_id: str
    map_id: str
    map_hash: str
    data_kind: DataKind
    scenario: MapScenario
    outcome: EpisodeOutcome

    def __post_init__(self) -> None:
        _required(self.episode_id, "episode_id")
        _required(self.map_id, "map_id")
        _sha256(self.map_hash, "map_hash")
        object.__setattr__(self, "data_kind", exactly_one(self.data_kind, DataKind, path="data_kind"))
        object.__setattr__(self, "scenario", exactly_one(self.scenario, MapScenario, path="scenario"))
        object.__setattr__(self, "outcome", exactly_one(self.outcome, EpisodeOutcome, path="outcome"))
        if self.outcome not in _scenario_domain(self.scenario):
            raise ResearchValidationError(
                "OUTCOME_OUTSIDE_SCENARIO_DOMAIN", "outcome is not allowed for scenario",
                path="outcome", expected=sorted(item.value for item in _scenario_domain(self.scenario)),
                actual=self.outcome.value,
            )
        super(EpisodeResult, self).__post_init__()

    @property
    def stratum(self) -> MapStratum:
        return MapStratum(self.data_kind, self.scenario)


@dataclass(frozen=True, slots=True)
class StratumAggregate(PersistedModel):
    data_kind: DataKind
    scenario: MapScenario
    outcome_counts: Mapping[str, int]
    episode_count: int

    def __post_init__(self) -> None:
        kind = exactly_one(self.data_kind, DataKind, path="data_kind")
        scenario = exactly_one(self.scenario, MapScenario, path="scenario")
        expected_keys = {item.value for item in _scenario_domain(scenario)}
        counts = dict(self.outcome_counts)
        if set(counts) != expected_keys or any(
            isinstance(value, bool) or not isinstance(value, int) or value < 0
            for value in counts.values()
        ):
            raise ResearchValidationError(
                "INVALID_AGGREGATE", "aggregate must contain nonnegative counts for exactly its domain",
                path="outcome_counts", expected=sorted(expected_keys), actual=counts,
            )
        if self.episode_count != sum(counts.values()):
            raise ResearchValidationError(
                "INVALID_AGGREGATE", "episode_count must equal outcome counts",
                path="episode_count", expected=sum(counts.values()), actual=self.episode_count,
            )
        object.__setattr__(self, "data_kind", kind)
        object.__setattr__(self, "scenario", scenario)
        object.__setattr__(self, "outcome_counts", MappingProxyType(counts))
        super(StratumAggregate, self).__post_init__()


def _scenario_domain(scenario: MapScenario) -> frozenset[EpisodeOutcome]:
    return INTERIOR_OUTCOMES if scenario is MapScenario.INTERIOR_CONTAINED else BOUNDARY_OUTCOMES


def _validate_boundary_markers(scenario: MapScenario, boundary_arc_ids: tuple[int, ...]) -> None:
    if scenario is MapScenario.INTERIOR_CONTAINED and boundary_arc_ids:
        raise ResearchValidationError(
            "INTERIOR_HAS_BOUNDARY_ARC", "Interior maps must have zero boundary arcs",
            path="boundary_arc_ids", expected=0, actual=len(boundary_arc_ids),
        )
    if scenario is MapScenario.BOUNDARY_ESCAPE and not boundary_arc_ids:
        raise ResearchValidationError(
            "BOUNDARY_ARC_REQUIRED", "Boundary maps require at least one boundary arc",
            path="boundary_arc_ids", expected=">=1", actual=0,
        )


def resolve_episode_outcome(
    *, capture: bool, boundary_escape: bool, timeout: bool
) -> EpisodeOutcome:
    """Resolve a physical step, with capture taking precedence over escape."""
    if capture:
        return EpisodeOutcome.CAPTURE
    if boundary_escape:
        return EpisodeOutcome.ESCAPE
    if timeout:
        return EpisodeOutcome.TIMEOUT
    raise ResearchValidationError(
        "MISSING_TERMINAL_OUTCOME", "at least one terminal condition must be true"
    )


class MapRegistry:
    """In-memory registration and map/outcome gate with content-addressed records."""

    def __init__(self) -> None:
        self._maps: dict[str, RegisteredMap] = {}

    def register_actual(self, spec: ActualOSMSpec) -> RegisteredMap:
        if not isinstance(spec, ActualOSMSpec):
            raise ResearchValidationError(
                "PROVENANCE_MISMATCH", "register_actual requires ActualOSMSpec", path="spec"
            )
        network_hash = content_hash(spec.network)
        if spec.provenance.network_content_hash != network_hash:
            raise ResearchValidationError(
                "PROVENANCE_MISMATCH", "network hash does not match the registered network",
                path="provenance.network_content_hash", expected=network_hash,
                actual=spec.provenance.network_content_hash,
            )
        network_edges = {
            edge for segment in spec.network.segments for edge in segment.source_edge_refs
        }
        if network_edges != set(spec.provenance.source_edge_ids):
            raise ResearchValidationError(
                "PROVENANCE_MISMATCH", "source edge identifiers do not match network lineage",
                path="provenance.source_edge_ids", expected=sorted(network_edges),
                actual=list(spec.provenance.source_edge_ids),
            )
        return self._register(
            map_id=spec.map_id,
            data_kind=spec.data_kind,
            scenario=spec.scenario,
            network=spec.network,
            network_hash=network_hash,
            provenance=spec.provenance,
        )

    def register_fixture(self, spec: FixtureSpec) -> RegisteredMap:
        if not isinstance(spec, FixtureSpec):
            raise ResearchValidationError(
                "PROVENANCE_MISMATCH", "register_fixture requires FixtureSpec", path="spec"
            )
        network_hash = content_hash(spec.network)
        if spec.provenance.fixture_content_hash != network_hash:
            raise ResearchValidationError(
                "PROVENANCE_MISMATCH", "fixture hash does not match the registered network",
                path="provenance.fixture_content_hash", expected=network_hash,
                actual=spec.provenance.fixture_content_hash,
            )
        return self._register(
            map_id=spec.map_id,
            data_kind=spec.data_kind,
            scenario=spec.scenario,
            network=spec.network,
            network_hash=network_hash,
            provenance=spec.provenance,
        )

    def _register(
        self, *, map_id: str, data_kind: DataKind, scenario: MapScenario,
        network: ModelNetwork, network_hash: str,
        provenance: ActualOSMProvenance | SyntheticFixtureProvenance,
    ) -> RegisteredMap:
        if map_id in self._maps:
            raise ResearchValidationError(
                "DUPLICATE_IDENTIFIER", "map_id is already registered", path="map_id", actual=map_id
            )
        boundary_arc_ids = tuple(segment.id for segment in network.segments if segment.crosses_boundary)
        _validate_boundary_markers(scenario, boundary_arc_ids)
        record = RegisteredMap(
            map_id=map_id, data_kind=data_kind, scenario=scenario,
            network_hash=network_hash, boundary_arc_ids=boundary_arc_ids,
            outcome_domain=_scenario_domain(scenario), provenance=provenance,
        )
        self._maps[map_id] = record
        return record

    def validate_episode(self, result: EpisodeResult) -> EpisodeResult:
        record = self._maps.get(result.map_id)
        if record is None:
            raise ResearchValidationError(
                "UNKNOWN_MAP", "episode references an unregistered map", path="map_id", actual=result.map_id
            )
        mismatches = {
            "map_hash": (record.content_hash, result.map_hash),
            "data_kind": (record.data_kind, result.data_kind),
            "scenario": (record.scenario, result.scenario),
        }
        for field_name, (expected, actual) in mismatches.items():
            if expected != actual:
                raise ResearchValidationError(
                    "STRATUM_ASSIGNMENT_MISMATCH",
                    "episode must be assigned exactly once to its map's kind×scenario stratum",
                    path=field_name,
                    expected=expected.value if hasattr(expected, "value") else expected,
                    actual=actual.value if hasattr(actual, "value") else actual,
                )
        if result.outcome not in record.outcome_domain:
            raise ResearchValidationError(
                "OUTCOME_OUTSIDE_SCENARIO_DOMAIN", "outcome is not allowed for registered map",
                path="outcome", expected=sorted(item.value for item in record.outcome_domain),
                actual=result.outcome.value,
            )
        return result

    def aggregate(self, results: Iterable[EpisodeResult]) -> tuple[StratumAggregate, ...]:
        grouped: dict[MapStratum, dict[str, int]] = {}
        seen: set[str] = set()
        for result in results:
            self.validate_episode(result)
            if result.episode_id in seen:
                raise ResearchValidationError(
                    "DUPLICATE_IDENTIFIER", "episode may be aggregated exactly once",
                    path="episode_id", actual=result.episode_id,
                )
            seen.add(result.episode_id)
            counts = grouped.setdefault(
                result.stratum,
                {outcome.value: 0 for outcome in _scenario_domain(result.scenario)},
            )
            counts[result.outcome.value] += 1
        return tuple(
            StratumAggregate(
                data_kind=stratum.data_kind, scenario=stratum.scenario,
                outcome_counts=counts, episode_count=sum(counts.values()),
            )
            for stratum, counts in sorted(
                grouped.items(), key=lambda item: (item[0].data_kind.value, item[0].scenario.value)
            )
        )

    def aggregate_single_stratum(
        self, results: Iterable[EpisodeResult]
    ) -> StratumAggregate:
        """Build one report table, rejecting cross-kind or cross-scenario input."""
        aggregates = self.aggregate(results)
        if not aggregates:
            raise ResearchValidationError(
                "EMPTY_AGGREGATE", "a single-stratum aggregate requires at least one episode",
                path="results",
            )
        if len(aggregates) != 1:
            strata = [
                {
                    "data_kind": aggregate.data_kind.value,
                    "scenario": aggregate.scenario.value,
                }
                for aggregate in aggregates
            ]
            raise ResearchValidationError(
                "MIXED_AGGREGATE_STRATA",
                "Actual OSM, fixture, Interior, and Boundary results require separate tables",
                path="results",
                expected="exactly one data-kind×scenario stratum",
                actual=strata,
            )
        return aggregates[0]

    def get(self, map_id: str) -> RegisteredMap:
        try:
            return self._maps[map_id]
        except KeyError as exc:
            raise ResearchValidationError(
                "UNKNOWN_MAP", "map is not registered", path="map_id", actual=map_id
            ) from exc


__all__ = (
    "ActualOSMProvenance",
    "ActualOSMSpec",
    "BOUNDARY_OUTCOMES",
    "EpisodeResult",
    "FixtureSpec",
    "INTERIOR_OUTCOMES",
    "MapRegistry",
    "MapStratum",
    "RegisteredMap",
    "StratumAggregate",
    "SyntheticFixtureProvenance",
    "resolve_episode_outcome",
)
