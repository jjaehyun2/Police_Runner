"""Property-based tests for exclusive, scenario-consistent map registration."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from string import ascii_lowercase, digits
from typing import Any

from hypothesis import given, settings, strategies as st
import pytest

from pursuit_evasion_rl.osm_demo.models import Intersection, ModelNetwork, Segment
from pursuit_evasion_rl.research.canonical import content_hash
from pursuit_evasion_rl.research.domain import DataKind, EpisodeOutcome, MapScenario
from pursuit_evasion_rl.research.errors import ResearchValidationError
from pursuit_evasion_rl.research.maps.registry import (
    ActualOSMProvenance,
    ActualOSMSpec,
    EpisodeResult,
    FixtureSpec,
    MapRegistry,
    SyntheticFixtureProvenance,
    resolve_episode_outcome,
)

# **Property 5: Map registration is exclusive and scenario-consistent**
# **Validates: Requirements 5.1–5.9, 19.2**

_PBT_SETTINGS = settings(max_examples=100, deadline=None)
_TOKEN_ALPHABET = ascii_lowercase + digits


@dataclass(frozen=True)
class MapCase:
    kind: DataKind
    scenario: MapScenario
    network: ModelNetwork
    edge_ref: str
    raw_hash: str
    fixture_seed: int


def _network(*, length_m: float, edge_ref: str, boundary: bool) -> ModelNetwork:
    return ModelNetwork(
        intersections=(
            Intersection(0, (0.0, 0.0), f"{edge_ref}/start", outgoing_segment_ids=(0,)),
            Intersection(1, (length_m, 0.0), f"{edge_ref}/end", incoming_segment_ids=(0,)),
        ),
        segments=(
            Segment(
                0, 0, 1, length_m, ((0.0, 0.0), (length_m, 0.0)),
                source_edge_refs=(edge_ref,), crosses_boundary=boundary,
            ),
        ),
    )


def _case(
    *, kind: DataKind, scenario: MapScenario, length_m: float,
    token: str, raw_hash: str, fixture_seed: int,
) -> MapCase:
    edge_ref = f"way/{token}/0"
    return MapCase(
        kind=kind,
        scenario=scenario,
        network=_network(
            length_m=length_m,
            edge_ref=edge_ref,
            boundary=scenario is MapScenario.BOUNDARY_ESCAPE,
        ),
        edge_ref=edge_ref,
        raw_hash=raw_hash,
        fixture_seed=fixture_seed,
    )


@st.composite
def map_cases(draw: st.DrawFn) -> MapCase:
    """Generate valid maps spanning both provenance kinds and both scenarios."""
    return _case(
        kind=draw(st.sampled_from(tuple(DataKind))),
        scenario=draw(st.sampled_from(tuple(MapScenario))),
        length_m=float(draw(st.integers(min_value=1, max_value=10_000))),
        token=draw(st.text(_TOKEN_ALPHABET, min_size=1, max_size=12)),
        raw_hash=draw(st.binary(min_size=32, max_size=32)).hex(),
        fixture_seed=draw(st.integers(min_value=-(2**31), max_value=2**31 - 1)),
    )


def _actual_provenance(case: MapCase, **changes: Any) -> ActualOSMProvenance:
    values: dict[str, Any] = {
        "query_geometry": {
            "type": "Polygon",
            "coordinates": [[[0, 0], [1, 0], [1, 1], [0, 0]]],
        },
        "source": "OpenStreetMap/Overpass",
        "acquired_at_utc": "2026-01-01T00:00:00Z",
        "raw_content_hash": case.raw_hash,
        "preprocessing_version": "coarsener-property-v1",
        "metric_crs": "EPSG:5179",
        "network_content_hash": content_hash(case.network),
        "source_edge_ids": (case.edge_ref,),
    }
    values.update(changes)
    return ActualOSMProvenance(**values)


def _fixture_provenance(case: MapCase, **changes: Any) -> SyntheticFixtureProvenance:
    values: dict[str, Any] = {
        "generator_path": "tests.research.properties.generated_line_network",
        "generator_version": "property-v1",
        "parameters": {"length_m": case.network.segments[0].length_m},
        "seed": case.fixture_seed,
        "fixture_content_hash": content_hash(case.network),
        "purpose": "Property 5 map registration validation",
    }
    values.update(changes)
    return SyntheticFixtureProvenance(**values)


def _spec(case: MapCase, map_id: str, **changes: Any) -> ActualOSMSpec | FixtureSpec:
    values: dict[str, Any] = {
        "map_id": map_id,
        "scenario": case.scenario,
        "network": case.network,
        "provenance": (
            _actual_provenance(case)
            if case.kind is DataKind.ACTUAL_OSM_MAP
            else _fixture_provenance(case)
        ),
        "data_kind": case.kind,
    }
    values.update(changes)
    return ActualOSMSpec(**values) if case.kind is DataKind.ACTUAL_OSM_MAP else FixtureSpec(**values)


def _register(registry: MapRegistry, case: MapCase, map_id: str):
    spec = _spec(case, map_id)
    if case.kind is DataKind.ACTUAL_OSM_MAP:
        return registry.register_actual(spec)
    return registry.register_fixture(spec)


def _episode(registered, episode_id: str, outcome: EpisodeOutcome) -> EpisodeResult:
    return EpisodeResult(
        episode_id=episode_id,
        map_id=registered.map_id,
        map_hash=registered.content_hash,
        data_kind=registered.data_kind,
        scenario=registered.scenario,
        outcome=outcome,
    )


@_PBT_SETTINGS
@given(case=map_cases())
def test_valid_registration_has_one_kind_scenario_and_fixed_domain(case: MapCase) -> None:
    registry = MapRegistry()
    registered = _register(registry, case, "generated-map")

    assert registered.data_kind is case.kind
    assert registered.scenario is case.scenario
    assert registered.stratum.data_kind is case.kind
    assert registered.stratum.scenario is case.scenario
    if case.kind is DataKind.ACTUAL_OSM_MAP:
        assert isinstance(registered.provenance, ActualOSMProvenance)
    else:
        assert isinstance(registered.provenance, SyntheticFixtureProvenance)

    if case.scenario is MapScenario.INTERIOR_CONTAINED:
        assert registered.boundary_arc_ids == ()
        assert registered.outcome_domain == {
            EpisodeOutcome.CAPTURE, EpisodeOutcome.TIMEOUT
        }
    else:
        assert registered.boundary_arc_ids == (0,)
        assert registered.outcome_domain == {
            EpisodeOutcome.CAPTURE, EpisodeOutcome.ESCAPE, EpisodeOutcome.TIMEOUT
        }


_CLASSIFICATION_DEFECTS = st.sampled_from(
    (
        ("data_kind", None),
        ("data_kind", [DataKind.ACTUAL_OSM_MAP, DataKind.SYNTHETIC_FIXTURE]),
        ("data_kind", "not-a-data-kind"),
        ("scenario", None),
        ("scenario", [MapScenario.INTERIOR_CONTAINED, MapScenario.BOUNDARY_ESCAPE]),
        ("scenario", "not-a-scenario"),
    )
)


@_PBT_SETTINGS
@given(case=map_cases(), defect=_CLASSIFICATION_DEFECTS)
def test_each_single_kind_or_scenario_classification_defect_is_rejected(
    case: MapCase, defect: tuple[str, Any]
) -> None:
    field_name, bad_value = defect
    with pytest.raises(ResearchValidationError) as raised:
        _spec(case, "classification-defect", **{field_name: bad_value})

    expected_codes = {
        None: "MISSING_CLASSIFICATION",
        "not-a-data-kind": "INVALID_CLASSIFICATION",
        "not-a-scenario": "INVALID_CLASSIFICATION",
    }
    if isinstance(bad_value, list):
        assert raised.value.code == "DUPLICATE_CLASSIFICATION"
    else:
        assert raised.value.code == expected_codes[bad_value]


@st.composite
def provenance_defects(draw: st.DrawFn, case: MapCase) -> tuple[str, Any]:
    """Choose one malformed or lineage-inconsistent field for the case's kind."""
    if case.kind is DataKind.ACTUAL_OSM_MAP:
        return draw(st.sampled_from((
            ("query_geometry", {}),
            ("source", ""),
            ("acquired_at_utc", "2026-01-01T00:00:00"),
            ("raw_content_hash", "bad"),
            ("preprocessing_version", ""),
            ("metric_crs", "EPSG:4326"),
            ("network_content_hash", "0" * 64),
            ("source_edge_ids", (f"{case.edge_ref}/different",)),
        )))
    return draw(st.sampled_from((
        ("generator_path", ""),
        ("generator_version", ""),
        ("parameters", None),
        ("seed", True),
        ("fixture_content_hash", "0" * 64),
        ("purpose", ""),
    )))


@_PBT_SETTINGS
@given(case=map_cases(), data=st.data())
def test_each_single_provenance_defect_closes_registration(
    case: MapCase, data: st.DataObject
) -> None:
    field_name, bad_value = data.draw(provenance_defects(case), label="provenance defect")
    registry = MapRegistry()

    with pytest.raises(ResearchValidationError):
        if case.kind is DataKind.ACTUAL_OSM_MAP:
            provenance = _actual_provenance(case, **{field_name: bad_value})
            registry.register_actual(
                ActualOSMSpec(
                    "provenance-defect", case.scenario, case.network, provenance
                )
            )
        else:
            provenance = _fixture_provenance(case, **{field_name: bad_value})
            registry.register_fixture(
                FixtureSpec(
                    "provenance-defect", case.scenario, case.network, provenance
                )
            )


@_PBT_SETTINGS
@given(case=map_cases())
def test_one_wrong_boundary_marker_always_rejects_registration(case: MapCase) -> None:
    wrong_network = _network(
        length_m=case.network.segments[0].length_m,
        edge_ref=case.edge_ref,
        boundary=case.scenario is MapScenario.INTERIOR_CONTAINED,
    )
    defective = MapCase(
        kind=case.kind,
        scenario=case.scenario,
        network=wrong_network,
        edge_ref=case.edge_ref,
        raw_hash=case.raw_hash,
        fixture_seed=case.fixture_seed,
    )

    with pytest.raises(ResearchValidationError) as raised:
        _register(MapRegistry(), defective, "boundary-defect")

    expected = (
        "INTERIOR_HAS_BOUNDARY_ARC"
        if case.scenario is MapScenario.INTERIOR_CONTAINED
        else "BOUNDARY_ARC_REQUIRED"
    )
    assert raised.value.code == expected


@_PBT_SETTINGS
@given(case=map_cases(), timeout=st.booleans())
def test_capture_precedence_and_out_of_domain_outcomes_are_enforced(
    case: MapCase, timeout: bool
) -> None:
    assert resolve_episode_outcome(
        capture=True, boundary_escape=True, timeout=timeout
    ) is EpisodeOutcome.CAPTURE

    with pytest.raises(ResearchValidationError) as raised:
        EpisodeResult(
            episode_id=f"invalid-{case.edge_ref}",
            map_id="interior-map",
            map_hash=content_hash(case.network),
            data_kind=case.kind,
            scenario=MapScenario.INTERIOR_CONTAINED,
            outcome=EpisodeOutcome.ESCAPE,
        )
    assert raised.value.code == "OUTCOME_OUTSIDE_SCENARIO_DOMAIN"


_AGGREGATION_DEFECTS = st.sampled_from(
    ("wrong_kind", "wrong_scenario", "wrong_map_hash", "duplicate_episode")
)


def _other_kind(kind: DataKind) -> DataKind:
    return (
        DataKind.SYNTHETIC_FIXTURE
        if kind is DataKind.ACTUAL_OSM_MAP
        else DataKind.ACTUAL_OSM_MAP
    )


def _other_scenario(scenario: MapScenario) -> MapScenario:
    return (
        MapScenario.BOUNDARY_ESCAPE
        if scenario is MapScenario.INTERIOR_CONTAINED
        else MapScenario.INTERIOR_CONTAINED
    )


def _different_hash(value: str) -> str:
    return ("0" if value[0] != "0" else "1") + value[1:]


@_PBT_SETTINGS
@given(case=map_cases(), defect=_AGGREGATION_DEFECTS)
def test_each_single_episode_assignment_defect_is_rejected_by_aggregation(
    case: MapCase, defect: str
) -> None:
    registry = MapRegistry()
    registered = _register(registry, case, "aggregate-defect-map")
    valid = _episode(registered, "episode", EpisodeOutcome.CAPTURE)

    if defect == "duplicate_episode":
        with pytest.raises(ResearchValidationError) as raised:
            registry.aggregate((valid, valid))
        assert raised.value.code == "DUPLICATE_IDENTIFIER"
        return

    changes = {
        "map_hash": valid.map_hash,
        "data_kind": valid.data_kind,
        "scenario": valid.scenario,
    }
    if defect == "wrong_kind":
        changes["data_kind"] = _other_kind(valid.data_kind)
    elif defect == "wrong_scenario":
        changes["scenario"] = _other_scenario(valid.scenario)
    else:
        changes["map_hash"] = _different_hash(valid.map_hash)

    defective = EpisodeResult(
        episode_id=valid.episode_id,
        map_id=valid.map_id,
        outcome=valid.outcome,
        **changes,
    )
    with pytest.raises(ResearchValidationError) as raised:
        registry.aggregate((defective,))
    assert raised.value.code == "STRATUM_ASSIGNMENT_MISMATCH"


@_PBT_SETTINGS
@given(
    base_length=st.integers(min_value=1, max_value=9_000),
    token=st.text(_TOKEN_ALPHABET, min_size=1, max_size=8),
    raw_hash=st.binary(min_size=32, max_size=32).map(bytes.hex),
    fixture_seed=st.integers(min_value=-(2**31), max_value=2**31 - 1),
    data=st.data(),
)
def test_aggregation_preserves_all_four_kind_by_scenario_strata(
    base_length: int,
    token: str,
    raw_hash: str,
    fixture_seed: int,
    data: st.DataObject,
) -> None:
    registry = MapRegistry()
    registered_maps = []
    expected: dict[tuple[DataKind, MapScenario], Counter[str]] = {}
    episodes: list[EpisodeResult] = []

    for index, (kind, scenario) in enumerate(
        (product for product in (
            (DataKind.ACTUAL_OSM_MAP, MapScenario.INTERIOR_CONTAINED),
            (DataKind.ACTUAL_OSM_MAP, MapScenario.BOUNDARY_ESCAPE),
            (DataKind.SYNTHETIC_FIXTURE, MapScenario.INTERIOR_CONTAINED),
            (DataKind.SYNTHETIC_FIXTURE, MapScenario.BOUNDARY_ESCAPE),
        ))
    ):
        case = _case(
            kind=kind,
            scenario=scenario,
            length_m=float(base_length + index + 1),
            token=f"{token}{index}",
            raw_hash=raw_hash,
            fixture_seed=fixture_seed + index,
        )
        registered = _register(registry, case, f"map-{index}")
        registered_maps.append(registered)
        allowed = tuple(registered.outcome_domain)
        outcomes = data.draw(
            st.lists(st.sampled_from(allowed), min_size=1, max_size=5),
            label=f"outcomes-{index}",
        )
        expected[(kind, scenario)] = Counter(outcome.value for outcome in outcomes)
        episodes.extend(
            _episode(registered, f"episode-{index}-{episode_index}", outcome)
            for episode_index, outcome in enumerate(outcomes)
        )

    aggregates = registry.aggregate(episodes)
    assert len(registered_maps) == 4
    assert len(aggregates) == 4
    assert {(item.data_kind, item.scenario) for item in aggregates} == set(expected)
    for aggregate in aggregates:
        key = (aggregate.data_kind, aggregate.scenario)
        observed = Counter({name: count for name, count in aggregate.outcome_counts.items() if count})
        assert observed == expected[key]
        assert aggregate.episode_count == sum(expected[key].values())
        if aggregate.scenario is MapScenario.INTERIOR_CONTAINED:
            assert "escape" not in aggregate.outcome_counts
