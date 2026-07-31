"""Task 2.1 unit tests for map provenance and scenario registration."""

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


def _network(*, boundary: bool = False, edge_ref: str = "way/1/0") -> ModelNetwork:
    return ModelNetwork(
        intersections=(
            Intersection(
                id=0, position_xy=(0.0, 0.0), source_signature="node/0",
                outgoing_segment_ids=(0,), incoming_segment_ids=(),
            ),
            Intersection(
                id=1, position_xy=(10.0, 0.0), source_signature="node/1",
                outgoing_segment_ids=(), incoming_segment_ids=(0,),
            ),
        ),
        segments=(
            Segment(
                id=0, start_id=0, end_id=1, length_m=10.0,
                geometry_xy=((0.0, 0.0), (10.0, 0.0)),
                source_edge_refs=(edge_ref,), crosses_boundary=boundary,
            ),
        ),
    )


def _actual_provenance(network: ModelNetwork, **changes) -> ActualOSMProvenance:
    values = {
        "query_geometry": {"type": "Polygon", "coordinates": [[[0, 0], [1, 0], [1, 1], [0, 0]]]},
        "source": "OpenStreetMap/Overpass",
        "acquired_at_utc": "2026-01-01T00:00:00Z",
        "raw_content_hash": "a" * 64,
        "preprocessing_version": "coarsener-1.0",
        "metric_crs": "EPSG:5179",
        "network_content_hash": content_hash(network),
        "source_edge_ids": ("way/1/0",),
    }
    values.update(changes)
    return ActualOSMProvenance(**values)


def _fixture_provenance(network: ModelNetwork, **changes) -> SyntheticFixtureProvenance:
    values = {
        "generator_path": "tests.fixtures.make_line_network",
        "generator_version": "1.0",
        "parameters": {"length_m": 10.0},
        "seed": 7,
        "fixture_content_hash": content_hash(network),
        "purpose": "map registry contract test",
    }
    values.update(changes)
    return SyntheticFixtureProvenance(**values)


def test_registers_actual_interior_and_fixture_boundary_with_fixed_domains():
    interior = _network()
    boundary = _network(boundary=True)
    registry = MapRegistry()
    actual = registry.register_actual(
        ActualOSMSpec(
            map_id="actual-interior", scenario=MapScenario.INTERIOR_CONTAINED,
            network=interior, provenance=_actual_provenance(interior),
        )
    )
    fixture = registry.register_fixture(
        FixtureSpec(
            map_id="fixture-boundary", scenario=MapScenario.BOUNDARY_ESCAPE,
            network=boundary, provenance=_fixture_provenance(boundary),
        )
    )
    assert actual.data_kind is DataKind.ACTUAL_OSM_MAP
    assert actual.boundary_arc_ids == ()
    assert actual.outcome_domain == {EpisodeOutcome.CAPTURE, EpisodeOutcome.TIMEOUT}
    assert fixture.data_kind is DataKind.SYNTHETIC_FIXTURE
    assert fixture.boundary_arc_ids == (0,)
    assert fixture.outcome_domain == {
        EpisodeOutcome.CAPTURE, EpisodeOutcome.ESCAPE, EpisodeOutcome.TIMEOUT
    }
    assert len(actual.provenance.query_geometry_hash) == 64
    assert len(actual.provenance.source_edge_hash) == 64


@pytest.mark.parametrize(
    ("field_name", "bad_value"),
    [
        ("query_geometry", {}),
        ("source", ""),
        ("acquired_at_utc", "2026-01-01T00:00:00"),
        ("raw_content_hash", "bad"),
        ("preprocessing_version", ""),
        ("metric_crs", "EPSG:4326"),
        ("network_content_hash", "bad"),
        ("source_edge_ids", ()),
    ],
)
def test_actual_osm_rejects_each_single_provenance_defect(field_name, bad_value):
    network = _network()
    with pytest.raises(ResearchValidationError):
        _actual_provenance(network, **{field_name: bad_value})


@pytest.mark.parametrize(
    ("field_name", "bad_value"),
    [
        ("generator_path", ""),
        ("generator_version", ""),
        ("parameters", None),
        ("seed", True),
        ("fixture_content_hash", "bad"),
        ("purpose", ""),
    ],
)
def test_fixture_rejects_each_single_provenance_defect(field_name, bad_value):
    network = _network()
    with pytest.raises(ResearchValidationError):
        _fixture_provenance(network, **{field_name: bad_value})


def test_registration_rejects_hash_and_source_edge_provenance_mismatch():
    network = _network()
    with pytest.raises(ResearchValidationError) as hash_error:
        MapRegistry().register_actual(
            ActualOSMSpec(
                map_id="bad-hash", scenario=MapScenario.INTERIOR_CONTAINED,
                network=network,
                provenance=_actual_provenance(network, network_content_hash="b" * 64),
            )
        )
    assert hash_error.value.code == "PROVENANCE_MISMATCH"

    with pytest.raises(ResearchValidationError) as edge_error:
        MapRegistry().register_actual(
            ActualOSMSpec(
                map_id="bad-edge", scenario=MapScenario.INTERIOR_CONTAINED,
                network=network,
                provenance=_actual_provenance(network, source_edge_ids=("way/other",)),
            )
        )
    assert edge_error.value.code == "PROVENANCE_MISMATCH"


def test_data_kind_is_exactly_one_and_must_match_provenance_type():
    network = _network()
    with pytest.raises(ResearchValidationError) as duplicate:
        ActualOSMSpec(
            map_id="ambiguous", scenario=MapScenario.INTERIOR_CONTAINED,
            network=network, provenance=_actual_provenance(network),
            data_kind=[DataKind.ACTUAL_OSM_MAP, DataKind.SYNTHETIC_FIXTURE],
        )
    assert duplicate.value.code == "DUPLICATE_CLASSIFICATION"

    with pytest.raises(ResearchValidationError) as mismatch:
        ActualOSMSpec(
            map_id="wrong-kind", scenario=MapScenario.INTERIOR_CONTAINED,
            network=network, provenance=_actual_provenance(network),
            data_kind=DataKind.SYNTHETIC_FIXTURE,
        )
    assert mismatch.value.code == "PROVENANCE_MISMATCH"


def test_scenario_rejects_a_single_wrong_boundary_marker():
    with pytest.raises(ResearchValidationError) as interior_error:
        network = _network(boundary=True)
        MapRegistry().register_fixture(
            FixtureSpec(
                map_id="interior-with-boundary", scenario=MapScenario.INTERIOR_CONTAINED,
                network=network, provenance=_fixture_provenance(network),
            )
        )
    assert interior_error.value.code == "INTERIOR_HAS_BOUNDARY_ARC"

    with pytest.raises(ResearchValidationError) as boundary_error:
        network = _network(boundary=False)
        MapRegistry().register_fixture(
            FixtureSpec(
                map_id="boundary-without-arc", scenario=MapScenario.BOUNDARY_ESCAPE,
                network=network, provenance=_fixture_provenance(network),
            )
        )
    assert boundary_error.value.code == "BOUNDARY_ARC_REQUIRED"


def test_capture_wins_a_simultaneous_capture_escape_step():
    assert resolve_episode_outcome(
        capture=True, boundary_escape=True, timeout=False
    ) is EpisodeOutcome.CAPTURE
    assert resolve_episode_outcome(
        capture=False, boundary_escape=True, timeout=False
    ) is EpisodeOutcome.ESCAPE
    with pytest.raises(ResearchValidationError) as missing:
        resolve_episode_outcome(capture=False, boundary_escape=False, timeout=False)
    assert missing.value.code == "MISSING_TERMINAL_OUTCOME"


def test_interior_escape_outcome_is_rejected():
    with pytest.raises(ResearchValidationError) as raised:
        EpisodeResult(
            episode_id="episode-bad", map_id="map", map_hash="a" * 64,
            data_kind=DataKind.SYNTHETIC_FIXTURE,
            scenario=MapScenario.INTERIOR_CONTAINED,
            outcome=EpisodeOutcome.ESCAPE,
        )
    assert raised.value.code == "OUTCOME_OUTSIDE_SCENARIO_DOMAIN"


def _episode(registered, episode_id: str, outcome: EpisodeOutcome) -> EpisodeResult:
    return EpisodeResult(
        episode_id=episode_id, map_id=registered.map_id, map_hash=registered.content_hash,
        data_kind=registered.data_kind, scenario=registered.scenario, outcome=outcome,
    )


def test_aggregate_keeps_kind_by_scenario_strata_separate():
    actual_network = _network()
    fixture_network = _network(boundary=True)
    registry = MapRegistry()
    actual = registry.register_actual(
        ActualOSMSpec(
            map_id="actual", scenario=MapScenario.INTERIOR_CONTAINED,
            network=actual_network, provenance=_actual_provenance(actual_network),
        )
    )
    fixture = registry.register_fixture(
        FixtureSpec(
            map_id="fixture", scenario=MapScenario.BOUNDARY_ESCAPE,
            network=fixture_network, provenance=_fixture_provenance(fixture_network),
        )
    )
    aggregates = registry.aggregate(
        (
            _episode(actual, "a-capture", EpisodeOutcome.CAPTURE),
            _episode(actual, "a-timeout", EpisodeOutcome.TIMEOUT),
            _episode(fixture, "f-escape", EpisodeOutcome.ESCAPE),
        )
    )
    assert len(aggregates) == 2
    by_kind = {aggregate.data_kind: aggregate for aggregate in aggregates}
    assert by_kind[DataKind.ACTUAL_OSM_MAP].outcome_counts == {
        "capture": 1, "timeout": 1
    }
    assert by_kind[DataKind.SYNTHETIC_FIXTURE].outcome_counts == {
        "capture": 0, "escape": 1, "timeout": 0
    }


def test_aggregate_rejects_wrong_stratum_and_duplicate_episode_assignment():
    network = _network()
    registry = MapRegistry()
    registered = registry.register_fixture(
        FixtureSpec(
            map_id="fixture", scenario=MapScenario.INTERIOR_CONTAINED,
            network=network, provenance=_fixture_provenance(network),
        )
    )
    valid = _episode(registered, "wrong", EpisodeOutcome.CAPTURE)
    wrong = EpisodeResult(
        episode_id=valid.episode_id,
        map_id=valid.map_id,
        map_hash=valid.map_hash,
        data_kind=DataKind.ACTUAL_OSM_MAP,
        scenario=valid.scenario,
        outcome=valid.outcome,
    )
    with pytest.raises(ResearchValidationError) as mismatch:
        registry.aggregate((wrong,))
    assert mismatch.value.code == "STRATUM_ASSIGNMENT_MISMATCH"

    episode = _episode(registered, "duplicate", EpisodeOutcome.TIMEOUT)
    with pytest.raises(ResearchValidationError) as duplicate:
        registry.aggregate((episode, episode))
    assert duplicate.value.code == "DUPLICATE_IDENTIFIER"
