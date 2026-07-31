"""Focused unit tests for versioned OSM demo domain contracts."""

from dataclasses import FrozenInstanceError
import json
from pathlib import Path

import pytest

from pursuit_evasion_rl.osm_demo.models import (
    DOMAIN_SCHEMA_VERSION,
    BoundedArea,
    DomainValidationError,
    Intersection,
    ModelNetwork,
    Segment,
    VehiclePlacement,
    validate_schema_version,
    validate_vehicle_placements,
)
from pursuit_evasion_rl.osm_demo.presets import DAEJEON_DRIVE_PRESET, NetworkPreset


def make_ring_network(*, virtual_segment: int | None = None) -> ModelNetwork:
    intersections = tuple(
        Intersection(
            id=index,
            position_xy=(float(index), 0.0),
            source_signature=f"node-{index}",
            outgoing_segment_ids=(index,),
            incoming_segment_ids=((index - 1) % 7,),
        )
        for index in range(7)
    )
    segments = tuple(
        Segment(
            id=index,
            start_id=index,
            end_id=(index + 1) % 7,
            length_m=1.0,
            geometry_xy=((float(index), 0.0), (float((index + 1) % 7), 0.0)),
            virtual=index == virtual_segment,
        )
        for index in range(7)
    )
    return ModelNetwork(intersections=intersections, segments=segments)


@pytest.mark.parametrize(
    ("changes", "code"),
    [
        ({"north": float("nan")}, "NON_FINITE_NUMBER"),
        ({"north": 36.0, "south": 36.1}, "INVALID_BBOX_ORDER"),
        ({"east": 126.0, "west": 127.0}, "INVALID_BBOX_ORDER"),
        ({"north": 91.0}, "COORDINATE_OUT_OF_RANGE"),
        ({"max_area_km2": 0.001}, "BBOX_AREA_EXCEEDED"),
    ],
)
def test_bounded_area_rejects_invalid_coordinates_and_area(changes, code):
    values = {
        "name": "test",
        "north": 36.36,
        "south": 36.35,
        "east": 127.40,
        "west": 127.39,
        "max_area_km2": 10.0,
    }
    values.update(changes)

    with pytest.raises(DomainValidationError) as raised:
        BoundedArea(**values)

    assert raised.value.code == code
    if code == "BBOX_AREA_EXCEEDED":
        assert raised.value.expected == changes["max_area_km2"]
        assert raised.value.actual > raised.value.expected


def test_schema_rejection_reports_supported_versions():
    with pytest.raises(DomainValidationError) as raised:
        validate_schema_version("999")

    assert raised.value.code == "UNSUPPORTED_SCHEMA_VERSION"
    assert raised.value.expected == [DOMAIN_SCHEMA_VERSION]
    assert DOMAIN_SCHEMA_VERSION in str(raised.value)


def test_model_network_must_be_nonempty():
    with pytest.raises(DomainValidationError) as raised:
        ModelNetwork(intersections=(), segments=())

    assert raised.value.code == "EMPTY_NETWORK"


def test_complete_vehicle_placement_accepts_exactly_six_police_and_fugitive():
    network = make_ring_network()
    police = tuple(VehiclePlacement(intersection_id=index) for index in range(6))

    validate_vehicle_placements(network, police, VehiclePlacement(intersection_id=6))


@pytest.mark.parametrize(
    ("police", "fugitive", "code"),
    [
        (tuple(VehiclePlacement(intersection_id=i) for i in range(5)), VehiclePlacement(intersection_id=6), "INVALID_POLICE_COUNT"),
        (tuple(VehiclePlacement(intersection_id=i) for i in range(6)), None, "MISSING_FUGITIVE"),
        (tuple(VehiclePlacement(intersection_id=i) for i in range(6)), VehiclePlacement(intersection_id=99), "INVALID_VEHICLE_POSITION"),
        (tuple(VehiclePlacement(intersection_id=i) for i in range(6)), VehiclePlacement(intersection_id=5), "DUPLICATE_VEHICLE_POSITION"),
    ],
)
def test_complete_vehicle_placement_rejects_invalid_teams(police, fugitive, code):
    with pytest.raises(DomainValidationError) as raised:
        validate_vehicle_placements(make_ring_network(), police, fugitive)

    assert raised.value.code == code


def test_virtual_segment_is_not_a_drivable_vehicle_position():
    police = tuple(VehiclePlacement(intersection_id=index) for index in range(6))

    with pytest.raises(DomainValidationError) as raised:
        validate_vehicle_placements(
            make_ring_network(virtual_segment=0),
            police,
            VehiclePlacement(segment_id=0, progress=0.5),
        )

    assert raised.value.code == "NON_DRIVABLE_VEHICLE_POSITION"


def test_daejeon_preset_matches_checked_configuration():
    config_path = (
        Path(__file__).parents[1]
        / "pursuit_evasion_rl"
        / "osm_demo"
        / "configs"
        / "daejeon_drive_v1.json"
    )
    config = json.loads(config_path.read_text(encoding="utf-8"))
    bbox = config["bbox"]

    assert DAEJEON_DRIVE_PRESET.name == config["name"]
    assert DAEJEON_DRIVE_PRESET.network_type == "drive"
    assert DAEJEON_DRIVE_PRESET.config_version == config["config_version"]
    assert (
        DAEJEON_DRIVE_PRESET.area.north,
        DAEJEON_DRIVE_PRESET.area.south,
        DAEJEON_DRIVE_PRESET.area.east,
        DAEJEON_DRIVE_PRESET.area.west,
    ) == (bbox["north"], bbox["south"], bbox["east"], bbox["west"])


def test_preset_rejects_unsupported_network_and_is_immutable():
    area = DAEJEON_DRIVE_PRESET.area
    with pytest.raises(DomainValidationError) as raised:
        NetworkPreset("bad", area, "walk", area.config_version)
    assert raised.value.code == "INVALID_NETWORK_TYPE"

    with pytest.raises(FrozenInstanceError):
        DAEJEON_DRIVE_PRESET.name = "changed"
