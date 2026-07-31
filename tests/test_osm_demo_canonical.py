"""Focused examples for canonical OSM artifact identities and cache keys."""

from dataclasses import dataclass
from decimal import Decimal
import hashlib
import json

import pytest

from pursuit_evasion_rl.osm_demo.canonical import (
    CanonicalSerializationError,
    artifact_identity_hash,
    build_cache_key,
    cache_key_payload,
    canonical_json,
    content_hash,
    sha256_bytes,
)
from pursuit_evasion_rl.osm_demo.models import BoundedArea


@pytest.fixture
def area() -> BoundedArea:
    return BoundedArea(
        name="cache-example",
        north=36.36,
        south=36.35,
        east=127.40,
        west=127.39,
        max_area_km2=10.0,
    )


def test_canonical_json_has_stable_mapping_set_and_utf8_encoding():
    first = {"z": {"beta", "alpha"}, "한글": 1, "a": [2, 1]}
    second = {"a": [2, 1], "한글": 1, "z": {"alpha", "beta"}}

    encoded = canonical_json(first)

    assert encoded == canonical_json(second)
    assert encoded == '{"a":[2,1],"z":["alpha","beta"],"한글":1}'.encode("utf-8")


def test_float_quantization_uses_documented_half_even_and_normalizes_zero():
    value = {
        "down": Decimal("1.2345678905"),
        "up": Decimal("1.2345678915"),
        "negative_zero": -0.0000000001,
    }

    assert json.loads(canonical_json(value)) == {
        "down": 1.23456789,
        "up": 1.234567892,
        "negative_zero": 0,
    }


@pytest.mark.parametrize("value", [float("nan"), float("inf"), float("-inf"), Decimal("NaN")])
def test_canonical_json_rejects_non_finite_numbers_at_any_depth(value):
    with pytest.raises(CanonicalSerializationError, match="finite"):
        canonical_json({"nested": [value]})


@dataclass(frozen=True)
class TimestampedArtifact:
    value: float
    acquired_at: str
    committed_at: str


def test_content_hash_includes_provenance_but_identity_hash_excludes_timestamps():
    earlier = TimestampedArtifact(12.25, "2025-01-01T00:00:00Z", "2025-01-01T00:01:00Z")
    later = TimestampedArtifact(12.25, "2026-02-03T04:05:06Z", "2026-02-03T04:06:00Z")

    assert content_hash(earlier) != content_hash(later)
    assert artifact_identity_hash(earlier) == artifact_identity_hash(later)


def test_sha256_helpers_match_exact_canonical_artifact_bytes():
    encoded = canonical_json({"a": 1})

    assert sha256_bytes(encoded) == hashlib.sha256(b'{"a":1}').hexdigest()
    assert content_hash({"a": 1}) == sha256_bytes(encoded)


def test_cache_key_contains_all_identity_inputs_and_no_timestamp(area):
    arguments = {
        "network_type": "drive",
        "raw_schema_version": "1.0",
        "coarsener_version": "coarsener-v1",
        "preprocessing_settings_hash": content_hash({"coordinate_precision": 9}),
    }

    payload = cache_key_payload(area, **arguments)
    key = build_cache_key(area, **arguments)

    assert payload == {
        "bbox": {"north": 36.36, "south": 36.35, "east": 127.4, "west": 127.39},
        **arguments,
    }
    assert not any(name.endswith("_at") for name in payload)
    assert key == content_hash(payload)
    assert len(key) == 64
    assert set(key) <= set("0123456789abcdef")


def test_cache_key_does_not_depend_on_area_name_or_provenance_configuration(area):
    renamed = BoundedArea(
        name="renamed",
        north=area.north,
        south=area.south,
        east=area.east,
        west=area.west,
        max_area_km2=20.0,
    )
    arguments = {
        "network_type": "drive",
        "raw_schema_version": "1.0",
        "coarsener_version": "coarsener-v1",
        "preprocessing_settings_hash": "settings-hash",
    }

    assert build_cache_key(area, **arguments) == build_cache_key(renamed, **arguments)


@pytest.mark.parametrize("decimals", [-1, 1.5, True])
def test_invalid_float_precision_is_rejected(decimals):
    with pytest.raises(CanonicalSerializationError, match="nonnegative integer"):
        canonical_json({"value": 1.0}, float_decimals=decimals)


def test_non_string_mapping_keys_and_empty_cache_fields_are_rejected(area):
    with pytest.raises(CanonicalSerializationError, match="keys must be strings"):
        canonical_json({1: "not-json-object-contract"})

    with pytest.raises(CanonicalSerializationError, match="nonempty strings"):
        build_cache_key(
            area,
            network_type="",
            raw_schema_version="1.0",
            coarsener_version="coarsener-v1",
            preprocessing_settings_hash="settings-hash",
        )
