"""Focused regressions for the migrated 21D/28D observation adapters."""

from __future__ import annotations

import numpy as np
import pytest

import osm_obs_aug as legacy_shim
from pursuit_evasion_rl.osm_demo.models import (
    DomainValidationError,
    Intersection,
    ModelNetwork,
    Segment,
    VehiclePlacement,
)
from pursuit_evasion_rl.research.canonical import content_hash
from pursuit_evasion_rl.research.variants.observations import (
    AUGMENTED_FIELD_ORDER,
    AUGMENTED_NORMALIZATION,
    DECISION_TIMING,
    Observation21DAdapter,
    Observation28DAdapter,
    canonical_observation_config_hash,
    observation_28d_contract,
)


def _network() -> ModelNetwork:
    positions = ((0.0, 0.0), (100.0, 0.0), (100.0, 100.0), (0.0, 100.0))
    segments = tuple(
        Segment(
            id=index,
            start_id=index,
            end_id=(index + 1) % 4,
            length_m=100.0,
            geometry_xy=(positions[index], positions[(index + 1) % 4]),
        )
        for index in range(4)
    )
    intersections = tuple(
        Intersection(
            id=index,
            position_xy=position,
            source_signature=f"fixture-{index}",
            outgoing_segment_ids=(index,),
            incoming_segment_ids=((index - 1) % 4,),
        )
        for index, position in enumerate(positions)
    )
    return ModelNetwork(intersections, segments)


def _fixture():
    network = _network()
    police = tuple(
        VehiclePlacement(intersection_id=index)
        for index in (0, 1, 2, 3, 0, 2)
    )
    fugitive = VehiclePlacement(intersection_id=1)
    return network, police, fugitive


def _observe(adapter, police_index, police, fugitive):
    return adapter.observe(
        police_index=police_index,
        police=police,
        fugitive=fugitive,
        step=7,
        max_steps=40,
        incoming_heading=0.25,
    )


def test_shim_and_package_adapter_are_bitwise_identical_on_same_fixture():
    network, police, fugitive = _fixture()
    package_adapter = Observation28DAdapter(network, clip_distance_m=500.0)
    shim_adapter = legacy_shim.AugmentedObs(network, clip_distance_m=500.0)

    assert legacy_shim.AugmentedObs is Observation28DAdapter
    for police_index in range(6):
        package_vector = _observe(package_adapter, police_index, police, fugitive)
        shim_vector = _observe(shim_adapter, police_index, police, fugitive)
        assert package_vector.dtype == np.float32
        assert package_vector.shape == (28,)
        assert package_vector.tobytes() == shim_vector.tobytes()


def test_contract_exposes_audited_order_timing_normalization_and_hash():
    contract = observation_28d_contract(clip_distance_m=500.0, near_radius_m=200.0)

    assert contract.added_fields == AUGMENTED_FIELD_ORDER
    assert contract.normalization == AUGMENTED_NORMALIZATION
    assert contract.calculation_timing == DECISION_TIMING
    assert contract.clip_distance_m == 500.0
    assert contract.near_radius_m == 200.0
    assert contract.final_vector_clip == (0.0, 1.0)
    assert contract.dimension == 28 and contract.base_dimension == 21
    assert contract.config_hash == canonical_observation_config_hash(500.0, 200.0)
    assert len(contract.config_hash) == 64
    assert contract.config_hash != canonical_observation_config_hash(500.0, 201.0)


def test_28d_preserves_21d_prefix_and_finally_clips_raw_negative_sine():
    network, police, fugitive = _fixture()
    adapter = Observation28DAdapter(network, clip_distance_m=500.0)
    base_adapter = Observation21DAdapter(network, clip_distance_m=500.0)

    # Officer 2 is directly above the fugitive: atan2 has negative sine.
    vector = _observe(adapter, 2, police, fugitive)
    base = _observe(base_adapter, 2, police, fugitive)

    np.testing.assert_array_equal(vector[:21], base)
    assert adapter._extra(2, police, fugitive)[1] == pytest.approx(-1.0)
    assert vector[22] == 0.0
    assert np.all(np.isfinite(vector))
    assert np.all((0.0 <= vector) & (vector <= 1.0))


def test_contract_hash_is_canonical_and_invalid_distances_are_rejected():
    contract = observation_28d_contract(500, 200)
    payload = {
        "profile_id": contract.profile_id,
        "base_profile_id": contract.base_profile_id,
        "base_dimension": contract.base_dimension,
        "dimension": contract.dimension,
        "added_fields": contract.added_fields,
        "calculation_timing": contract.calculation_timing,
        "clip_distance_m": contract.clip_distance_m,
        "near_radius_m": contract.near_radius_m,
        "normalization": contract.normalization,
        "final_vector_clip": contract.final_vector_clip,
        "dtype": contract.dtype,
    }
    assert contract.config_hash == content_hash(payload)

    with pytest.raises(DomainValidationError):
        observation_28d_contract(0.0, 200.0)
    with pytest.raises(DomainValidationError):
        Observation28DAdapter(_network(), near_radius_m=float("nan"))
