"""Offline tests for the legacy and OSM topology observation profiles.

These cover the OSM topology profile invariants required by Requirements 7.2,
7.6, 7.8, 8.4 and 8.11: exactly 21 finite float32 values in [0, 1], stable
action-slot layout, ID-free features and relabel invariance. No external OSM
access is performed.
"""

from __future__ import annotations

import numpy as np
import pytest

from pursuit_evasion_rl.osm_demo.models import (
    DomainValidationError,
    Intersection,
    ModelNetwork,
    Segment,
    VehiclePlacement,
)
from pursuit_evasion_rl.osm_demo.observations import (
    ACTION_DIM,
    LEGACY_FIELD_ORDER,
    LEGACY_PROFILE_ID,
    OBSERVATION_DIM,
    OSM_PROFILE_ID,
    OSMTopologyV1Adapter,
    flatten_legacy_grid_v0,
    legacy_grid_v0_contract,
    osm_topology_v1_contract,
)


def _segment(seg_id, start, end, geometry, length, *, crosses=False):
    return Segment(
        id=seg_id,
        start_id=start,
        end_id=end,
        length_m=length,
        geometry_xy=geometry,
        crosses_boundary=crosses,
    )


def _build_network():
    """A small directed network: 0->1, 1->2, 1->3, 2->3 (3 is a bbox boundary)."""
    positions = {0: (0.0, 0.0), 1: (100.0, 0.0), 2: (200.0, 0.0), 3: (200.0, 100.0)}
    segments = (
        _segment(0, 0, 1, ((0.0, 0.0), (100.0, 0.0)), 100.0),
        _segment(1, 1, 2, ((100.0, 0.0), (200.0, 0.0)), 100.0),
        _segment(2, 1, 3, ((100.0, 0.0), (200.0, 100.0)), 141.42),
        _segment(3, 2, 3, ((200.0, 0.0), (200.0, 100.0)), 100.0, crosses=True),
    )
    outgoing = {0: (0,), 1: (1, 2), 2: (3,), 3: ()}
    incoming = {0: (), 1: (0,), 2: (1,), 3: (2, 3)}
    intersections = tuple(
        Intersection(
            id=i,
            position_xy=positions[i],
            source_signature=f"sig-{i}",
            boundary_kind="bbox" if i == 3 else None,
            outgoing_segment_ids=outgoing[i],
            incoming_segment_ids=incoming[i],
        )
        for i in range(4)
    )
    return ModelNetwork(intersections, segments)


def _police_at(*intersection_ids):
    return tuple(VehiclePlacement(intersection_id=i) for i in intersection_ids)


# --------------------------------------------------------------------------
# Legacy profile
# --------------------------------------------------------------------------


def test_legacy_flatten_reproduces_sorted_key_vector():
    observation = {
        "agent_id_onehot": [0, 0, 1, 0, 0, 0],
        "current_step": [7],
        "my_position": [3],
        "my_progress": [0.25],
        "neighbors": [4, 5, -1, -1, -1],
        "nearest_boundary_dist": [2],
        "other_positions": [9, 8, 7, 6, 5, 4],
    }
    vector = flatten_legacy_grid_v0(observation)
    assert vector.shape == (OBSERVATION_DIM,)
    assert vector.dtype == np.float32
    # ascending key order: agent_id_onehot, current_step, my_position, my_progress,
    # nearest_boundary_dist, neighbors, other_positions
    expected = np.array(
        [0, 0, 1, 0, 0, 0, 7, 3, 0.25, 2, 4, 5, -1, -1, -1, 9, 8, 7, 6, 5, 4],
        dtype=np.float32,
    )
    np.testing.assert_array_equal(vector, expected)
    assert legacy_grid_v0_contract().profile_id == LEGACY_PROFILE_ID


def test_legacy_flatten_rejects_missing_field():
    with pytest.raises(DomainValidationError):
        flatten_legacy_grid_v0({"current_step": [1]})


def test_legacy_field_order_matches_glossary():
    assert LEGACY_FIELD_ORDER == (
        "agent_id_onehot",
        "current_step",
        "my_position",
        "my_progress",
        "nearest_boundary_dist",
        "neighbors",
        "other_positions",
    )


# --------------------------------------------------------------------------
# OSM topology profile
# --------------------------------------------------------------------------


def test_osm_observation_shape_finiteness_and_range():
    network = _build_network()
    adapter = OSMTopologyV1Adapter(network, clip_distance_m=500.0)
    police = _police_at(0, 1, 2, 3, 0, 1)
    fugitive = VehiclePlacement(segment_id=1, progress=0.5)
    for index in range(6):
        vector = adapter.observe(
            police_index=index,
            police=police,
            fugitive=fugitive,
            step=3,
            max_steps=50,
        )
        assert vector.shape == (OBSERVATION_DIM,)
        assert vector.dtype == np.float32
        assert np.all(np.isfinite(vector))
        assert np.all(vector >= 0.0) and np.all(vector <= 1.0)
        # police one-hot occupies the first six slots
        assert vector[index] == pytest.approx(1.0)
        assert vector[:6].sum() == pytest.approx(1.0)


def test_osm_contract_matches_dimensions():
    contract = osm_topology_v1_contract(500.0)
    assert contract.profile_id == OSM_PROFILE_ID
    assert sum(contract.field_sizes) == OBSERVATION_DIM
    assert ACTION_DIM == 6


def test_unreachable_and_padded_slots_saturate_to_one():
    network = _build_network()
    adapter = OSMTopologyV1Adapter(network, clip_distance_m=500.0)
    # An officer sitting on boundary intersection 3 has no outgoing segments and
    # cannot reach any other agent, so every distance/action slot saturates.
    police = _police_at(3, 0, 1, 2, 0, 1)
    fugitive = VehiclePlacement(intersection_id=0)
    vector = adapter.observe(
        police_index=0, police=police, fugitive=fugitive, step=0, max_steps=10
    )
    # distance to fugitive, distance to boundary, all five action slots and all
    # other-agent distances are unreachable from a sink intersection.
    assert vector[6] == pytest.approx(0.0)  # step ratio 0
    assert vector[7] == pytest.approx(0.0)  # progress at intersection
    assert vector[8] == pytest.approx(1.0)  # self -> fugitive unreachable
    assert np.all(vector[10:15] == pytest.approx(1.0))  # action-slot scores padded
    assert np.all(vector[15:21] == pytest.approx(1.0))  # other-agent distances


def test_directed_distance_normalization_is_monotone():
    network = _build_network()
    adapter = OSMTopologyV1Adapter(network, clip_distance_m=1000.0)
    # police_0 at intersection 0, fugitive at intersection 1 => distance 100 / 1000
    police = _police_at(0, 2, 3, 1, 2, 3)
    fugitive = VehiclePlacement(intersection_id=1)
    vector = adapter.observe(
        police_index=0, police=police, fugitive=fugitive, step=10, max_steps=20
    )
    assert vector[6] == pytest.approx(0.5)  # step ratio 10/20
    assert vector[8] == pytest.approx(0.1)  # 100m / 1000m clip


def test_relabel_invariance_example():
    """Requirement 8.11/7.8: relabeling storage IDs leaves the vector unchanged."""
    network = _build_network()
    adapter = OSMTopologyV1Adapter(network, clip_distance_m=500.0)
    police = _police_at(0, 1, 2, 3, 0, 1)
    fugitive = VehiclePlacement(segment_id=1, progress=0.4)
    baseline = [
        adapter.observe(
            police_index=i, police=police, fugitive=fugitive, step=5, max_steps=40
        )
        for i in range(6)
    ]

    # Permute intersection and segment storage IDs while preserving topology and
    # geometry, then relabel every placement accordingly.
    int_perm = {0: 2, 1: 0, 2: 3, 3: 1}
    seg_perm = {0: 3, 1: 1, 2: 2, 3: 0}
    relabeled = _relabel(network, int_perm, seg_perm)
    relabeled_adapter = OSMTopologyV1Adapter(relabeled, clip_distance_m=500.0)
    relabeled_police = _police_at(*(int_perm[p.intersection_id] for p in police))
    relabeled_fugitive = VehiclePlacement(
        segment_id=seg_perm[fugitive.segment_id], progress=fugitive.progress
    )
    relabeled_obs = [
        relabeled_adapter.observe(
            police_index=i,
            police=relabeled_police,
            fugitive=relabeled_fugitive,
            step=5,
            max_steps=40,
        )
        for i in range(6)
    ]

    for original, permuted in zip(baseline, relabeled_obs):
        np.testing.assert_allclose(original, permuted, rtol=0, atol=1e-6)


def test_observe_rejects_bad_agent_index():
    network = _build_network()
    adapter = OSMTopologyV1Adapter(network, clip_distance_m=500.0)
    police = _police_at(0, 1, 2, 3, 0, 1)
    with pytest.raises(DomainValidationError):
        adapter.observe(
            police_index=6,
            police=police,
            fugitive=VehiclePlacement(intersection_id=0),
            step=0,
            max_steps=10,
        )


def _relabel(network, int_perm, seg_perm):
    positions = {item.id: item.position_xy for item in network.intersections}
    boundary = {item.id: item.boundary_kind for item in network.intersections}
    signatures = {item.id: item.source_signature for item in network.intersections}

    new_segments = []
    for segment in network.segments:
        new_segments.append(
            Segment(
                id=seg_perm[segment.id],
                start_id=int_perm[segment.start_id],
                end_id=int_perm[segment.end_id],
                length_m=segment.length_m,
                geometry_xy=segment.geometry_xy,
                crosses_boundary=segment.crosses_boundary,
            )
        )
    new_segments.sort(key=lambda s: s.id)

    new_intersections = []
    for old_id in range(len(network.intersections)):
        new_id = int_perm[old_id]
        outgoing = tuple(s.id for s in new_segments if s.start_id == new_id)
        incoming = tuple(s.id for s in new_segments if s.end_id == new_id)
        new_intersections.append(
            Intersection(
                id=new_id,
                position_xy=positions[old_id],
                source_signature=signatures[old_id],
                boundary_kind=boundary[old_id],
                outgoing_segment_ids=outgoing,
                incoming_segment_ids=incoming,
            )
        )
    new_intersections.sort(key=lambda i: i.id)
    return ModelNetwork(tuple(new_intersections), tuple(new_segments))
