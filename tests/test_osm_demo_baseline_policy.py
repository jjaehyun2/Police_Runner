"""Offline tests for the deterministic shortest-path Baseline_Police.

These cover the non-learned team baseline behind the uniform ``PolicePolicy``
protocol (design section 7; Requirements 8.1-8.10, 14.3) without any external
OSM access or checkpoint:

* each officer independently selects the valid outgoing segment minimizing the
  directed distance to the fugitive's position,
* ties are broken by ascending action-slot order (byte-for-byte deterministic),
* an officer with no exit that can reach the fugitive stays,
* six atomic recommendations are always returned, and the baseline satisfies the
  same ``PolicePolicy`` protocol as the learned recommender (interchangeable).
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
from pursuit_evasion_rl.osm_demo.policies import (
    BASELINE_PROFILE_ID,
    STAY_ACTION,
    BaselinePolicePolicy,
    PolicePolicy,
    build_action_mask,
)


# --------------------------------------------------------------------------
# Network builders
# --------------------------------------------------------------------------


def _segment(seg_id, start, end, length, *, crosses=False):
    # Geometry is a simple two-point stub; the baseline uses ``length_m`` for
    # directed distances, not the geometric extent.
    geometry = ((float(start), 0.0), (float(end), 1.0))
    return Segment(
        id=seg_id,
        start_id=start,
        end_id=end,
        length_m=length,
        geometry_xy=geometry,
        crosses_boundary=crosses,
    )


def _network(intersection_count, segments, outgoing, incoming, boundary=()):
    intersections = tuple(
        Intersection(
            id=i,
            position_xy=(float(i), float(i)),
            source_signature=f"sig-{i}",
            boundary_kind="bbox" if i in boundary else None,
            outgoing_segment_ids=outgoing.get(i, ()),
            incoming_segment_ids=incoming.get(i, ()),
        )
        for i in range(intersection_count)
    )
    return ModelNetwork(intersections, tuple(segments))


def _diamond(short_len, long_len):
    """0 branches to 1 and 2; both rejoin at fugitive node 3.

    Path 0->1->3 costs ``short_len`` per hop; path 0->2->3 costs ``short_len``
    then ``long_len``. With ``long_len == short_len`` the two exits tie.
    """
    segments = (
        _segment(0, 0, 1, short_len),
        _segment(1, 0, 2, short_len),
        _segment(2, 1, 3, short_len),
        _segment(3, 2, 3, long_len),
    )
    outgoing = {0: (0, 1), 1: (2,), 2: (3,)}
    incoming = {1: (0,), 2: (1,), 3: (2, 3)}
    return _network(4, segments, outgoing, incoming, boundary=())


def _police_at(*intersection_ids):
    return tuple(VehiclePlacement(intersection_id=i) for i in intersection_ids)


# --------------------------------------------------------------------------
# Shortest-path selection (Requirement 14.3)
# --------------------------------------------------------------------------


def test_baseline_selects_shortest_directed_path_to_fugitive():
    # 0->1->3 costs 20; 0->2->3 costs 10 + 1000 = 1010. Officer must pick 0->1.
    network = _diamond(short_len=10.0, long_len=1000.0)
    policy = BaselinePolicePolicy(network)

    recommendations = policy.recommend(
        police=_police_at(0, 0, 0, 0, 0, 0),
        fugitive=VehiclePlacement(intersection_id=3),
        step=0,
        max_steps=10,
    )
    assert len(recommendations) == 6
    for recommendation in recommendations:
        # Segment 0 is the 0->1 exit (start of the cheap path to the fugitive).
        assert recommendation.segment_id == 0
        assert recommendation.next_intersection_id == 1
        assert recommendation.valid is True
        assert recommendation.profile == BASELINE_PROFILE_ID


def test_baseline_breaks_ties_by_lowest_action_slot():
    # Symmetric diamond: both exits from 0 reach fugitive node 3 at equal cost.
    network = _diamond(short_len=100.0, long_len=100.0)
    policy = BaselinePolicePolicy(network)

    _, ordered = build_action_mask(network, 0)
    assert len(ordered) == 2  # two tied outgoing exits

    recommendation = policy.recommend(
        police=_police_at(0, 0, 0, 0, 0, 0),
        fugitive=VehiclePlacement(intersection_id=3),
        step=0,
        max_steps=10,
    )[0]
    # The tie must resolve to the first ordered action slot, deterministically.
    assert recommendation.action_index == 0
    assert recommendation.segment_id == ordered[0]


def test_baseline_is_deterministic_across_repeated_calls():
    network = _diamond(short_len=100.0, long_len=100.0)
    policy = BaselinePolicePolicy(network)
    police = _police_at(0, 0, 0, 0, 0, 0)
    fugitive = VehiclePlacement(intersection_id=3)

    first = policy.recommend(police=police, fugitive=fugitive, step=0, max_steps=10)
    second = policy.recommend(police=police, fugitive=fugitive, step=3, max_steps=10)
    assert [r.as_dict() for r in first] == [r.as_dict() for r in second]


# --------------------------------------------------------------------------
# Stay when the fugitive is unreachable (Requirement 9.9 interaction)
# --------------------------------------------------------------------------


def test_baseline_stays_when_no_exit_reaches_the_fugitive():
    # One-way chain 0->1->2; fugitive sits "behind" at 0. From 1 the only exit
    # (1->2) can never reach 0, so the officer stays.
    segments = (
        _segment(0, 0, 1, 50.0),
        _segment(1, 1, 2, 50.0),
    )
    outgoing = {0: (0,), 1: (1,)}
    incoming = {1: (0,), 2: (1,)}
    network = _network(3, segments, outgoing, incoming)
    policy = BaselinePolicePolicy(network)

    recommendation = policy.recommend(
        police=_police_at(1, 1, 1, 1, 1, 1),
        fugitive=VehiclePlacement(intersection_id=0),
        step=0,
        max_steps=10,
    )[0]
    assert recommendation.action_index == STAY_ACTION
    assert recommendation.segment_id is None
    assert recommendation.next_intersection_id == 1


def test_baseline_officers_on_segments_decide_at_the_downstream_intersection():
    # Officer riding segment 0 (0->1) decides at intersection 1 after arriving.
    network = _diamond(short_len=10.0, long_len=1000.0)
    policy = BaselinePolicePolicy(network)
    police = (VehiclePlacement(segment_id=0, progress=0.5),) + _police_at(0, 0, 0, 0, 0)

    recommendations = policy.recommend(
        police=police,
        fugitive=VehiclePlacement(intersection_id=3),
        step=0,
        max_steps=10,
    )
    # From intersection 1 the only exit is segment 2 (1->3) toward the fugitive.
    assert recommendations[0].segment_id == 2
    assert recommendations[0].next_intersection_id == 3


# --------------------------------------------------------------------------
# Uniform protocol / validation (Requirements 8.2-8.3, 14.3)
# --------------------------------------------------------------------------


def test_baseline_satisfies_police_policy_protocol():
    network = _diamond(short_len=10.0, long_len=10.0)
    policy = BaselinePolicePolicy(network)
    assert isinstance(policy, PolicePolicy)
    assert policy.experimental is False


def test_baseline_rejects_wrong_police_count():
    network = _diamond(short_len=10.0, long_len=10.0)
    policy = BaselinePolicePolicy(network)
    with pytest.raises(DomainValidationError) as excinfo:
        policy.recommend(
            police=_police_at(0, 0, 0),  # only three officers
            fugitive=VehiclePlacement(intersection_id=3),
            step=0,
            max_steps=10,
        )
    assert excinfo.value.code == "INVALID_POLICE_COUNT"
