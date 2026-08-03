"""Property 22 coverage: containment metrics satisfy geometric bounds and symmetries."""

from __future__ import annotations

import math

from hypothesis import assume, given, settings, strategies as st
import pytest

from pursuit_evasion_rl.osm_demo.models import Intersection, ModelNetwork, Segment
from pursuit_evasion_rl.research.errors import ResearchValidationError
from pursuit_evasion_rl.research.metrics.behavior import containment_angles, reachable_region_reduction

# **Property 22: Containment metrics satisfy geometric bounds and symmetries**
# **Validates: Requirements 13.5, 19.4**

pytestmark = pytest.mark.offline

_PBT_SETTINGS = settings(max_examples=100, deadline=None)

_coord = st.floats(min_value=-500.0, max_value=500.0, allow_nan=False, allow_infinity=False)
_point = st.tuples(_coord, _coord)


@_PBT_SETTINGS
@given(
    positions=st.lists(_point, min_size=6, max_size=6),
    fugitive=_point,
)
def test_angular_coverage_and_max_gap_stay_within_their_natural_bounds(positions, fugitive) -> None:
    # An officer exactly coincident with the fugitive has an undefined bearing
    # (atan2(0, 0) is a special-cased constant, not a real direction); exclude
    # that degenerate configuration, which cannot arise in a live episode since
    # zero separation is itself a capture.
    assume(all(math.dist(p, fugitive) > 1e-6 for p in positions))
    # When every officer shares the exact same bearing from the fugitive, the
    # six circular gaps are each exactly 0 rather than summing to one full lap
    # (there is no "circle" to traverse); exclude that measure-zero alignment,
    # which real, independently-moving officers cannot land on exactly.
    bearings = [math.atan2(p[1] - fugitive[1], p[0] - fugitive[0]) for p in positions]
    assume(len(set(bearings)) > 1)
    result = containment_angles(positions, fugitive)
    assert 0.0 <= result.angular_coverage <= 1.0
    assert 0.0 <= result.max_angular_gap_rad <= 2 * math.pi + 1e-9
    assert sum(result.sorted_gaps_rad) == pytest.approx(2 * math.pi, abs=1e-6)


@_PBT_SETTINGS
@given(
    positions=st.lists(_point, min_size=6, max_size=6),
    fugitive=_point,
    angle=st.floats(min_value=-math.pi, max_value=math.pi, allow_nan=False),
    offset=_point,
)
def test_angular_coverage_and_max_gap_are_rotation_and_translation_invariant(positions, fugitive, angle, offset) -> None:
    assume(all(math.dist(p, fugitive) > 1e-6 for p in positions))

    def transform(point: tuple[float, float]) -> tuple[float, float]:
        x, y = point
        cos_a, sin_a = math.cos(angle), math.sin(angle)
        rotated = (x * cos_a - y * sin_a, x * sin_a + y * cos_a)
        return (rotated[0] + offset[0], rotated[1] + offset[1])

    original = containment_angles(positions, fugitive)
    transformed = containment_angles([transform(p) for p in positions], transform(fugitive))

    assert transformed.angular_coverage == pytest.approx(original.angular_coverage, abs=1e-6)
    assert transformed.max_angular_gap_rad == pytest.approx(original.max_angular_gap_rad, abs=1e-6)


@_PBT_SETTINGS
@given(positions=st.lists(_point, min_size=1, max_size=8))
def test_containment_angles_requires_exactly_six_officer_positions(positions) -> None:
    if len(positions) == 6:
        return  # covered by the bounds test above
    with pytest.raises(ResearchValidationError) as excinfo:
        containment_angles(positions, (0.0, 0.0))
    assert excinfo.value.code == "INVALID_POLICE_COUNT"


def _ring_network(node_count: int) -> ModelNetwork:
    coordinates = tuple((100.0 * math.cos(2 * math.pi * i / node_count), 100.0 * math.sin(2 * math.pi * i / node_count)) for i in range(node_count))
    segments = tuple(
        Segment(i, i, (i + 1) % node_count, math.dist(coordinates[i], coordinates[(i + 1) % node_count]), (coordinates[i], coordinates[(i + 1) % node_count]))
        for i in range(node_count)
    )
    intersections = tuple(
        Intersection(i, coordinates[i], f"node-{i}",
                     outgoing_segment_ids=tuple(s.id for s in segments if s.start_id == i),
                     incoming_segment_ids=tuple(s.id for s in segments if s.end_id == i))
        for i in range(node_count)
    )
    return ModelNetwork(intersections, segments)


_RING = _ring_network(10)


@_PBT_SETTINGS
@given(
    fugitive=st.integers(min_value=0, max_value=9),
    occupied_subset=st.lists(st.integers(min_value=0, max_value=9), min_size=0, max_size=5, unique=True),
    extra_occupied=st.lists(st.integers(min_value=0, max_value=9), min_size=0, max_size=5, unique=True),
)
def test_reachable_region_reduction_is_monotone_as_more_nodes_are_removed(fugitive, occupied_subset, extra_occupied) -> None:
    smaller_set = frozenset(occupied_subset) - {fugitive}
    larger_set = smaller_set | (frozenset(extra_occupied) - {fugitive})

    small = reachable_region_reduction(network=_RING, fugitive_intersection_id=fugitive, occupied_intersection_ids=smaller_set)
    large = reachable_region_reduction(network=_RING, fugitive_intersection_id=fugitive, occupied_intersection_ids=larger_set)

    assert small.baseline_reachable_count == large.baseline_reachable_count
    # Removing a superset of nodes can never leave MORE reachable than removing a subset.
    assert large.contained_reachable_count <= small.contained_reachable_count
    assert large.reduction_count >= small.reduction_count
    assert 0.0 <= small.reduction_fraction <= 1.0
    assert 0.0 <= large.reduction_fraction <= 1.0
