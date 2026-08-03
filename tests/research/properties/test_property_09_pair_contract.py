"""Property 9 coverage: paired status is equivalent to complete case-contract equality."""

from __future__ import annotations

import math

from hypothesis import given, settings, strategies as st
import pytest

from pursuit_evasion_rl.osm_demo.models import (
    EpisodeConfig,
    EpisodeOutcome,
    Intersection,
    ModelNetwork,
    Segment,
    VehiclePlacement,
)
from pursuit_evasion_rl.research.domain import DataKind, MapScenario
from pursuit_evasion_rl.research.evaluation.paired import (
    CASE_CONTRACT_FIELDS,
    EpisodeCase,
    FugitiveRNGSpec,
    TerminationConfig,
    compare_cases,
)

# **Property 9: Paired status is equivalent to complete case-contract equality**
# **Validates: Requirements 8.1, 8.8, 19.5**

pytestmark = pytest.mark.offline

_PBT_SETTINGS = settings(max_examples=100, deadline=None)

_NODE_COUNT = 8


def _ring_network() -> ModelNetwork:
    coordinates = tuple((100.0 * math.cos(2 * math.pi * i / _NODE_COUNT), 100.0 * math.sin(2 * math.pi * i / _NODE_COUNT)) for i in range(_NODE_COUNT))
    segments = tuple(
        Segment(i, i, (i + 1) % _NODE_COUNT, math.dist(coordinates[i], coordinates[(i + 1) % _NODE_COUNT]), (coordinates[i], coordinates[(i + 1) % _NODE_COUNT]))
        for i in range(_NODE_COUNT)
    )
    intersections = tuple(
        Intersection(
            i, coordinates[i], f"node-{i}",
            outgoing_segment_ids=tuple(s.id for s in segments if s.start_id == i),
            incoming_segment_ids=tuple(s.id for s in segments if s.end_id == i),
        )
        for i in range(_NODE_COUNT)
    )
    return ModelNetwork(intersections, segments)


_NETWORK = _ring_network()


def _base_case(*, map_hash: str = "a" * 64, fugitive_seed: int = 0, timeout: int = 38, max_steps: int = 40) -> EpisodeCase:
    police = tuple(VehiclePlacement(intersection_id=i) for i in range(6))
    fugitive = VehiclePlacement(intersection_id=6)
    return EpisodeCase(
        map_hash=map_hash,
        data_kind=DataKind.SYNTHETIC_FIXTURE,
        scenario=MapScenario.INTERIOR_CONTAINED,
        police=police,
        fugitive=fugitive,
        fugitive_rng=FugitiveRNGSpec(stream_id="evader", seed=fugitive_seed),
        environment=EpisodeConfig(dt_s=1.0, police_speed_mps=16.0, fugitive_speed_mps=9.0, capture_radius_m=25.0, max_steps=max_steps),
        termination=TerminationConfig(
            outcome_domain=frozenset({EpisodeOutcome.CAPTURE, EpisodeOutcome.TIMEOUT}), timeout_step_limit=timeout,
        ),
    )


_hex64 = st.text("0123456789abcdef", min_size=64, max_size=64)


@_PBT_SETTINGS
@given(map_hash=_hex64, fugitive_seed=st.integers(min_value=0, max_value=10_000))
def test_identical_cases_always_compare_paired(map_hash: str, fugitive_seed: int) -> None:
    left = _base_case(map_hash=map_hash, fugitive_seed=fugitive_seed)
    right = _base_case(map_hash=map_hash, fugitive_seed=fugitive_seed)
    status = compare_cases(left, right)
    assert status.paired
    assert status.mismatches == ()
    assert status.case_hash == left.case_hash == right.case_hash


@_PBT_SETTINGS
@given(
    map_hash=_hex64,
    fugitive_seed=st.integers(min_value=0, max_value=10_000),
    field=st.sampled_from(CASE_CONTRACT_FIELDS),
    other_seed=st.integers(min_value=0, max_value=10_000),
)
def test_perturbing_exactly_one_field_always_reports_that_field_as_the_only_mismatch(
    map_hash: str, fugitive_seed: int, field: str, other_seed: int
) -> None:
    left = _base_case(map_hash=map_hash, fugitive_seed=fugitive_seed)
    if field == "map_hash":
        right = _base_case(map_hash="b" * 64 if map_hash != "b" * 64 else "c" * 64, fugitive_seed=fugitive_seed)
    elif field == "data_kind":
        base = _base_case(map_hash=map_hash, fugitive_seed=fugitive_seed)
        right = EpisodeCase(
            map_hash=base.map_hash, data_kind=DataKind.ACTUAL_OSM_MAP, scenario=base.scenario,
            police=base.police, fugitive=base.fugitive,
            fugitive_rng=base.fugitive_rng, environment=base.environment, termination=base.termination,
        )
    elif field == "scenario":
        base = _base_case(map_hash=map_hash, fugitive_seed=fugitive_seed)
        right = EpisodeCase(
            map_hash=base.map_hash, data_kind=base.data_kind, scenario=MapScenario.BOUNDARY_ESCAPE,
            police=base.police, fugitive=base.fugitive, fugitive_rng=base.fugitive_rng,
            environment=base.environment,
            termination=TerminationConfig(
                outcome_domain=frozenset({EpisodeOutcome.CAPTURE, EpisodeOutcome.ESCAPE, EpisodeOutcome.TIMEOUT}),
                timeout_step_limit=base.termination.timeout_step_limit,
            ),
        )
    elif field == "fugitive_rng":
        alternate = other_seed if other_seed != fugitive_seed else other_seed + 1
        right = _base_case(map_hash=map_hash, fugitive_seed=alternate)
    elif field == "termination":
        right = _base_case(map_hash=map_hash, fugitive_seed=fugitive_seed, timeout=39)
    elif field == "environment":
        right = _base_case(map_hash=map_hash, fugitive_seed=fugitive_seed, max_steps=45, timeout=38)
    elif field == "fugitive":
        base = _base_case(map_hash=map_hash, fugitive_seed=fugitive_seed)
        right = EpisodeCase(
            map_hash=base.map_hash, data_kind=base.data_kind, scenario=base.scenario,
            police=base.police, fugitive=VehiclePlacement(intersection_id=7),
            fugitive_rng=base.fugitive_rng, environment=base.environment, termination=base.termination,
        )
    elif field.startswith("police_"):
        index = int(field.split("_")[1])
        base = _base_case(map_hash=map_hash, fugitive_seed=fugitive_seed)
        police = list(base.police)
        police[index] = VehiclePlacement(intersection_id=7)  # node 7 is unused by any base-case placement
        right = EpisodeCase(
            map_hash=base.map_hash, data_kind=base.data_kind, scenario=base.scenario,
            police=tuple(police), fugitive=base.fugitive,
            fugitive_rng=base.fugitive_rng, environment=base.environment, termination=base.termination,
        )
    else:  # pragma: no cover - CASE_CONTRACT_FIELDS is exhaustively handled above
        pytest.fail(f"unhandled contract field {field}")

    status = compare_cases(left, right)
    assert not status.paired
    assert status.case_hash is None
    assert field in status.mismatched_fields


@_PBT_SETTINGS
@given(map_hash=_hex64, fugitive_seed=st.integers(min_value=0, max_value=10_000))
def test_paired_iff_case_hashes_agree(map_hash: str, fugitive_seed: int) -> None:
    left = _base_case(map_hash=map_hash, fugitive_seed=fugitive_seed)
    same = _base_case(map_hash=map_hash, fugitive_seed=fugitive_seed)
    different = _base_case(map_hash=map_hash, fugitive_seed=fugitive_seed + 1)

    assert (left.case_hash == same.case_hash) == compare_cases(left, same).paired
    assert (left.case_hash == different.case_hash) == compare_cases(left, different).paired
