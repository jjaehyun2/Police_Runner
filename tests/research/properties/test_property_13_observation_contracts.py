"""Property 13 coverage for observation variant contracts and ID independence."""

from __future__ import annotations

from dataclasses import dataclass
import math

from hypothesis import assume, given, settings, strategies as st
import numpy as np
import pytest

from pursuit_evasion_rl.osm_demo.models import (
    POLICE_COUNT,
    Intersection,
    ModelNetwork,
    Segment,
    VehiclePlacement,
)
from pursuit_evasion_rl.osm_demo.observations import OSM_FIELD_ORDER, OSM_FIELD_SIZES
from pursuit_evasion_rl.research.variants.observations import (
    AUGMENTED_FIELD_ORDER,
    OBSERVATION_21D_DIM,
    OBSERVATION_28D_DIM,
    Observation21DAdapter,
    Observation28DAdapter,
    actor_capacity,
    compare_observation_capacities,
)

# **Property 13: Observation variants preserve contracts and ID independence**
# **Validates: Requirements 9.1-9.2, 9.6, 19.4**

pytestmark = [pytest.mark.property, pytest.mark.offline]

_PBT_SETTINGS = settings(max_examples=100, deadline=None)

# Normalized features live in [0, 1]; 1e-6 is far below any identifier leak but
# still absorbs the float32 rounding of order-dependent team means.
_TOLERANCE = 1e-6

_AUDITED_HIDDEN_DIMS = (128, 128)
_AUDITED_ACTION_DIM = 6


def _field_spans() -> dict[str, tuple[int, int]]:
    spans: dict[str, tuple[int, int]] = {}
    offset = 0
    for name, size in zip(OSM_FIELD_ORDER, OSM_FIELD_SIZES, strict=True):
        spans[name] = (offset, offset + size)
        offset += size
    return spans


_FIELD_SPANS = _field_spans()
_ONEHOT_START = _FIELD_SPANS["police_onehot"][0]
_OTHERS_START = _FIELD_SPANS["other_agent_distances"][0]
# Contiguous block of purely own-state/geometric features between the identity
# one-hot and the per-officer distance block.
_OWN_STATE_SPAN = (
    _FIELD_SPANS["step_ratio"][0],
    _FIELD_SPANS["action_slot_fugitive_scores"][1],
)

# Distinct, irregularly spaced lattice points keep every generated segment
# non-degenerate and avoid accidental geometric symmetries.
_LATTICE = tuple(
    (float(column) * 137.0, float(row) * 149.0) for row in range(4) for column in range(4)
)


@dataclass(frozen=True)
class ObservationCase:
    """One fully specified pre-action state snapshot for both variants."""

    network: ModelNetwork
    police: tuple[VehiclePlacement, ...]
    fugitive: VehiclePlacement
    step: int
    max_steps: int
    incoming_heading: float | None


def _build_network(
    positions: tuple[tuple[float, float], ...],
    edges: tuple[tuple[int, int], ...],
    boundary_flags: tuple[bool, ...],
) -> ModelNetwork:
    segments = tuple(
        Segment(
            id=index,
            start_id=start,
            end_id=end,
            length_m=math.dist(positions[start], positions[end]),
            geometry_xy=(positions[start], positions[end]),
        )
        for index, (start, end) in enumerate(edges)
    )
    intersections = tuple(
        Intersection(
            id=node,
            position_xy=position,
            source_signature=f"property13-signature-{node}",
            boundary_kind="bbox" if boundary_flags[node] else None,
            outgoing_segment_ids=tuple(item.id for item in segments if item.start_id == node),
            incoming_segment_ids=tuple(item.id for item in segments if item.end_id == node),
        )
        for node, position in enumerate(positions)
    )
    return ModelNetwork(intersections, segments)


@st.composite
def networks(draw: st.DrawFn) -> ModelNetwork:
    """Generate a directed, always-connected road network with optional chords."""
    node_count = draw(st.integers(min_value=4, max_value=6))
    order = draw(st.permutations(range(len(_LATTICE))))
    positions = tuple(_LATTICE[index] for index in order[:node_count])
    edges = [(node, (node + 1) % node_count) for node in range(node_count)]
    chords = draw(
        st.lists(
            st.tuples(
                st.integers(min_value=0, max_value=node_count - 1),
                st.integers(min_value=0, max_value=node_count - 1),
            ),
            max_size=4,
        )
    )
    for start, end in chords:
        if start != end and (start, end) not in edges:
            edges.append((start, end))
    boundary_flags = draw(
        st.lists(st.booleans(), min_size=node_count, max_size=node_count)
    )
    return _build_network(positions, tuple(edges), tuple(boundary_flags))


@st.composite
def placements(draw: st.DrawFn, network: ModelNetwork) -> VehiclePlacement:
    if draw(st.booleans()):
        return VehiclePlacement(
            segment_id=draw(st.integers(min_value=0, max_value=len(network.segments) - 1)),
            progress=draw(
                st.floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False)
            ),
        )
    return VehiclePlacement(
        intersection_id=draw(st.integers(min_value=0, max_value=len(network.intersections) - 1))
    )


@st.composite
def observation_cases(draw: st.DrawFn) -> ObservationCase:
    """Generate a network state: topology, six officers, a fugitive and episode timing."""
    network = draw(networks())
    police = tuple(draw(placements(network)) for _ in range(POLICE_COUNT))
    fugitive = draw(placements(network))
    max_steps = draw(st.integers(min_value=1, max_value=400))
    return ObservationCase(
        network=network,
        police=police,
        fugitive=fugitive,
        step=draw(st.integers(min_value=0, max_value=max_steps)),
        max_steps=max_steps,
        incoming_heading=draw(
            st.one_of(
                st.none(),
                st.floats(
                    min_value=-math.pi,
                    max_value=math.pi,
                    allow_nan=False,
                    allow_infinity=False,
                ),
            )
        ),
    )


def _observe(adapter, case: ObservationCase, *, police_index: int, police, fugitive=None):
    return adapter.observe(
        police_index=police_index,
        police=police,
        fugitive=case.fugitive if fugitive is None else fugitive,
        step=case.step,
        max_steps=case.max_steps,
        incoming_heading=case.incoming_heading,
    )


def _relabel_police(
    police: tuple[VehiclePlacement, ...], permutation
) -> tuple[VehiclePlacement, ...]:
    """Move the officer holding slot ``old`` to slot ``permutation[old]``."""
    relabeled: list[VehiclePlacement | None] = [None] * len(police)
    for old_slot, new_slot in enumerate(permutation):
        relabeled[new_slot] = police[old_slot]
    return tuple(relabeled)


def _other_ranks(observer: int) -> dict[int, int]:
    """Position of each other officer slot inside ``other_agent_distances``."""
    return {
        slot: rank
        for rank, slot in enumerate(slot for slot in range(POLICE_COUNT) if slot != observer)
    }


def _relabel_network(
    network: ModelNetwork, node_permutation, segment_permutation
) -> ModelNetwork:
    """Rewrite canonical storage IDs while preserving topology and geometry."""
    segments = tuple(
        sorted(
            (
                Segment(
                    id=segment_permutation[segment.id],
                    start_id=node_permutation[segment.start_id],
                    end_id=node_permutation[segment.end_id],
                    length_m=segment.length_m,
                    geometry_xy=segment.geometry_xy,
                )
                for segment in network.segments
            ),
            key=lambda item: item.id,
        )
    )
    intersections = tuple(
        sorted(
            (
                Intersection(
                    id=node_permutation[item.id],
                    position_xy=item.position_xy,
                    source_signature=item.source_signature,
                    boundary_kind=item.boundary_kind,
                    outgoing_segment_ids=tuple(
                        segment_permutation[value] for value in item.outgoing_segment_ids
                    ),
                    incoming_segment_ids=tuple(
                        segment_permutation[value] for value in item.incoming_segment_ids
                    ),
                )
                for item in network.intersections
            ),
            key=lambda item: item.id,
        )
    )
    return ModelNetwork(intersections, segments)


def _relabel_placement(
    placement: VehiclePlacement, node_permutation, segment_permutation
) -> VehiclePlacement:
    if placement.segment_id is not None:
        return VehiclePlacement(
            segment_id=segment_permutation[placement.segment_id],
            progress=placement.progress,
        )
    return VehiclePlacement(intersection_id=node_permutation[placement.intersection_id])


def test_declared_field_layout_sums_to_the_published_variant_dimensions() -> None:
    assert sum(OSM_FIELD_SIZES) == OBSERVATION_21D_DIM == 21
    assert len(AUGMENTED_FIELD_ORDER) == 7
    assert OBSERVATION_21D_DIM + len(AUGMENTED_FIELD_ORDER) == OBSERVATION_28D_DIM == 28
    # The only identity-bearing 21D field is the positional action-slot one-hot;
    # no declared field is a raw or canonical storage identifier.
    assert "police_onehot" in OSM_FIELD_ORDER
    assert not any(
        name == "id" or name.endswith("_id") or name.endswith("_ids")
        for name in OSM_FIELD_ORDER + AUGMENTED_FIELD_ORDER
    )


@_PBT_SETTINGS
@given(case=observation_cases())
def test_both_variants_emit_declared_dimensions_that_are_finite_and_unit_normalized(
    case: ObservationCase,
) -> None:
    adapter_21 = Observation21DAdapter(case.network)
    adapter_28 = Observation28DAdapter(case.network)
    for observer in range(POLICE_COUNT):
        vector_21 = _observe(adapter_21, case, police_index=observer, police=case.police)
        vector_28 = _observe(adapter_28, case, police_index=observer, police=case.police)
        for vector, dimension in (
            (vector_21, OBSERVATION_21D_DIM),
            (vector_28, OBSERVATION_28D_DIM),
        ):
            assert vector.shape == (dimension,)
            assert vector.dtype == np.float32
            assert np.all(np.isfinite(vector))
            assert np.all(vector >= 0.0)
            assert np.all(vector <= 1.0)
        # The 28D variant is a strict extension of the audited 21D prefix.
        assert np.array_equal(vector_28[:OBSERVATION_21D_DIM], vector_21)


@_PBT_SETTINGS
@given(case=observation_cases(), permutation=st.permutations(range(POLICE_COUNT)))
def test_relabeling_officer_slots_permutes_only_the_identity_onehot(
    case: ObservationCase, permutation
) -> None:
    """Observing the same physical officer under a different slot label is content-identical.

    For a bijection ``permutation`` on slots 0..5 and a police tuple relabeled so
    the officer holding slot ``s`` now holds slot ``permutation[s]``:

    * the one-hot block is exactly the same vector permuted by ``permutation``;
    * every own-state/geometric feature is unchanged;
    * the distance to another officer follows that officer to its new slot;
    * every 28D team feature is unchanged.
    """
    assume(list(permutation) != list(range(POLICE_COUNT)))
    relabeled_police = _relabel_police(case.police, permutation)
    adapter_21 = Observation21DAdapter(case.network)
    adapter_28 = Observation28DAdapter(case.network)
    own_start, own_end = _OWN_STATE_SPAN

    for adapter in (adapter_21, adapter_28):
        for observer in range(POLICE_COUNT):
            original = _observe(adapter, case, police_index=observer, police=case.police)
            moved = _observe(
                adapter,
                case,
                police_index=permutation[observer],
                police=relabeled_police,
            )

            assert original[_ONEHOT_START + observer] == 1.0
            assert moved[_ONEHOT_START + permutation[observer]] == 1.0
            for slot in range(POLICE_COUNT):
                assert (
                    original[_ONEHOT_START + slot]
                    == moved[_ONEHOT_START + permutation[slot]]
                )

            np.testing.assert_allclose(
                original[own_start:own_end],
                moved[own_start:own_end],
                rtol=0.0,
                atol=_TOLERANCE,
            )

            original_ranks = _other_ranks(observer)
            moved_ranks = _other_ranks(permutation[observer])
            for other in range(POLICE_COUNT):
                if other == observer:
                    continue
                assert original[_OTHERS_START + original_ranks[other]] == pytest.approx(
                    moved[_OTHERS_START + moved_ranks[permutation[other]]],
                    abs=_TOLERANCE,
                )
            fugitive_slot = _OTHERS_START + POLICE_COUNT - 1
            assert original[fugitive_slot] == pytest.approx(
                moved[fugitive_slot], abs=_TOLERANCE
            )

            np.testing.assert_allclose(
                original[OBSERVATION_21D_DIM:],
                moved[OBSERVATION_21D_DIM:],
                rtol=0.0,
                atol=_TOLERANCE,
            )


@_PBT_SETTINGS
@given(case=observation_cases(), seed=st.integers(min_value=0, max_value=2**31 - 1))
def test_bijective_canonical_id_relabeling_leaves_both_variants_unchanged(
    case: ObservationCase, seed: int
) -> None:
    """No raw or canonical storage identifier reaches either observation vector."""
    node_count = len(case.network.intersections)
    segment_count = len(case.network.segments)
    generator = np.random.default_rng(seed)
    node_permutation = generator.permutation(node_count).tolist()
    segment_permutation = generator.permutation(segment_count).tolist()
    assume(node_permutation != list(range(node_count)))

    relabeled_network = _relabel_network(
        case.network, node_permutation, segment_permutation
    )
    relabeled_police = tuple(
        _relabel_placement(placement, node_permutation, segment_permutation)
        for placement in case.police
    )
    relabeled_fugitive = _relabel_placement(
        case.fugitive, node_permutation, segment_permutation
    )

    for factory in (Observation21DAdapter, Observation28DAdapter):
        original_adapter = factory(case.network)
        relabeled_adapter = factory(relabeled_network)
        for observer in range(POLICE_COUNT):
            original = _observe(
                original_adapter, case, police_index=observer, police=case.police
            )
            relabeled = _observe(
                relabeled_adapter,
                case,
                police_index=observer,
                police=relabeled_police,
                fugitive=relabeled_fugitive,
            )
            np.testing.assert_allclose(original, relabeled, rtol=0.0, atol=_TOLERANCE)


@_PBT_SETTINGS
@given(
    hidden_width=st.integers(min_value=8, max_value=256),
    layer_count=st.integers(min_value=1, max_value=3),
    action_dim=st.integers(min_value=2, max_value=12),
    num_officers=st.integers(min_value=1, max_value=12),
)
def test_capacity_accounting_detects_every_silent_observation_dimension_change(
    hidden_width: int, layer_count: int, action_dim: int, num_officers: int
) -> None:
    """A wrong declared dimension can never pass as a capacity-matched condition."""
    hidden_dims = (hidden_width,) * layer_count
    shared = {
        "action_dim": action_dim,
        "num_officers": num_officers,
        "hidden_dims": hidden_dims,
    }

    base = actor_capacity(OBSERVATION_21D_DIM, **shared)
    augmented = actor_capacity(OBSERVATION_28D_DIM, **shared)
    extra_features = OBSERVATION_28D_DIM - OBSERVATION_21D_DIM
    assert augmented.parameter_count - base.parameter_count == extra_features * hidden_width

    comparison = compare_observation_capacities(
        base_observation_dim=OBSERVATION_21D_DIM,
        matched_observation_dim=OBSERVATION_28D_DIM,
        **shared,
    )
    assert comparison.relative_parameter_difference > 0.0

    identical = compare_observation_capacities(
        base_observation_dim=OBSERVATION_21D_DIM,
        matched_observation_dim=OBSERVATION_21D_DIM,
        **shared,
    )
    assert identical.relative_parameter_difference == 0.0
    assert identical.is_capacity_matched

    # The detector is live at the audited default width: swapping 21D for 28D
    # without re-tuning hidden_dims is reported as a capacity mismatch.
    audited = compare_observation_capacities(
        base_observation_dim=OBSERVATION_21D_DIM,
        matched_observation_dim=OBSERVATION_28D_DIM,
        action_dim=_AUDITED_ACTION_DIM,
        num_officers=POLICE_COUNT,
        hidden_dims=_AUDITED_HIDDEN_DIMS,
    )
    assert not audited.is_capacity_matched
