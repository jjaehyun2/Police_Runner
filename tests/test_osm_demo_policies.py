"""Offline tests for action masks, learned-policy loading and atomic decoding.

These cover Requirements 6.4-6.5 and 8.1-8.6 without any external OSM access:

* the six-entry mask maps five ordered exits plus stay (6.4),
* masking is applied before probabilities and a nonexistent/masked exit is a
  classified error (6.5),
* atomic ActionRecommendation outputs carry every mandatory field (8.2),
* an incomplete actor output fails the whole request with no partial results
  (8.3),
* structural/observation-profile compatibility gates the verified OSM route and
  legacy-direct inference is experimental-only and opt-in (8.1, 8.5, 8.6),
* a genuinely saved [21,128,128,6] checkpoint loads on CPU with weights-only
  safety and yields six complete recommendations (8.1-8.2).
"""

from __future__ import annotations

import numpy as np
import pytest

from pursuit_evasion_rl.osm_demo.checkpoint import CheckpointInspector
from pursuit_evasion_rl.osm_demo.models import (
    DomainValidationError,
    Intersection,
    ModelNetwork,
    Segment,
    VehiclePlacement,
)
from pursuit_evasion_rl.osm_demo.observations import (
    OSMTopologyV1Adapter,
    osm_topology_v1_contract,
)
from pursuit_evasion_rl.osm_demo.policies import (
    ACTION_DIM,
    STAY_ACTION,
    ActionRecommendation,
    LearnedPolicePolicy,
    LegacyDirectPolicePolicy,
    build_action_mask,
    decode_recommendation,
    masked_probabilities,
    osm_execution_config,
)

torch = pytest.importorskip("torch")


# --------------------------------------------------------------------------
# Fixtures
# --------------------------------------------------------------------------


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
    """Directed network: 0->1, 1->2, 1->3, 2->3 (intersection 3 is a bbox boundary)."""
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


class _ActorModule(torch.nn.Module):
    """Mirror of the training MLPNetwork actor (``self.network`` Sequential)."""

    def __init__(self, obs_dim=21, action_dim=6, hidden=(128, 128)):
        super().__init__()
        layers = []
        previous = obs_dim
        for size in hidden:
            layers.append(torch.nn.Linear(previous, size))
            layers.append(torch.nn.ReLU())
            previous = size
        layers.append(torch.nn.Linear(previous, action_dim))
        self.network = torch.nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)


def _save_checkpoint(path):
    torch.manual_seed(0)
    actor = _ActorModule()
    torch.save({"police_actor": actor.state_dict()}, path)
    return path


def _osm_manifest(adapter):
    """A dict manifest declaring the osm_topology_v1 profile and architecture."""
    return {
        "observation_contract": adapter.contract,
        "architecture": {"observation_dim": 21, "hidden_dims": (128, 128), "action_dim": 6},
        "police_count": 6,
        "fixed_max_degree": 5,
        "training_networks": ["train-net-hash"],
    }


def _pass_report(path, adapter):
    inspection = CheckpointInspector().inspect(
        path,
        _osm_manifest(adapter),
        selected_observation_contract=adapter.contract,
        execution_config=osm_execution_config(),
        network_hash="",
    )
    return inspection.report


# --------------------------------------------------------------------------
# Masking (Requirement 6.4)
# --------------------------------------------------------------------------


def test_mask_maps_ordered_exits_plus_stay():
    network = _build_network()
    mask, ordered = build_action_mask(network, 1)
    assert mask.shape == (ACTION_DIM,)
    assert len(ordered) == 2  # intersection 1 has two outgoing exits
    assert bool(mask[0]) and bool(mask[1])
    assert not bool(mask[2]) and not bool(mask[3]) and not bool(mask[4])
    assert bool(mask[STAY_ACTION])  # stay is always valid


def test_sink_intersection_only_allows_stay():
    network = _build_network()
    mask, ordered = build_action_mask(network, 3)  # boundary sink, no exits
    assert ordered == ()
    assert bool(mask[STAY_ACTION])
    assert not mask[:STAY_ACTION].any()


# --------------------------------------------------------------------------
# Mask-before-probabilities (Requirement 6.5)
# --------------------------------------------------------------------------


def test_masked_slots_receive_zero_probability():
    mask = np.array([True, True, False, False, False, True])
    logits = np.array([1.0, 1.0, 50.0, 50.0, 50.0, 1.0])  # masked slots have huge logits
    probabilities = masked_probabilities(logits, mask)
    assert probabilities[2] == 0.0 and probabilities[3] == 0.0 and probabilities[4] == 0.0
    assert probabilities.sum() == pytest.approx(1.0)
    # Greedy selection ignores the masked high-logit slots.
    assert int(np.argmax(np.where(mask, probabilities, -np.inf))) in {0, 1, STAY_ACTION}


def test_masked_action_selection_is_an_error():
    network = _build_network()
    mask, ordered = build_action_mask(network, 1)  # slots 0,1 and stay valid
    probabilities = np.full(ACTION_DIM, 1.0 / ACTION_DIM)
    with pytest.raises(DomainValidationError) as excinfo:
        decode_recommendation(
            agent_id="police_0",
            network=network,
            decision_intersection_id=1,
            mask=mask,
            ordered_segment_ids=ordered,
            probabilities=probabilities,
            profile="osm_topology_v1",
            compatibility_report_id="report",
            action_index=3,  # masked slot
        )
    assert excinfo.value.code == "MASKED_ACTION_SELECTED"


def test_nonexistent_exit_selection_is_an_error():
    """A mask bit set for a slot without a backing segment must error (6.5)."""
    network = _build_network()
    # Intersection 1 has exactly two ordered exits; slot 2 has no segment.
    ordered = (1, 2)
    mask = np.array([True, True, True, False, False, True])  # slot 2 wrongly enabled
    probabilities = np.full(ACTION_DIM, 1.0 / ACTION_DIM)
    with pytest.raises(DomainValidationError) as excinfo:
        decode_recommendation(
            agent_id="police_0",
            network=network,
            decision_intersection_id=1,
            mask=mask,
            ordered_segment_ids=ordered,
            probabilities=probabilities,
            profile="osm_topology_v1",
            compatibility_report_id="report",
            action_index=2,
        )
    assert excinfo.value.code == "NONEXISTENT_EXIT_SELECTED"


# --------------------------------------------------------------------------
# Atomic recommendation decoding (Requirement 8.2)
# --------------------------------------------------------------------------


def test_atomic_recommendation_has_all_fields_for_movement():
    network = _build_network()
    mask, ordered = build_action_mask(network, 1)
    logits = np.array([10.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    probabilities = masked_probabilities(logits, mask)
    recommendation = decode_recommendation(
        agent_id="police_0",
        network=network,
        decision_intersection_id=1,
        mask=mask,
        ordered_segment_ids=ordered,
        probabilities=probabilities,
        profile="osm_topology_v1",
        compatibility_report_id="report-123",
    )
    assert isinstance(recommendation, ActionRecommendation)
    assert recommendation.action_index == 0
    assert recommendation.segment_id == ordered[0]
    segment = next(s for s in network.segments if s.id == ordered[0])
    assert recommendation.next_intersection_id == segment.end_id
    assert recommendation.valid is True
    assert 0.0 <= recommendation.probability <= 1.0
    assert recommendation.profile == "osm_topology_v1"
    assert recommendation.compatibility_report_id == "report-123"


def test_stay_recommendation_uses_null_segment():
    network = _build_network()
    mask, ordered = build_action_mask(network, 3)  # sink -> only stay
    probabilities = masked_probabilities(np.zeros(ACTION_DIM), mask)
    recommendation = decode_recommendation(
        agent_id="police_0",
        network=network,
        decision_intersection_id=3,
        mask=mask,
        ordered_segment_ids=ordered,
        probabilities=probabilities,
        profile="osm_topology_v1",
        compatibility_report_id="report",
    )
    assert recommendation.action_index == STAY_ACTION
    assert recommendation.segment_id is None
    assert recommendation.next_intersection_id == 3


# --------------------------------------------------------------------------
# Whole-request failure on incomplete output (Requirement 8.3)
# --------------------------------------------------------------------------


def test_incomplete_actor_output_fails_whole_request(tmp_path):
    network = _build_network()
    adapter = OSMTopologyV1Adapter(network, clip_distance_m=500.0)
    path = _save_checkpoint(tmp_path / "actor.pt")
    report = _pass_report(path, adapter)

    def bad_actor(_observation):
        return np.array([1.0, 2.0, 3.0])  # only three logits, not ACTION_DIM

    policy = LearnedPolicePolicy(network, adapter, bad_actor, report=report)
    police = _police_at(1, 1, 1, 1, 1, 1)
    with pytest.raises(DomainValidationError) as excinfo:
        policy.recommend(
            police=police,
            fugitive=VehiclePlacement(intersection_id=2),
            step=0,
            max_steps=10,
        )
    assert excinfo.value.code == "INCOMPLETE_ACTOR_OUTPUT"


def test_recommend_returns_six_complete_recommendations(tmp_path):
    network = _build_network()
    adapter = OSMTopologyV1Adapter(network, clip_distance_m=500.0)
    path = _save_checkpoint(tmp_path / "actor.pt")
    report = _pass_report(path, adapter)

    def actor(_observation):
        return np.array([10.0, 0.0, 0.0, 0.0, 0.0, 0.0])  # prefer slot 0 when valid

    policy = LearnedPolicePolicy(network, adapter, actor, report=report)
    police = _police_at(1, 1, 1, 1, 1, 1)
    recommendations = policy.recommend(
        police=police,
        fugitive=VehiclePlacement(intersection_id=2),
        step=1,
        max_steps=20,
    )
    assert len(recommendations) == 6
    for index, recommendation in enumerate(recommendations):
        assert recommendation.agent_id == f"police_{index}"
        assert recommendation.action_index == 0  # intersection 1 has a valid slot 0
        assert recommendation.segment_id is not None
        assert recommendation.compatibility_report_id == report.report_id


# --------------------------------------------------------------------------
# Compatibility gating (Requirements 8.1, 8.5, 8.6)
# --------------------------------------------------------------------------


def test_from_checkpoint_loads_actor_and_recommends(tmp_path):
    network = _build_network()
    adapter = OSMTopologyV1Adapter(network, clip_distance_m=500.0)
    path = _save_checkpoint(tmp_path / "actor.pt")
    policy = LearnedPolicePolicy.from_checkpoint(
        network,
        path,
        adapter=adapter,
        manifest=_osm_manifest(adapter),
    )
    recommendations = policy.recommend(
        police=_police_at(0, 1, 2, 3, 1, 2),
        fugitive=VehiclePlacement(intersection_id=1),
        step=2,
        max_steps=40,
    )
    assert len(recommendations) == 6
    assert all(r.profile == "osm_topology_v1" for r in recommendations)


def test_missing_manifest_blocks_verified_osm_route(tmp_path):
    """No osm profile evidence => the verified OSM route is refused (8.6)."""
    network = _build_network()
    adapter = OSMTopologyV1Adapter(network, clip_distance_m=500.0)
    path = _save_checkpoint(tmp_path / "actor.pt")
    with pytest.raises(DomainValidationError) as excinfo:
        LearnedPolicePolicy.from_checkpoint(network, path, adapter=adapter)
    assert excinfo.value.code == "SEMANTIC_PROFILE_REQUIRED"


def test_legacy_direct_requires_opt_in(tmp_path):
    network = _build_network()
    adapter = OSMTopologyV1Adapter(network, clip_distance_m=500.0)
    path = _save_checkpoint(tmp_path / "actor.pt")
    report = _pass_report(path, adapter)

    def actor(_observation):
        return np.zeros(ACTION_DIM)

    with pytest.raises(DomainValidationError) as excinfo:
        LegacyDirectPolicePolicy(
            network, adapter, actor, report=report, allow_experimental_legacy=False
        )
    assert excinfo.value.code == "LEGACY_INFERENCE_NOT_OPTED_IN"

    # Opting in is permitted and remains experimental / legacy-profiled.
    policy = LegacyDirectPolicePolicy(
        network, adapter, actor, report=report, allow_experimental_legacy=True
    )
    assert policy.experimental is True
    assert policy.profile == "legacy_grid_v0"
