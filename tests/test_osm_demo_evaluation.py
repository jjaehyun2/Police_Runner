"""Focused offline tests for the pre-registered paired evaluation entry point.

Covers config freezing, seen/unseen separation, paired learned-vs-baseline
comparison and the non-overstatement guarantee that successful criteria are
never auto-verified (design section 8 flow 4 and section 9; Requirements
10.7-10.8, 14.1-14.6).  All networks are built in process from committed offline
fixtures; no external OSM access, torch checkpoints or real training are used.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from pursuit_evasion_rl.osm_demo.canonical import content_hash
from pursuit_evasion_rl.osm_demo.claims import ClaimKind, ClaimRegistry
from pursuit_evasion_rl.osm_demo.coarsening import coarsen_raw_graph, prepare_model_network
from pursuit_evasion_rl.osm_demo.evaluation import (
    SEEN_PARTITION,
    UNSEEN_PARTITION,
    AreaEvaluation,
    PairedEvaluationResult,
    episode_config_from_plan,
    run_paired_evaluation,
)
from pursuit_evasion_rl.osm_demo.experiments import (
    EvidenceState,
    ExperimentPlan,
    RegionSplits,
)
from pursuit_evasion_rl.osm_demo.fixtures import _edge, _node  # offline fixture helpers
from pursuit_evasion_rl.osm_demo.models import (
    POLICE_COUNT,
    ClaimStatus,
    DomainValidationError,
    EpisodeConfig,
    RawOSMGraph,
    RunMode,
)
from pursuit_evasion_rl.osm_demo.policies import (
    ACTION_DIM,
    STAY_ACTION,
    BaselinePolicePolicy,
    build_action_mask,
    decode_recommendation,
)

pytestmark = pytest.mark.offline

UPDATED_AT = "2024-01-01T00:00:00Z"


# ---------------------------------------------------------------------------
# Offline network builders (two distinct placeable grids => distinct hashes)
# ---------------------------------------------------------------------------
def _grid_raw(rows: int, cols: int, *, spacing: float = 100.0) -> RawOSMGraph:
    positions: dict[str, tuple[float, float]] = {}
    for row in range(rows):
        for col in range(cols):
            positions[f"n{row}_{col}"] = (col * spacing, row * spacing)
    nodes = tuple(_node(source_id, *position) for source_id, position in positions.items())
    edges = []
    for row in range(rows):
        for col in range(cols):
            here = f"n{row}_{col}"
            if col + 1 < cols:
                right = f"n{row}_{col + 1}"
                edges.append(_edge(here, right, f"h-{here}-{right}", positions, oneway=False, road_class="secondary"))
                edges.append(_edge(right, here, f"h-{right}-{here}", positions, oneway=False, road_class="secondary"))
            if row + 1 < rows:
                down = f"n{row + 1}_{col}"
                edges.append(_edge(here, down, f"v-{here}-{down}", positions, oneway=False, road_class="secondary"))
                edges.append(_edge(down, here, f"v-{down}-{here}", positions, oneway=False, road_class="secondary"))
    return RawOSMGraph(nodes=nodes, edges=tuple(edges))


def _network(rows: int, cols: int):
    return prepare_model_network(coarsen_raw_graph(_grid_raw(rows, cols))).network


class _StayPolicy:
    """A trivial deterministic policy where every officer stays.

    Satisfies the same ``PolicePolicy`` protocol as the baseline recommender so
    the paired evaluation can compare two genuinely different policies on
    identical placements and fugitive randomness.
    """

    profile = "stay_test_v0"
    experimental = False

    def __init__(self, network) -> None:
        self.network = network
        self._segments = {segment.id: segment for segment in network.segments}

    def _decision(self, placement) -> int:
        if placement.segment_id is not None:
            return int(self._segments[placement.segment_id].end_id)
        return int(placement.intersection_id)

    def recommend(self, *, police, fugitive, step, max_steps, incoming_headings=None):
        recommendations = []
        for index in range(POLICE_COUNT):
            heading = None if incoming_headings is None else incoming_headings[index]
            decision_id = self._decision(police[index])
            mask, ordered = build_action_mask(self.network, decision_id, heading)
            probabilities = np.zeros(ACTION_DIM, dtype=np.float64)
            probabilities[STAY_ACTION] = 1.0
            recommendations.append(
                decode_recommendation(
                    agent_id=f"police_{index}",
                    network=self.network,
                    decision_intersection_id=decision_id,
                    mask=mask,
                    ordered_segment_ids=ordered,
                    probabilities=probabilities,
                    profile=self.profile,
                    compatibility_report_id=self.profile,
                    action_index=STAY_ACTION,
                )
            )
        return tuple(recommendations)


# ---------------------------------------------------------------------------
# Plan / network fixtures
# ---------------------------------------------------------------------------
@pytest.fixture(scope="module")
def networks():
    seen = _network(3, 3)
    unseen = _network(3, 4)
    seen_hash = content_hash(seen)
    unseen_hash = content_hash(unseen)
    assert seen_hash != unseen_hash
    return {
        "seen": {seen_hash: seen},
        "unseen": {unseen_hash: unseen},
        "seen_hash": seen_hash,
        "unseen_hash": unseen_hash,
    }


def _make_plan(networks, *, success_criteria, episode_count=2, seeds=(1, 2), minimum=1) -> ExperimentPlan:
    splits = RegionSplits(
        train={networks["seen_hash"]},
        validation={"net-validation"},
        test={networks["unseen_hash"]},
    )
    return ExperimentPlan.create(
        plan_id="plan-eval",
        region_splits=splits,
        checkpoint_candidates=("ckpt-osm",),
        fugitive_rules={"kind": "heuristic"},
        capture_radius_m=20.0,
        max_steps=12,
        episode_count=episode_count,
        seeds=seeds,
        success_criteria=success_criteria,
        evaluation_config={"paired": True, "dt_s": 1.0, "police_speed_mps": 15.0, "fugitive_speed_mps": 10.0},
        confidence_interval_method="wilson",
        minimum_evaluation_episodes=minimum,
    )


def _learned_factory(network):
    return BaselinePolicePolicy(network)


def _baseline_factory(network):
    return _StayPolicy(network)


def _run(networks, plan) -> PairedEvaluationResult:
    return run_paired_evaluation(
        plan=plan,
        seen_networks=networks["seen"],
        unseen_networks=networks["unseen"],
        learned_policy_factory=_learned_factory,
        baseline_policy_factory=_baseline_factory,
        checkpoint_hash="ckpt-osm",
        code_version="v1",
        run_id="run-eval",
    )


# ---------------------------------------------------------------------------
# Config freezing (Requirement 14.1)
# ---------------------------------------------------------------------------
def test_episode_config_is_frozen_from_plan(networks):
    plan = _make_plan(networks, success_criteria={"min_capture_rate": 0.0})
    config = episode_config_from_plan(plan)
    assert config.capture_radius_m == plan.capture_radius_m
    assert config.max_steps == plan.max_steps
    assert config.deterministic is True


def test_checkpoint_must_be_pre_registered(networks):
    plan = _make_plan(networks, success_criteria={"min_capture_rate": 0.0})
    with pytest.raises(DomainValidationError) as raised:
        run_paired_evaluation(
            plan=plan,
            seen_networks=networks["seen"],
            unseen_networks=networks["unseen"],
            learned_policy_factory=_learned_factory,
            baseline_policy_factory=_baseline_factory,
            checkpoint_hash="not-registered",
            code_version="v1",
            run_id="run-eval",
        )
    assert raised.value.code == "CHECKPOINT_NOT_PRE_REGISTERED"


# ---------------------------------------------------------------------------
# Seen/unseen separation (Requirement 14.2)
# ---------------------------------------------------------------------------
def test_networks_must_belong_to_frozen_split(networks):
    plan = _make_plan(networks, success_criteria={"min_capture_rate": 0.0})
    # Passing the unseen (test-split) network as a "seen" network is rejected.
    with pytest.raises(DomainValidationError) as raised:
        run_paired_evaluation(
            plan=plan,
            seen_networks=networks["unseen"],
            unseen_networks=networks["unseen"],
            learned_policy_factory=_learned_factory,
            baseline_policy_factory=_baseline_factory,
            checkpoint_hash="ckpt-osm",
            code_version="v1",
            run_id="run-eval",
        )
    assert raised.value.code == "NETWORK_OUTSIDE_SPLIT"


def test_seen_and_unseen_results_are_separated(networks):
    plan = _make_plan(networks, success_criteria={"min_capture_rate": 0.0})
    result = _run(networks, plan)
    assert len(result.seen_evaluations) == 1
    assert len(result.unseen_evaluations) == 1
    assert result.seen_evaluations[0].partition == SEEN_PARTITION
    assert result.unseen_evaluations[0].partition == UNSEEN_PARTITION
    assert result.seen_evaluations[0].network_hash == networks["seen_hash"]
    assert result.unseen_evaluations[0].network_hash == networks["unseen_hash"]


# ---------------------------------------------------------------------------
# Paired comparison and equivalent accounting (Requirements 14.3-14.4)
# ---------------------------------------------------------------------------
def test_paired_episodes_share_placement_and_fugitive_stream(networks):
    plan = _make_plan(networks, success_criteria={"min_capture_rate": 0.0})
    result = _run(networks, plan)
    for area in result.seen_evaluations + result.unseen_evaluations:
        assert len(area.pairs) == plan.episode_count
        for pair in area.pairs:
            # Both policies see identical seven-vehicle initial placement.
            assert pair.learned.initial_state == pair.baseline.initial_state
            assert pair.learned.hashes["initial_state"] == pair.baseline.hashes["initial_state"]
            # Distinguished only by policy profile.
            assert pair.learned.hashes["policy_profile"] != pair.baseline.hashes["policy_profile"]


def test_metrics_use_equivalent_accounting(networks):
    plan = _make_plan(networks, success_criteria={"min_capture_rate": 0.0})
    result = _run(networks, plan)
    area = result.unseen_evaluations[0]
    for metrics in (area.learned_metrics, area.baseline_metrics):
        # Identical confidence method and conserved counts for both policies.
        assert metrics.latency["confidence_interval_method"] == plan.confidence_interval_method
        assert sum(metrics.outcomes.values()) == plan.episode_count
        assert set(metrics.outcomes) == {"capture", "escape", "timeout"}


def test_evaluation_is_reproducible(networks):
    plan = _make_plan(networks, success_criteria={"min_capture_rate": 0.0})
    first = _run(networks, plan)
    second = _run(networks, plan)
    assert first.run_manifest == second.run_manifest
    assert first.evidence == second.evidence


# ---------------------------------------------------------------------------
# No auto-verification (Requirements 10.7-10.8, 14.5-14.6)
# ---------------------------------------------------------------------------
def test_run_manifest_is_experimental_evaluation(networks):
    plan = _make_plan(networks, success_criteria={"min_capture_rate": 0.0})
    result = _run(networks, plan)
    manifest = result.run_manifest
    assert manifest.mode is RunMode.OSM_EVALUATION
    assert manifest.claim_status is ClaimStatus.EXPERIMENTAL
    assert manifest.plan_id == plan.plan_id
    assert manifest.checkpoint_hash == "ckpt-osm"
    assert tuple(manifest.seeds) == plan.seeds


def test_met_criteria_produce_complete_but_unverified_evidence(networks):
    # min_capture_rate >= 0 always holds, so criteria are met and evidence is
    # complete -- yet the entry point never promotes a claim to verified.
    plan = _make_plan(networks, success_criteria={"min_capture_rate": 0.0})
    result = _run(networks, plan)
    assert result.criteria_met is True
    assert result.success_criteria_met is True
    assert result.evidence.state is EvidenceState.COMPLETE
    assert result.run_manifest.claim_status is ClaimStatus.EXPERIMENTAL

    # The emitted evidence is well-formed and *eligible* for a later explicit
    # human verification, but registering the inputs does not verify anything.
    registry = ClaimRegistry()
    result.register_claim_inputs(
        registry, claim_name="osm-capture-rate", reason="paired eval", updated_at=UPDATED_AT
    )
    assert registry.osm_verification_failures(result.evidence) == ()
    assert result.evidence in registry.verified_osm_evidence()

    # No claim was auto-created as verified; a human must transition explicitly.
    record = registry.register_claim(
        "osm-capture-rate", ClaimKind.OSM_PERFORMANCE, reason="init", updated_at=UPDATED_AT
    )
    assert record.status is not ClaimStatus.VERIFIED


def test_unmet_criteria_produce_failed_evidence(networks):
    # A capture rate >= 1.5 is impossible, so criteria fail: evidence is FAILED,
    # success is not reported and the claim cannot become verified (Req. 14.5).
    plan = _make_plan(networks, success_criteria={"min_capture_rate": 1.5})
    result = _run(networks, plan)
    assert result.criteria_met is False
    assert result.success_criteria_met is False
    assert result.evidence.state is EvidenceState.FAILED

    registry = ClaimRegistry()
    result.register_claim_inputs(
        registry, claim_name="osm-capture-rate", reason="paired eval", updated_at=UPDATED_AT
    )
    # Failed evidence is excluded from verifiable OSM evidence.
    assert "success-criteria-not-met" in registry.osm_verification_failures(result.evidence)
    assert result.evidence not in registry.verified_osm_evidence()


def test_unknown_success_criterion_is_rejected(networks):
    plan = _make_plan(networks, success_criteria={"mystery_metric": 0.5})
    with pytest.raises(DomainValidationError) as raised:
        _run(networks, plan)
    assert raised.value.code == "UNKNOWN_SUCCESS_CRITERION"
