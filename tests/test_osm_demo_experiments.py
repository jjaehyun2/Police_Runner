"""Focused offline tests for experiment plans, run lineage and the claim registry."""

import pytest

from pursuit_evasion_rl.osm_demo.claims import ClaimKind, ClaimRegistry
from pursuit_evasion_rl.osm_demo.experiments import (
    OSM_OBSERVATION_PROFILE,
    EvidenceState,
    ExperimentEvidence,
    ExperimentPlan,
    ImplementationEvidence,
    RegionSplits,
    ReproducibilityRecord,
    TrainingRunLineage,
)
from pursuit_evasion_rl.osm_demo.models import (
    ClaimStatus,
    DomainValidationError,
    RunLineage,
    RunManifest,
    RunMode,
)

UPDATED_AT = "2024-01-01T00:00:00Z"


def make_region_splits() -> RegionSplits:
    return RegionSplits(train={"net-train"}, validation={"net-val"}, test={"net-test"})


def make_plan(**overrides) -> ExperimentPlan:
    kwargs = dict(
        plan_id="plan-1",
        region_splits=make_region_splits(),
        checkpoint_candidates=("ckpt-osm",),
        fugitive_rules={"kind": "heuristic"},
        capture_radius_m=15.0,
        max_steps=100,
        episode_count=2,
        seeds=(1, 2),
        success_criteria={"min_capture_rate": 0.5},
        evaluation_config={"paired": True},
        confidence_interval_method="wilson",
        minimum_evaluation_episodes=1,
    )
    kwargs.update(overrides)
    return ExperimentPlan.create(**kwargs)


def make_eval_run(**overrides) -> RunManifest:
    lineage = RunLineage(
        split_manifest_hash="split-1",
        train_network_hashes={"net-train"},
        validation_network_hashes={"net-val"},
        test_network_hashes={"net-test"},
        code_version="v1",
    )
    kwargs = dict(
        run_id="run-eval",
        mode=RunMode.OSM_EVALUATION,
        network_hash="net-test",
        observation_profile=OSM_OBSERVATION_PROFILE,
        seeds=(1, 2),
        policy_kind="learned",
        claim_status=ClaimStatus.EXPERIMENTAL,
        checkpoint_hash="ckpt-osm",
        plan_id="plan-1",
        artifacts={"metrics": "h1"},
        run_config={"max_steps": 100},
        lineage=lineage,
    )
    kwargs.update(overrides)
    return RunManifest(**kwargs)


def make_repro(**overrides) -> ReproducibilityRecord:
    kwargs = dict(
        checkpoint_hash="ckpt-osm",
        network_hashes={"net-test"},
        run_config_hash="rc-hash",
        code_hash="code-hash",
        input_artifact_hashes={"metrics": "h1"},
    )
    kwargs.update(overrides)
    return ReproducibilityRecord(**kwargs)


def make_evidence(plan: ExperimentPlan, **overrides) -> ExperimentEvidence:
    kwargs = dict(
        evidence_id="ev-1",
        run_manifest=make_eval_run(),
        plan_hash=plan.frozen_config_hash,
        state=EvidenceState.COMPLETE,
        episode_count=2,
        training_region_hashes={"net-train"},
        evaluation_region_hashes={"net-test"},
        success_criteria_met=True,
        confidence_interval_method="wilson",
        artifact_hashes={"metrics": "h1"},
        reproducibility=make_repro(),
    )
    kwargs.update(overrides)
    return ExperimentEvidence(**kwargs)


# --- Experiment plan immutability and completeness -------------------------------


def test_plan_create_freezes_matching_config_hash():
    plan = make_plan()
    assert plan.frozen_config_hash
    # Reconstructing with the same content reproduces the frozen hash.
    assert make_plan().frozen_config_hash == plan.frozen_config_hash


def test_plan_rejects_tampered_frozen_hash():
    plan = make_plan()
    with pytest.raises(DomainValidationError) as raised:
        ExperimentPlan(
            plan_id=plan.plan_id,
            frozen_config_hash="deadbeef",
            region_splits=plan.region_splits,
            checkpoint_candidates=plan.checkpoint_candidates,
            fugitive_rules=plan.fugitive_rules,
            capture_radius_m=plan.capture_radius_m,
            max_steps=plan.max_steps,
            episode_count=plan.episode_count,
            seeds=plan.seeds,
            success_criteria=plan.success_criteria,
            evaluation_config=plan.evaluation_config,
            confidence_interval_method=plan.confidence_interval_method,
            minimum_evaluation_episodes=plan.minimum_evaluation_episodes,
        )
    assert raised.value.code == "PLAN_HASH_MISMATCH"


def test_plan_requires_one_seed_per_episode():
    with pytest.raises(DomainValidationError) as raised:
        make_plan(seeds=(1,))
    assert raised.value.code == "SEED_COUNT_MISMATCH"


def test_region_splits_reject_overlap():
    with pytest.raises(DomainValidationError) as raised:
        RegionSplits(train={"a"}, validation={"b"}, test={"a"})
    assert raised.value.code == "OVERLAPPING_REGION_SPLITS"


def test_registry_rejects_conflicting_plan_registration():
    registry = ClaimRegistry()
    registry.register_plan(make_plan())
    with pytest.raises(DomainValidationError) as raised:
        registry.register_plan(make_plan(max_steps=200))
    assert raised.value.code == "IMMUTABLE_PLAN_CONFLICT"


# --- Verified OSM evidence eligibility -------------------------------------------


def test_complete_planned_evidence_is_eligible_and_verifies_claim():
    registry = ClaimRegistry()
    plan = registry.register_plan(make_plan())
    evidence = registry.register_evidence(make_evidence(plan))

    assert registry.osm_verification_failures(evidence) == ()
    assert evidence in registry.verified_osm_evidence()

    registry.register_claim(
        "osm-capture-rate", ClaimKind.OSM_PERFORMANCE, reason="init", updated_at=UPDATED_AT
    )
    record = registry.transition(
        "osm-capture-rate",
        ClaimStatus.VERIFIED,
        evidence_ids=("ev-1",),
        reason="pre-registered evaluation met criteria",
        updated_at=UPDATED_AT,
    )
    assert record.status is ClaimStatus.VERIFIED
    assert record.evidence_ids == ("ev-1",)


def test_legacy_direct_evidence_is_excluded_from_verified_osm():
    registry = ClaimRegistry()
    plan = registry.register_plan(make_plan())
    legacy_run = make_eval_run(
        run_id="run-legacy",
        mode=RunMode.LEGACY_DIRECT,
        observation_profile="legacy_grid_v0",
        plan_id=None,
        lineage=None,
    )
    evidence = registry.register_evidence(
        make_evidence(plan, evidence_id="ev-legacy", run_manifest=legacy_run)
    )

    reasons = registry.osm_verification_failures(evidence)
    assert "legacy-direct-excluded" in reasons
    assert evidence not in registry.verified_osm_evidence()


@pytest.mark.parametrize(
    ("overrides", "expected_reason"),
    [
        ({"state": EvidenceState.PRELIMINARY, "success_criteria_met": False}, "evidence-preliminary"),
        ({"state": EvidenceState.FAILED, "success_criteria_met": False}, "evidence-failed"),
        ({"reproducibility": None}, "unreproducible-artifacts"),
    ],
)
def test_preliminary_failed_and_unreproducible_evidence_excluded(overrides, expected_reason):
    registry = ClaimRegistry()
    plan = registry.register_plan(make_plan())
    evidence = registry.register_evidence(make_evidence(plan, evidence_id="ev-x", **overrides))

    reasons = registry.osm_verification_failures(evidence)
    assert expected_reason in reasons
    assert evidence not in registry.verified_osm_evidence()


def test_transition_to_verified_rejects_ineligible_evidence():
    registry = ClaimRegistry()
    plan = registry.register_plan(make_plan())
    registry.register_evidence(make_evidence(plan, evidence_id="ev-bad", reproducibility=None))
    registry.register_claim(
        "osm-capture-rate", ClaimKind.OSM_PERFORMANCE, reason="init", updated_at=UPDATED_AT
    )

    with pytest.raises(DomainValidationError) as raised:
        registry.transition(
            "osm-capture-rate",
            ClaimStatus.VERIFIED,
            evidence_ids=("ev-bad",),
            reason="attempt",
            updated_at=UPDATED_AT,
        )
    assert raised.value.code == "INELIGIBLE_OSM_EVIDENCE"


def test_success_criteria_requires_complete_state():
    plan = make_plan()
    with pytest.raises(DomainValidationError) as raised:
        make_evidence(plan, state=EvidenceState.PRELIMINARY, success_criteria_met=True)
    assert raised.value.code == "INCOMPLETE_SUCCESS_EVIDENCE"


# --- Feature claim guards --------------------------------------------------------


def test_feature_claim_requires_passed_reproducible_automated_checks():
    registry = ClaimRegistry()
    registry.register_claim(
        "coarsener", ClaimKind.FEATURE, reason="init", updated_at=UPDATED_AT
    )
    failing = registry.register_evidence(
        ImplementationEvidence(
            evidence_id="impl-fail",
            checks=("pytest",),
            artifact_hashes={"suite": "h"},
            passed=False,
            reproducible=True,
        )
    )
    assert failing.evidence_id == "impl-fail"
    with pytest.raises(DomainValidationError) as raised:
        registry.transition(
            "coarsener",
            ClaimStatus.IMPLEMENTED,
            evidence_ids=("impl-fail",),
            reason="attempt",
            updated_at=UPDATED_AT,
        )
    assert raised.value.code == "AUTOMATED_CHECK_REQUIRED"

    registry.register_evidence(
        ImplementationEvidence(
            evidence_id="impl-ok",
            checks=("pytest",),
            artifact_hashes={"suite": "h"},
            passed=True,
            reproducible=True,
        )
    )
    record = registry.transition(
        "coarsener",
        ClaimStatus.IMPLEMENTED,
        evidence_ids=("impl-ok",),
        reason="checks passed",
        updated_at=UPDATED_AT,
    )
    assert record.status is ClaimStatus.IMPLEMENTED


def test_synthetic_grid_only_generalization_claim_is_unsupported():
    registry = ClaimRegistry()
    record = registry.register_claim(
        "osm-generalization",
        ClaimKind.OSM_GENERALIZATION,
        reason="grid trained only",
        updated_at=UPDATED_AT,
        synthetic_grid_only=True,
    )
    assert record.status is ClaimStatus.UNSUPPORTED


# --- Training mode / split distinctions ------------------------------------------


def test_training_lineage_separates_finetune_and_from_scratch():
    splits = make_region_splits()
    finetune_run = RunManifest(
        run_id="run-ft",
        mode=RunMode.OSM_FINETUNE,
        network_hash="net-train",
        observation_profile=OSM_OBSERVATION_PROFILE,
        seeds=(7,),
        policy_kind="learned",
        claim_status=ClaimStatus.EXPERIMENTAL,
        checkpoint_hash="ckpt-ft",
        run_config={"epochs": 1},
        lineage=RunLineage(
            split_manifest_hash="split-1",
            train_network_hashes={"net-train"},
            validation_network_hashes={"net-val"},
            test_network_hashes={"net-test"},
            code_version="v1",
            parent_checkpoint_hash="legacy-parent",
        ),
    )
    lineage = TrainingRunLineage(
        run_manifest=finetune_run,
        region_splits=splits,
        parent_checkpoint_hash="legacy-parent",
    )
    assert lineage.has_complete_unseen_test_evidence is False

    with pytest.raises(DomainValidationError) as raised:
        TrainingRunLineage(
            run_manifest=finetune_run,
            region_splits=splits,
            parent_checkpoint_hash=None,
        )
    assert raised.value.code == "MISSING_PARENT_CHECKPOINT"
