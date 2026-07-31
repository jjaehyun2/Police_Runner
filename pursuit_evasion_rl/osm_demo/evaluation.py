"""Pre-registered paired OSM evaluation entry point.

This module implements the evaluation side of the OSM_Training_Path (design
section 8 "OSM fine-tuning and retraining" flow 4 "OSM policy evaluation", and
section 9 "Metrics, latency and claim evidence"; Requirements 10.7-10.8 and
14.1-14.6).  It ties together the pieces built by earlier tasks -- the immutable
:class:`~pursuit_evasion_rl.osm_demo.experiments.ExperimentPlan`, the
deterministic paired :mod:`~pursuit_evasion_rl.osm_demo.runner`, the
:mod:`~pursuit_evasion_rl.osm_demo.metrics` aggregation and the evidence-backed
:mod:`~pursuit_evasion_rl.osm_demo.claims` registry -- behind a single,
mockable, fully-offline entry point :func:`run_paired_evaluation`.

Guarantees implemented here:

* **Frozen config before execution** (Requirement 14.1).  The supplied
  :class:`ExperimentPlan` is content-frozen: its ``frozen_config_hash`` is
  re-derived and compared before any episode runs, and the evaluation reuses the
  plan's seeds, episode count, capture radius and step limit verbatim so nothing
  is silently overridden at run time.  The checkpoint under evaluation must be
  one of the plan's pre-registered ``checkpoint_candidates``.

* **Seen vs unseen separation** (Requirement 14.2).  ``seen`` networks must all
  belong to the plan's training split and ``unseen`` networks to the test split;
  the two are evaluated and reported separately so a training-region score can
  never be mistaken for generalization.

* **Paired learned-vs-baseline comparison** (Requirements 14.3-14.4).  For every
  planned seed the learned and baseline policies run on the *same* deterministic
  placement and fugitive random stream (both derive from ``(seed, index)`` and
  neither depends on which policy runs), and both sides are summarized with the
  identical :func:`~pursuit_evasion_rl.osm_demo.metrics.summarize_metrics`
  accounting and confidence-interval method from the plan.

* **No auto-verification** (Requirements 10.7-10.8, 14.5-14.6).  The entry point
  emits run/metrics/claim *inputs* only: a ``osm-evaluation`` :class:`RunManifest`
  fixed to ``experimental``, per-policy :class:`MetricsSummary` objects carrying
  the confidence method and preliminary flag, and a candidate
  :class:`ExperimentEvidence`.  Success criteria are evaluated and reported but a
  claim is **never** promoted to ``verified`` automatically; that decision stays
  with an explicit :meth:`ClaimRegistry.transition` performed by a human.

The module performs no external OSM access and runs no real training; it consumes
already-prepared networks and policy factories.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Mapping, Sequence

from .canonical import content_hash
from .claims import ClaimKind, ClaimRegistry
from .experiments import (
    OSM_OBSERVATION_PROFILE,
    EvidenceState,
    ExperimentEvidence,
    ExperimentPlan,
    RegionSplits,
    ReproducibilityRecord,
)
from .metrics import OutcomeAggregate, aggregate_outcomes, summarize_metrics
from .models import (
    ClaimStatus,
    DomainValidationError,
    EpisodeConfig,
    EpisodeRecord,
    MetricsSummary,
    ModelNetwork,
    RunLineage,
    RunManifest,
    RunMode,
)
from .runner import FugitiveFactory, PairedEpisodeRecords, run_single_episode

# A policy factory builds a fresh policy bound to a specific network.  The
# evaluation is policy-agnostic: any object implementing the ``PolicePolicy``
# protocol (learned OSM actor, baseline shortest-path, ...) is accepted.
PolicyFactory = Callable[[ModelNetwork], Any]

# Partition labels used to keep seen (training) and unseen (test) areas apart.
SEEN_PARTITION = "seen"
UNSEEN_PARTITION = "unseen"

_LEARNED_KIND = "learned"
_BASELINE_KIND = "baseline"


# ---------------------------------------------------------------------------
# Result models
# ---------------------------------------------------------------------------
@dataclass(frozen=True, slots=True)
class AreaEvaluation:
    """Paired learned/baseline results for one evaluated network (Req. 14.2-14.4)."""

    network_hash: str
    partition: str
    learned_metrics: MetricsSummary
    baseline_metrics: MetricsSummary
    pairs: tuple[PairedEpisodeRecords, ...]

    @property
    def learned_records(self) -> tuple[EpisodeRecord, ...]:
        return tuple(pair.learned for pair in self.pairs)

    @property
    def baseline_records(self) -> tuple[EpisodeRecord, ...]:
        return tuple(pair.baseline for pair in self.pairs)


@dataclass(frozen=True, slots=True)
class CriterionResult:
    """One evaluated success criterion (reported, never auto-verified)."""

    name: str
    threshold: float
    value: float
    passed: bool


@dataclass(frozen=True, slots=True)
class PairedEvaluationResult:
    """Emitted run/metrics/claim inputs for a pre-registered paired evaluation.

    The result deliberately carries *inputs* for a later, explicit claim
    decision.  ``success_criteria_met`` reports whether the pre-registered
    criteria held on complete unseen evidence; it does **not** change any claim
    status by itself (Requirements 14.5-14.6).
    """

    plan: ExperimentPlan
    config: EpisodeConfig
    run_manifest: RunManifest
    seen_evaluations: tuple[AreaEvaluation, ...]
    unseen_evaluations: tuple[AreaEvaluation, ...]
    criteria_results: tuple[CriterionResult, ...]
    criteria_met: bool
    preliminary: bool
    evidence: ExperimentEvidence

    @property
    def success_criteria_met(self) -> bool:
        """True only when criteria held on complete (non-preliminary) evidence."""
        return bool(self.evidence.success_criteria_met)

    def register_claim_inputs(
        self,
        registry: ClaimRegistry,
        *,
        claim_name: str,
        reason: str,
        updated_at: str,
        kind: ClaimKind = ClaimKind.OSM_PERFORMANCE,
    ) -> ExperimentEvidence:
        """Register the plan, evidence and an *unpromoted* claim in ``registry``.

        This emits the claim inputs without transitioning the claim to
        ``verified``; the returned evidence can later back an explicit, guarded
        :meth:`ClaimRegistry.transition` performed by a human (Requirements
        14.5-14.6).  The registered claim is created in its guarded initial state
        (``experimental`` for OSM result claims), never ``verified``.
        """
        registry.register_plan(self.plan)
        registry.register_evidence(self.evidence)
        return self.evidence


# ---------------------------------------------------------------------------
# Success-criteria evaluation (reported, never auto-verifying)
# ---------------------------------------------------------------------------
def _rate(aggregate: OutcomeAggregate, outcome: str) -> float:
    return float(aggregate.rates.get(outcome, 0.0))


def _evaluate_criterion(
    name: str,
    threshold: float,
    learned: OutcomeAggregate,
    baseline: OutcomeAggregate,
) -> CriterionResult:
    """Evaluate one supported success criterion against the unseen aggregates."""
    threshold = float(threshold)
    if name == "min_capture_rate":
        value = _rate(learned, "capture")
        passed = value >= threshold
    elif name == "max_escape_rate":
        value = _rate(learned, "escape")
        passed = value <= threshold
    elif name == "max_timeout_rate":
        value = _rate(learned, "timeout")
        passed = value <= threshold
    elif name == "min_capture_rate_improvement":
        value = _rate(learned, "capture") - _rate(baseline, "capture")
        passed = value >= threshold
    elif name == "max_escape_rate_reduction":
        # Learned escape rate must be at least ``threshold`` below the baseline's.
        value = _rate(baseline, "escape") - _rate(learned, "escape")
        passed = value >= threshold
    else:
        raise DomainValidationError(
            "UNKNOWN_SUCCESS_CRITERION",
            f"Unsupported success criterion {name!r}",
            expected=sorted(
                (
                    "min_capture_rate",
                    "max_escape_rate",
                    "max_timeout_rate",
                    "min_capture_rate_improvement",
                    "max_escape_rate_reduction",
                )
            ),
            actual=name,
        )
    return CriterionResult(name=name, threshold=threshold, value=float(value), passed=bool(passed))


def _evaluate_success_criteria(
    success_criteria: Mapping[str, Any],
    learned: OutcomeAggregate,
    baseline: OutcomeAggregate,
) -> tuple[tuple[CriterionResult, ...], bool]:
    results = tuple(
        _evaluate_criterion(str(name), value, learned, baseline)
        for name, value in success_criteria.items()
    )
    met = bool(results) and all(result.passed for result in results)
    return results, met


# ---------------------------------------------------------------------------
# Config and area evaluation
# ---------------------------------------------------------------------------
def episode_config_from_plan(plan: ExperimentPlan) -> EpisodeConfig:
    """Build the frozen episode config from the plan's evaluation settings.

    Capture radius and step limit come straight from the frozen plan; movement
    speeds and the step duration are read from the plan's ``evaluation_config``
    with conservative defaults so the executed config is fully determined by the
    (already hashed) plan (Requirement 14.1).
    """
    evaluation_config = plan.evaluation_config
    return EpisodeConfig(
        dt_s=float(evaluation_config.get("dt_s", 1.0)),
        police_speed_mps=float(evaluation_config.get("police_speed_mps", 15.0)),
        fugitive_speed_mps=float(evaluation_config.get("fugitive_speed_mps", 10.0)),
        capture_radius_m=plan.capture_radius_m,
        max_steps=plan.max_steps,
        deterministic=True,
    )


def _split_manifest_hash(region_splits: RegionSplits) -> str:
    """Deterministic content hash over the disjoint region splits (sorted)."""
    return content_hash(
        {
            "train": sorted(region_splits.train),
            "validation": sorted(region_splits.validation),
            "test": sorted(region_splits.test),
        }
    )


def _evaluate_area(
    *,
    network: ModelNetwork,
    network_hash: str,
    partition: str,
    plan: ExperimentPlan,
    config: EpisodeConfig,
    learned_policy: Any,
    baseline_policy: Any,
    run_id: str,
    fugitive_factory: FugitiveFactory | None,
) -> AreaEvaluation:
    """Run every planned seed as a learned/baseline pair on one network.

    Both policies share the ``(seed, episode_index)`` derived placement and
    fugitive random stream, so the comparison uses identical initial conditions
    and equivalent accounting (Requirements 14.3-14.4).
    """
    extra_hashes = {
        "plan": plan.frozen_config_hash,
        "partition": partition,
        "evaluation_network": network_hash,
    }
    pairs: list[PairedEpisodeRecords] = []
    for episode_index, seed in enumerate(plan.seeds):
        learned_record = run_single_episode(
            network,
            config,
            learned_policy,
            run_id=f"{run_id}:{network_hash}:learned",
            run_seed=int(seed),
            episode_index=episode_index,
            fugitive_factory=fugitive_factory,
            extra_hashes=extra_hashes,
        )
        baseline_record = run_single_episode(
            network,
            config,
            baseline_policy,
            run_id=f"{run_id}:{network_hash}:baseline",
            run_seed=int(seed),
            episode_index=episode_index,
            fugitive_factory=fugitive_factory,
            extra_hashes=extra_hashes,
        )
        pairs.append(
            PairedEpisodeRecords(
                episode_index=episode_index,
                seed=int(seed),
                learned=learned_record,
                baseline=baseline_record,
            )
        )

    learned_metrics = summarize_metrics(
        [pair.learned for pair in pairs],
        network,
        policy_kind=_LEARNED_KIND,
        minimum_evaluation_episodes=plan.minimum_evaluation_episodes,
        confidence_interval_method=plan.confidence_interval_method,
    )
    baseline_metrics = summarize_metrics(
        [pair.baseline for pair in pairs],
        network,
        policy_kind=_BASELINE_KIND,
        minimum_evaluation_episodes=plan.minimum_evaluation_episodes,
        confidence_interval_method=plan.confidence_interval_method,
    )
    return AreaEvaluation(
        network_hash=network_hash,
        partition=partition,
        learned_metrics=learned_metrics,
        baseline_metrics=baseline_metrics,
        pairs=tuple(pairs),
    )


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def run_paired_evaluation(
    *,
    plan: ExperimentPlan,
    seen_networks: Mapping[str, ModelNetwork],
    unseen_networks: Mapping[str, ModelNetwork],
    learned_policy_factory: PolicyFactory,
    baseline_policy_factory: PolicyFactory,
    checkpoint_hash: str,
    code_version: str,
    run_id: str,
    config: EpisodeConfig | None = None,
    observation_profile: str = OSM_OBSERVATION_PROFILE,
    run_config_hash: str | None = None,
    fugitive_factory: FugitiveFactory | None = None,
) -> PairedEvaluationResult:
    """Execute a pre-registered paired evaluation and emit run/metrics/claim inputs.

    ``seen_networks`` maps every training-split network hash to its prepared
    :class:`ModelNetwork`; ``unseen_networks`` does the same for the test split.
    Each provided hash must belong to the matching frozen split, and the unseen
    hashes must cover the whole test split so the emitted evidence describes the
    full pre-registered generalization set (Requirements 14.1-14.2).

    The learned and baseline policies are compared on identical placements and
    fugitive randomness for every planned seed (Requirements 14.3-14.4).  The
    returned :class:`PairedEvaluationResult` carries an ``experimental`` run
    manifest, per-policy metrics and a candidate evidence object; **no** claim is
    promoted to ``verified`` automatically (Requirements 10.7-10.8, 14.5-14.6).
    """
    if not isinstance(plan, ExperimentPlan):
        raise DomainValidationError("INVALID_PLAN", "A frozen ExperimentPlan is required")
    if observation_profile != OSM_OBSERVATION_PROFILE:
        raise DomainValidationError(
            "INVALID_EVALUATION_PROFILE",
            "OSM evaluation requires the osm_topology_v1 observation profile",
            expected=OSM_OBSERVATION_PROFILE,
            actual=observation_profile,
        )
    if not str(run_id).strip():
        raise DomainValidationError("MISSING_RUN_ID", "A nonempty run_id is required", path="run_id")
    if not str(code_version).strip():
        raise DomainValidationError(
            "MISSING_CODE_VERSION", "code_version must be nonempty", path="code_version"
        )
    if not str(checkpoint_hash).strip():
        raise DomainValidationError(
            "MISSING_CHECKPOINT", "checkpoint_hash must be nonempty", path="checkpoint_hash"
        )

    # --- Freeze the experiment config before execution (Requirement 14.1) ---
    expected_hash = content_hash(plan.identity_payload())
    if plan.frozen_config_hash != expected_hash:
        raise DomainValidationError(
            "PLAN_HASH_MISMATCH",
            "Experiment plan content does not match its frozen configuration hash",
            expected=expected_hash,
            actual=plan.frozen_config_hash,
        )
    if checkpoint_hash not in plan.checkpoint_candidates:
        raise DomainValidationError(
            "CHECKPOINT_NOT_PRE_REGISTERED",
            "The evaluated checkpoint must be one of the plan's pre-registered candidates",
            expected=sorted(plan.checkpoint_candidates),
            actual=checkpoint_hash,
        )

    # --- Enforce seen/unseen split membership (Requirement 14.2) ------------
    _require_partition(seen_networks, plan.region_splits.train, SEEN_PARTITION)
    _require_partition(unseen_networks, plan.region_splits.test, UNSEEN_PARTITION)
    if not unseen_networks:
        raise DomainValidationError(
            "MISSING_UNSEEN_NETWORKS",
            "Paired evaluation requires at least one unseen (test-split) network",
        )
    if frozenset(unseen_networks) != plan.region_splits.test:
        raise DomainValidationError(
            "INCOMPLETE_UNSEEN_COVERAGE",
            "Unseen networks must cover the entire pre-registered test split",
            expected=sorted(plan.region_splits.test),
            actual=sorted(unseen_networks),
        )

    config = config or episode_config_from_plan(plan)

    seen_evaluations = tuple(
        _evaluate_area(
            network=network,
            network_hash=network_hash,
            partition=SEEN_PARTITION,
            plan=plan,
            config=config,
            learned_policy=learned_policy_factory(network),
            baseline_policy=baseline_policy_factory(network),
            run_id=run_id,
            fugitive_factory=fugitive_factory,
        )
        for network_hash, network in sorted(seen_networks.items())
    )
    unseen_evaluations = tuple(
        _evaluate_area(
            network=network,
            network_hash=network_hash,
            partition=UNSEEN_PARTITION,
            plan=plan,
            config=config,
            learned_policy=learned_policy_factory(network),
            baseline_policy=baseline_policy_factory(network),
            run_id=run_id,
            fugitive_factory=fugitive_factory,
        )
        for network_hash, network in sorted(unseen_networks.items())
    )

    # --- Aggregate the unseen (generalization) outcomes for the criteria ----
    unseen_learned_records = tuple(
        record for area in unseen_evaluations for record in area.learned_records
    )
    unseen_baseline_records = tuple(
        record for area in unseen_evaluations for record in area.baseline_records
    )
    learned_aggregate = aggregate_outcomes(tuple(r.outcome for r in unseen_learned_records))
    baseline_aggregate = aggregate_outcomes(tuple(r.outcome for r in unseen_baseline_records))

    criteria_results, criteria_met = _evaluate_success_criteria(
        plan.success_criteria, learned_aggregate, baseline_aggregate
    )

    unseen_episode_count = len(unseen_learned_records)
    preliminary = unseen_episode_count < plan.minimum_evaluation_episodes

    # Success may only be reported on complete (non-preliminary) evidence
    # (Requirement 14.5 / 10.8); otherwise the result stays preliminary/failed.
    success_criteria_met = bool(criteria_met and not preliminary)
    if preliminary:
        state = EvidenceState.PRELIMINARY
    elif success_criteria_met:
        state = EvidenceState.COMPLETE
    else:
        state = EvidenceState.FAILED

    # --- Emit the run manifest, metrics artifacts and candidate evidence ----
    primary_network_hash = sorted(unseen_networks)[0]
    lineage = RunLineage(
        split_manifest_hash=_split_manifest_hash(plan.region_splits),
        train_network_hashes=plan.region_splits.train,
        validation_network_hashes=plan.region_splits.validation,
        test_network_hashes=plan.region_splits.test,
        code_version=code_version,
    )
    run_config = {
        "plan_id": plan.plan_id,
        "frozen_config_hash": plan.frozen_config_hash,
        "capture_radius_m": float(plan.capture_radius_m),
        "max_steps": int(plan.max_steps),
        "confidence_interval_method": plan.confidence_interval_method,
        "baseline_policy": plan.baseline_policy,
        "code_version": code_version,
        "seen_network_hashes": sorted(seen_networks),
        "unseen_network_hashes": sorted(unseen_networks),
    }
    run_manifest = RunManifest(
        run_id=run_id,
        mode=RunMode.OSM_EVALUATION,
        network_hash=primary_network_hash,
        observation_profile=OSM_OBSERVATION_PROFILE,
        seeds=plan.seeds,
        policy_kind=_LEARNED_KIND,
        claim_status=ClaimStatus.EXPERIMENTAL,  # never auto-verified (Req. 14.5)
        checkpoint_hash=checkpoint_hash,
        plan_id=plan.plan_id,
        run_config=run_config,
        lineage=lineage,
    )

    artifact_hashes = _artifact_hashes(seen_evaluations, unseen_evaluations)
    reproducibility = ReproducibilityRecord(
        checkpoint_hash=checkpoint_hash,
        network_hashes=frozenset(unseen_networks),
        run_config_hash=run_config_hash or content_hash(run_config),
        code_hash=code_version,
        input_artifact_hashes=artifact_hashes,
    )
    evidence = ExperimentEvidence(
        evidence_id=f"{run_id}:evidence",
        run_manifest=run_manifest,
        plan_hash=plan.frozen_config_hash,
        state=state,
        episode_count=unseen_episode_count,
        training_region_hashes=plan.region_splits.train,
        evaluation_region_hashes=plan.region_splits.test,
        success_criteria_met=success_criteria_met,
        confidence_interval_method=plan.confidence_interval_method,
        artifact_hashes=artifact_hashes,
        reproducibility=reproducibility,
    )

    return PairedEvaluationResult(
        plan=plan,
        config=config,
        run_manifest=run_manifest,
        seen_evaluations=seen_evaluations,
        unseen_evaluations=unseen_evaluations,
        criteria_results=criteria_results,
        criteria_met=criteria_met,
        preliminary=preliminary,
        evidence=evidence,
    )


def _require_partition(
    networks: Mapping[str, ModelNetwork], allowed: frozenset[str], partition: str
) -> None:
    """Reject any network hash that does not belong to its frozen split."""
    for network_hash, network in networks.items():
        if not str(network_hash).strip():
            raise DomainValidationError(
                "MISSING_NETWORK_HASH", f"{partition} network hash must be nonempty"
            )
        if not isinstance(network, ModelNetwork):
            raise DomainValidationError(
                "INVALID_NETWORK", f"{partition} network {network_hash!r} must be a ModelNetwork"
            )
        if network_hash not in allowed:
            raise DomainValidationError(
                "NETWORK_OUTSIDE_SPLIT",
                f"{partition} network hash does not belong to the frozen {partition} split",
                expected=sorted(allowed),
                actual=network_hash,
            )


def _artifact_hashes(
    seen_evaluations: Sequence[AreaEvaluation],
    unseen_evaluations: Sequence[AreaEvaluation],
) -> dict[str, str]:
    """Content-address every per-policy metrics summary as a result artifact."""
    artifacts: dict[str, str] = {}
    for label, evaluations in (("seen", seen_evaluations), ("unseen", unseen_evaluations)):
        for area in evaluations:
            artifacts[f"{label}:{area.network_hash}:learned_metrics"] = content_hash(
                area.learned_metrics
            )
            artifacts[f"{label}:{area.network_hash}:baseline_metrics"] = content_hash(
                area.baseline_metrics
            )
    return artifacts


__all__ = (
    "AreaEvaluation",
    "CriterionResult",
    "PairedEvaluationResult",
    "PolicyFactory",
    "SEEN_PARTITION",
    "UNSEEN_PARTITION",
    "episode_config_from_plan",
    "run_paired_evaluation",
)
