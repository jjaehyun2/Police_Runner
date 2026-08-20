"""Remediated-condition training for ONE seed, real Daejeon OSM networks.

Identical launch shape to ``run_seed.py`` (5 independent processes, seeds
0-4), but registers the NEW Condition ``real_scale_daejeon_remediated_v1``
instead of mutating the audited default (protocol C4: coefficient changes are
a new Condition, never an in-place edit).  Differences from the main study:

  A. step rewards use directed ROAD-GRAPH distance (RemediatedStepReward)
  B. symmetric regress multiplier 1.0 (existing symmetric ablation arm)
  C. own term decoupled from fugitive motion (post-step fugitive fixed)
  D. TIMEOUT closes as a truncation (bootstrap preserved)
  E. officers decide at their arrival intersection (evader parity)
  F. u-turn suppression ON, with the sampling-time logit bias stored and
     re-applied during PPO recomputation (bias-drop bug fixed in
     masked_mappo/_forward_batch)
  G. team delta credited to the argmin officer; capture bonus split
     (capturer 1.0, others NONCAPTURER_CAPTURE_SHARE)

Paired evaluation against the same lowest-legal-action baseline over the same
500 sealed episode cases, so results are directly comparable with the
``real_scale_daejeon_main_study`` reports.
"""
from __future__ import annotations

import json
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from pursuit_evasion_rl.research.canonical import canonical_json
from pursuit_evasion_rl.research.domain import DataKind, ExecutionStatus, MapScenario
from pursuit_evasion_rl.research.experiments.main_study import (
    LEGACY_BEST_V2_PATH,
    MainStudyConditionSpec,
    require_main_study_admission,
    train_and_evaluate_seed,
)
from pursuit_evasion_rl.research.experiments.pilot import PilotConditionOutcome, PilotReport
from pursuit_evasion_rl.research.maps.snapshots import OfflineSnapshotStore, load_snapshot_spec
from pursuit_evasion_rl.research.maps.splits import DataHandle, HandleKind, SplitScope, TuningDataView
from pursuit_evasion_rl.research.protocol import ProtocolStore
from pursuit_evasion_rl.research.quality import build_attestation
from pursuit_evasion_rl.research.runs.manifest import RunManifestStore, RunProvenance
from pursuit_evasion_rl.research.training.trainer import TrainerConfig
from pursuit_evasion_rl.research.variants.placement import PlacementCurriculumConfig, PlacementStyle
from pursuit_evasion_rl.research.variants.remediation import (
    REMEDIATED_CONDITION_ID,
    remediated_trainer_kwargs,
)

REPO_ROOT = Path(__file__).resolve().parents[3]
CONDITION_ID = REMEDIATED_CONDITION_ID
SCRATCH = Path(__file__).resolve().parent / "output"


class _LowestLegalActionPolicy:
    policy_id = "lowest-legal-action-v1"

    def act(self, observation) -> int:
        return min(index for index, legal in enumerate(observation.action_mask) if legal)


def _load_pilot_report(path: Path) -> PilotReport:
    payload = json.loads(path.read_text(encoding="utf-8"))
    outcomes = tuple(
        PilotConditionOutcome(
            condition_id=item["condition_id"], execution_status=ExecutionStatus(item["execution_status"]),
            status_reason=item["status_reason"], run_id=item["run_id"],
            resource_actual=dict(item.get("resource_actual", {})), error_artifact_hash=item.get("error_artifact_hash"),
        )
        for item in payload["outcomes"]
    )
    return PilotReport(admission_hash=payload["admission_hash"], outcomes=outcomes)


def main() -> int:
    seed = int(sys.argv[1])
    log = lambda msg: print(f"[remediated seed {seed}] {msg}", flush=True)

    store = OfflineSnapshotStore(str(REPO_ROOT))
    train_spec = load_snapshot_spec(REPO_ROOT / "configs/research/maps/daejeon_train.yaml")
    validation_spec = load_snapshot_spec(REPO_ROOT / "configs/research/maps/daejeon_validation.yaml")
    train_network = store.import_snapshot(train_spec).snapshot.network
    validation_network = store.import_snapshot(validation_spec).snapshot.network
    log(f"networks loaded: train={len(train_network.intersections)} validation={len(validation_network.intersections)}")

    attestation = build_attestation(
        repo_root=REPO_ROOT, junit_output_dir=REPO_ROOT / ".quality-gate",
        generated_at_utc="2026-08-01T00:00:00Z",
    )
    log(f"attestation eligible={attestation.pilot_admission_eligible}")
    protocol = ProtocolStore(REPO_ROOT / "artifacts/research/protocols").read(
        "20260731T193445-df6a1df8902d"
    )
    pilot_report = _load_pilot_report(SCRATCH / "pilot_report.json")
    token = require_main_study_admission(
        quality_attestation=attestation, protocol=protocol, pilot_report=pilot_report,
        best_v2_path=LEGACY_BEST_V2_PATH,
    )
    log("admission token ready")

    tuning_data = TuningDataView(
        train=(DataHandle(SplitScope.TRAIN, HandleKind.MAP, "daejeon-train-v1"),),
        validation=(DataHandle(SplitScope.VALIDATION, HandleKind.MAP, "daejeon-validation-v1"),),
    )
    config = TrainerConfig(updates=35000, episodes_per_update=1, max_steps=450)
    remediation = remediated_trainer_kwargs()
    spec = MainStudyConditionSpec(
        condition_id=CONDITION_ID,
        scenario=MapScenario.BOUNDARY_ESCAPE,
        train_network=train_network,
        validation_network=validation_network,
        tuning_data=tuning_data,
        config=config,
        placement_config=PlacementCurriculumConfig(style=PlacementStyle.GLOBAL),
        baseline_policy_factory=lambda _network: _LowestLegalActionPolicy(),
        episodes_per_seed=500,
        data_kind=DataKind.ACTUAL_OSM_MAP,
        reward_components=remediation["reward_components"],
        stabilization_condition=remediation["stabilization_condition"],
        step_reward_fn=remediation["step_reward_fn"],
        timeout_bootstrap=remediation["timeout_bootstrap"],
        arrival_decisions=remediation["arrival_decisions"],
    )
    runs = RunManifestStore(REPO_ROOT / "artifacts/research/runs")

    def _provenance() -> RunProvenance:
        return RunProvenance(
            code_hash=attestation.code_hash_at_attestation, dirty_tree=False,
            dependency_hash=attestation.code_hash_at_attestation, runtime="python3.11+torch2.8.0+cpu",
            device="cpu", map_hash=train_spec.network_content_hash, split="train", seed=seed,
            input_hashes={"train_map": train_spec.network_content_hash, "validation_map": validation_spec.network_content_hash},
        )

    log("starting train_and_evaluate_seed (35000 updates, 500 paired episodes)")
    t0 = time.time()
    total_updates = config.updates
    progress_state = {"last_logged": 0}

    def _progress(record) -> None:
        n = record.update_index
        if n - progress_state["last_logged"] < 1000 and n != total_updates:
            return
        progress_state["last_logged"] = n
        elapsed = time.time() - t0
        rate = elapsed / n
        eta_h = rate * (total_updates - n) / 3600
        log(
            f"update {n}/{total_updates} "
            f"train_capture_rate={record.train_capture_rate:.3f} "
            f"validation_capture_rate={record.validation_capture_rate:.3f} "
            f"selected={record.selected_checkpoint} "
            f"elapsed={elapsed/60:.1f}min eta={eta_h:.2f}h"
        )

    outcome = train_and_evaluate_seed(
        token, spec, seed, runs=runs, output_root=str(REPO_ROOT / "artifacts/research/checkpoints"),
        provenance_factory=_provenance, progress_callback=_progress,
    )
    dt = time.time() - t0
    log(f"finished in {dt:.1f}s status={outcome.execution_status} reason={outcome.status_reason}")

    out_path = SCRATCH / f"remediated_seed{seed}_outcome.json"
    out_path.write_bytes(canonical_json(outcome) + b"\n")
    log(f"wrote {out_path}")
    return 0 if outcome.execution_status is ExecutionStatus.COMPLETED else 1


if __name__ == "__main__":
    raise SystemExit(main())
