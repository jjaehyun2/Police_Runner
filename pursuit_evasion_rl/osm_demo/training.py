"""Manifest-driven OSM fine-tune and from-scratch training entry points.

This module implements the orchestration side of the OSM_Training_Path
(design section 8, "OSM fine-tuning and retraining"; Requirements 8.7-8.10 and
14.2).  It deliberately does **not** run any real, long MAPPO training loop.
Instead it wires together the pieces already built by earlier tasks -- the
immutable domain models (:mod:`.models`), experiment lineage
(:mod:`.experiments`), the ``osm_topology_v1`` observation contract
(:mod:`.observations`) and safe checkpoint inspection/loading
(:mod:`.checkpoint`, :mod:`.policies`) -- behind two small, mockable entry
points:

* :func:`finetune_osm_policy` -- initializes **only shape-compatible** actor
  weights from a structurally compatible parent checkpoint, records the legacy
  parent hash, and trains with the ``osm_topology_v1`` profile.
* :func:`train_osm_policy_from_scratch` -- independent random initialization of
  the identical MAPPO architecture on the same disjoint region splits.

Both entry points:

* reuse the MAPPO architecture: six police, observation dimension 21, action
  dimension 6 and MLP hidden layers ``[128, 128]`` (Requirement 8.7);
* require the ``osm_topology_v1`` observation profile and reject any other
  profile (Requirements 8.7, 8.11);
* require pairwise-disjoint train/validation/test region manifests -- overlap is
  a hard failure via :class:`~pursuit_evasion_rl.osm_demo.experiments.RegionSplits`
  and :class:`~pursuit_evasion_rl.osm_demo.models.RunLineage` (Requirements 8.7,
  14.2);
* save **distinct** lineage, normalization/action-order contracts and checkpoint
  manifests for the two run kinds (Requirement 8.8); and
* leave the produced checkpoint ``experimental`` with no completed unseen-test
  evidence, because training completion alone is not verified performance
  (Requirement 8.10).

Real training is injected through a :data:`TrainingHook`.  The default hook is a
deterministic mock that produces a correctly shaped ``police_actor`` state dict
so manifests and checkpoint bytes are reproducible offline without GPUs, OSM
access or long optimization runs.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from .canonical import canonical_json, content_hash
from .checkpoint import (
    EXPECTED_ARCHITECTURE,
    EXPECTED_LAYER_SHAPES,
    CheckpointInspector,
    _actor_state,
    _file_sha256,
    _safe_cpu_load,
)
from .experiments import OSM_OBSERVATION_PROFILE, RegionSplits, TrainingRunLineage
from .models import (
    POLICE_COUNT,
    ArchitectureSpec,
    CheckpointManifest,
    ClaimStatus,
    DomainValidationError,
    ObservationContract,
    RunLineage,
    RunManifest,
    RunMode,
)
from .observations import DEFAULT_CLIP_DISTANCE_M, OSM_PROFILE_ID, osm_topology_v1_contract
from .policies import _strip_network_prefix

# The MAPPO architecture reused by both OSM training paths (Requirement 8.7).
OSM_TRAINING_ARCHITECTURE: ArchitectureSpec = EXPECTED_ARCHITECTURE

# Ordered ``police_actor`` parameter names paired with their required shapes.
# The order matters: it must reproduce ``EXPECTED_LAYER_SHAPES`` so the inferred
# 21->128->128->6 structure round-trips through :mod:`.checkpoint`.
_NAMED_LAYER_SHAPES: tuple[tuple[str, tuple[int, ...]], ...] = (
    ("network.0.weight", EXPECTED_LAYER_SHAPES[0]),
    ("network.0.bias", EXPECTED_LAYER_SHAPES[1]),
    ("network.2.weight", EXPECTED_LAYER_SHAPES[2]),
    ("network.2.bias", EXPECTED_LAYER_SHAPES[3]),
    ("network.4.weight", EXPECTED_LAYER_SHAPES[4]),
    ("network.4.bias", EXPECTED_LAYER_SHAPES[5]),
)


@dataclass(frozen=True, slots=True)
class TrainingContext:
    """Immutable inputs handed to a :data:`TrainingHook`.

    A hook receives everything needed to run (or mock) an optimization loop and
    must return the final ``police_actor`` parameter state.  ``initial_actor_state``
    is populated only for fine-tune runs and contains exactly the shape-compatible
    parent weights (Requirement 8.8); it is ``None`` for from-scratch runs.
    """

    mode: RunMode
    run_id: str
    architecture: ArchitectureSpec
    observation_contract: ObservationContract
    region_splits: RegionSplits
    training_network_hash: str
    seeds: tuple[int, ...]
    code_version: str
    parent_checkpoint_hash: str | None
    initial_actor_state: Mapping[str, Any] | None


# A training hook maps a context to the final ``police_actor`` parameter mapping
# (parameter name -> tensor).  The mapping must contain the ordered
# 21->128->128->6 weight/bias tensors.
TrainingHook = Callable[[TrainingContext], Mapping[str, Any]]


@dataclass(frozen=True, slots=True)
class TrainingArtifacts:
    """Deterministic artifacts produced by an OSM training entry point."""

    run_manifest: RunManifest
    checkpoint_manifest: CheckpointManifest
    training_lineage: TrainingRunLineage
    observation_contract: ObservationContract
    checkpoint_path: Path
    checkpoint_hash: str
    split_manifest_hash: str
    manifest_paths: Mapping[str, Path]

    def as_dict(self) -> dict[str, Any]:
        return {
            "run_id": self.run_manifest.run_id,
            "mode": self.run_manifest.mode.value,
            "checkpoint_hash": self.checkpoint_hash,
            "split_manifest_hash": self.split_manifest_hash,
            "observation_profile": self.run_manifest.observation_profile,
            "checkpoint_path": str(self.checkpoint_path),
            "manifest_paths": {name: str(path) for name, path in self.manifest_paths.items()},
        }


# ---------------------------------------------------------------------------
# Split identity
# ---------------------------------------------------------------------------


def split_manifest_hash(region_splits: RegionSplits) -> str:
    """Return a deterministic content hash over the disjoint region splits.

    The hash is stable under set iteration order because the partitions are
    serialized as sorted lists (Requirements 8.7, 14.2).
    """
    return content_hash(
        {
            "train": sorted(region_splits.train),
            "validation": sorted(region_splits.validation),
            "test": sorted(region_splits.test),
        }
    )


# ---------------------------------------------------------------------------
# Default deterministic mock training hook
# ---------------------------------------------------------------------------


def _context_seed(context: TrainingContext) -> int:
    """Derive a stable non-negative seed from the run mode and seed list."""
    digest = content_hash(
        {"mode": context.mode.value, "seeds": list(context.seeds), "run_id": context.run_id}
    )
    return int(digest[:16], 16)


def default_mock_training_hook(context: TrainingContext) -> Mapping[str, Any]:
    """Produce a correctly shaped, deterministic ``police_actor`` state dict.

    This is a stand-in for a real MAPPO optimization loop.  From-scratch runs use
    a fully independent deterministic initialization derived from the run seed;
    fine-tune runs start from the shape-compatible parent weights and apply a
    small deterministic adjustment so the two paths produce genuinely different
    checkpoints while staying reproducible offline.
    """
    import numpy as np
    import torch

    rng = np.random.default_rng(_context_seed(context))
    state: dict[str, Any] = {}
    for name, shape in _NAMED_LAYER_SHAPES:
        base = (rng.standard_normal(size=shape) * 0.01).astype(np.float32)
        parent = None
        if context.initial_actor_state is not None:
            candidate = context.initial_actor_state.get(name)
            if candidate is not None:
                parent = np.asarray(candidate, dtype=np.float32)
        if parent is not None and parent.shape == tuple(shape):
            # Fine-tune: only shape-compatible weights are reused (Requirement 8.8).
            base = (parent + base).astype(np.float32)
        state[name] = torch.from_numpy(np.ascontiguousarray(base))
    return state


# ---------------------------------------------------------------------------
# Parent checkpoint handling for fine-tuning
# ---------------------------------------------------------------------------


def _load_shape_compatible_parent(parent_checkpoint_path: str | Path) -> tuple[str, dict[str, Any]]:
    """Inspect a parent checkpoint and return its hash and shape-compatible weights.

    The parent must expose the exact 21->128->128->6 ``police_actor`` structure;
    otherwise fine-tuning is refused instead of silently reshaping weights
    (Requirements 7.1-7.5, 8.8).  Only tensors whose shape matches the expected
    layer are returned, so incompatible extras never seed the new actor.
    """
    path = Path(parent_checkpoint_path)
    inspection = CheckpointInspector().inspect(path)
    if inspection.report.as_dict()["inference_blocked"]:
        raise DomainValidationError(
            "INCOMPATIBLE_PARENT_CHECKPOINT",
            "Fine-tune parent checkpoint is not structurally compatible with the MAPPO architecture",
            expected=list(EXPECTED_LAYER_SHAPES),
            actual=inspection.report.as_dict()["mismatches"],
        )
    if inspection.inferred_contract.layer_shapes != EXPECTED_LAYER_SHAPES:
        raise DomainValidationError(
            "INCOMPATIBLE_PARENT_CHECKPOINT",
            "Fine-tune parent checkpoint actor shapes must match 21->128->128->6",
            expected=list(EXPECTED_LAYER_SHAPES),
            actual=[list(shape) for shape in inspection.inferred_contract.layer_shapes],
        )

    payload = _safe_cpu_load(path)
    _, actor_state = _actor_state(payload)
    stripped = _strip_network_prefix(actor_state)  # bare ``0.weight`` style keys
    expected = {name: tuple(shape) for name, shape in _NAMED_LAYER_SHAPES}
    compatible: dict[str, Any] = {}
    import numpy as np

    for bare_key, tensor in stripped.items():
        named = f"network.{bare_key}"
        if named not in expected:
            continue
        array = np.asarray(tensor, dtype=np.float32)
        if array.shape == expected[named]:
            compatible[named] = array
    if len(compatible) != len(_NAMED_LAYER_SHAPES):
        raise DomainValidationError(
            "INCOMPATIBLE_PARENT_CHECKPOINT",
            "Fine-tune parent checkpoint is missing shape-compatible actor parameters",
            expected=[name for name, _ in _NAMED_LAYER_SHAPES],
            actual=sorted(compatible),
        )
    return inspection.checkpoint_hash, compatible


# ---------------------------------------------------------------------------
# Checkpoint persistence
# ---------------------------------------------------------------------------


def _validate_actor_state(actor_state: Mapping[str, Any]) -> dict[str, Any]:
    """Ensure the hook returned the ordered, correctly shaped actor parameters."""
    import numpy as np
    import torch

    expected = {name: tuple(shape) for name, shape in _NAMED_LAYER_SHAPES}
    missing = [name for name in expected if name not in actor_state]
    if missing:
        raise DomainValidationError(
            "INCOMPLETE_TRAINED_ACTOR",
            "Training hook did not return every required police_actor parameter",
            expected=list(expected),
            actual=sorted(actor_state),
        )
    ordered: dict[str, Any] = {}
    for name, shape in _NAMED_LAYER_SHAPES:
        tensor = actor_state[name]
        array = np.asarray(tensor.detach().cpu().numpy() if hasattr(tensor, "detach") else tensor, dtype=np.float32)
        if array.shape != tuple(shape):
            raise DomainValidationError(
                "INVALID_TRAINED_ACTOR_SHAPE",
                f"police_actor parameter {name!r} has the wrong shape",
                expected=list(shape),
                actual=list(array.shape),
            )
        ordered[name] = torch.from_numpy(np.ascontiguousarray(array))
    return ordered


def _save_actor_checkpoint(path: Path, actor_state: Mapping[str, Any]) -> str:
    """Persist the actor state as a weights-only-loadable checkpoint and hash it."""
    import torch

    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"police_actor": dict(actor_state)}, path)
    return _file_sha256(path)


# ---------------------------------------------------------------------------
# Manifest construction
# ---------------------------------------------------------------------------


def _run_lineage(
    *,
    split_hash: str,
    region_splits: RegionSplits,
    code_version: str,
    parent_checkpoint_hash: str | None,
) -> RunLineage:
    return RunLineage(
        split_manifest_hash=split_hash,
        train_network_hashes=region_splits.train,
        validation_network_hashes=region_splits.validation,
        test_network_hashes=region_splits.test,
        code_version=code_version,
        parent_checkpoint_hash=parent_checkpoint_hash,
    )


def _lineage_contract(
    *,
    mode: RunMode,
    run_id: str,
    split_hash: str,
    seeds: Sequence[int],
    code_version: str,
    parent_checkpoint_hash: str | None,
    observation_contract: ObservationContract,
) -> dict[str, Any]:
    """Normalization/action-order contract recorded alongside the checkpoint."""
    return {
        "mode": mode.value,
        "run_id": run_id,
        "split_manifest_hash": split_hash,
        "seeds": list(seeds),
        "code_version": code_version,
        "parent_checkpoint_hash": parent_checkpoint_hash,
        "observation_profile": observation_contract.profile_id,
        "normalization": dict(observation_contract.normalization),
        "action_ordering": observation_contract.action_ordering,
        "architecture": {
            "observation_dim": OSM_TRAINING_ARCHITECTURE.observation_dim,
            "hidden_dims": list(OSM_TRAINING_ARCHITECTURE.hidden_dims),
            "action_dim": OSM_TRAINING_ARCHITECTURE.action_dim,
            "police_count": POLICE_COUNT,
        },
    }


def _write_manifests(
    output_dir: Path,
    *,
    run_manifest: RunManifest,
    checkpoint_manifest: CheckpointManifest,
    training_lineage: TrainingRunLineage,
    observation_contract: ObservationContract,
) -> dict[str, Path]:
    """Write every manifest as canonical JSON so bytes are reproducible."""
    output_dir.mkdir(parents=True, exist_ok=True)
    documents = {
        "run_manifest": run_manifest,
        "checkpoint_manifest": checkpoint_manifest,
        "training_lineage": training_lineage,
        "observation_contract": observation_contract,
    }
    paths: dict[str, Path] = {}
    for name, document in documents.items():
        target = output_dir / f"{name}.json"
        target.write_bytes(canonical_json(document))
        paths[name] = target
    return paths


# ---------------------------------------------------------------------------
# Entry points
# ---------------------------------------------------------------------------


def _run_osm_training(
    *,
    mode: RunMode,
    run_id: str,
    region_splits: RegionSplits,
    training_network_hash: str,
    seeds: Sequence[int],
    code_version: str,
    output_dir: str | Path,
    observation_profile: str,
    clip_distance_m: float,
    parent_checkpoint_path: str | Path | None,
    training_hook: TrainingHook | None,
) -> TrainingArtifacts:
    if mode not in {RunMode.OSM_FINETUNE, RunMode.OSM_FROM_SCRATCH}:
        raise DomainValidationError(
            "INVALID_TRAINING_MODE",
            "Only osm-finetune and osm-from-scratch modes are supported here",
            actual=getattr(mode, "value", mode),
        )
    if observation_profile != OSM_PROFILE_ID:
        # Requirements 8.7/8.11: OSM training requires the ID-free topology profile.
        raise DomainValidationError(
            "INVALID_TRAINING_PROFILE",
            "OSM training requires the osm_topology_v1 observation profile",
            expected=OSM_PROFILE_ID,
            actual=observation_profile,
        )
    if not str(run_id).strip():
        raise DomainValidationError("MISSING_RUN_ID", "A nonempty run_id is required", path="run_id")
    if not str(code_version).strip():
        raise DomainValidationError("MISSING_CODE_VERSION", "code_version must be nonempty", path="code_version")
    seeds = tuple(int(seed) for seed in seeds)
    if not seeds:
        raise DomainValidationError("MISSING_SEEDS", "At least one deterministic seed is required")
    if training_network_hash not in region_splits.train:
        # Disjoint-split enforcement: training must draw from the training split.
        raise DomainValidationError(
            "TRAINING_NETWORK_OUTSIDE_SPLIT",
            "The training network hash must belong to the registered training split",
            expected=sorted(region_splits.train),
            actual=training_network_hash,
        )

    parent_checkpoint_hash: str | None = None
    initial_actor_state: Mapping[str, Any] | None = None
    if mode is RunMode.OSM_FINETUNE:
        if parent_checkpoint_path is None:
            raise DomainValidationError(
                "MISSING_PARENT_CHECKPOINT",
                "Fine-tune runs require a structurally compatible parent checkpoint",
            )
        parent_checkpoint_hash, initial_actor_state = _load_shape_compatible_parent(
            parent_checkpoint_path
        )
    elif parent_checkpoint_path is not None:
        raise DomainValidationError(
            "UNEXPECTED_PARENT_CHECKPOINT",
            "From-scratch runs perform an independent initialization and take no parent checkpoint",
        )

    observation_contract = osm_topology_v1_contract(clip_distance_m)
    split_hash = split_manifest_hash(region_splits)

    context = TrainingContext(
        mode=mode,
        run_id=run_id,
        architecture=OSM_TRAINING_ARCHITECTURE,
        observation_contract=observation_contract,
        region_splits=region_splits,
        training_network_hash=training_network_hash,
        seeds=seeds,
        code_version=code_version,
        parent_checkpoint_hash=parent_checkpoint_hash,
        initial_actor_state=initial_actor_state,
    )

    hook = training_hook or default_mock_training_hook
    actor_state = _validate_actor_state(hook(context))

    output_dir = Path(output_dir)
    checkpoint_path = output_dir / f"{run_id}.pt"
    checkpoint_hash = _save_actor_checkpoint(checkpoint_path, actor_state)

    lineage_contract = _lineage_contract(
        mode=mode,
        run_id=run_id,
        split_hash=split_hash,
        seeds=seeds,
        code_version=code_version,
        parent_checkpoint_hash=parent_checkpoint_hash,
        observation_contract=observation_contract,
    )

    run_lineage = _run_lineage(
        split_hash=split_hash,
        region_splits=region_splits,
        code_version=code_version,
        parent_checkpoint_hash=parent_checkpoint_hash,
    )

    run_manifest = RunManifest(
        run_id=run_id,
        mode=mode,
        network_hash=training_network_hash,
        observation_profile=OSM_PROFILE_ID,
        seeds=seeds,
        policy_kind="learned",
        claim_status=ClaimStatus.EXPERIMENTAL,  # Requirement 8.10
        checkpoint_hash=checkpoint_hash,
        run_config={
            "architecture": lineage_contract["architecture"],
            "clip_distance_m": float(clip_distance_m),
            "code_version": code_version,
        },
        lineage=run_lineage,
    )

    checkpoint_manifest = CheckpointManifest(
        checkpoint_hash=checkpoint_hash,
        architecture=OSM_TRAINING_ARCHITECTURE,
        police_count=POLICE_COUNT,
        observation_contract=observation_contract,
        training_networks=tuple(sorted(region_splits.train)),
        split_manifest_hash=split_hash,
        lineage=lineage_contract,
        code_version=code_version,
        claim_status=ClaimStatus.EXPERIMENTAL,  # Requirement 8.10
    )

    # Training completion alone is experimental: no completed unseen-test evidence
    # and therefore no checkpoint_manifest_hash promotion here (Requirement 8.10).
    training_lineage = TrainingRunLineage(
        run_manifest=run_manifest,
        region_splits=region_splits,
        parent_checkpoint_hash=parent_checkpoint_hash,
    )

    manifest_paths = _write_manifests(
        output_dir,
        run_manifest=run_manifest,
        checkpoint_manifest=checkpoint_manifest,
        training_lineage=training_lineage,
        observation_contract=observation_contract,
    )

    return TrainingArtifacts(
        run_manifest=run_manifest,
        checkpoint_manifest=checkpoint_manifest,
        training_lineage=training_lineage,
        observation_contract=observation_contract,
        checkpoint_path=checkpoint_path,
        checkpoint_hash=checkpoint_hash,
        split_manifest_hash=split_hash,
        manifest_paths=manifest_paths,
    )


def finetune_osm_policy(
    *,
    run_id: str,
    region_splits: RegionSplits,
    training_network_hash: str,
    seeds: Sequence[int],
    code_version: str,
    parent_checkpoint_path: str | Path,
    output_dir: str | Path,
    observation_profile: str = OSM_PROFILE_ID,
    clip_distance_m: float = DEFAULT_CLIP_DISTANCE_M,
    training_hook: TrainingHook | None = None,
) -> TrainingArtifacts:
    """Fine-tune the MAPPO police actor from a shape-compatible parent checkpoint.

    Only shape-compatible actor weights are used as the initialization and the
    legacy parent checkpoint hash is recorded in both the run and checkpoint
    lineage (Requirement 8.8).  Training uses the ``osm_topology_v1`` profile on
    the disjoint region splits and yields an ``experimental`` checkpoint
    (Requirements 8.7, 8.10).
    """
    return _run_osm_training(
        mode=RunMode.OSM_FINETUNE,
        run_id=run_id,
        region_splits=region_splits,
        training_network_hash=training_network_hash,
        seeds=seeds,
        code_version=code_version,
        output_dir=output_dir,
        observation_profile=observation_profile,
        clip_distance_m=clip_distance_m,
        parent_checkpoint_path=parent_checkpoint_path,
        training_hook=training_hook,
    )


def train_osm_policy_from_scratch(
    *,
    run_id: str,
    region_splits: RegionSplits,
    training_network_hash: str,
    seeds: Sequence[int],
    code_version: str,
    output_dir: str | Path,
    observation_profile: str = OSM_PROFILE_ID,
    clip_distance_m: float = DEFAULT_CLIP_DISTANCE_M,
    training_hook: TrainingHook | None = None,
) -> TrainingArtifacts:
    """Train the identical MAPPO architecture from an independent initialization.

    The from-scratch baseline uses the same six-police, 21->128->128->6 MAPPO
    architecture and disjoint region splits but takes no parent checkpoint, so
    its lineage and manifests stay distinct from the fine-tune path
    (Requirements 8.7, 8.8).
    """
    return _run_osm_training(
        mode=RunMode.OSM_FROM_SCRATCH,
        run_id=run_id,
        region_splits=region_splits,
        training_network_hash=training_network_hash,
        seeds=seeds,
        code_version=code_version,
        output_dir=output_dir,
        observation_profile=observation_profile,
        clip_distance_m=clip_distance_m,
        parent_checkpoint_path=None,
        training_hook=training_hook,
    )


__all__ = (
    "OSM_TRAINING_ARCHITECTURE",
    "TrainingArtifacts",
    "TrainingContext",
    "TrainingHook",
    "default_mock_training_hook",
    "finetune_osm_policy",
    "split_manifest_hash",
    "train_osm_policy_from_scratch",
)
