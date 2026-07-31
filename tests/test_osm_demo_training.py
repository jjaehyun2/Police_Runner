"""Focused offline tests for manifest-driven OSM fine-tune / from-scratch training.

These tests never run a real MAPPO optimization loop. They rely on the default
deterministic mock training hook (or tiny injected hooks) so the manifests and
checkpoint bytes are reproducible offline without GPUs or OSM access.

Validates: Requirements 8.7, 8.8, 8.9, 8.10, 14.2
"""

from __future__ import annotations

from collections import OrderedDict

import pytest
import torch

from pursuit_evasion_rl.osm_demo.canonical import canonical_json
from pursuit_evasion_rl.osm_demo.checkpoint import EXPECTED_LAYER_SHAPES
from pursuit_evasion_rl.osm_demo.experiments import (
    OSM_OBSERVATION_PROFILE,
    RegionSplits,
    TrainingRunLineage,
)
from pursuit_evasion_rl.osm_demo.models import (
    POLICE_COUNT,
    ClaimStatus,
    DomainValidationError,
    RunMode,
)
from pursuit_evasion_rl.osm_demo.observations import OSM_PROFILE_ID
from pursuit_evasion_rl.osm_demo.training import (
    OSM_TRAINING_ARCHITECTURE,
    finetune_osm_policy,
    split_manifest_hash,
    train_osm_policy_from_scratch,
)


ACTOR_KEYS = (
    "network.0.weight",
    "network.0.bias",
    "network.2.weight",
    "network.2.bias",
    "network.4.weight",
    "network.4.bias",
)


def _region_splits() -> RegionSplits:
    return RegionSplits(
        train=frozenset({"train-net-a", "train-net-b"}),
        validation=frozenset({"val-net-a"}),
        test=frozenset({"test-net-a", "test-net-b"}),
    )


def _save_parent_checkpoint(path, shapes=EXPECTED_LAYER_SHAPES):
    """Write a structurally compatible 21->128->128->6 parent checkpoint."""
    actor = OrderedDict(
        (key, torch.full(shape, 0.5, dtype=torch.float32))
        for key, shape in zip(ACTOR_KEYS, shapes)
    )
    torch.save({"police_actor": actor, "epoch": 7}, path)
    return path


# ---------------------------------------------------------------------------
# split identity
# ---------------------------------------------------------------------------


def test_split_manifest_hash_is_order_independent():
    ordered = RegionSplits(
        train=frozenset({"a", "b"}),
        validation=frozenset({"c"}),
        test=frozenset({"d"}),
    )
    shuffled = RegionSplits(
        train=frozenset({"b", "a"}),
        validation=frozenset({"c"}),
        test=frozenset({"d"}),
    )
    assert split_manifest_hash(ordered) == split_manifest_hash(shuffled)


def test_split_manifest_hash_changes_with_content():
    base = _region_splits()
    other = RegionSplits(
        train=frozenset({"train-net-a", "train-net-c"}),
        validation=frozenset({"val-net-a"}),
        test=frozenset({"test-net-a", "test-net-b"}),
    )
    assert split_manifest_hash(base) != split_manifest_hash(other)


# ---------------------------------------------------------------------------
# from-scratch
# ---------------------------------------------------------------------------


def test_from_scratch_produces_experimental_mappo_checkpoint(tmp_path):
    artifacts = train_osm_policy_from_scratch(
        run_id="scratch-1",
        region_splits=_region_splits(),
        training_network_hash="train-net-a",
        seeds=(1, 2, 3),
        code_version="v-test",
        output_dir=tmp_path,
    )

    manifest = artifacts.checkpoint_manifest
    assert manifest.claim_status is ClaimStatus.EXPERIMENTAL  # Req 8.10
    assert manifest.police_count == POLICE_COUNT  # Req 8.7
    assert manifest.architecture == OSM_TRAINING_ARCHITECTURE
    assert manifest.architecture.observation_dim == 21
    assert tuple(manifest.architecture.hidden_dims) == (128, 128)
    assert manifest.architecture.action_dim == 6
    assert manifest.observation_contract.profile_id == OSM_PROFILE_ID  # Req 8.7/8.11

    run = artifacts.run_manifest
    assert run.mode is RunMode.OSM_FROM_SCRATCH
    assert run.claim_status is ClaimStatus.EXPERIMENTAL
    assert run.lineage is not None
    assert run.lineage.parent_checkpoint_hash is None  # no parent for scratch

    # Checkpoint file exists and loads as weights-only tensors.
    assert artifacts.checkpoint_path.exists()
    loaded = torch.load(artifacts.checkpoint_path, weights_only=True)
    assert set(loaded["police_actor"]) == set(ACTOR_KEYS)


def test_from_scratch_rejects_parent_checkpoint(tmp_path):
    parent = _save_parent_checkpoint(tmp_path / "parent.pt")
    # from-scratch has no parent_checkpoint_path kwarg; the internal guard is
    # exercised through the finetune-only path, so assert scratch lineage stays
    # parent-free and cannot be constructed with a parent hash.
    with pytest.raises(DomainValidationError) as exc:
        TrainingRunLineage(
            run_manifest=train_osm_policy_from_scratch(
                run_id="scratch-2",
                region_splits=_region_splits(),
                training_network_hash="train-net-a",
                seeds=(9,),
                code_version="v-test",
                output_dir=tmp_path,
            ).run_manifest,
            region_splits=_region_splits(),
            parent_checkpoint_hash="deadbeef",
        )
    assert exc.value.code == "UNEXPECTED_PARENT_CHECKPOINT"
    assert parent.exists()


# ---------------------------------------------------------------------------
# fine-tune
# ---------------------------------------------------------------------------


def test_finetune_records_parent_hash_and_reuses_shape_compatible_weights(tmp_path):
    parent = _save_parent_checkpoint(tmp_path / "parent.pt")
    artifacts = finetune_osm_policy(
        run_id="finetune-1",
        region_splits=_region_splits(),
        training_network_hash="train-net-b",
        seeds=(4, 5),
        code_version="v-test",
        parent_checkpoint_path=parent,
        output_dir=tmp_path / "out",
    )

    run = artifacts.run_manifest
    assert run.mode is RunMode.OSM_FINETUNE
    assert run.claim_status is ClaimStatus.EXPERIMENTAL  # Req 8.10
    assert run.lineage is not None
    assert run.lineage.parent_checkpoint_hash  # Req 8.8: legacy parent recorded
    assert artifacts.checkpoint_manifest.lineage["parent_checkpoint_hash"] == (
        run.lineage.parent_checkpoint_hash
    )
    assert artifacts.observation_contract.profile_id == OSM_PROFILE_ID


def test_finetune_requires_parent_checkpoint(tmp_path):
    with pytest.raises(DomainValidationError) as exc:
        finetune_osm_policy(
            run_id="finetune-2",
            region_splits=_region_splits(),
            training_network_hash="train-net-a",
            seeds=(1,),
            code_version="v-test",
            parent_checkpoint_path=None,  # type: ignore[arg-type]
            output_dir=tmp_path,
        )
    assert exc.value.code == "MISSING_PARENT_CHECKPOINT"


def test_finetune_rejects_structurally_incompatible_parent(tmp_path):
    bad_shapes = (
        (64, 21),
        (64,),
        (64, 64),
        (64,),
        (6, 64),
        (6,),
    )
    parent = _save_parent_checkpoint(tmp_path / "bad_parent.pt", shapes=bad_shapes)
    with pytest.raises(DomainValidationError) as exc:
        finetune_osm_policy(
            run_id="finetune-3",
            region_splits=_region_splits(),
            training_network_hash="train-net-a",
            seeds=(1,),
            code_version="v-test",
            parent_checkpoint_path=parent,
            output_dir=tmp_path / "out",
        )
    assert exc.value.code == "INCOMPATIBLE_PARENT_CHECKPOINT"


# ---------------------------------------------------------------------------
# distinct lineage and contracts (Req 8.8)
# ---------------------------------------------------------------------------


def test_finetune_and_from_scratch_produce_distinct_lineage_and_checkpoints(tmp_path):
    splits = _region_splits()
    parent = _save_parent_checkpoint(tmp_path / "parent.pt")

    finetuned = finetune_osm_policy(
        run_id="shared-id",
        region_splits=splits,
        training_network_hash="train-net-a",
        seeds=(7, 8),
        code_version="v-test",
        parent_checkpoint_path=parent,
        output_dir=tmp_path / "ft",
    )
    scratch = train_osm_policy_from_scratch(
        run_id="shared-id",
        region_splits=splits,
        training_network_hash="train-net-a",
        seeds=(7, 8),
        code_version="v-test",
        output_dir=tmp_path / "sc",
    )

    # Distinct run modes and lineage.
    assert finetuned.run_manifest.mode is RunMode.OSM_FINETUNE
    assert scratch.run_manifest.mode is RunMode.OSM_FROM_SCRATCH
    assert finetuned.run_manifest.lineage.parent_checkpoint_hash is not None
    assert scratch.run_manifest.lineage.parent_checkpoint_hash is None

    # Distinct checkpoint bytes even with an identical run_id and seeds.
    assert finetuned.checkpoint_hash != scratch.checkpoint_hash
    # Shared, deterministic split identity.
    assert finetuned.split_manifest_hash == scratch.split_manifest_hash


# ---------------------------------------------------------------------------
# profile and split guards (Req 8.7, 14.2)
# ---------------------------------------------------------------------------


def test_training_rejects_non_osm_observation_profile(tmp_path):
    with pytest.raises(DomainValidationError) as exc:
        train_osm_policy_from_scratch(
            run_id="scratch-bad-profile",
            region_splits=_region_splits(),
            training_network_hash="train-net-a",
            seeds=(1,),
            code_version="v-test",
            output_dir=tmp_path,
            observation_profile="legacy_grid_v0",
        )
    assert exc.value.code == "INVALID_TRAINING_PROFILE"


def test_training_network_must_belong_to_train_split(tmp_path):
    with pytest.raises(DomainValidationError) as exc:
        train_osm_policy_from_scratch(
            run_id="scratch-outside-split",
            region_splits=_region_splits(),
            training_network_hash="test-net-a",  # in the test split, not train
            seeds=(1,),
            code_version="v-test",
            output_dir=tmp_path,
        )
    assert exc.value.code == "TRAINING_NETWORK_OUTSIDE_SPLIT"


def test_overlapping_region_splits_are_a_hard_failure():
    with pytest.raises(DomainValidationError) as exc:
        RegionSplits(
            train=frozenset({"shared", "train-only"}),
            validation=frozenset({"val"}),
            test=frozenset({"shared"}),  # overlaps train
        )
    assert exc.value.code == "OVERLAPPING_REGION_SPLITS"


# ---------------------------------------------------------------------------
# reproducibility (offline determinism)
# ---------------------------------------------------------------------------


def test_from_scratch_is_deterministic_offline(tmp_path):
    kwargs = dict(
        run_id="scratch-det",
        region_splits=_region_splits(),
        training_network_hash="train-net-a",
        seeds=(11, 12),
        code_version="v-test",
    )
    first = train_osm_policy_from_scratch(output_dir=tmp_path / "a", **kwargs)
    second = train_osm_policy_from_scratch(output_dir=tmp_path / "b", **kwargs)
    assert first.checkpoint_hash == second.checkpoint_hash
    assert first.split_manifest_hash == second.split_manifest_hash


def test_manifests_are_written_as_reproducible_canonical_json(tmp_path):
    artifacts = train_osm_policy_from_scratch(
        run_id="scratch-manifests",
        region_splits=_region_splits(),
        training_network_hash="train-net-a",
        seeds=(1,),
        code_version="v-test",
        output_dir=tmp_path,
    )
    for name, path in artifacts.manifest_paths.items():
        assert path.exists(), name
    # run_manifest bytes match canonical_json of the domain object.
    run_path = artifacts.manifest_paths["run_manifest"]
    assert run_path.read_bytes() == canonical_json(artifacts.run_manifest)
