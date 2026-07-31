"""Focused unit tests for safe checkpoint inspection and compatibility reports."""

from __future__ import annotations

from collections import OrderedDict
import hashlib
import json

import pytest
import torch

from pursuit_evasion_rl.osm_demo.checkpoint import (
    EXPECTED_LAYER_SHAPES,
    CheckpointInspector,
)
from pursuit_evasion_rl.osm_demo.models import CheckStatus, ObservationContract


ACTOR_KEYS = (
    "network.0.weight",
    "network.0.bias",
    "network.2.weight",
    "network.2.bias",
    "network.4.weight",
    "network.4.bias",
)


def actor_state(shapes=EXPECTED_LAYER_SHAPES):
    return OrderedDict(
        (key, torch.zeros(shape, dtype=torch.float32))
        for key, shape in zip(ACTOR_KEYS, shapes)
    )


def save_checkpoint(path, *, actor=None):
    torch.save(
        {
            "police_actor": actor or actor_state(),
            "police_critic": {"network.0.weight": torch.zeros((128, 21))},
            "epoch": 12,
        },
        path,
    )


@pytest.fixture
def osm_contract():
    return ObservationContract(
        profile_id="osm_topology_v1",
        fields=(
            "agent_id_onehot",
            "normalized_time",
            "segment_progress",
            "fugitive_directed_distance",
            "boundary_directed_distance",
            "outgoing_fugitive_scores",
            "team_and_fugitive_distances",
        ),
        field_sizes=(6, 1, 1, 1, 1, 5, 6),
        dtype="float32",
        padding_value=1.0,
        normalization={
            "range": [0.0, 1.0],
            "unreachable": 1.0,
            "distance_scale": "manifest_clipping_distance",
        },
        action_ordering="turn_angle_then_bearing_length_hash;stay=5",
    )


def checks_by_code(inspection, section):
    return {check.code: check for check in getattr(inspection.report, section)}


def complete_manifest(checkpoint_hash, contract, *, network_hash="network-a"):
    return {
        "checkpoint_hash": checkpoint_hash,
        "architecture": {
            "observation_dim": 21,
            "hidden_dims": [128, 128],
            "action_dim": 6,
            "fixed_max_degree": 5,
        },
        "police_count": 6,
        "observation_contract": {
            "profile_id": contract.profile_id,
            "fields": list(contract.fields),
            "field_sizes": list(contract.field_sizes),
            "dtype": contract.dtype,
            "padding_value": contract.padding_value,
            "normalization": dict(contract.normalization),
            "action_ordering": contract.action_ordering,
        },
        "training_networks": [network_hash],
        "performance_evidence": {"run_id": "reproducible-run-1"},
    }


def test_valid_actor_without_manifest_infers_only_structure(tmp_path):
    path = tmp_path / "actor.pt"
    save_checkpoint(path)

    inspection = CheckpointInspector().inspect(path)

    assert inspection.checkpoint_hash == hashlib.sha256(path.read_bytes()).hexdigest()
    assert inspection.inferred_contract.layer_shapes == EXPECTED_LAYER_SHAPES
    assert inspection.inferred_contract.observation_dim == 21
    assert inspection.inferred_contract.hidden_dims == (128, 128)
    assert inspection.inferred_contract.action_dim == 6
    assert inspection.inferred_contract.unknown == {
        "police_count",
        "observation_semantics",
        "training_region",
        "performance",
    }
    structure = checks_by_code(inspection, "structure")
    semantics = checks_by_code(inspection, "semantics")
    assert structure["CHECKPOINT_LAYER_SHAPES"].status is CheckStatus.PASS
    assert structure["POLICE_COUNT"].status is CheckStatus.UNKNOWN
    assert semantics["OBSERVATION_PROFILE"].status is CheckStatus.UNKNOWN
    assert semantics["TRAINING_REGION_PROVENANCE"].status is CheckStatus.UNKNOWN
    assert semantics["PERFORMANCE_EVIDENCE"].status is CheckStatus.UNKNOWN
    assert inspection.report.overall is CheckStatus.UNKNOWN
    json.dumps(inspection.as_dict())


def test_complete_matching_manifest_and_runtime_produce_sectioned_pass_report(
    tmp_path, osm_contract
):
    path = tmp_path / "actor.pt"
    save_checkpoint(path)
    checkpoint_hash = hashlib.sha256(path.read_bytes()).hexdigest()
    runtime = {
        "police_count": 6,
        "fixed_max_degree": 5,
        "observation_dim": 21,
        "hidden_dims": [128, 128],
        "action_dim": 6,
    }

    inspection = CheckpointInspector().inspect(
        path,
        complete_manifest(checkpoint_hash, osm_contract),
        selected_observation_contract=osm_contract,
        execution_config=runtime,
        network_hash="network-a",
    )

    assert inspection.report.overall is CheckStatus.PASS
    report = inspection.report.as_dict()
    assert set(report["sections"]) == {"structure", "semantics", "network", "execution"}
    assert not report["inference_blocked"]
    assert report["verified_osm_inference_allowed"]
    assert report["mismatches"] == []


def test_manifest_and_runtime_mismatches_include_expected_and_actual_values(
    tmp_path, osm_contract
):
    path = tmp_path / "actor.pt"
    save_checkpoint(path)
    manifest = complete_manifest("0" * 64, osm_contract)
    manifest["architecture"]["hidden_dims"] = [64, 64]
    manifest["police_count"] = 4
    manifest["fixed_max_degree"] = 7
    manifest["observation_contract"]["profile_id"] = "legacy_grid_v0"

    inspection = CheckpointInspector().inspect(
        path,
        manifest,
        selected_observation_contract=osm_contract,
        execution_config={
            "police_count": 5,
            "fixed_max_degree": 5,
            "observation_dim": 21,
            "hidden_dims": [128, 128],
            "action_dim": 6,
        },
    )

    assert inspection.report.overall is CheckStatus.FAIL
    report = inspection.report.as_dict()
    mismatch_codes = {item["code"] for item in report["mismatches"]}
    assert {
        "MANIFEST_CHECKPOINT_HASH",
        "MANIFEST_ACTOR_LAYER_SHAPES",
        "POLICE_COUNT",
        "FIXED_MAX_DEGREE",
        "EXECUTION_POLICE_COUNT",
    } <= mismatch_codes
    police_mismatch = next(item for item in report["mismatches"] if item["code"] == "POLICE_COUNT")
    assert police_mismatch["expected"] == 6
    assert police_mismatch["actual"] == 4
    assert checks_by_code(inspection, "semantics")["OBSERVATION_PROFILE"].status is CheckStatus.WARNING
    assert report["inference_blocked"]


def test_prefixed_state_dict_is_recognized_without_loading_model_classes(tmp_path):
    path = tmp_path / "prefixed.pt"
    prefixed = OrderedDict(
        (f"module.police_actor.{key}", value) for key, value in actor_state().items()
    )
    prefixed["module.police_critic.network.0.weight"] = torch.zeros((128, 21))
    torch.save({"state_dict": prefixed}, path)

    inspection = CheckpointInspector().inspect(path)

    assert inspection.inferred_contract.actor_key == "state_dict.police_actor"
    assert inspection.inferred_contract.layer_shapes == EXPECTED_LAYER_SHAPES
    assert checks_by_code(inspection, "structure")["CHECKPOINT_LAYER_SHAPES"].status is CheckStatus.PASS


def test_nested_repository_checkpoint_is_inspected_but_config_semantics_remain_unknown(tmp_path):
    path = tmp_path / "training-checkpoint.pt"
    torch.save(
        {
            "model_state": {
                "police_actor": actor_state(),
                "police_critic": {"network.0.weight": torch.zeros((128, 21))},
            },
            # These values are not a Checkpoint_Manifest and must not be used
            # as semantic proof.
            "config": {"num_police": 6, "training_region": "synthetic-grid"},
        },
        path,
    )

    inspection = CheckpointInspector().inspect(path)

    assert inspection.inferred_contract.actor_key == "model_state.police_actor"
    assert inspection.inferred_contract.layer_shapes == EXPECTED_LAYER_SHAPES
    assert inspection.inferred_contract.unknown == {
        "police_count",
        "observation_semantics",
        "training_region",
        "performance",
    }
    assert checks_by_code(inspection, "structure")["POLICE_COUNT"].status is CheckStatus.UNKNOWN
    assert checks_by_code(inspection, "semantics")["TRAINING_REGION_PROVENANCE"].status is CheckStatus.UNKNOWN


def test_warning_mismatches_are_machine_readable(tmp_path, osm_contract):
    path = tmp_path / "legacy.pt"
    save_checkpoint(path)
    checkpoint_hash = hashlib.sha256(path.read_bytes()).hexdigest()
    manifest = complete_manifest(checkpoint_hash, osm_contract)
    manifest["observation_contract"]["profile_id"] = "legacy_grid_v0"

    inspection = CheckpointInspector().inspect(
        path,
        manifest,
        selected_observation_contract=osm_contract,
    )

    warning = next(
        item for item in inspection.report.as_dict()["mismatches"]
        if item["code"] == "OBSERVATION_PROFILE"
    )
    assert warning == {
        "section": "semantics",
        "code": "OBSERVATION_PROFILE",
        "status": "warning",
        "expected": "osm_topology_v1",
        "actual": "legacy_grid_v0",
        "message": "Legacy observation semantics are not compatible with the OSM observation profile",
    }


def test_incompatible_actor_shape_and_corrupt_file_return_fail_reports(tmp_path):
    bad_shape_path = tmp_path / "bad-shape.pt"
    bad_shapes = list(EXPECTED_LAYER_SHAPES)
    bad_shapes[0] = (128, 20)
    save_checkpoint(bad_shape_path, actor=actor_state(tuple(bad_shapes)))

    incompatible = CheckpointInspector().inspect(bad_shape_path)

    structure = checks_by_code(incompatible, "structure")
    assert structure["CHECKPOINT_OBSERVATION_DIM"].status is CheckStatus.FAIL
    assert structure["CHECKPOINT_LAYER_SHAPES"].status is CheckStatus.FAIL
    assert structure["CHECKPOINT_OBSERVATION_DIM"].expected == 21
    assert structure["CHECKPOINT_OBSERVATION_DIM"].actual == 20

    corrupt_path = tmp_path / "corrupt.pt"
    corrupt_path.write_bytes(b"not a torch checkpoint")
    corrupt = CheckpointInspector().inspect(corrupt_path)

    assert corrupt.report.overall is CheckStatus.FAIL
    assert corrupt.report.structure[0].code == "CHECKPOINT_SAFE_INSPECTION_FAILED"
    assert corrupt.report.structure[0].status is CheckStatus.FAIL
    assert corrupt.inferred_contract.observation_dim is None
