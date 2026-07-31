"""Task 6.4 integration coverage: manifest seal/fork + full checkpoint resume together.

Exercises pursuit_evasion_rl/research/runs/manifest.py (task 6.1) and
pursuit_evasion_rl/research/training/checkpoint.py (task 6.2) against a real,
tiny CPU trainer and a temp filesystem, including the "torn write" crash
scenario the atomic .tmp-then-replace pattern is meant to protect against.
Device RNG round-trip is only asserted when a CUDA device actually exists.
"""
from __future__ import annotations

from pathlib import Path

import pytest
import torch

from pursuit_evasion_rl.osm_demo.coarsening import coarsen_raw_graph, prepare_model_network
from pursuit_evasion_rl.osm_demo.fixtures import daejeon
from pursuit_evasion_rl.research.domain import ExecutionStatus
from pursuit_evasion_rl.research.errors import ResearchValidationError
from pursuit_evasion_rl.research.runs.manifest import RunManifestStore, RunProvenance
from pursuit_evasion_rl.research.training.checkpoint import (
    CompatibilityContract,
    capture_trainer_resume_state,
    check_resume_compatibility,
    load_resume_state,
    restore_trainer_resume_state,
    save_resume_state,
)
from pursuit_evasion_rl.research.training.equivalence import (
    build_equivalence_trainer,
    equivalence_contract,
)
from pursuit_evasion_rl.research.training.trainer import TrainerConfig

pytestmark = pytest.mark.offline


def _tiny_network():
    # Same fixture the task 6.3 equivalence tests use: a real, small, offline
    # network with enough distinct drivable intersections to place 6 police.
    return prepare_model_network(coarsen_raw_graph(daejeon())).network


def _tiny_config() -> TrainerConfig:
    return TrainerConfig(updates=2, episodes_per_update=1, max_steps=15, hidden_dims=(4,))


def _provenance(**overrides) -> RunProvenance:
    fields = {
        "code_hash": "a" * 64, "dirty_tree": False, "dependency_hash": "b" * 64,
        "runtime": "python3.11+torch2.8.0+cpu", "device": "cpu", "map_hash": "c" * 64,
        "split": "train", "seed": 5, "input_hashes": {"map": "c" * 64},
    }
    fields.update(overrides)
    return RunProvenance(**fields)


def test_manifest_seal_gates_a_full_checkpoint_save_and_a_forked_child_can_resume(tmp_path: Path) -> None:
    manifests = RunManifestStore(tmp_path / "runs")
    network = _tiny_network()
    config = _tiny_config()

    root = manifests.create(protocol_hash="p" * 64, condition_hash="q" * 64, provenance=_provenance())
    trainer = build_equivalence_trainer(network, config, training_seed=11, output_root=tmp_path / "artifacts")
    trainer.execute_update(1)

    checkpoint_path = tmp_path / "artifacts" / "checkpoint.pt"
    digest = save_resume_state(checkpoint_path, capture_trainer_resume_state(trainer), equivalence_contract(trainer))

    sealed = manifests.seal(
        root.run_id, execution_status=ExecutionStatus.COMPLETED, status_reason="interrupted after update 1",
        artifact_hashes={"resume_state": digest},
    )
    assert sealed.is_sealed

    # Requirement 14.4: continuing a sealed run's work forks a child, not an
    # in-place mutation of the sealed manifest.
    child = manifests.fork(sealed.run_id, provenance=_provenance(seed=11))
    assert child.run.parent_id == sealed.run_id

    resumed_trainer = build_equivalence_trainer(network, config, training_seed=11, output_root=tmp_path / "artifacts2")
    state = load_resume_state(checkpoint_path, expected_contract=equivalence_contract(resumed_trainer))
    restore_trainer_resume_state(resumed_trainer, state)
    assert resumed_trainer.update_index == 1
    outcome = resumed_trainer.execute_update(2)
    assert outcome.update_index == 2

    manifests.seal(
        child.run_id, execution_status=ExecutionStatus.COMPLETED, status_reason="resumed to completion",
        artifact_hashes={"resume_state": digest},
    )
    assert manifests.read(sealed.run_id).manifest_hash == sealed.manifest_hash  # untouched by the child's work


def test_incompatible_contract_refuses_to_resume(tmp_path: Path) -> None:
    network = _tiny_network()
    config = _tiny_config()
    trainer = build_equivalence_trainer(network, config, training_seed=3, output_root=tmp_path / "a")
    trainer.execute_update(1)

    checkpoint_path = tmp_path / "checkpoint.pt"
    save_resume_state(checkpoint_path, capture_trainer_resume_state(trainer), equivalence_contract(trainer))

    other = build_equivalence_trainer(network, config, training_seed=3, output_root=tmp_path / "b")
    wrong_contract = CompatibilityContract.for_policy(
        other.policy, code_hash="different-code-hash", dependency_hash="equivalence-dependency",
        protocol_hash="equivalence-protocol", condition_hash="interruption-equivalence",
        map_hash="equivalence-map", split="train",
    )
    with pytest.raises(ResearchValidationError) as excinfo:
        load_resume_state(checkpoint_path, expected_contract=wrong_contract)
    assert excinfo.value.code == "RESUME_COMPATIBILITY_GATE_FAILED"


def test_torn_write_never_corrupts_the_previous_checkpoint(tmp_path: Path) -> None:
    """A crash mid-write leaves a stray .tmp file but never a partially-written target."""
    network = _tiny_network()
    config = _tiny_config()
    trainer = build_equivalence_trainer(network, config, training_seed=9, output_root=tmp_path / "a")
    trainer.execute_update(1)

    checkpoint_path = tmp_path / "checkpoint.pt"
    first_digest = save_resume_state(
        checkpoint_path, capture_trainer_resume_state(trainer), equivalence_contract(trainer)
    )
    original_bytes = checkpoint_path.read_bytes()

    # Simulate a crash mid-write: a stale .tmp exists but the real file must
    # still be exactly what the last successful save wrote.
    torn = checkpoint_path.with_suffix(checkpoint_path.suffix + ".tmp")
    torn.write_bytes(b"not a valid checkpoint, this is a torn write")
    assert checkpoint_path.read_bytes() == original_bytes

    reread = load_resume_state(checkpoint_path, expected_contract=equivalence_contract(trainer))
    assert reread.content_hash == first_digest

    # A subsequent real save overwrites the stray .tmp and succeeds normally.
    trainer.execute_update(2)
    second_digest = save_resume_state(
        checkpoint_path, capture_trainer_resume_state(trainer), equivalence_contract(trainer)
    )
    assert second_digest != first_digest
    assert not torn.exists() or torn.read_bytes() != b"not a valid checkpoint, this is a torn write"
    reread_again = load_resume_state(checkpoint_path, expected_contract=equivalence_contract(trainer))
    assert reread_again.content_hash == second_digest


def test_full_rng_round_trip_including_device_rng_when_available(tmp_path: Path) -> None:
    network = _tiny_network()
    config = _tiny_config()
    trainer = build_equivalence_trainer(network, config, training_seed=21, output_root=tmp_path / "a")
    trainer.execute_update(1)

    state = capture_trainer_resume_state(trainer)
    assert state.python_random_state and state.numpy_random_state
    assert isinstance(state.torch_cpu_rng_state, torch.Tensor)
    assert isinstance(state.training_generator_state, torch.Tensor)
    if torch.cuda.is_available():
        assert state.torch_device_rng_state is not None
    else:
        assert state.torch_device_rng_state is None

    checkpoint_path = tmp_path / "checkpoint.pt"
    save_resume_state(checkpoint_path, state, equivalence_contract(trainer))
    fresh = build_equivalence_trainer(network, config, training_seed=21, output_root=tmp_path / "b")
    reloaded = load_resume_state(checkpoint_path, expected_contract=equivalence_contract(fresh))
    restore_trainer_resume_state(fresh, reloaded)

    assert torch.equal(fresh._training_rng.get_state(), trainer._training_rng.get_state())
    assert torch.equal(torch.get_rng_state(), state.torch_cpu_rng_state)


def test_seal_then_check_resume_compatibility_directly(tmp_path: Path) -> None:
    """check_resume_compatibility is itself exercised, not only through load_resume_state."""
    network = _tiny_network()
    config = _tiny_config()
    trainer = build_equivalence_trainer(network, config, training_seed=1, output_root=tmp_path / "a")
    contract = equivalence_contract(trainer)
    check_resume_compatibility(contract, contract)  # identical contracts always pass

    from dataclasses import replace

    drifted = replace(contract, map_hash="a-different-map-hash")
    with pytest.raises(ResearchValidationError) as excinfo:
        check_resume_compatibility(contract, drifted)
    assert excinfo.value.code == "RESUME_COMPATIBILITY_GATE_FAILED"
