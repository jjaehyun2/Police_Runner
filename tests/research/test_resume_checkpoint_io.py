"""Task 6.2 full Resume_State checkpoint IO and compatibility gate."""
from __future__ import annotations

from dataclasses import replace
import random

import numpy as np
import pytest
import torch

from pursuit_evasion_rl.osm_demo.coarsening import coarsen_raw_graph, prepare_model_network
from pursuit_evasion_rl.osm_demo.fixtures import daejeon
from pursuit_evasion_rl.research.errors import ResearchValidationError
from pursuit_evasion_rl.research.maps import DataHandle, HandleKind, SplitScope, TuningDataView
from pursuit_evasion_rl.research.smdp import AsyncDecisionTransitionBuffer, DecisionEpoch
from pursuit_evasion_rl.research.training.checkpoint import (
    OPTIONAL_RESUME_FIELDS,
    REQUIRED_RESUME_FIELDS,
    CompatibilityContract,
    PendingTransitionSnapshot,
    ResumeState,
    capture_generator_state,
    capture_numpy_random_state,
    capture_python_random_state,
    capture_trainer_resume_state,
    check_resume_compatibility,
    load_resume_state,
    policy_hidden_dims,
    restore_generator_state,
    restore_trainer_resume_state,
    resume_state_content_hash,
    save_resume_state,
)
from pursuit_evasion_rl.research.training.trainer import ResearchTrainer, TrainerConfig

pytestmark = pytest.mark.offline


def _network():
    return prepare_model_network(coarsen_raw_graph(daejeon())).network


def _trainer(tmp_path, *, seed: int = 17, run_id: str = "checkpoint-run") -> ResearchTrainer:
    return ResearchTrainer(
        train_network=_network(),
        validation_network=_network(),
        tuning_data=TuningDataView(
            train=(DataHandle(SplitScope.TRAIN, HandleKind.MAP, "fixture-train"),),
            validation=(DataHandle(SplitScope.VALIDATION, HandleKind.MAP, "fixture-validation"),),
        ),
        config=TrainerConfig(updates=1, episodes_per_update=1, validation_episodes=1, max_steps=4, hidden_dims=(4,)),
        condition_id="checkpoint-fixture",
        training_seed=seed,
        output_root=tmp_path / "artifacts",
        run_id=run_id,
    )


def _pending_snapshots() -> tuple[PendingTransitionSnapshot, ...]:
    """Pending SMDP decisions taken mid-rollout, straight out of the real buffer."""
    buffer = AsyncDecisionTransitionBuffer(gamma=0.9, max_virtual_hops=4, actor_obs_dim=2, critic_context_dim=3, action_count=3)
    buffer.start_decisions({
        officer: DecisionEpoch.create(
            actor_obs=(officer / 10.0, 1.0),
            action=officer % 2,
            action_mask=(True, True, False),
            old_log_prob=-0.25 - officer / 100.0,
            critic_context=(0.5, float(officer), 1.0),
        )
        for officer in range(6)
    })
    buffer.record_physical_step((1.0, 2.0, 3.0, 4.0, 5.0, 6.0), virtual_hops=[0] * 6)
    buffer.record_physical_step((0.5, 0.5, 0.5, 0.5, 0.5, 0.5), virtual_hops=[1, 0, 0, 0, 0, 0])
    return tuple(PendingTransitionSnapshot.from_pending(item) for item in buffer.pending)


def _fully_populated_state(trainer: ResearchTrainer) -> ResumeState:
    """A Resume_State where every optional field carries real, non-null data."""
    evader = np.random.default_rng(777)
    evader.random(3)
    placement = np.random.default_rng(31)
    placement.random(5)
    sampler = np.random.default_rng(99)
    return replace(
        capture_trainer_resume_state(
            trainer,
            parent_run_id="20260731T101500-abcdef123456",
            scheduler_state={"last_epoch": 4, "base_lrs": [3e-4]},
            scaler_state={"scale": 65536.0, "growth_tracker": 2},
            curriculum_state={"stage": 2, "promoted_at_update": 7},
            hysteresis_state={"target": 3, "steps_held": 2, "switch_count": 1, "min_hold_k": 3, "margin": 0.25},
            running_normalization={"mean": [0.0, 1.0], "var": [1.0, 2.0], "count": 128},
            pending_transitions=[item.to_pending() for item in _pending_snapshots()],
            rollout_buffer={"episode_seed": 11, "completed": 3, "partial_rewards": [0.5, 0.25]},
            evader_rng=evader,
            placement_rng=placement,
            sampler_rng=sampler,
        ),
        episode_index=9,
        update_index=4,
        environment_step_index=812,
    )


def _contract(trainer: ResearchTrainer) -> CompatibilityContract:
    return CompatibilityContract.for_policy(
        trainer.policy,
        code_hash="code-a", dependency_hash="deps-a", protocol_hash="protocol-a",
        condition_hash="condition-a", map_hash="map-a", split="train",
    )


# ---------------------------------------------------------------------------
# Round trip (Requirement 14.5)
# ---------------------------------------------------------------------------


def test_every_resume_field_round_trips_including_optional_and_pending_smdp_state(tmp_path):
    trainer = _trainer(tmp_path)
    state = _fully_populated_state(trainer)
    path = tmp_path / "resume" / "state.pt"

    digest = save_resume_state(path, state, _contract(trainer))
    loaded = load_resume_state(path, expected_contract=_contract(trainer))

    assert digest == resume_state_content_hash(loaded) == loaded.content_hash
    assert not list(path.parent.glob("*.tmp"))

    for name in ("actor", "critic"):
        for key, tensor in state.policy_state_dict[name].items():
            assert torch.equal(loaded.policy_state_dict[name][key], tensor)
    assert torch.equal(loaded.torch_cpu_rng_state, state.torch_cpu_rng_state)
    assert torch.equal(loaded.training_generator_state, state.training_generator_state)
    assert (loaded.episode_index, loaded.update_index, loaded.environment_step_index) == (9, 4, 812)
    assert loaded.training_seed == trainer.training_seed
    assert loaded.parent_run_id == "20260731T101500-abcdef123456"
    for name in ("python_random_state", "numpy_random_state", "scheduler_state", "scaler_state",
                 "evader_rng_state", "placement_rng_state", "sampler_rng_state", "curriculum_state",
                 "hysteresis_state", "running_normalization", "rollout_buffer"):
        assert dict(getattr(loaded, name)) == dict(getattr(state, name)), name

    assert loaded.pending_transitions == state.pending_transitions
    for snapshot, original in zip(loaded.pending_transitions, state.pending_transitions):
        restored = snapshot.to_pending()
        assert restored.epoch.actor_obs == original.actor_obs
        assert restored.epoch.stored_mask_bytes == original.stored_mask_bytes
        assert restored.epoch.old_log_prob == original.old_log_prob
        assert restored.epoch.critic_context == original.critic_context
        assert restored.discounted_reward == original.discounted_reward
        assert restored.duration_steps == original.duration_steps


def test_schema_carries_every_optional_field_as_explicit_null_when_unused(tmp_path):
    trainer = _trainer(tmp_path)
    state = capture_trainer_resume_state(trainer)
    path = tmp_path / "sparse.pt"

    save_resume_state(path, state, _contract(trainer))
    payload = torch.load(path, map_location="cpu", weights_only=False)

    for name in OPTIONAL_RESUME_FIELDS:
        assert name in payload["state"], f"{name} must be serialized, never omitted"
    unset = set(OPTIONAL_RESUME_FIELDS) - {"torch_device_rng_state"}
    assert all(payload["state"][name] is None for name in unset)
    assert load_resume_state(path, expected_contract=_contract(trainer)).parent_run_id is None


def test_device_rng_is_captured_when_available_and_never_crashes_without_a_device(tmp_path):
    state = capture_trainer_resume_state(_trainer(tmp_path))
    if torch.cuda.is_available():
        assert state.torch_device_rng_state is not None
        assert len(state.torch_device_rng_state["cuda"]) == torch.cuda.device_count()
    else:
        assert state.torch_device_rng_state is None


def test_dedicated_numpy_generator_streams_restore_their_exact_next_draws():
    source = np.random.default_rng(2026)
    source.random(4)
    captured = capture_generator_state(source)
    expected = source.random(3)

    revived = np.random.default_rng(1)
    restore_generator_state(revived, captured)
    assert np.array_equal(revived.random(3), expected)


# ---------------------------------------------------------------------------
# Content hash
# ---------------------------------------------------------------------------


def test_content_hash_is_stable_across_identical_saves_and_moves_with_any_change(tmp_path):
    trainer = _trainer(tmp_path)
    state = _fully_populated_state(trainer)

    first = save_resume_state(tmp_path / "a.pt", state, _contract(trainer))
    second = save_resume_state(tmp_path / "b.pt", state, _contract(trainer))
    assert first == second

    assert resume_state_content_hash(replace(state, update_index=state.update_index + 1)) != first
    assert resume_state_content_hash(replace(state, episode_index=state.episode_index + 1)) != first
    assert resume_state_content_hash(replace(state, scheduler_state=None)) != first
    assert resume_state_content_hash(replace(state, pending_transitions=None)) != first

    mutated = {key: dict(value) for key, value in state.policy_state_dict.items()}
    key = next(iter(mutated["actor"]))
    mutated["actor"][key] = mutated["actor"][key] + 1.0
    assert resume_state_content_hash(replace(state, policy_state_dict=mutated)) != first

    generator = state.training_generator_state.clone()
    generator[0] = (int(generator[0].item()) + 1) % 256
    assert resume_state_content_hash(replace(state, training_generator_state=generator)) != first


def test_resume_state_snapshot_does_not_alias_live_trainer_tensors(tmp_path):
    trainer = _trainer(tmp_path)
    state = capture_trainer_resume_state(trainer)
    before = resume_state_content_hash(state)

    trainer.execute_update(1)

    assert resume_state_content_hash(state) == before


# ---------------------------------------------------------------------------
# Compatibility gate (Requirement 14.6-14.8)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("field_name", ("code_hash", "dependency_hash", "protocol_hash", "condition_hash", "map_hash", "split"))
def test_identity_drift_refuses_to_resume_the_original_run(tmp_path, field_name):
    trainer = _trainer(tmp_path)
    path = tmp_path / "state.pt"
    save_resume_state(path, capture_trainer_resume_state(trainer), _contract(trainer))
    incompatible = replace(_contract(trainer), **{field_name: "drifted"})

    with pytest.raises(ResearchValidationError) as error:
        load_resume_state(path, expected_contract=incompatible)
    assert error.value.code == "RESUME_COMPATIBILITY_GATE_FAILED"
    assert error.value.path == field_name


@pytest.mark.parametrize(
    ("field_name", "value"),
    (
        ("actor_obs_dim", 27),
        ("critic_context_dim", 12),
        ("action_dim", 5),
        ("num_officers", 4),
        ("hidden_dims", (8, 8)),
    ),
)
def test_tensor_shape_drift_refuses_to_resume_the_original_run(tmp_path, field_name, value):
    trainer = _trainer(tmp_path)
    path = tmp_path / "state.pt"
    save_resume_state(path, capture_trainer_resume_state(trainer), _contract(trainer))
    incompatible = replace(_contract(trainer), **{field_name: value})

    with pytest.raises(ResearchValidationError) as error:
        load_resume_state(path, expected_contract=incompatible)
    assert error.value.code == "RESUME_TENSOR_SHAPE_MISMATCH"
    assert error.value.path == field_name


def test_contract_reads_real_policy_shapes(tmp_path):
    trainer = _trainer(tmp_path)
    contract = _contract(trainer)
    assert contract.actor_obs_dim == trainer.policy.actor_obs_dim
    assert contract.critic_context_dim == trainer.policy.critic_context_dim
    assert contract.action_dim == trainer.policy.action_dim
    assert contract.num_officers == trainer.policy.num_officers
    assert contract.hidden_dims == policy_hidden_dims(trainer.policy) == trainer.config.hidden_dims


def test_expected_content_hash_mismatch_is_a_gate_failure(tmp_path):
    trainer = _trainer(tmp_path)
    saved = replace(_contract(trainer), resume_state_hash="a" * 64)
    current = replace(_contract(trainer), resume_state_hash="b" * 64)

    with pytest.raises(ResearchValidationError) as error:
        check_resume_compatibility(saved, current)
    assert error.value.code == "RESUME_STATE_HASH_MISMATCH"


# ---------------------------------------------------------------------------
# Incomplete / corrupted checkpoints must fail loudly
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("field_name", REQUIRED_RESUME_FIELDS)
def test_each_individually_missing_required_field_is_rejected(tmp_path, field_name):
    trainer = _trainer(tmp_path)
    path = tmp_path / "state.pt"
    save_resume_state(path, _fully_populated_state(trainer), _contract(trainer))

    payload = torch.load(path, map_location="cpu", weights_only=False)
    payload["state"][field_name] = None
    torch.save(payload, path)

    with pytest.raises(ResearchValidationError) as error:
        load_resume_state(path, expected_contract=_contract(trainer))
    assert error.value.code == "INCOMPLETE_RESUME_STATE"
    assert field_name in error.value.actual


@pytest.mark.parametrize("field_name", ("training_generator_state", "torch_cpu_rng_state", "policy_state_dict", "episode_index"))
def test_tampered_state_fails_the_content_hash(tmp_path, field_name):
    trainer = _trainer(tmp_path)
    path = tmp_path / "state.pt"
    save_resume_state(path, _fully_populated_state(trainer), _contract(trainer))

    payload = torch.load(path, map_location="cpu", weights_only=False)
    if field_name == "episode_index":
        payload["state"][field_name] += 1
    elif field_name == "policy_state_dict":
        key = next(iter(payload["state"][field_name]["actor"]))
        payload["state"][field_name]["actor"][key] = payload["state"][field_name]["actor"][key] + 1.0
    else:
        tensor = payload["state"][field_name].clone()
        tensor[0] = (int(tensor[0].item()) + 1) % 256
        payload["state"][field_name] = tensor
    torch.save(payload, path)

    with pytest.raises(ResearchValidationError) as error:
        load_resume_state(path, expected_contract=_contract(trainer))
    assert error.value.code == "RESUME_STATE_HASH_MISMATCH"


def test_torn_write_is_reported_as_a_corrupt_checkpoint(tmp_path):
    trainer = _trainer(tmp_path)
    path = tmp_path / "state.pt"
    save_resume_state(path, capture_trainer_resume_state(trainer), _contract(trainer))
    path.write_bytes(path.read_bytes()[: len(path.read_bytes()) // 2])

    with pytest.raises(ResearchValidationError) as error:
        load_resume_state(path, expected_contract=_contract(trainer))
    assert error.value.code == "CORRUPT_RESUME_STATE"


def test_missing_checkpoint_is_reported_rather_than_silently_starting_fresh(tmp_path):
    trainer = _trainer(tmp_path)
    with pytest.raises(ResearchValidationError) as error:
        load_resume_state(tmp_path / "absent.pt", expected_contract=_contract(trainer))
    assert error.value.code == "RESUME_STATE_NOT_FOUND"


def test_unknown_extra_field_is_rejected(tmp_path):
    trainer = _trainer(tmp_path)
    path = tmp_path / "state.pt"
    save_resume_state(path, capture_trainer_resume_state(trainer), _contract(trainer))

    payload = torch.load(path, map_location="cpu", weights_only=False)
    payload["state"]["smuggled_field"] = 1
    torch.save(payload, path)

    with pytest.raises(ResearchValidationError) as error:
        load_resume_state(path, expected_contract=_contract(trainer))
    assert error.value.code == "UNKNOWN_RESUME_STATE_FIELD"


# ---------------------------------------------------------------------------
# Trainer integration
# ---------------------------------------------------------------------------


def test_capture_and_restore_reinstates_every_trainer_owned_stream(tmp_path):
    trainer = _trainer(tmp_path, run_id="capture-run")
    trainer.execute_update(1)
    random.seed(4)
    np.random.seed(5)
    state = capture_trainer_resume_state(trainer, parent_run_id="parent-run-1")

    expected = (random.random(), float(np.random.random()), torch.rand(1).item())

    revived = _trainer(tmp_path, run_id="revived-run")
    restore_trainer_resume_state(revived, state)

    assert (revived.episode_index, revived.update_index, revived.environment_step_index) == (
        trainer.episode_index, trainer.update_index, trainer.environment_step_index,
    )
    assert torch.equal(revived._training_rng.get_state(), state.training_generator_state)
    for name in ("actor", "critic"):
        for key, tensor in state.policy_state_dict[name].items():
            assert torch.equal(revived.policy.state_dict()[name][key], tensor)
    assert (random.random(), float(np.random.random()), torch.rand(1).item()) == expected


def test_restore_refuses_a_state_from_a_different_training_seed(tmp_path):
    state = capture_trainer_resume_state(_trainer(tmp_path, seed=17, run_id="seed-a"))
    other = _trainer(tmp_path, seed=18, run_id="seed-b")

    with pytest.raises(ResearchValidationError) as error:
        restore_trainer_resume_state(other, state)
    assert error.value.code == "RESUME_COMPATIBILITY_GATE_FAILED"
    assert error.value.path == "training_seed"


def test_capture_reads_the_trainers_real_indices(tmp_path):
    trainer = _trainer(tmp_path)
    trainer.execute_update(1)
    state = capture_trainer_resume_state(trainer)

    assert state.update_index == 1
    assert state.episode_index == trainer.config.episodes_per_update + trainer.config.validation_episodes
    assert state.environment_step_index > 0
    assert state.training_seed == trainer.training_seed
    assert dict(state.python_random_state) == capture_python_random_state()
    assert dict(state.numpy_random_state) == capture_numpy_random_state()
