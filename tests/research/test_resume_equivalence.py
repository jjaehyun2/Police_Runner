"""Task 6.3 interruption-equivalence over every interruption point 0 <= K <= N."""
from __future__ import annotations

from copy import deepcopy
from dataclasses import replace

import pytest
import torch

from pursuit_evasion_rl.osm_demo.coarsening import coarsen_raw_graph, prepare_model_network
from pursuit_evasion_rl.osm_demo.fixtures import daejeon
from pursuit_evasion_rl.research.errors import ResearchValidationError
from pursuit_evasion_rl.research.training.checkpoint import restore_trainer_resume_state
from pursuit_evasion_rl.research.training.equivalence import (
    DEFAULT_GPU_ATOL,
    DEFAULT_GPU_RTOL,
    assert_equivalent,
    compare_traces,
    compare_traces_on_gpu_path,
    run_continuous,
    run_interrupted,
    trace_hash,
)
from pursuit_evasion_rl.research.training.trainer import TrainerConfig

pytestmark = pytest.mark.offline

UPDATES = 3
TRAINING_SEED = 5


def _config() -> TrainerConfig:
    return TrainerConfig(
        updates=UPDATES, episodes_per_update=1, validation_episodes=1, max_steps=6, hidden_dims=(4,),
    )


@pytest.fixture(scope="module")
def network():
    return prepare_model_network(coarsen_raw_graph(daejeon())).network


@pytest.fixture(scope="module")
def continuous(network, tmp_path_factory):
    root = tmp_path_factory.mktemp("continuous")
    return run_continuous(network, _config(), TRAINING_SEED, updates=UPDATES, output_root=root)


# ---------------------------------------------------------------------------
# Every interruption point is equivalent to the continuous run
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("interrupt_after", range(UPDATES + 1))
def test_every_interruption_point_matches_the_continuous_run_bitwise(network, continuous, tmp_path, interrupt_after):
    interrupted = run_interrupted(
        network, _config(), TRAINING_SEED,
        updates=UPDATES, interrupt_after=interrupt_after,
        checkpoint_path=tmp_path / "resume.pt", output_root=tmp_path / "runs",
    )

    report = assert_equivalent(continuous, interrupted)

    assert report.equivalent and report.mode == "cpu_bitwise"
    assert report.update_count == UPDATES
    assert report.continuous_hash == report.interrupted_hash == trace_hash(interrupted)
    assert [item.update_index for item in interrupted] == list(range(1, UPDATES + 1))


@pytest.mark.parametrize("interrupt_after", range(UPDATES + 1))
def test_gpu_tolerance_path_runs_when_a_device_exists_and_skips_cleanly_otherwise(
    network, continuous, tmp_path, interrupt_after
):
    interrupted = run_interrupted(
        network, _config(), TRAINING_SEED,
        updates=UPDATES, interrupt_after=interrupt_after,
        checkpoint_path=tmp_path / "resume.pt", output_root=tmp_path / "runs",
    )

    report = compare_traces_on_gpu_path(continuous, interrupted, rtol=DEFAULT_GPU_RTOL, atol=DEFAULT_GPU_ATOL)

    assert report.equivalent and report.mode == "gpu_tolerance"
    assert (report.rtol, report.atol) == (DEFAULT_GPU_RTOL, DEFAULT_GPU_ATOL)
    if torch.cuda.is_available():
        assert report.skipped_reason is None
    else:
        assert report.skipped_reason == "no CUDA device is available"


def test_continuous_trace_is_sensitive_enough_to_notice_a_different_seed(network, continuous, tmp_path):
    other = run_continuous(network, _config(), TRAINING_SEED + 1, updates=UPDATES, output_root=tmp_path / "other")
    assert not compare_traces(continuous, other).equivalent


# ---------------------------------------------------------------------------
# Omitting one RNG/state component must be detected, never silently accepted
# ---------------------------------------------------------------------------


def _omit_training_generator(trainer, state):
    """Restore everything except the training RNG stream, as a fresh run would have it."""
    restore_trainer_resume_state(trainer, state)
    trainer._training_rng.manual_seed(trainer.training_seed)


def _omit_optimizer_state(trainer, state):
    fresh = deepcopy(trainer.policy.optimizer.state_dict())
    restore_trainer_resume_state(trainer, state)
    trainer.policy.optimizer.load_state_dict(fresh)


def _omit_policy_parameters(trainer, state):
    fresh = deepcopy(trainer.policy.state_dict())
    restore_trainer_resume_state(trainer, state)
    trainer.policy.load_state_dict(fresh)


def _corrupt_training_generator(trainer, state):
    tampered = state.training_generator_state.clone()
    tampered[0] = (int(tampered[0].item()) + 1) % 256
    restore_trainer_resume_state(trainer, replace(state, training_generator_state=tampered))


@pytest.mark.parametrize(
    ("name", "broken_restore"),
    (
        ("training_generator_state", _omit_training_generator),
        ("optimizer_state_dict", _omit_optimizer_state),
        ("policy_state_dict", _omit_policy_parameters),
        ("corrupted_training_generator", _corrupt_training_generator),
    ),
)
def test_a_resume_that_drops_one_state_component_is_reported_as_a_divergence(
    network, continuous, tmp_path, name, broken_restore
):
    interrupted = run_interrupted(
        network, _config(), TRAINING_SEED,
        updates=UPDATES, interrupt_after=1,
        checkpoint_path=tmp_path / "resume.pt", output_root=tmp_path / "runs",
        restore=broken_restore,
    )

    report = compare_traces(continuous, interrupted)
    assert not report.equivalent, f"omitting {name} must not be silently accepted"
    assert report.divergences
    assert report.continuous_hash != report.interrupted_hash

    with pytest.raises(ResearchValidationError) as error:
        assert_equivalent(continuous, interrupted)
    assert error.value.code == "INTERRUPTION_DIVERGENCE"
    assert error.value.expected == report.continuous_hash
    assert error.value.actual == report.interrupted_hash
    assert error.value.as_dict()["details"]["divergences"]
    assert error.value.as_dict()["details"]["record_hash"]


def test_divergence_is_not_absorbed_by_the_gpu_tolerance_path(network, continuous, tmp_path):
    interrupted = run_interrupted(
        network, _config(), TRAINING_SEED,
        updates=UPDATES, interrupt_after=1,
        checkpoint_path=tmp_path / "resume.pt", output_root=tmp_path / "runs",
        restore=_omit_training_generator,
    )

    if torch.cuda.is_available():
        assert not compare_traces_on_gpu_path(continuous, interrupted).equivalent
    assert not compare_traces(continuous, interrupted, rtol=DEFAULT_GPU_RTOL, atol=DEFAULT_GPU_ATOL).equivalent


def test_failure_record_is_content_addressed_and_hashes_its_verdict(network, continuous, tmp_path):
    interrupted = run_interrupted(
        network, _config(), TRAINING_SEED,
        updates=UPDATES, interrupt_after=2,
        checkpoint_path=tmp_path / "resume.pt", output_root=tmp_path / "runs",
        restore=_omit_optimizer_state,
    )
    failed = compare_traces(continuous, interrupted)
    passed = compare_traces(continuous, continuous)

    assert failed.content_hash != passed.content_hash
    assert failed.content_hash == compare_traces(continuous, interrupted).content_hash
    assert not failed.equivalent and passed.equivalent


# ---------------------------------------------------------------------------
# Harness input validation
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("interrupt_after", (-1, UPDATES + 1))
def test_interruption_point_outside_zero_to_n_is_rejected(network, tmp_path, interrupt_after):
    with pytest.raises(ResearchValidationError) as error:
        run_interrupted(
            network, _config(), TRAINING_SEED,
            updates=UPDATES, interrupt_after=interrupt_after,
            checkpoint_path=tmp_path / "resume.pt", output_root=tmp_path / "runs",
        )
    assert error.value.code == "INVALID_EQUIVALENCE_RANGE"
