"""Property 27 coverage: interruption and resumption are observationally equivalent."""

from __future__ import annotations

from copy import deepcopy
from functools import lru_cache

from hypothesis import HealthCheck, assume, given, settings, strategies as st
import pytest
import torch

from pursuit_evasion_rl.osm_demo.coarsening import coarsen_raw_graph, prepare_model_network
from pursuit_evasion_rl.osm_demo.fixtures import daejeon
from pursuit_evasion_rl.research.errors import ResearchValidationError
from pursuit_evasion_rl.research.training.checkpoint import restore_trainer_resume_state
from pursuit_evasion_rl.research.training.equivalence import (
    assert_equivalent,
    compare_traces,
    run_continuous,
    run_interrupted,
)
from pursuit_evasion_rl.research.training.trainer import TrainerConfig

# **Property 27: Interruption and resumption are observationally equivalent**
# **Validates: Requirements 14.7, 19.6**

pytestmark = pytest.mark.offline

# This property spins up a real tiny CPU trainer per example, so keep the
# example count and the (N, K, seed) space small enough to stay fast while
# still exercising every interruption point at least once. The network is
# expensive to build (coarsening a real fixture graph) but is otherwise fixed
# across every example, so it is built exactly once and reused.
_PBT_SETTINGS = settings(max_examples=8, deadline=None, suppress_health_check=[HealthCheck.function_scoped_fixture])

_N = 3


@lru_cache(maxsize=1)
def _network():
    return prepare_model_network(coarsen_raw_graph(daejeon())).network


def _config() -> TrainerConfig:
    return TrainerConfig(updates=_N, episodes_per_update=1, validation_episodes=1, max_steps=6, hidden_dims=(4,))


@_PBT_SETTINGS
@given(interrupt_after=st.integers(min_value=0, max_value=_N), training_seed=st.integers(min_value=1, max_value=9999))
def test_every_interruption_point_is_equivalent_to_the_continuous_run(tmp_path_factory, interrupt_after, training_seed) -> None:
    network = _network()
    config = _config()
    continuous = run_continuous(network, config, training_seed, updates=_N, output_root=tmp_path_factory.mktemp("continuous"))
    interrupted = run_interrupted(
        network, config, training_seed, updates=_N, interrupt_after=interrupt_after,
        checkpoint_path=tmp_path_factory.mktemp("checkpoints") / "resume.pt",
        output_root=tmp_path_factory.mktemp("interrupted"),
    )
    report = assert_equivalent(continuous, interrupted)
    assert report.equivalent
    assert report.mode == "cpu_bitwise"
    assert report.update_count == _N


@_PBT_SETTINGS
@given(training_seed=st.integers(min_value=1, max_value=9999), other_seed=st.integers(min_value=1, max_value=9999))
def test_a_different_seed_is_detected_as_a_divergence_not_silently_accepted(
    tmp_path_factory, training_seed, other_seed
) -> None:
    """The comparison must actually be sensitive: two different seeds must not compare equal."""
    assume(training_seed != other_seed)
    network = _network()
    config = _config()
    first = run_continuous(network, config, training_seed, updates=_N, output_root=tmp_path_factory.mktemp("a"))
    second = run_continuous(network, config, other_seed, updates=_N, output_root=tmp_path_factory.mktemp("b"))
    report = compare_traces(first, second)
    assert not report.equivalent
    assert report.divergences
    with pytest.raises(ResearchValidationError) as excinfo:
        assert_equivalent(first, second)
    assert excinfo.value.code == "INTERRUPTION_DIVERGENCE"


# Only streams the current trainer actually consumes after construction can
# produce an observable divergence in a short trace. `training_generator_state`
# drives every action sample during rollouts (Requirement 14.5's sampler RNG),
# and `optimizer_state_dict` drives every PPO parameter update. The global
# `torch_cpu_rng_state`, by contrast, is only consumed once, at policy weight
# initialization inside the trainer constructor -- by the time a checkpoint is
# saved and restored mid-training, nothing in `execute_update` reads from it
# again, so corrupting it is schema-complete (Requirement 14.5) but not
# observable from THIS trainer's trace; it is deliberately excluded here.
_DROP_STREAMS = ("training_generator_state", "optimizer_state_dict")


def _restore_dropping(stream: str):
    def _restore(trainer, state) -> None:
        patched = deepcopy(state)
        if stream == "training_generator_state":
            object.__setattr__(patched, "training_generator_state", torch.Generator(device="cpu").manual_seed(999_999).get_state())
        elif stream == "optimizer_state_dict":
            object.__setattr__(patched, "optimizer_state_dict", {"state": {}, "param_groups": deepcopy(dict(state.optimizer_state_dict)["param_groups"])})
        restore_trainer_resume_state(trainer, patched)

    return _restore


@_PBT_SETTINGS
@given(
    interrupt_after=st.integers(min_value=1, max_value=_N - 1),
    training_seed=st.integers(min_value=1, max_value=9999),
    dropped_stream=st.sampled_from(_DROP_STREAMS),
)
def test_omitting_any_single_rng_or_state_stream_on_resume_is_detected_as_a_failure(
    tmp_path_factory, interrupt_after, training_seed, dropped_stream
) -> None:
    """A resume that silently drops one stream must never pass as equivalent."""
    network = _network()
    config = _config()
    continuous = run_continuous(network, config, training_seed, updates=_N, output_root=tmp_path_factory.mktemp("continuous"))
    interrupted = run_interrupted(
        network, config, training_seed, updates=_N, interrupt_after=interrupt_after,
        checkpoint_path=tmp_path_factory.mktemp("checkpoints") / "resume.pt",
        output_root=tmp_path_factory.mktemp("interrupted"),
        restore=_restore_dropping(dropped_stream),
    )
    report = compare_traces(continuous, interrupted)
    assert not report.equivalent
    assert report.divergences
    with pytest.raises(ResearchValidationError):
        assert_equivalent(continuous, interrupted)
