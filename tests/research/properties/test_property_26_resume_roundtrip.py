"""Property 26 coverage: Resume_State serialization is a complete round trip."""

from __future__ import annotations

from string import hexdigits

from hypothesis import given, settings, strategies as st
import pytest
import torch

from pursuit_evasion_rl.research.errors import ResearchValidationError
from pursuit_evasion_rl.research.smdp import DecisionEpoch, PendingDecision
from pursuit_evasion_rl.research.training.checkpoint import (
    REQUIRED_RESUME_FIELDS,
    CompatibilityContract,
    PendingTransitionSnapshot,
    ResumeState,
    capture_numpy_random_state,
    capture_python_random_state,
    load_resume_state,
    resume_state_content_hash,
    save_resume_state,
)

# **Property 26: Resume state serialization is a complete round trip**
# **Validates: Requirements 14.5-14.6, 14.8, 19.6**

pytestmark = pytest.mark.offline

_PBT_SETTINGS = settings(max_examples=100, deadline=None)

_hex64 = st.text(hexdigits.lower(), min_size=64, max_size=64)
_dim = st.integers(min_value=1, max_value=8)
_index = st.integers(min_value=0, max_value=10_000)


def _tiny_tensor_dict(width: int) -> dict[str, torch.Tensor]:
    return {"weight": torch.randn(width, width), "bias": torch.randn(width)}


@st.composite
def resume_states(draw: st.DrawFn) -> ResumeState:
    width = draw(_dim)
    include_pending = draw(st.booleans())
    pending = None
    if include_pending:
        officer_count = draw(st.integers(min_value=1, max_value=3))
        pending = tuple(
            PendingTransitionSnapshot(
                officer_id=officer,
                actor_obs=tuple(draw(st.floats(min_value=-5.0, max_value=5.0, allow_nan=False, allow_infinity=False)) for _ in range(2)),
                action=draw(st.integers(min_value=0, max_value=1)),
                stored_mask_bytes=bytes([1, 1]),
                old_log_prob=draw(st.floats(min_value=-5.0, max_value=0.0, allow_nan=False, allow_infinity=False)),
                critic_context=tuple(draw(st.floats(min_value=-5.0, max_value=5.0, allow_nan=False, allow_infinity=False)) for _ in range(2)),
                discounted_reward=draw(st.floats(min_value=-10.0, max_value=10.0, allow_nan=False, allow_infinity=False)),
                duration_steps=draw(st.integers(min_value=0, max_value=20)),
            )
            for officer in range(officer_count)
        )
    return ResumeState(
        policy_state_dict={"actor": _tiny_tensor_dict(width), "critic": _tiny_tensor_dict(width)},
        optimizer_state_dict={"state": {}, "param_groups": [{"lr": draw(st.floats(min_value=1e-5, max_value=1e-2))}]},
        episode_index=draw(_index),
        update_index=draw(_index),
        environment_step_index=draw(_index),
        training_seed=draw(st.integers(min_value=0, max_value=2**31 - 1)),
        python_random_state=capture_python_random_state(),
        numpy_random_state=capture_numpy_random_state(),
        torch_cpu_rng_state=torch.get_rng_state(),
        training_generator_state=torch.Generator(device="cpu").manual_seed(draw(st.integers(min_value=0, max_value=1000))).get_state(),
        pending_transitions=pending,
        parent_run_id=draw(st.one_of(st.none(), st.text(hexdigits.lower(), min_size=8, max_size=20))),
    )


def _contract(*, resume_state_hash: str | None = None) -> CompatibilityContract:
    return CompatibilityContract(
        code_hash="a" * 64, dependency_hash="b" * 64, protocol_hash="c" * 64, condition_hash="d" * 64,
        map_hash="e" * 64, split="train", actor_obs_dim=4, critic_context_dim=8, action_dim=6, num_officers=6,
        hidden_dims=(4, 4), resume_state_hash=resume_state_hash,
    )


@_PBT_SETTINGS
@given(state=resume_states())
def test_saving_and_loading_reproduces_every_field_and_the_same_content_hash(tmp_path_factory, state) -> None:
    path = tmp_path_factory.mktemp("resume") / "checkpoint.pt"
    digest = save_resume_state(path, state, _contract())
    assert digest == resume_state_content_hash(state)

    reloaded = load_resume_state(path, expected_contract=_contract())
    assert reloaded.content_hash == digest
    assert reloaded.episode_index == state.episode_index
    assert reloaded.update_index == state.update_index
    assert reloaded.environment_step_index == state.environment_step_index
    assert reloaded.training_seed == state.training_seed
    assert torch.equal(reloaded.torch_cpu_rng_state, state.torch_cpu_rng_state)
    assert torch.equal(reloaded.training_generator_state, state.training_generator_state)
    for key in ("actor", "critic"):
        for tensor_name in state.policy_state_dict[key]:
            assert torch.equal(reloaded.policy_state_dict[key][tensor_name], state.policy_state_dict[key][tensor_name])
    if state.pending_transitions is None:
        assert reloaded.pending_transitions is None
    else:
        assert len(reloaded.pending_transitions) == len(state.pending_transitions)
        for left, right in zip(reloaded.pending_transitions, state.pending_transitions):
            assert left == right
            # Round-trips back into a usable PendingDecision, not just data.
            assert isinstance(left.to_pending(), PendingDecision)


@_PBT_SETTINGS
@given(state=resume_states(), missing_field=st.sampled_from(REQUIRED_RESUME_FIELDS))
def test_a_checkpoint_missing_any_required_field_refuses_to_resume(tmp_path_factory, state, missing_field) -> None:
    from pursuit_evasion_rl.research.training.checkpoint import _state_payload, _state_from_payload

    payload = _state_payload(state)
    payload[missing_field] = None
    with pytest.raises(ResearchValidationError) as excinfo:
        _state_from_payload(payload)
    assert excinfo.value.code == "INCOMPLETE_RESUME_STATE"


@_PBT_SETTINGS
@given(state=resume_states())
def test_a_corrupted_or_incomplete_checkpoint_file_fails_loudly_not_partially(tmp_path_factory, state) -> None:
    path = tmp_path_factory.mktemp("resume") / "checkpoint.pt"
    save_resume_state(path, state, _contract())

    corrupted = path.with_name("corrupted.pt")
    corrupted.write_bytes(b"not a valid torch checkpoint")
    with pytest.raises(ResearchValidationError) as excinfo:
        load_resume_state(corrupted, expected_contract=_contract())
    assert excinfo.value.code == "CORRUPT_RESUME_STATE"

    missing = path.with_name("does-not-exist.pt")
    with pytest.raises(ResearchValidationError) as missing_excinfo:
        load_resume_state(missing, expected_contract=_contract())
    assert missing_excinfo.value.code == "RESUME_STATE_NOT_FOUND"


@_PBT_SETTINGS
@given(state=resume_states())
def test_repeated_saves_of_the_same_in_memory_state_hash_identically(tmp_path_factory, state) -> None:
    first = resume_state_content_hash(state)
    second = resume_state_content_hash(state)
    assert first == second
    path = tmp_path_factory.mktemp("resume") / "a.pt"
    other_path = tmp_path_factory.mktemp("resume2") / "b.pt"
    digest_a = save_resume_state(path, state, _contract())
    digest_b = save_resume_state(other_path, state, _contract())
    assert digest_a == digest_b == first
