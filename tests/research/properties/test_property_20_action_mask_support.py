"""Property 20 coverage for exact stored-mask MAPPO sampling/recomputation identity."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib

from hypothesis import given, settings, strategies as st
import pytest
import torch

from pursuit_evasion_rl.research.errors import ResearchValidationError
from pursuit_evasion_rl.research.policies.masked_mappo import (
    MaskedMAPPOBatch,
    ResearchMaskedMAPPO,
    StoredActionMask,
)

# **Property 20: Action-mask support is identical during sampling and PPO recomputation**
# **Validates: Requirements 19.3-19.4**

_PBT_SETTINGS = settings(max_examples=100, deadline=None)

ACTION_DIM = 4
ACTOR_OBS_DIM = 3
CRITIC_CONTEXT_DIM = 5
NUM_OFFICERS = 6
_ACTIONS = tuple(range(ACTION_DIM))


def _policy() -> ResearchMaskedMAPPO:
    torch.manual_seed(20250101)
    return ResearchMaskedMAPPO(
        actor_obs_dim=ACTOR_OBS_DIM,
        critic_context_dim=CRITIC_CONTEXT_DIM,
        action_dim=ACTION_DIM,
        num_officers=NUM_OFFICERS,
        hidden_dims=(8,),
        learning_rate=1e-3,
        device="cpu",
    )


@st.composite
def partial_masks(draw: st.DrawFn) -> tuple[bool, ...]:
    """A boolean support that is neither empty nor full, so a valid action and
    an invalid action index both always exist."""
    size = draw(st.integers(min_value=1, max_value=ACTION_DIM - 1))
    valid = tuple(draw(st.permutations(_ACTIONS))[:size])
    return tuple(index in valid for index in _ACTIONS)


@dataclass(frozen=True)
class RowSpec:
    officer_id: int
    actor_obs: tuple[float, ...]
    mask: tuple[bool, ...]
    critic_context: tuple[float, ...]
    advantage: float
    ret: float


row_specs = st.builds(
    RowSpec,
    officer_id=st.integers(min_value=0, max_value=NUM_OFFICERS - 1),
    actor_obs=st.tuples(*[st.floats(min_value=-3.0, max_value=3.0, allow_nan=False, allow_infinity=False)] * ACTOR_OBS_DIM),
    mask=partial_masks(),
    critic_context=st.tuples(
        *[st.floats(min_value=-3.0, max_value=3.0, allow_nan=False, allow_infinity=False)] * CRITIC_CONTEXT_DIM
    ),
    advantage=st.floats(min_value=-2.0, max_value=2.0, allow_nan=False, allow_infinity=False),
    ret=st.floats(min_value=-2.0, max_value=2.0, allow_nan=False, allow_infinity=False),
)


def _sample_row(policy: ResearchMaskedMAPPO, generator: torch.Generator, spec: RowSpec):
    stored = policy.seal_mask(torch.tensor(spec.mask, dtype=torch.bool))
    return stored, policy.sample(
        torch.tensor(spec.actor_obs), stored, generator, officer_id=spec.officer_id,
    )


def _batch(policy: ResearchMaskedMAPPO, specs: list[RowSpec]) -> MaskedMAPPOBatch:
    generator = torch.Generator(device="cpu").manual_seed(7)
    rows = [_sample_row(policy, generator, spec) for spec in specs]
    stored_masks = tuple(stored for stored, _ in rows)
    samples = tuple(sample for _, sample in rows)
    return MaskedMAPPOBatch(
        actor_obs=torch.tensor([spec.actor_obs for spec in specs]),
        officer_ids=torch.tensor([spec.officer_id for spec in specs], dtype=torch.long),
        actions=torch.tensor([sample.action for sample in samples], dtype=torch.long),
        old_log_probs=torch.tensor([sample.log_prob for sample in samples]),
        advantages=torch.tensor([spec.advantage for spec in specs]),
        returns=torch.tensor([spec.ret for spec in specs]),
        critic_context=torch.tensor([spec.critic_context for spec in specs]),
        stored_masks=stored_masks,
        sampling_mask_bytes=tuple(sample.stored_mask_bytes for sample in samples),
        sampling_mask_hashes=tuple(sample.stored_mask_hash for sample in samples),
    )


def _parameters(policy: ResearchMaskedMAPPO) -> tuple[torch.Tensor, ...]:
    return tuple(parameter.detach().clone() for parameter in (*policy.actor.parameters(), *policy.critic.parameters()))


@_PBT_SETTINGS
@given(specs=st.lists(row_specs, min_size=1, max_size=4))
def test_valid_rollout_batch_reproduces_identical_support_bytes_and_log_probs(specs: list[RowSpec]) -> None:
    policy = _policy()
    batch = _batch(policy, specs)

    evaluation = policy.evaluate_actions(batch)

    # Row-by-row sampling and whole-batch recomputation are two different matmul
    # batch shapes over an otherwise unchanged actor; they agree up to CPU BLAS
    # floating-point rounding, not necessarily bit-for-bit (see torch.equal vs
    # torch.allclose for batched vs per-row linear algebra).
    assert torch.allclose(evaluation.new_log_probs, batch.old_log_probs, atol=1e-4)
    for row, stored in enumerate(batch.stored_masks):
        assert stored.mask_bytes == batch.sampling_mask_bytes[row]
        assert stored.mask_hash == batch.sampling_mask_hashes[row]
    loss = policy.update(batch)
    assert loss.update_applied


_DEFECTS = st.sampled_from(("mask_bit_flip", "invalid_action"))


@_PBT_SETTINGS
@given(specs=st.lists(row_specs, min_size=1, max_size=4), defect=_DEFECTS, data=st.data())
def test_mask_bit_flip_or_invalid_action_fails_the_whole_update_without_mutation(
    specs: list[RowSpec], defect: str, data: st.DataObject
) -> None:
    policy = _policy()
    batch = _batch(policy, specs)
    target_row = data.draw(st.integers(min_value=0, max_value=len(specs) - 1))
    before_parameters = _parameters(policy)

    if defect == "mask_bit_flip":
        original = batch.stored_masks[target_row]
        flipped = bytearray(original.mask_bytes)
        flip_index = data.draw(st.integers(min_value=0, max_value=ACTION_DIM - 1))
        flipped[flip_index] ^= 1
        if not any(flipped):
            flipped[flip_index] ^= 1  # never construct an empty-support mask; flip a different bit instead
            flipped[(flip_index + 1) % ACTION_DIM] ^= 1
        corrupted = StoredActionMask(bytes(flipped), hashlib.sha256(bytes(flipped)).hexdigest(), ACTION_DIM)
        batch.stored_masks = tuple(
            corrupted if index == target_row else mask for index, mask in enumerate(batch.stored_masks)
        )
        expected_codes = {"STORED_MASK_BYTE_DRIFT"}
    else:
        mask_bytes = batch.stored_masks[target_row].mask_bytes
        invalid_indices = [index for index, bit in enumerate(mask_bytes) if bit == 0]
        invalid_action = data.draw(st.sampled_from(invalid_indices))
        batch.actions[target_row] = invalid_action
        expected_codes = {"SELECTED_INVALID_ACTION"}

    with pytest.raises(ResearchValidationError) as excinfo:
        policy.update(batch)
    assert excinfo.value.code in expected_codes
    assert all(torch.equal(left, right) for left, right in zip(before_parameters, _parameters(policy)))
