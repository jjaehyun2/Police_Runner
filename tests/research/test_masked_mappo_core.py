"""Deterministic CPU checks for task 4.2 exact stored-mask MAPPO."""

from __future__ import annotations

import math

import pytest
import torch

from pursuit_evasion_rl.research.errors import ResearchValidationError
from pursuit_evasion_rl.research.policies.masked_mappo import (
    MaskedMAPPOBatch,
    ResearchMaskedMAPPO,
    StoredActionMask,
)

pytestmark = pytest.mark.offline


def _policy() -> ResearchMaskedMAPPO:
    torch.manual_seed(1402)
    return ResearchMaskedMAPPO(
        actor_obs_dim=3,
        critic_context_dim=7,
        action_dim=4,
        num_officers=6,
        hidden_dims=(8,),
        learning_rate=1e-3,
        device="cpu",
    )


def _rollout(policy: ResearchMaskedMAPPO) -> tuple[MaskedMAPPOBatch, tuple]:
    observations = torch.tensor([[0.1, 0.2, 0.3], [0.5, -0.2, 0.7]])
    masks = (
        policy.seal_mask(torch.tensor([True, False, True, False])),
        policy.seal_mask(torch.tensor([False, True, True, False])),
    )
    generator = torch.Generator(device="cpu").manual_seed(91)
    samples = tuple(
        policy.sample(observations[index], masks[index], generator, officer_id=index + 1)
        for index in range(2)
    )
    batch = MaskedMAPPOBatch(
        actor_obs=observations,
        officer_ids=torch.tensor([1, 2], dtype=torch.long),
        actions=torch.tensor([sample.action for sample in samples], dtype=torch.long),
        old_log_probs=torch.tensor([sample.log_prob for sample in samples]),
        advantages=torch.tensor([0.4, -0.3]),
        returns=torch.tensor([0.25, -0.1]),
        critic_context=torch.arange(14, dtype=torch.float32).reshape(2, 7) / 14.0,
        stored_masks=masks,
        sampling_mask_bytes=tuple(sample.stored_mask_bytes for sample in samples),
        sampling_mask_hashes=tuple(sample.stored_mask_hash for sample in samples),
    )
    return batch, samples


def _parameters(policy: ResearchMaskedMAPPO) -> tuple[torch.Tensor, ...]:
    return tuple(parameter.detach().clone() for parameter in (*policy.actor.parameters(), *policy.critic.parameters()))


def test_unchanged_parameters_reproduce_log_probs_and_invalid_probability_is_exact_zero():
    policy = _policy()
    batch, samples = _rollout(policy)

    evaluation = policy.evaluate_actions(batch)

    assert torch.equal(evaluation.new_log_probs, batch.old_log_probs)
    for row, sample in enumerate(samples):
        stored = batch.stored_masks[row]
        assert sample.support_bytes == sample.stored_mask_bytes == stored.mask_bytes
        assert sample.support_hash == sample.stored_mask_hash == stored.mask_hash
        invalid = ~stored.as_tensor()
        assert torch.equal(sample.probabilities[invalid], torch.zeros(invalid.sum()))
        assert torch.equal(evaluation.probabilities[row, invalid], torch.zeros(invalid.sum()))
    loss = policy.update(batch)
    assert loss.update_applied
    assert all(math.isfinite(value) for value in (loss.policy_loss, loss.value_loss, loss.entropy, loss.total_loss))


def test_actor_and_centralized_critic_have_explicit_privilege_separation():
    policy = _policy()
    batch, _ = _rollout(policy)
    captured: dict[str, torch.Tensor] = {}

    def capture_actor(_module, args):
        captured["actor"] = args[0].detach().clone()

    def capture_critic(_module, args):
        captured["critic"] = args[0].detach().clone()

    actor_hook = policy.actor.network[0].register_forward_pre_hook(capture_actor)
    critic_hook = policy.critic.network[0].register_forward_pre_hook(capture_critic)
    try:
        policy.evaluate_actions(batch)
    finally:
        actor_hook.remove()
        critic_hook.remove()

    assert captured["actor"].shape == (2, policy.actor_obs_dim + policy.num_officers)
    assert torch.equal(captured["actor"][:, : policy.actor_obs_dim], batch.actor_obs)
    expected_one_hot = torch.nn.functional.one_hot(batch.officer_ids, 6).to(torch.float32)
    assert torch.equal(captured["actor"][:, policy.actor_obs_dim :], expected_one_hot)
    assert captured["critic"].shape == (2, policy.critic_context_dim)
    assert torch.equal(captured["critic"], batch.critic_context)
    assert captured["actor"].shape[1] != policy.critic_context_dim


def test_empty_support_and_selected_invalid_action_fail_before_mutation():
    policy = _policy()
    with pytest.raises(ResearchValidationError) as empty:
        policy.seal_mask(torch.zeros(4, dtype=torch.bool))
    assert empty.value.code == "EMPTY_ACTION_SUPPORT"

    batch, _ = _rollout(policy)
    before = _parameters(policy)
    batch.actions[0] = 1
    with pytest.raises(ResearchValidationError) as invalid:
        policy.update(batch)
    assert invalid.value.code == "SELECTED_INVALID_ACTION"
    assert all(torch.equal(left, right) for left, right in zip(before, _parameters(policy)))


def test_mask_byte_drift_fails_the_whole_update_without_optimizer_mutation():
    policy = _policy()
    batch, _ = _rollout(policy)
    policy.update(batch)  # Populate Adam state so rollback/non-mutation covers it too.
    before_parameters = _parameters(policy)
    before_optimizer = {
        key: value.detach().clone()
        for state in policy.optimizer.state.values()
        for key, value in state.items()
        if isinstance(value, torch.Tensor)
    }
    changed = policy.seal_mask(torch.tensor([True, True, True, False]))
    batch.stored_masks = (changed, batch.stored_masks[1])

    with pytest.raises(ResearchValidationError) as drift:
        policy.update(batch)

    assert drift.value.code == "STORED_MASK_BYTE_DRIFT"
    assert all(torch.equal(left, right) for left, right in zip(before_parameters, _parameters(policy)))
    after_optimizer = {
        key: value
        for state in policy.optimizer.state.values()
        for key, value in state.items()
        if isinstance(value, torch.Tensor)
    }
    assert before_optimizer.keys() == after_optimizer.keys()
    assert all(torch.equal(before_optimizer[key], after_optimizer[key]) for key in before_optimizer)


def test_hash_drift_malformed_context_and_nonfinite_ratio_are_fail_closed():
    with pytest.raises(ResearchValidationError) as hash_drift:
        StoredActionMask(b"\x01\x00", "0" * 64, 2)
    assert hash_drift.value.code == "STORED_MASK_HASH_DRIFT"

    policy = _policy()
    batch, _ = _rollout(policy)
    before = _parameters(policy)
    batch.critic_context = torch.zeros((2, 6))
    with pytest.raises(ResearchValidationError) as context:
        policy.update(batch)
    assert context.value.code == "MALFORMED_CENTRALIZED_CONTEXT"
    assert all(torch.equal(left, right) for left, right in zip(before, _parameters(policy)))

    batch, _ = _rollout(policy)
    batch.old_log_probs[0] = -3.0e38
    with pytest.raises(ResearchValidationError) as nonfinite:
        policy.update(batch)
    assert nonfinite.value.code == "NONFINITE_VALUE"
    assert all(torch.equal(left, right) for left, right in zip(before, _parameters(policy)))
