"""Leakage-safe hybrid-SMDP trainer for exact stored-mask MAPPO."""
from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import re
from typing import Sequence
from uuid import uuid4

import numpy as np
import torch

from pursuit_evasion_rl.osm_demo.environment import FUGITIVE_ID, OSMRoadPursuitEnv, STAY_ACTION
from pursuit_evasion_rl.osm_demo.metrics import placement_position
from pursuit_evasion_rl.osm_demo.models import EpisodeConfig, EpisodeOutcome, ModelNetwork, POLICE_COUNT
from pursuit_evasion_rl.research.errors import ResearchValidationError
from pursuit_evasion_rl.research.maps.splits import TuningDataView
from pursuit_evasion_rl.research.policies.baselines import GoalEvader, _Graph
from pursuit_evasion_rl.research.policies.masked_mappo import (
    MaskedMAPPOBatch, PPOLoss, ResearchMaskedMAPPO, StoredActionMask,
)
from pursuit_evasion_rl.research.smdp import AsyncDecisionTransitionBuffer, DecisionEpoch, DecisionTransition
from pursuit_evasion_rl.research.variants.observations import OBSERVATION_28D_DIM, Observation28DAdapter

TRAINER_SCHEMA_VERSION = "research-masked-mappo-trainer-v1"
DEFAULT_OUTPUT_ROOT = Path("artifacts/research/checkpoints")
_RUN_COMPONENT = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")


@dataclass(frozen=True, slots=True)
class TrainerConfig:
    updates: int
    episodes_per_update: int = 1
    validation_episodes: int = 1
    max_steps: int = 450
    police_speed_mps: float = 16.0
    fugitive_speed_mps: float = 9.0
    capture_radius_m: float = 25.0
    vision_range_m: float = 150.0
    clip_distance_m: float = 2000.0
    near_radius_m: float = 200.0
    learning_rate: float = 3e-4
    gamma: float = 0.99
    capture_bonus: float = 20.0
    time_penalty: float = 0.02
    distance_scale_m: float = 50.0
    team_coefficient: float = 1.0
    own_coefficient: float = 0.6
    regress_multiplier: float = 1.6
    entropy_coefficient: float = 0.01
    ppo_epochs: int = 1
    hidden_dims: tuple[int, ...] = (128, 128)
    max_virtual_hops: int = 64

    def __post_init__(self) -> None:
        integer_fields = (self.updates, self.episodes_per_update, self.validation_episodes, self.max_steps, self.ppo_epochs)
        if any(isinstance(value, bool) or value <= 0 for value in integer_fields):
            raise ResearchValidationError("INVALID_TRAINER_CONFIGURATION", "trainer counts must be positive integers")
        if not 0.0 <= self.gamma <= 1.0:
            raise ResearchValidationError("INVALID_TRAINER_CONFIGURATION", "gamma must be in [0, 1]")
        if self.max_virtual_hops < 0 or any(size <= 0 for size in self.hidden_dims):
            raise ResearchValidationError("INVALID_TRAINER_CONFIGURATION", "hidden dimensions and virtual-hop bound are invalid")


@dataclass(frozen=True, slots=True)
class EpisodeRollout:
    transitions: tuple[DecisionTransition, ...]
    outcome: EpisodeOutcome
    physical_steps: int
    legal_action_records: tuple[tuple[bytes, int], ...]

    @property
    def all_actions_legal(self) -> bool:
        return all(mask[action] == 1 for mask, action in self.legal_action_records)


@dataclass(frozen=True, slots=True)
class UpdateOutcome:
    """One update's result before validation-driven checkpoint selection."""

    update_index: int
    transition_count: int
    train_capture_rate: float
    validation_capture_rate: float
    losses: tuple[PPOLoss, ...]
    all_actions_legal: bool


@dataclass(frozen=True, slots=True)
class UpdateRecord:
    update_index: int
    transition_count: int
    train_capture_rate: float
    validation_capture_rate: float
    selected_checkpoint: bool
    losses: tuple[PPOLoss, ...]


@dataclass(frozen=True, slots=True)
class TrainingResult:
    run_id: str
    run_directory: Path
    checkpoint_path: Path
    manifest_path: Path
    history: tuple[UpdateRecord, ...]
    all_actions_legal: bool


def _safe_component(value: str, field: str) -> str:
    if not _RUN_COMPONENT.fullmatch(value):
        raise ResearchValidationError("INVALID_RUN_PATH_COMPONENT", f"{field} is not a safe path component", path=field, actual=value)
    return value


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json_atomic(path: Path, payload: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False), encoding="utf-8")
    temporary.replace(path)


class ResearchTrainer:
    """Train only from train capabilities and select checkpoints only on validation.

    The constructor intentionally has no test or cross-city parameter. The only
    data capability accepted is :class:`TuningDataView`, whose public surface is
    restricted to train and validation handles.
    """

    def __init__(
        self,
        *,
        train_network: ModelNetwork,
        validation_network: ModelNetwork,
        tuning_data: TuningDataView,
        config: TrainerConfig,
        condition_id: str,
        training_seed: int,
        output_root: str | Path = DEFAULT_OUTPUT_ROOT,
        run_id: str | None = None,
        policy: ResearchMaskedMAPPO | None = None,
    ) -> None:
        if not isinstance(tuning_data, TuningDataView):
            raise ResearchValidationError("INVALID_TUNING_VIEW", "trainer requires a leakage-safe TuningDataView")
        if not tuning_data.train or not tuning_data.validation:
            raise ResearchValidationError("INCOMPLETE_TUNING_VIEW", "train and validation handles are both required")
        if not isinstance(train_network, ModelNetwork) or not isinstance(validation_network, ModelNetwork):
            raise ResearchValidationError("INVALID_TRAINING_NETWORK", "train and validation inputs must be ModelNetwork values")
        self.train_network = train_network
        self.validation_network = validation_network
        self.tuning_data = tuning_data
        self.config = config
        self.condition_id = _safe_component(condition_id, "condition_id")
        self.training_seed = int(training_seed)
        generated = f"{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S')}-{uuid4().hex[:12]}"
        self.run_id = _safe_component(run_id or generated, "run_id")
        self.output_root = Path(output_root)
        self.run_directory = self.output_root / self.condition_id / str(self.training_seed) / self.run_id
        self._validate_output_path()
        self.episode_config = EpisodeConfig(
            dt_s=1.0,
            police_speed_mps=config.police_speed_mps,
            fugitive_speed_mps=config.fugitive_speed_mps,
            capture_radius_m=config.capture_radius_m,
            max_steps=config.max_steps,
        )
        critic_dim = OBSERVATION_28D_DIM * POLICE_COUNT
        torch.manual_seed(self.training_seed)
        self.policy = policy or ResearchMaskedMAPPO(
            actor_obs_dim=OBSERVATION_28D_DIM,
            critic_context_dim=critic_dim,
            action_dim=6,
            num_officers=POLICE_COUNT,
            hidden_dims=config.hidden_dims,
            learning_rate=config.learning_rate,
            entropy_coefficient=config.entropy_coefficient,
            device="cpu",
        )
        if self.policy.actor_obs_dim != OBSERVATION_28D_DIM or self.policy.critic_context_dim != critic_dim:
            raise ResearchValidationError("TRAINER_POLICY_CONTRACT_MISMATCH", "policy dimensions do not match trainer observation/context contract")
        self._training_rng = torch.Generator(device="cpu").manual_seed(self.training_seed)
        self.episode_index = 0
        self.update_index = 0
        self.environment_step_index = 0

    def set_progress(self, *, episode_index: int, update_index: int, environment_step_index: int) -> None:
        """Restore the resume indices (see ``training/checkpoint.py``)."""
        for name, value in (
            ("episode_index", episode_index),
            ("update_index", update_index),
            ("environment_step_index", environment_step_index),
        ):
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ResearchValidationError("INVALID_RESUME_INDEX", f"{name} must be a nonnegative integer", path=name, actual=value)
        self.episode_index = episode_index
        self.update_index = update_index
        self.environment_step_index = environment_step_index

    def _validate_output_path(self) -> None:
        resolved = self.run_directory.resolve()
        protected = (Path("checkpoints/osm_mappo").resolve(), Path("checkpoints/osm_mappo/best_v2.pt").resolve())
        if any(resolved == item or item in resolved.parents or resolved in item.parents for item in protected):
            raise ResearchValidationError("PROTECTED_CHECKPOINT_PATH", "research runs cannot overlap legacy checkpoint paths", actual=str(resolved))
        if self.run_directory.exists():
            raise ResearchValidationError("RUN_DIRECTORY_EXISTS", "run directory identities are immutable and cannot be reused", actual=str(self.run_directory))

    @staticmethod
    def _actor_observations(adapter: Observation28DAdapter, state, max_steps: int) -> tuple[np.ndarray, ...]:
        return tuple(
            adapter.observe(
                police_index=index,
                police=state.police,
                fugitive=state.fugitive,
                step=state.step,
                max_steps=max_steps,
                incoming_heading=state.incoming_headings.get(f"police_{index}"),
            )
            for index in range(POLICE_COUNT)
        )

    @staticmethod
    def _critic_context(observations: Sequence[np.ndarray]) -> tuple[float, ...]:
        return tuple(float(value) for observation in observations for value in observation)

    @staticmethod
    def _positions(network: ModelNetwork, state) -> tuple[tuple[float, float], ...]:
        return tuple(placement_position(network, placement) for placement in state.police)

    def _step_rewards(self, network: ModelNetwork, before, after, captured: bool) -> tuple[float, ...]:
        fugitive_before = placement_position(network, before.fugitive)
        fugitive_after = placement_position(network, after.fugitive)
        old_distances = tuple(math.dist(position, fugitive_before) for position in self._positions(network, before))
        new_distances = tuple(math.dist(position, fugitive_after) for position in self._positions(network, after))
        team_term = (min(old_distances) - min(new_distances)) / self.config.distance_scale_m
        rewards = []
        for old, new in zip(old_distances, new_distances):
            delta = (old - new) / self.config.distance_scale_m
            own = self.config.own_coefficient * (delta if delta >= 0.0 else self.config.regress_multiplier * delta)
            reward = self.config.team_coefficient * team_term + own - self.config.time_penalty
            if captured:
                reward += self.config.capture_bonus
            rewards.append(reward)
        return tuple(rewards)

    def _rollout(self, network: ModelNetwork, *, seed: int, generator: torch.Generator) -> EpisodeRollout:
        env = OSMRoadPursuitEnv(network, self.episode_config)
        env.reset(seed=seed)
        adapter = Observation28DAdapter(
            network,
            clip_distance_m=self.config.clip_distance_m,
            near_radius_m=self.config.near_radius_m,
        )
        graph = _Graph(network)
        evader = GoalEvader(
            network,
            rng=np.random.default_rng(seed + 777),
            graph=graph,
            vision_range_m=self.config.vision_range_m,
        )
        buffer = AsyncDecisionTransitionBuffer(
            gamma=self.config.gamma,
            max_virtual_hops=self.config.max_virtual_hops,
            actor_obs_dim=self.policy.actor_obs_dim,
            critic_context_dim=self.policy.critic_context_dim,
            action_count=self.policy.action_dim,
        )
        legal_records: list[tuple[bytes, int]] = []
        first_epoch = True
        while env.outcome is None:
            state = env.episode_state()
            observations = self._actor_observations(adapter, state, self.episode_config.max_steps)
            context = self._critic_context(observations)
            masks = env.action_masks()
            decision_ids = tuple(
                index for index in range(POLICE_COUNT)
                if first_epoch or bool(masks[f"police_{index}"][:-1].any())
            )
            actions = {f"police_{index}": STAY_ACTION for index in range(POLICE_COUNT)}
            epochs: dict[int, DecisionEpoch] = {}
            for officer_id in decision_ids:
                mask = self.policy.seal_mask(masks[f"police_{officer_id}"])
                sample = self.policy.sample(
                    torch.from_numpy(observations[officer_id]), mask, generator,
                    officer_id=officer_id,
                )
                epochs[officer_id] = DecisionEpoch.create(
                    actor_obs=observations[officer_id],
                    action=sample.action,
                    action_mask=sample.stored_mask_bytes,
                    old_log_prob=sample.log_prob,
                    critic_context=context,
                )
                actions[f"police_{officer_id}"] = sample.action
                legal_records.append((sample.stored_mask_bytes, sample.action))
            buffer.start_decisions(epochs)
            first_epoch = False
            actions[FUGITIVE_ID] = evader.env_provider(network, self._positions(network, state))
            _, _, _, _, info = env.step(actions)
            next_state = env.episode_state()
            rewards = self._step_rewards(network, state, next_state, env.outcome is EpisodeOutcome.CAPTURE)
            virtual_counts = [0] * POLICE_COUNT
            for event in info["events"]:
                for officer_id in range(POLICE_COUNT):
                    if event.startswith(f"police_{officer_id}:virtual_hop:"):
                        virtual_counts[officer_id] += 1
            buffer.record_physical_step(rewards, virtual_hops=virtual_counts)
        final_observations = self._actor_observations(adapter, env.episode_state(), self.episode_config.max_steps)
        final_context = self._critic_context(final_observations)
        buffer.close_terminal(tuple(final_context for _ in range(POLICE_COUNT)))
        rollout = EpisodeRollout(
            transitions=buffer.drain_completed(),
            outcome=env.outcome or EpisodeOutcome.TIMEOUT,
            physical_steps=env.episode_state().step,
            legal_action_records=tuple(legal_records),
        )
        if not rollout.all_actions_legal:
            raise ResearchValidationError("ILLEGAL_ROLLOUT_ACTION", "rollout emitted an action outside its stored support")
        self.episode_index += 1
        self.environment_step_index += rollout.physical_steps
        return rollout

    def _batch(self, transitions: Sequence[DecisionTransition]) -> MaskedMAPPOBatch:
        if not transitions:
            raise ResearchValidationError("EMPTY_TRAINING_ROLLOUT", "an optimizer update requires decision transitions")
        actor_obs = torch.tensor([item.actor_obs for item in transitions], dtype=torch.float32)
        contexts = torch.tensor([item.critic_context for item in transitions], dtype=torch.float32)
        next_contexts = torch.tensor([item.next_critic_context for item in transitions], dtype=torch.float32)
        with torch.no_grad():
            values = self.policy.critic_values(contexts)
            next_values = self.policy.critic_values(next_contexts)
        returns = torch.tensor(
            [item.bootstrap_target(float(next_values[index].item())) for index, item in enumerate(transitions)],
            dtype=torch.float32,
        )
        advantages = returns - values.detach().cpu()
        if advantages.numel() > 1:
            deviation = advantages.std(unbiased=False)
            if float(deviation.item()) > 0.0:
                advantages = (advantages - advantages.mean()) / (deviation + 1e-8)
        stored_masks = tuple(
            StoredActionMask(item.stored_mask_bytes, item.stored_mask_hash, self.policy.action_dim)
            for item in transitions
        )
        return MaskedMAPPOBatch(
            actor_obs=actor_obs,
            officer_ids=torch.tensor([item.officer_id for item in transitions], dtype=torch.long),
            actions=torch.tensor([item.action for item in transitions], dtype=torch.long),
            old_log_probs=torch.tensor([item.old_log_prob for item in transitions], dtype=torch.float32),
            advantages=advantages,
            returns=returns,
            critic_context=contexts,
            stored_masks=stored_masks,
            sampling_mask_bytes=tuple(item.stored_mask_bytes for item in transitions),
            sampling_mask_hashes=tuple(item.stored_mask_hash for item in transitions),
        )

    def _validation_score(self, update_index: int) -> float:
        captures = 0
        for episode in range(self.config.validation_episodes):
            validation_rng = torch.Generator(device="cpu").manual_seed(
                self.training_seed * 1_000_003 + update_index * 10_007 + episode
            )
            rollout = self._rollout(
                self.validation_network,
                seed=self.training_seed + 1_000_000 + episode,
                generator=validation_rng,
            )
            captures += rollout.outcome is EpisodeOutcome.CAPTURE
        return captures / self.config.validation_episodes

    def _save_checkpoint(self, path: Path, *, update_index: int, validation_score: float) -> None:
        payload = {
            "schema_version": TRAINER_SCHEMA_VERSION,
            "run_id": self.run_id,
            "condition_id": self.condition_id,
            "training_seed": self.training_seed,
            "update_index": update_index,
            "selection_source": "validation",
            "validation_score": validation_score,
            "actor_obs_dim": self.policy.actor_obs_dim,
            "critic_context_dim": self.policy.critic_context_dim,
            "action_dim": self.policy.action_dim,
            "num_officers": self.policy.num_officers,
            "hidden_dims": self.config.hidden_dims,
            "observation": {
                "profile": "osm_topology_augmented_v1",
                "clip_distance_m": self.config.clip_distance_m,
                "near_radius_m": self.config.near_radius_m,
            },
            "model": self.policy.state_dict(),
            "optimizer": self.policy.optimizer.state_dict(),
        }
        temporary = path.with_suffix(path.suffix + ".tmp")
        torch.save(payload, temporary)
        temporary.replace(path)

    def execute_update(self, update_index: int) -> UpdateOutcome:
        """Run one update's rollouts, PPO epochs and validation pass.

        This owns no run-directory or checkpoint-selection state, so a resume
        harness can drive updates one at a time without ``train``'s artifacts.
        """
        transitions: list[DecisionTransition] = []
        captures = 0
        legal = True
        for episode in range(self.config.episodes_per_update):
            rollout = self._rollout(
                self.train_network,
                seed=self.training_seed + (update_index - 1) * self.config.episodes_per_update + episode,
                generator=self._training_rng,
            )
            transitions.extend(rollout.transitions)
            captures += rollout.outcome is EpisodeOutcome.CAPTURE
            legal = legal and rollout.all_actions_legal
        batch = self._batch(transitions)
        losses = tuple(self.policy.update(batch) for _ in range(self.config.ppo_epochs))
        validation_score = self._validation_score(update_index)
        self.update_index = update_index
        return UpdateOutcome(
            update_index=update_index,
            transition_count=len(transitions),
            train_capture_rate=captures / self.config.episodes_per_update,
            validation_capture_rate=validation_score,
            losses=losses,
            all_actions_legal=legal,
        )

    def train(self) -> TrainingResult:
        self.run_directory.mkdir(parents=True, exist_ok=False)
        manifest_path = self.run_directory / "manifest.json"
        checkpoint_path = self.run_directory / "best_validation.pt"
        base_manifest = {
            "schema_version": TRAINER_SCHEMA_VERSION,
            "run_id": self.run_id,
            "condition_id": self.condition_id,
            "training_seed": self.training_seed,
            "state": "draft",
            "created_at_utc": _utc_now(),
            "data_capabilities": {
                "train": [handle.identifier for handle in self.tuning_data.train],
                "validation": [handle.identifier for handle in self.tuning_data.validation],
            },
            "selection_source": "validation",
            "output_root": str(self.output_root),
            "config": asdict(self.config),
        }
        _write_json_atomic(manifest_path, base_manifest)
        history: list[UpdateRecord] = []
        legal = True
        best_validation = -math.inf
        for update_index in range(1, self.config.updates + 1):
            outcome = self.execute_update(update_index)
            legal = legal and outcome.all_actions_legal
            selected = outcome.validation_capture_rate > best_validation
            if selected:
                best_validation = outcome.validation_capture_rate
                self._save_checkpoint(
                    checkpoint_path, update_index=update_index, validation_score=outcome.validation_capture_rate,
                )
            history.append(UpdateRecord(
                update_index=update_index,
                transition_count=outcome.transition_count,
                train_capture_rate=outcome.train_capture_rate,
                validation_capture_rate=outcome.validation_capture_rate,
                selected_checkpoint=selected,
                losses=outcome.losses,
            ))
        if not checkpoint_path.is_file():
            raise ResearchValidationError("MISSING_VALIDATION_CHECKPOINT", "validation selection produced no checkpoint")
        sealed_manifest = dict(base_manifest)
        sealed_manifest.update({
            "state": "sealed",
            "sealed_at_utc": _utc_now(),
            "execution_status": "completed",
            "completed_updates": len(history),
            "checkpoint": {
                "relative_path": checkpoint_path.name,
                "sha256": _sha256_file(checkpoint_path),
                "selection_source": "validation",
                "best_validation_score": best_validation,
            },
            "all_actions_legal": legal,
        })
        _write_json_atomic(manifest_path, sealed_manifest)
        return TrainingResult(
            run_id=self.run_id,
            run_directory=self.run_directory,
            checkpoint_path=checkpoint_path,
            manifest_path=manifest_path,
            history=tuple(history),
            all_actions_legal=legal,
        )


PaperTrainer = ResearchTrainer

__all__ = (
    "DEFAULT_OUTPUT_ROOT", "EpisodeRollout", "PaperTrainer", "ResearchTrainer",
    "TRAINER_SCHEMA_VERSION", "TrainerConfig", "TrainingResult", "UpdateOutcome", "UpdateRecord",
)
