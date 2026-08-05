"""멀티 에이전트 강화학습 알고리즘 래퍼 모듈.

BaseAlgorithm 추상 클래스와 MAPPO, MADDPG 구현체를 제공한다.
PyTorch 기반 MLP 네트워크를 사용하며, 경찰 팀은 파라미터 공유,
도망자는 별도 정책을 사용한다.
"""

from __future__ import annotations

import logging
import os
from abc import ABC, abstractmethod
from collections import deque
from collections.abc import Sequence
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical

logger = logging.getLogger(__name__)


class MLPNetwork(nn.Module):
    """간단한 MLP 네트워크 (Actor 또는 Critic용)."""

    def __init__(self, input_dim: int, output_dim: int, hidden_dims: list[int] | None = None) -> None:
        super().__init__()
        if hidden_dims is None:
            hidden_dims = [64, 64]

        layers: list[nn.Module] = []
        prev_dim = input_dim
        for h_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, h_dim))
            layers.append(nn.ReLU())
            prev_dim = h_dim
        layers.append(nn.Linear(prev_dim, output_dim))

        self.network = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)


class BaseAlgorithm(ABC):
    """멀티 에이전트 RL 알고리즘 기본 클래스."""

    @abstractmethod
    def train_step(self, batch: dict) -> dict[str, float]:
        """배치 데이터로 한 스텝 학습을 수행한다.

        Args:
            batch: 학습 데이터 배치. 키:
                - observations: 에이전트별 관측값
                - actions: 에이전트별 행동
                - rewards: 에이전트별 보상
                - next_observations: 다음 관측값
                - dones: 에피소드 종료 여부

        Returns:
            학습 메트릭 딕셔너리 (loss, policy_loss, value_loss 등)
        """
        ...

    @abstractmethod
    def get_action(self, agent_id: str, observation: dict) -> int:
        """주어진 관측값에 대해 행동을 선택한다.

        Args:
            agent_id: 에이전트 식별자
            observation: 에이전트 관측값 딕셔너리

        Returns:
            선택된 행동 인덱스
        """
        ...

    @abstractmethod
    def save(self, path: str) -> None:
        """모델 파라미터를 저장한다.

        Args:
            path: 저장 경로
        """
        ...

    @abstractmethod
    def load(self, path: str) -> None:
        """모델 파라미터를 로드한다.

        Args:
            path: 로드 경로
        """
        ...


def _flatten_observation(observation: dict) -> np.ndarray:
    """관측값 딕셔너리를 1D numpy 배열로 평탄화한다."""
    parts = []
    for key in sorted(observation.keys()):
        arr = np.asarray(observation[key], dtype=np.float32).flatten()
        parts.append(arr)
    return np.concatenate(parts)


# 무효 행동 logit 값. -inf 대신 큰 음수를 쓰면 마스크가 전부 False인
# 행에서도 NaN 없이 균등 분포로 떨어진다.
_MASK_FILL_VALUE = -1e9


def _apply_action_mask(logits: torch.Tensor, mask: np.ndarray) -> torch.Tensor:
    """무효 행동의 logit을 큰 음수로 채운다.

    Args:
        logits: (batch, action_dim) 로짓 텐서
        mask: (batch, action_dim) boolean 마스크. True가 유효 행동.

    Returns:
        마스크가 적용된 로짓 텐서
    """
    mask_tensor = torch.as_tensor(np.asarray(mask, dtype=bool), device=logits.device)
    return logits.masked_fill(~mask_tensor, _MASK_FILL_VALUE)


def _get_obs_dim(num_police: int, max_degree: int) -> int:
    """관측 공간의 총 차원을 계산한다.

    관측값 구조:
        my_position: (1,)
        neighbors: (max_degree,)
        other_positions: (num_police,)  # 다른 에이전트 수 = num_police
        nearest_boundary_dist: (1,)
        current_step: (1,)
        prev_police_actions: (num_police,)
    """
    return 1 + max_degree + num_police + 1 + 1 + num_police


class MAPPOAlgorithm(BaseAlgorithm):
    """MAPPO (Multi-Agent PPO) 래퍼.

    경찰 팀은 하나의 공유 정책(파라미터 공유)을 사용하고,
    도망자는 별도 정책을 사용한다.
    중앙집중형 가치 함수(centralized value function)를 구현한다.
    """

    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        num_police: int,
        learning_rate: float = 3e-4,
        discount_factor: float = 0.99,
        epsilon: float = 0.1,
        hidden_dims: list[int] | None = None,
    ) -> None:
        """MAPPOAlgorithm 초기화.

        Args:
            obs_dim: 관측값 차원 (평탄화된 크기)
            action_dim: 행동 공간 크기
            num_police: 경찰 에이전트 수
            learning_rate: 학습률
            discount_factor: 할인 계수
            epsilon: PPO 클리핑 계수
            hidden_dims: 히든 레이어 차원 리스트
        """
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.num_police = num_police
        self.lr = learning_rate
        self.gamma = discount_factor
        self.epsilon = epsilon

        if hidden_dims is None:
            hidden_dims = [64, 64]

        # 경찰 공유 정책 (Actor + Critic)
        self.police_actor = MLPNetwork(obs_dim, action_dim, hidden_dims)
        self.police_critic = MLPNetwork(obs_dim, 1, hidden_dims)

        # 도망자 별도 정책
        self.fugitive_actor = MLPNetwork(obs_dim, action_dim, hidden_dims)
        self.fugitive_critic = MLPNetwork(obs_dim, 1, hidden_dims)

        # 옵티마이저
        self.police_optimizer = optim.Adam(
            list(self.police_actor.parameters()) + list(self.police_critic.parameters()),
            lr=self.lr,
        )
        self.fugitive_optimizer = optim.Adam(
            list(self.fugitive_actor.parameters()) + list(self.fugitive_critic.parameters()),
            lr=self.lr,
        )

    def get_action(self, agent_id: str, observation: dict, action_mask: np.ndarray | None = None) -> int:
        """관측값에 기반하여 행동을 선택한다 (확률적 샘플링 + 행동 마스크).

        Args:
            agent_id: 에이전트 식별자
            observation: 관측값 딕셔너리
            action_mask: 유효 행동 마스크 (boolean 배열). None이면 마스크 미적용.

        Returns:
            선택된 행동 인덱스
        """
        obs_flat = _flatten_observation(observation)
        obs_tensor = torch.FloatTensor(obs_flat).unsqueeze(0)

        if agent_id.startswith("police"):
            actor = self.police_actor
        else:
            actor = self.fugitive_actor

        with torch.no_grad():
            logits = actor(obs_tensor)

            # 행동 마스크 적용: 무효 행동의 logit을 큰 음수로 설정
            # (get_log_prob / _ppo_update의 재계산 경로와 동일한 방식)
            if action_mask is not None:
                logits = _apply_action_mask(logits, np.asarray(action_mask)[None, :])

            dist = Categorical(logits=logits)
            action = dist.sample()

        return action.item()

    def train_step(self, batch: dict) -> dict[str, float]:
        """PPO 알고리즘으로 한 스텝 학습을 수행한다.

        Args:
            batch: 학습 배치 데이터. 키:
                - police_obs: list[np.ndarray] 경찰 관측값
                - police_actions: list[int] 경찰 행동
                - police_rewards: list[float] 경찰 보상
                - police_old_log_probs: list[float] 이전 로그 확률
                - police_dones: list[bool] (선택) 궤적 경계. 여러 경찰/에피소드의
                  경험을 이어붙였다면 각 궤적의 마지막 인덱스를 True로 준다.
                  없으면 전체를 하나의 궤적으로 취급한다.
                - police_masks: list[np.ndarray] (선택) 샘플링에 쓴 행동 마스크
                - fugitive_obs: list[np.ndarray] 도망자 관측값
                - fugitive_actions: list[int] 도망자 행동
                - fugitive_rewards: list[float] 도망자 보상
                - fugitive_old_log_probs: list[float] 이전 로그 확률
                - fugitive_dones: list[bool] (선택) 궤적 경계
                - fugitive_masks: list[np.ndarray] (선택) 행동 마스크

        Returns:
            학습 메트릭 딕셔너리
        """
        metrics: dict[str, float] = {}

        # 경찰 팀 업데이트
        police_loss = self._ppo_update(
            actor=self.police_actor,
            critic=self.police_critic,
            optimizer=self.police_optimizer,
            obs_list=batch.get("police_obs", []),
            actions_list=batch.get("police_actions", []),
            rewards_list=batch.get("police_rewards", []),
            old_log_probs_list=batch.get("police_old_log_probs", []),
            dones_list=batch.get("police_dones"),
            masks_list=batch.get("police_masks"),
        )
        metrics["police_policy_loss"] = police_loss["policy_loss"]
        metrics["police_value_loss"] = police_loss["value_loss"]

        # 도망자 업데이트
        fugitive_loss = self._ppo_update(
            actor=self.fugitive_actor,
            critic=self.fugitive_critic,
            optimizer=self.fugitive_optimizer,
            obs_list=batch.get("fugitive_obs", []),
            actions_list=batch.get("fugitive_actions", []),
            rewards_list=batch.get("fugitive_rewards", []),
            old_log_probs_list=batch.get("fugitive_old_log_probs", []),
            dones_list=batch.get("fugitive_dones"),
            masks_list=batch.get("fugitive_masks"),
        )
        metrics["fugitive_policy_loss"] = fugitive_loss["policy_loss"]
        metrics["fugitive_value_loss"] = fugitive_loss["value_loss"]

        return metrics

    def _ppo_update(
        self,
        actor: MLPNetwork,
        critic: MLPNetwork,
        optimizer: optim.Optimizer,
        obs_list: list[np.ndarray],
        actions_list: list[int],
        rewards_list: list[float],
        old_log_probs_list: list[float],
        dones_list: Sequence[bool] | None = None,
        masks_list: Sequence[np.ndarray] | None = None,
    ) -> dict[str, float]:
        """PPO 클리핑 업데이트를 수행한다.

        Args:
            dones_list: 궤적 경계 (각 궤적 마지막 인덱스가 True). None이면
                전체를 하나의 궤적으로 본다.
            masks_list: 샘플링 시 사용한 행동 마스크. 주어지면 로그 확률을
                동일한 마스크 분포에서 재계산한다 (ratio 편향 제거).
        """
        if not obs_list:
            return {"policy_loss": 0.0, "value_loss": 0.0}

        obs_tensor = torch.FloatTensor(np.array(obs_list))
        actions_tensor = torch.LongTensor(actions_list)
        old_log_probs_tensor = torch.FloatTensor(old_log_probs_list)

        # 할인 누적 보상 (returns) 계산
        returns = self._compute_returns(rewards_list, dones_list)
        returns_tensor = torch.FloatTensor(returns)

        # 가치 추정
        values = critic(obs_tensor).squeeze(-1)
        advantages = returns_tensor - values.detach()

        # 어드밴티지 정규화
        if advantages.numel() > 1:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # 새 로그 확률 계산 (샘플링과 동일한 마스크 분포에서)
        logits = actor(obs_tensor)
        if masks_list is not None and len(masks_list) > 0:
            logits = _apply_action_mask(logits, np.asarray(masks_list))
        dist = Categorical(logits=logits)
        new_log_probs = dist.log_prob(actions_tensor)

        # PPO 클리핑
        ratio = torch.exp(new_log_probs - old_log_probs_tensor)
        clipped_ratio = torch.clamp(ratio, 1.0 - self.epsilon, 1.0 + self.epsilon)
        policy_loss = -torch.min(ratio * advantages, clipped_ratio * advantages).mean()

        # 가치 손실
        value_loss = nn.functional.mse_loss(values, returns_tensor)

        # 엔트로피 보너스
        entropy_loss = -dist.entropy().mean() * 0.01

        # 총 손실
        total_loss = policy_loss + 0.5 * value_loss + entropy_loss

        optimizer.zero_grad()
        total_loss.backward()
        nn.utils.clip_grad_norm_(
            list(actor.parameters()) + list(critic.parameters()), max_norm=0.5
        )
        optimizer.step()

        return {
            "policy_loss": policy_loss.item(),
            "value_loss": value_loss.item(),
        }

    def _compute_returns(
        self, rewards: list[float], dones: Sequence[bool] | None = None
    ) -> list[float]:
        """할인 누적 보상을 계산한다.

        Args:
            rewards: 보상 시퀀스. 여러 궤적을 이어붙인 것일 수 있다.
            dones: 각 인덱스가 궤적의 마지막 스텝인지 여부. 주어지면
                dones[i]가 True인 지점에서 누적값을 끊어, 서로 다른
                에이전트/에피소드 사이로 할인이 새지 않게 한다.
                None이면 전체를 하나의 궤적으로 취급한다 (기존 동작).

        Returns:
            인덱스별 할인 누적 보상
        """
        n = len(rewards)
        returns: list[float] = [0.0] * n
        g = 0.0
        for i in range(n - 1, -1, -1):
            # 궤적 경계: 다음(i+1) 스텝의 값을 끌고 오지 않는다.
            if dones is not None and i < len(dones) and dones[i]:
                g = 0.0
            g = rewards[i] + self.gamma * g
            returns[i] = g
        return returns

    def get_log_prob(
        self,
        agent_id: str,
        observation: dict,
        action: int,
        action_mask: np.ndarray | None = None,
    ) -> float:
        """특정 행동의 로그 확률을 반환한다.

        Args:
            agent_id: 에이전트 식별자
            observation: 관측값 딕셔너리
            action: 로그 확률을 구할 행동
            action_mask: 유효 행동 마스크. get_action()의 샘플링 분포와
                동일하게 맞추려면 샘플링에 쓴 마스크를 그대로 넘겨야 한다.
        """
        obs_flat = _flatten_observation(observation)
        obs_tensor = torch.FloatTensor(obs_flat).unsqueeze(0)

        if agent_id.startswith("police"):
            actor = self.police_actor
        else:
            actor = self.fugitive_actor

        with torch.no_grad():
            logits = actor(obs_tensor)
            if action_mask is not None:
                logits = _apply_action_mask(logits, np.asarray(action_mask)[None, :])
            dist = Categorical(logits=logits)
            log_prob = dist.log_prob(torch.tensor(action))

        return log_prob.item()

    def save(self, path: str) -> None:
        """모델 체크포인트를 저장한다."""
        os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)
        checkpoint = {
            "police_actor": self.police_actor.state_dict(),
            "police_critic": self.police_critic.state_dict(),
            "fugitive_actor": self.fugitive_actor.state_dict(),
            "fugitive_critic": self.fugitive_critic.state_dict(),
            "police_optimizer": self.police_optimizer.state_dict(),
            "fugitive_optimizer": self.fugitive_optimizer.state_dict(),
        }
        torch.save(checkpoint, path)
        logger.info("MAPPO 체크포인트 저장: %s", path)

    def load(self, path: str) -> None:
        """모델 체크포인트를 로드한다."""
        checkpoint = torch.load(path, map_location="cpu")
        self.police_actor.load_state_dict(checkpoint["police_actor"])
        self.police_critic.load_state_dict(checkpoint["police_critic"])
        self.fugitive_actor.load_state_dict(checkpoint["fugitive_actor"])
        self.fugitive_critic.load_state_dict(checkpoint["fugitive_critic"])
        self.police_optimizer.load_state_dict(checkpoint["police_optimizer"])
        self.fugitive_optimizer.load_state_dict(checkpoint["fugitive_optimizer"])
        logger.info("MAPPO 체크포인트 로드: %s", path)


class MADDPGAlgorithm(BaseAlgorithm):
    """MADDPG (Multi-Agent DDPG) 래퍼.

    각 에이전트가 독립적인 Actor를 가지고,
    중앙집중형 Critic(모든 에이전트의 관측+행동을 입력)을 사용한다.
    이산 행동 공간에서는 Gumbel-Softmax를 통해 미분 가능하게 처리한다.
    """

    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        num_police: int,
        learning_rate: float = 3e-4,
        discount_factor: float = 0.99,
        epsilon: float = 0.1,
        replay_buffer_size: int = 100000,
        batch_size: int = 64,
        hidden_dims: list[int] | None = None,
    ) -> None:
        """MADDPGAlgorithm 초기화.

        Args:
            obs_dim: 관측값 차원
            action_dim: 행동 공간 크기
            num_police: 경찰 에이전트 수
            learning_rate: 학습률
            discount_factor: 할인 계수
            epsilon: 탐험률 (epsilon-greedy)
            replay_buffer_size: 리플레이 버퍼 크기
            batch_size: 미니배치 크기
            hidden_dims: 히든 레이어 차원 리스트
        """
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.num_police = num_police
        self.lr = learning_rate
        self.gamma = discount_factor
        self.epsilon = epsilon
        self.batch_size = batch_size

        if hidden_dims is None:
            hidden_dims = [64, 64]

        num_agents = num_police + 1  # 경찰 + 도망자

        # 각 에이전트의 Actor (경찰은 공유, 도망자는 별도)
        self.police_actor = MLPNetwork(obs_dim, action_dim, hidden_dims)
        self.fugitive_actor = MLPNetwork(obs_dim, action_dim, hidden_dims)

        # 중앙집중형 Critic: 모든 에이전트의 obs + action을 입력
        critic_input_dim = num_agents * (obs_dim + action_dim)
        self.police_critic = MLPNetwork(critic_input_dim, 1, hidden_dims)
        self.fugitive_critic = MLPNetwork(critic_input_dim, 1, hidden_dims)

        # 타겟 네트워크
        self.police_actor_target = MLPNetwork(obs_dim, action_dim, hidden_dims)
        self.fugitive_actor_target = MLPNetwork(obs_dim, action_dim, hidden_dims)
        self.police_critic_target = MLPNetwork(critic_input_dim, 1, hidden_dims)
        self.fugitive_critic_target = MLPNetwork(critic_input_dim, 1, hidden_dims)

        # 타겟 네트워크 초기 동기화
        self._hard_update(self.police_actor_target, self.police_actor)
        self._hard_update(self.fugitive_actor_target, self.fugitive_actor)
        self._hard_update(self.police_critic_target, self.police_critic)
        self._hard_update(self.fugitive_critic_target, self.fugitive_critic)

        # 옵티마이저
        self.police_actor_optimizer = optim.Adam(self.police_actor.parameters(), lr=self.lr)
        self.police_critic_optimizer = optim.Adam(self.police_critic.parameters(), lr=self.lr)
        self.fugitive_actor_optimizer = optim.Adam(self.fugitive_actor.parameters(), lr=self.lr)
        self.fugitive_critic_optimizer = optim.Adam(self.fugitive_critic.parameters(), lr=self.lr)

        # 리플레이 버퍼
        self.replay_buffer: deque[dict] = deque(maxlen=replay_buffer_size)

        # 소프트 업데이트 계수
        self.tau = 0.01

    def get_action(self, agent_id: str, observation: dict, action_mask: np.ndarray | None = None) -> int:
        """epsilon-greedy 정책으로 행동을 선택한다 (행동 마스크 적용)."""
        if np.random.random() < self.epsilon:
            # 탐험: 유효 행동 중에서만 랜덤 선택
            if action_mask is not None:
                valid_actions = np.where(action_mask)[0]
                if len(valid_actions) > 0:
                    return int(np.random.choice(valid_actions))
            return np.random.randint(0, self.action_dim)

        obs_flat = _flatten_observation(observation)
        obs_tensor = torch.FloatTensor(obs_flat).unsqueeze(0)

        if agent_id.startswith("police"):
            actor = self.police_actor
        else:
            actor = self.fugitive_actor

        with torch.no_grad():
            logits = actor(obs_tensor)

            # 행동 마스크 적용
            if action_mask is not None:
                mask_tensor = torch.BoolTensor(action_mask).unsqueeze(0)
                logits = logits.masked_fill(~mask_tensor, float("-inf"))

            action = logits.argmax(dim=-1)

        return action.item()

    def store_transition(self, transition: dict) -> None:
        """경험을 리플레이 버퍼에 저장한다.

        Args:
            transition: 전환 데이터 딕셔너리. 키:
                - obs: dict[str, np.ndarray] 에이전트별 관측값 (평탄화)
                - actions: dict[str, int] 에이전트별 행동
                - rewards: dict[str, float] 에이전트별 보상
                - next_obs: dict[str, np.ndarray] 다음 관측값 (평탄화)
                - dones: dict[str, bool] 종료 여부
        """
        self.replay_buffer.append(transition)

    def train_step(self, batch: dict) -> dict[str, float]:
        """리플레이 버퍼에서 미니배치를 샘플링하여 학습한다.

        Args:
            batch: 미사용 (내부 리플레이 버퍼에서 샘플링).
                   호환성을 위해 인터페이스 유지.

        Returns:
            학습 메트릭 딕셔너리
        """
        if len(self.replay_buffer) < self.batch_size:
            return {"police_critic_loss": 0.0, "fugitive_critic_loss": 0.0}

        # 미니배치 샘플링
        indices = np.random.choice(len(self.replay_buffer), self.batch_size, replace=False)
        mini_batch = [self.replay_buffer[i] for i in indices]

        agent_ids = mini_batch[0]["obs"].keys()

        # 배치 데이터 구성
        all_obs = {aid: [] for aid in agent_ids}
        all_actions = {aid: [] for aid in agent_ids}
        all_rewards = {aid: [] for aid in agent_ids}
        all_next_obs = {aid: [] for aid in agent_ids}
        all_dones = {aid: [] for aid in agent_ids}

        for transition in mini_batch:
            for aid in agent_ids:
                all_obs[aid].append(transition["obs"][aid])
                all_actions[aid].append(transition["actions"][aid])
                all_rewards[aid].append(transition["rewards"][aid])
                all_next_obs[aid].append(transition["next_obs"][aid])
                all_dones[aid].append(float(transition["dones"][aid]))

        metrics: dict[str, float] = {}

        # 경찰 Critic 업데이트
        police_critic_loss = self._update_critic(
            critic=self.police_critic,
            critic_target=self.police_critic_target,
            optimizer=self.police_critic_optimizer,
            all_obs=all_obs,
            all_actions=all_actions,
            all_rewards=all_rewards,
            all_next_obs=all_next_obs,
            all_dones=all_dones,
            agent_ids=list(agent_ids),
            target_agent_type="police",
        )
        metrics["police_critic_loss"] = police_critic_loss

        # 도망자 Critic 업데이트
        fugitive_critic_loss = self._update_critic(
            critic=self.fugitive_critic,
            critic_target=self.fugitive_critic_target,
            optimizer=self.fugitive_critic_optimizer,
            all_obs=all_obs,
            all_actions=all_actions,
            all_rewards=all_rewards,
            all_next_obs=all_next_obs,
            all_dones=all_dones,
            agent_ids=list(agent_ids),
            target_agent_type="fugitive",
        )
        metrics["fugitive_critic_loss"] = fugitive_critic_loss

        # 타겟 네트워크 소프트 업데이트
        self._soft_update(self.police_actor_target, self.police_actor)
        self._soft_update(self.fugitive_actor_target, self.fugitive_actor)
        self._soft_update(self.police_critic_target, self.police_critic)
        self._soft_update(self.fugitive_critic_target, self.fugitive_critic)

        return metrics

    def _update_critic(
        self,
        critic: MLPNetwork,
        critic_target: MLPNetwork,
        optimizer: optim.Optimizer,
        all_obs: dict[str, list],
        all_actions: dict[str, list],
        all_rewards: dict[str, list],
        all_next_obs: dict[str, list],
        all_dones: dict[str, list],
        agent_ids: list[str],
        target_agent_type: str,
    ) -> float:
        """Critic 네트워크를 업데이트한다."""
        batch_size = len(all_obs[agent_ids[0]])

        # 현재 obs + actions 결합 (critic 입력)
        current_inputs = []
        next_inputs = []
        for aid in agent_ids:
            obs_t = torch.FloatTensor(np.array(all_obs[aid]))
            action_onehot = torch.zeros(batch_size, self.action_dim)
            for i, a in enumerate(all_actions[aid]):
                action_onehot[i, a] = 1.0
            current_inputs.append(obs_t)
            current_inputs.append(action_onehot)

            # 타겟 행동 계산
            next_obs_t = torch.FloatTensor(np.array(all_next_obs[aid]))
            if aid.startswith("police"):
                target_actor = self.police_actor_target
            else:
                target_actor = self.fugitive_actor_target
            with torch.no_grad():
                next_logits = target_actor(next_obs_t)
                next_action_onehot = torch.zeros(batch_size, self.action_dim)
                next_action_indices = next_logits.argmax(dim=-1)
                for i in range(batch_size):
                    next_action_onehot[i, next_action_indices[i]] = 1.0
            next_inputs.append(next_obs_t)
            next_inputs.append(next_action_onehot)

        current_input_tensor = torch.cat(current_inputs, dim=-1)
        next_input_tensor = torch.cat(next_inputs, dim=-1)

        # 보상 및 done 선택 (target_agent_type에 해당하는 첫 에이전트 기준)
        target_aid = None
        for aid in agent_ids:
            if aid.startswith(target_agent_type):
                target_aid = aid
                break
        if target_aid is None:
            target_aid = agent_ids[0]

        rewards_tensor = torch.FloatTensor(all_rewards[target_aid])
        dones_tensor = torch.FloatTensor(all_dones[target_aid])

        # 타겟 Q값 계산
        with torch.no_grad():
            next_q = critic_target(next_input_tensor).squeeze(-1)
            target_q = rewards_tensor + self.gamma * (1.0 - dones_tensor) * next_q

        # 현재 Q값
        current_q = critic(current_input_tensor).squeeze(-1)

        # Critic 손실
        critic_loss = nn.functional.mse_loss(current_q, target_q)

        optimizer.zero_grad()
        critic_loss.backward()
        nn.utils.clip_grad_norm_(critic.parameters(), max_norm=0.5)
        optimizer.step()

        return critic_loss.item()

    def _soft_update(self, target: nn.Module, source: nn.Module) -> None:
        """타겟 네트워크 소프트 업데이트 (Polyak averaging)."""
        for t_param, s_param in zip(target.parameters(), source.parameters()):
            t_param.data.copy_(self.tau * s_param.data + (1.0 - self.tau) * t_param.data)

    def _hard_update(self, target: nn.Module, source: nn.Module) -> None:
        """타겟 네트워크 하드 업데이트 (완전 복사)."""
        target.load_state_dict(source.state_dict())

    def save(self, path: str) -> None:
        """모델 체크포인트를 저장한다."""
        os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)
        checkpoint = {
            "police_actor": self.police_actor.state_dict(),
            "police_critic": self.police_critic.state_dict(),
            "fugitive_actor": self.fugitive_actor.state_dict(),
            "fugitive_critic": self.fugitive_critic.state_dict(),
            "police_actor_target": self.police_actor_target.state_dict(),
            "fugitive_actor_target": self.fugitive_actor_target.state_dict(),
            "police_critic_target": self.police_critic_target.state_dict(),
            "fugitive_critic_target": self.fugitive_critic_target.state_dict(),
        }
        torch.save(checkpoint, path)
        logger.info("MADDPG 체크포인트 저장: %s", path)

    def load(self, path: str) -> None:
        """모델 체크포인트를 로드한다."""
        checkpoint = torch.load(path, map_location="cpu")
        self.police_actor.load_state_dict(checkpoint["police_actor"])
        self.police_critic.load_state_dict(checkpoint["police_critic"])
        self.fugitive_actor.load_state_dict(checkpoint["fugitive_actor"])
        self.fugitive_critic.load_state_dict(checkpoint["fugitive_critic"])
        self.police_actor_target.load_state_dict(checkpoint["police_actor_target"])
        self.fugitive_actor_target.load_state_dict(checkpoint["fugitive_actor_target"])
        self.police_critic_target.load_state_dict(checkpoint["police_critic_target"])
        self.fugitive_critic_target.load_state_dict(checkpoint["fugitive_critic_target"])
        logger.info("MADDPG 체크포인트 로드: %s", path)
