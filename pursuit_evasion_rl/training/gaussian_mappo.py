"""연속 행동 공간용 Gaussian MAPPO 알고리즘.

GaussianMLPActor와 GaussianMAPPOAlgorithm을 제공한다.
경찰 팀은 파라미터를 공유하고, 도망자는 별도 정책을 사용한다.
"""

from __future__ import annotations

import logging
import os

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Normal

from pursuit_evasion_rl.training.algorithms import BaseAlgorithm, MLPNetwork

logger = logging.getLogger(__name__)


class GaussianMLPActor(nn.Module):
    """연속 행동 공간을 위한 Gaussian 정책 네트워크.

    평균(mu)과 로그 표준편차(log_std)를 출력한다.
    action_dim=2: [speed, heading]
    """

    def __init__(
        self, obs_dim: int, action_dim: int, hidden_dims: list[int] | None = None
    ) -> None:
        super().__init__()
        if hidden_dims is None:
            hidden_dims = [128, 128]

        layers: list[nn.Module] = []
        prev_dim = obs_dim
        for h_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, h_dim))
            layers.append(nn.ReLU())
            prev_dim = h_dim

        self.shared = nn.Sequential(*layers)
        self.mean_head = nn.Linear(prev_dim, action_dim)
        self.log_std_head = nn.Linear(prev_dim, action_dim)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """순전파: 평균과 로그 표준편차를 반환한다.

        Args:
            x: 관측값 텐서, shape=(batch, obs_dim)

        Returns:
            (mean, log_std) 튜플, 각 shape=(batch, action_dim)
        """
        features = self.shared(x)
        mean = self.mean_head(features)
        log_std = self.log_std_head(features)
        # log_std 범위 제한 (안정성)
        log_std = torch.clamp(log_std, min=-5.0, max=2.0)
        return mean, log_std


class GaussianMAPPOAlgorithm(BaseAlgorithm):
    """연속 행동 공간용 MAPPO (Gaussian 정책).

    경찰 팀은 파라미터를 공유하고, 도망자는 별도 정책을 사용한다.
    """

    def __init__(
        self,
        obs_dim: int = 51,
        action_dim: int = 2,
        num_police: int = 5,
        learning_rate: float = 1e-4,
        discount_factor: float = 0.99,
        clip_epsilon: float = 0.2,
        entropy_coeff: float = 0.01,
        hidden_dims: list[int] | None = None,
        max_speed_police: float = 8.0,
        max_speed_fugitive: float = 10.0,
    ) -> None:
        """GaussianMAPPOAlgorithm 초기화.

        Args:
            obs_dim: 관측값 차원 (기본 51)
            action_dim: 행동 차원 (기본 2: speed, heading)
            num_police: 경찰 수
            learning_rate: 학습률
            discount_factor: 할인 계수
            clip_epsilon: PPO 클리핑 계수
            entropy_coeff: 엔트로피 보너스 계수
            hidden_dims: 히든 레이어 차원 (기본 [128, 128])
            max_speed_police: 경찰 최대 속도
            max_speed_fugitive: 도망자 최대 속도
        """
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.num_police = num_police
        self.lr = learning_rate
        self.gamma = discount_factor
        self.clip_epsilon = clip_epsilon
        self.entropy_coeff = entropy_coeff
        self.max_speed_police = max_speed_police
        self.max_speed_fugitive = max_speed_fugitive

        if hidden_dims is None:
            hidden_dims = [128, 128]
        self.hidden_dims = hidden_dims

        # 경찰 공유 정책 (Actor + Critic)
        self.police_actor = GaussianMLPActor(obs_dim, action_dim, hidden_dims)
        self.police_critic = MLPNetwork(obs_dim, 1, hidden_dims)

        # 도망자 별도 정책
        self.fugitive_actor = GaussianMLPActor(obs_dim, action_dim, hidden_dims)
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

    def get_action(self, agent_id: str, observation: np.ndarray) -> np.ndarray:
        """관측값에 기반하여 연속 행동을 샘플링한다.

        Gaussian 분포에서 샘플링 후 유효 범위로 클램핑한다.

        Args:
            agent_id: 에이전트 식별자
            observation: 관측 벡터 (shape=(51,) 또는 dict)

        Returns:
            행동 벡터 [speed, heading], shape=(2,)
        """
        # observation이 dict인 경우 flatten
        if isinstance(observation, dict):
            parts = []
            for key in sorted(observation.keys()):
                parts.append(np.asarray(observation[key], dtype=np.float32).flatten())
            obs_flat = np.concatenate(parts)
        else:
            obs_flat = np.asarray(observation, dtype=np.float32)

        # NaN/Inf 방지
        obs_flat = np.nan_to_num(obs_flat, nan=0.0, posinf=1.0, neginf=-1.0)

        obs_tensor = torch.FloatTensor(obs_flat).unsqueeze(0)

        if agent_id.startswith("police"):
            actor = self.police_actor
            max_speed = self.max_speed_police
        else:
            actor = self.fugitive_actor
            max_speed = self.max_speed_fugitive

        with torch.no_grad():
            mean, log_std = actor(obs_tensor)
            std = log_std.exp()

            # NaN 방지: mean 또는 std에 NaN/Inf가 있으면 기본값 사용
            if torch.isnan(mean).any() or torch.isinf(mean).any():
                mean = torch.zeros_like(mean)
            if torch.isnan(std).any() or torch.isinf(std).any():
                std = torch.ones_like(std) * 0.5

            dist = Normal(mean, std + 1e-6)
            action = dist.sample()

        action_np = action.squeeze(0).numpy()

        # 유효 범위로 클램핑: speed [0, max_speed], heading [0, 2π]
        action_np[0] = float(np.clip(action_np[0], 0.0, max_speed))
        action_np[1] = float(action_np[1] % (2.0 * np.pi))
        if action_np[1] < 0:
            action_np[1] += 2.0 * np.pi

        return action_np.astype(np.float32)

    def get_log_prob(self, agent_id: str, obs: np.ndarray, action: np.ndarray) -> float:
        """특정 행동의 로그 확률을 반환한다.

        Args:
            agent_id: 에이전트 식별자
            obs: 관측 벡터
            action: 행동 벡터 [speed, heading]

        Returns:
            로그 확률 스칼라
        """
        obs_flat = np.nan_to_num(np.asarray(obs, dtype=np.float32), nan=0.0, posinf=1.0, neginf=-1.0)
        obs_tensor = torch.FloatTensor(obs_flat).unsqueeze(0)
        action_tensor = torch.FloatTensor(np.asarray(action, dtype=np.float32)).unsqueeze(0)

        if agent_id.startswith("police"):
            actor = self.police_actor
        else:
            actor = self.fugitive_actor

        with torch.no_grad():
            mean, log_std = actor(obs_tensor)
            std = log_std.exp()

            # NaN 방지
            if torch.isnan(mean).any() or torch.isinf(mean).any():
                return 0.0
            if torch.isnan(std).any() or torch.isinf(std).any():
                std = torch.ones_like(std) * 0.5

            dist = Normal(mean, std + 1e-6)
            log_prob = dist.log_prob(action_tensor).sum(dim=-1)

        result = float(log_prob.item())
        if np.isnan(result) or np.isinf(result):
            return 0.0
        return result

    def train_step(self, batch: dict) -> dict[str, float]:
        """PPO with Gaussian log prob으로 한 스텝 학습을 수행한다.

        Args:
            batch: 학습 데이터 배치. 키:
                - police_obs: list[np.ndarray]
                - police_actions: list[np.ndarray]
                - police_rewards: list[float]
                - police_old_log_probs: list[float]
                - fugitive_obs: list[np.ndarray]
                - fugitive_actions: list[np.ndarray]
                - fugitive_rewards: list[float]
                - fugitive_old_log_probs: list[float]

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
        )
        metrics["fugitive_policy_loss"] = fugitive_loss["policy_loss"]
        metrics["fugitive_value_loss"] = fugitive_loss["value_loss"]

        return metrics

    def _ppo_update(
        self,
        actor: GaussianMLPActor,
        critic: MLPNetwork,
        optimizer: optim.Optimizer,
        obs_list: list[np.ndarray],
        actions_list: list[np.ndarray],
        rewards_list: list[float],
        old_log_probs_list: list[float],
    ) -> dict[str, float]:
        """Gaussian PPO 클리핑 업데이트를 수행한다."""
        if not obs_list:
            return {"policy_loss": 0.0, "value_loss": 0.0}

        obs_tensor = torch.FloatTensor(np.array(obs_list))
        actions_tensor = torch.FloatTensor(np.array(actions_list))
        old_log_probs_tensor = torch.FloatTensor(old_log_probs_list)

        # 할인 누적 보상 계산
        returns = self._compute_returns(rewards_list)
        returns_tensor = torch.FloatTensor(returns)

        # 가치 추정
        values = critic(obs_tensor).squeeze(-1)
        advantages = returns_tensor - values.detach()

        # 어드밴티지 정규화
        if advantages.numel() > 1:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # 새 로그 확률 계산 (Gaussian)
        mean, log_std = actor(obs_tensor)
        std = log_std.exp()
        dist = Normal(mean, std)
        new_log_probs = dist.log_prob(actions_tensor).sum(dim=-1)

        # PPO 클리핑
        ratio = torch.exp(new_log_probs - old_log_probs_tensor)
        clipped_ratio = torch.clamp(ratio, 1.0 - self.clip_epsilon, 1.0 + self.clip_epsilon)
        policy_loss = -torch.min(ratio * advantages, clipped_ratio * advantages).mean()

        # 가치 손실
        value_loss = nn.functional.mse_loss(values, returns_tensor)

        # 엔트로피 보너스
        entropy = dist.entropy().sum(dim=-1).mean()
        entropy_loss = -entropy * self.entropy_coeff

        # 총 손실
        total_loss = policy_loss + 0.5 * value_loss + entropy_loss

        # NaN 체크: 손실이 NaN이면 이 스텝 건너뛰기
        if torch.isnan(total_loss) or torch.isinf(total_loss):
            return {"policy_loss": 0.0, "value_loss": 0.0}

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

    def _compute_returns(self, rewards: list[float]) -> list[float]:
        """할인 누적 보상을 계산한다."""
        returns: list[float] = []
        g = 0.0
        for r in reversed(rewards):
            g = r + self.gamma * g
            returns.insert(0, g)
        return returns

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
            "config": {
                "obs_dim": self.obs_dim,
                "action_dim": self.action_dim,
                "num_police": self.num_police,
                "hidden_dims": self.hidden_dims,
                "clip_epsilon": self.clip_epsilon,
                "entropy_coeff": self.entropy_coeff,
            },
        }
        torch.save(checkpoint, path)
        logger.info("GaussianMAPPO 체크포인트 저장: %s", path)

    def load(self, path: str) -> None:
        """모델 체크포인트를 로드한다."""
        checkpoint = torch.load(path, map_location="cpu")
        self.police_actor.load_state_dict(checkpoint["police_actor"])
        self.police_critic.load_state_dict(checkpoint["police_critic"])
        self.fugitive_actor.load_state_dict(checkpoint["fugitive_actor"])
        self.fugitive_critic.load_state_dict(checkpoint["fugitive_critic"])
        self.police_optimizer.load_state_dict(checkpoint["police_optimizer"])
        self.fugitive_optimizer.load_state_dict(checkpoint["fugitive_optimizer"])
        logger.info("GaussianMAPPO 체크포인트 로드: %s", path)
