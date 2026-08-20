"""Self-Play 학습 매니저 모듈.

경찰 팀과 도망자가 서로 대전하며 학습하는 Self-Play 구조를 관리한다.
에피소드 실행, 경험 수집, 에이전트 업데이트를 담당한다.
"""

from __future__ import annotations

import logging
from collections import deque
from typing import Any

import numpy as np

from pursuit_evasion_rl.training.algorithms import (
    BaseAlgorithm,
    MADDPGAlgorithm,
    MAPPOAlgorithm,
    _flatten_observation,
)

logger = logging.getLogger(__name__)


def _stack_masks(masks: list) -> np.ndarray | None:
    """행동 마스크 리스트를 (N, action_dim) 배열로 만든다.

    마스크가 비었거나 하나라도 None이면 None을 반환해, 업데이트 쪽에서
    마스크 없이 (기존 동작대로) 처리하게 한다.

    Args:
        masks: 스텝별 행동 마스크 리스트

    Returns:
        boolean 배열 또는 None
    """
    if not masks or any(m is None for m in masks):
        return None
    return np.asarray(masks, dtype=bool)


class SelfPlayManager:
    """Self-Play 학습 매니저.

    경찰 팀과 도망자 에이전트가 동일 환경에서 대전하며 학습한다.
    에피소드를 실행하고, 경험(trajectories)을 수집하며,
    양 팀의 정책을 업데이트한다.

    Attributes:
        env: 추격-도주 환경
        police_algo: 경찰 팀 알고리즘
        fugitive_algo: 도망자 알고리즘
        config: 학습 설정 딕셔너리
    """

    def __init__(
        self,
        env: Any,
        police_algo: BaseAlgorithm,
        fugitive_algo: BaseAlgorithm,
        config: dict,
    ) -> None:
        """SelfPlayManager 초기화.

        Args:
            env: PursuitEnvironment 인스턴스
            police_algo: 경찰 팀 학습 알고리즘
            fugitive_algo: 도망자 학습 알고리즘
            config: 학습 설정. 키:
                - discount_factor: 할인 계수
                - batch_size: 배치 크기
                - win_rate_window: 승률 계산 윈도우 크기
        """
        self.env = env
        self.police_algo = police_algo
        self.fugitive_algo = fugitive_algo
        self.config = config

        # 통계 추적
        self._win_rate_window = config.get("win_rate_window", 100)
        self._episode_results: deque[str] = deque(maxlen=self._win_rate_window)
        self._episode_rewards: deque[float] = deque(maxlen=self._win_rate_window)
        self._episode_lengths: deque[int] = deque(maxlen=self._win_rate_window)
        self._total_episodes = 0

    def run_episode(self) -> dict:
        """에피소드 하나를 실행하고 경험(trajectories)을 수집한다.

        Returns:
            에피소드 결과 딕셔너리:
                - trajectories: 에이전트별 경험 데이터
                - total_reward_police: 경찰 팀 총 보상
                - total_reward_fugitive: 도망자 총 보상
                - episode_length: 에피소드 길이
                - termination_reason: 종료 원인
        """
        observations, info = self.env.reset()

        # 에이전트 ID 목록
        agent_ids = list(observations.keys())
        police_ids = [aid for aid in agent_ids if aid.startswith("police")]
        fugitive_id = "fugitive"

        # 경험 버퍼
        trajectories: dict[str, dict[str, list]] = {
            aid: {
                "obs": [],
                "actions": [],
                "rewards": [],
                "log_probs": [],
                "masks": [],
            }
            for aid in agent_ids
        }

        total_reward_police = 0.0
        total_reward_fugitive = 0.0
        step_count = 0
        done = False
        termination_reason = "timeout"

        while not done:
            # 행동 선택 (행동 마스크 적용)
            action_masks = self.env.get_action_masks()
            actions: dict[str, int] = {}
            for aid in agent_ids:
                obs = observations[aid]
                mask = action_masks.get(aid)
                if aid.startswith("police"):
                    action = self.police_algo.get_action(aid, obs, action_mask=mask)
                else:
                    action = self.fugitive_algo.get_action(aid, obs, action_mask=mask)
                actions[aid] = action

                # 경험 저장 (관측값 평탄화)
                obs_flat = _flatten_observation(obs)
                trajectories[aid]["obs"].append(obs_flat)
                trajectories[aid]["actions"].append(action)
                # 샘플링에 쓴 마스크를 저장해야 업데이트 때 같은 분포로
                # 로그 확률을 재계산할 수 있다.
                trajectories[aid]["masks"].append(mask)

                # MAPPO의 경우 로그 확률 저장 (마스크 분포 기준)
                if isinstance(self.police_algo, MAPPOAlgorithm):
                    if aid.startswith("police"):
                        log_prob = self.police_algo.get_log_prob(
                            aid, obs, action, action_mask=mask
                        )
                    else:
                        log_prob = self.fugitive_algo.get_log_prob(
                            aid, obs, action, action_mask=mask
                        )
                    trajectories[aid]["log_probs"].append(log_prob)

            # 환경 스텝 실행
            next_observations, rewards, terminated, truncated, step_info = self.env.step(actions)

            # 보상 기록
            for aid in agent_ids:
                trajectories[aid]["rewards"].append(rewards.get(aid, 0.0))

            # 총 보상 누적
            for pid in police_ids:
                total_reward_police += rewards.get(pid, 0.0)
            total_reward_fugitive += rewards.get(fugitive_id, 0.0)

            # MADDPG 리플레이 버퍼에 전환 저장
            if isinstance(self.police_algo, MADDPGAlgorithm):
                transition = {
                    "obs": {aid: _flatten_observation(observations[aid]) for aid in agent_ids},
                    "actions": actions,
                    "rewards": rewards,
                    "next_obs": {aid: _flatten_observation(next_observations[aid]) for aid in agent_ids},
                    "dones": {aid: terminated.get(aid, False) or truncated.get(aid, False) for aid in agent_ids},
                }
                self.police_algo.store_transition(transition)

            # 종료 확인
            any_terminated = any(terminated.values())
            any_truncated = any(truncated.values())
            done = any_terminated or any_truncated

            if done:
                # 종료 원인 추출
                for aid in agent_ids:
                    reason = step_info.get(aid, {}).get("termination_reason")
                    if reason is not None:
                        termination_reason = reason
                        break

            observations = next_observations
            step_count += 1

        # 경찰 팀 평균 보상
        total_reward_police /= max(len(police_ids), 1)

        # 통계 업데이트
        self._total_episodes += 1
        self._episode_results.append(termination_reason)
        self._episode_rewards.append(total_reward_police + total_reward_fugitive)
        self._episode_lengths.append(step_count)

        return {
            "trajectories": trajectories,
            "total_reward_police": total_reward_police,
            "total_reward_fugitive": total_reward_fugitive,
            "episode_length": step_count,
            "termination_reason": termination_reason,
        }

    def update_agents(self, trajectories: dict) -> dict:
        """수집된 경험으로 양 팀의 정책을 업데이트한다.

        Args:
            trajectories: run_episode()에서 반환된 trajectories 딕셔너리

        Returns:
            학습 메트릭 딕셔너리
        """
        if isinstance(self.police_algo, MAPPOAlgorithm):
            return self._update_mappo(trajectories)
        elif isinstance(self.police_algo, MADDPGAlgorithm):
            return self._update_maddpg()
        else:
            # 일반 BaseAlgorithm 인터페이스 사용
            return self.police_algo.train_step(trajectories)

    def _update_mappo(self, trajectories: dict) -> dict:
        """MAPPO 알고리즘으로 양 팀을 업데이트한다."""
        agent_ids = list(trajectories.keys())
        police_ids = [aid for aid in agent_ids if aid.startswith("police")]
        fugitive_id = "fugitive"

        # 경찰 팀 배치 구성 (모든 경찰 경험을 모음 - 파라미터 공유)
        # 경찰별 궤적을 이어붙이므로, 각 경찰 궤적의 마지막 인덱스를 done으로
        # 표시해 할인 누적이 다른 경찰의 경험으로 새지 않게 한다.
        police_obs: list[np.ndarray] = []
        police_actions: list[int] = []
        police_rewards: list[float] = []
        police_old_log_probs: list[float] = []
        police_masks: list = []
        police_dones: list[bool] = []

        for pid in police_ids:
            traj = trajectories[pid]
            n = len(traj["obs"])
            if n == 0:
                continue
            police_obs.extend(traj["obs"])
            police_actions.extend(traj["actions"])
            police_rewards.extend(traj["rewards"])
            police_old_log_probs.extend(traj["log_probs"])
            police_masks.extend(traj.get("masks", [None] * n))
            police_dones.extend([False] * (n - 1) + [True])

        # 도망자 배치 구성 (단일 궤적이므로 마지막만 done)
        fugitive_traj = trajectories.get(fugitive_id, {})
        fugitive_obs = fugitive_traj.get("obs", [])
        fugitive_actions = fugitive_traj.get("actions", [])
        fugitive_rewards = fugitive_traj.get("rewards", [])
        fugitive_old_log_probs = fugitive_traj.get("log_probs", [])
        fugitive_masks = fugitive_traj.get("masks", [])
        n_fug = len(fugitive_obs)
        fugitive_dones = [False] * max(n_fug - 1, 0) + ([True] if n_fug else [])

        batch = {
            "police_obs": police_obs,
            "police_actions": police_actions,
            "police_rewards": police_rewards,
            "police_old_log_probs": police_old_log_probs,
            "police_dones": police_dones,
            "police_masks": _stack_masks(police_masks),
            "fugitive_obs": fugitive_obs,
            "fugitive_actions": fugitive_actions,
            "fugitive_rewards": fugitive_rewards,
            "fugitive_old_log_probs": fugitive_old_log_probs,
            "fugitive_dones": fugitive_dones,
            "fugitive_masks": _stack_masks(fugitive_masks),
        }

        return self.police_algo.train_step(batch)

    def _update_maddpg(self) -> dict:
        """MADDPG 알고리즘으로 양 팀을 업데이트한다 (리플레이 버퍼 기반)."""
        return self.police_algo.train_step({})

    def get_stats(self) -> dict:
        """현재까지의 학습 통계를 반환한다.

        Returns:
            통계 딕셔너리:
                - total_episodes: 총 에피소드 수
                - police_win_rate: 경찰 승률 (최근 N 에피소드)
                - fugitive_win_rate: 도망자 승률 (최근 N 에피소드)
                - timeout_rate: 타임아웃 비율 (최근 N 에피소드)
                - avg_reward: 평균 보상 (최근 N 에피소드)
                - avg_episode_length: 평균 에피소드 길이 (최근 N 에피소드)
        """
        total = len(self._episode_results)
        if total == 0:
            return {
                "total_episodes": 0,
                "police_win_rate": 0.0,
                "fugitive_win_rate": 0.0,
                "timeout_rate": 0.0,
                "avg_reward": 0.0,
                "avg_episode_length": 0.0,
            }

        police_wins = sum(1 for r in self._episode_results if r == "police_win")
        fugitive_wins = sum(1 for r in self._episode_results if r == "fugitive_win")
        timeouts = sum(1 for r in self._episode_results if r == "timeout")

        return {
            "total_episodes": self._total_episodes,
            "police_win_rate": police_wins / total,
            "fugitive_win_rate": fugitive_wins / total,
            "timeout_rate": timeouts / total,
            "avg_reward": float(np.mean(self._episode_rewards)) if self._episode_rewards else 0.0,
            "avg_episode_length": float(np.mean(self._episode_lengths)) if self._episode_lengths else 0.0,
        }
