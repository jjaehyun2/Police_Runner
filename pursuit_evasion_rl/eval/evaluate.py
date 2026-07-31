"""모델 평가 모듈.

Evaluator 클래스를 제공한다.
학습된 모델의 성능을 평가하고, 휴리스틱 에이전트와 비교한다.
"""

from __future__ import annotations

import logging
import os
from typing import Any

import numpy as np

from pursuit_evasion_rl.env.pursuit_env import PursuitEnvironment
from pursuit_evasion_rl.eval.heuristic_agents import (
    HeuristicFugitiveAgent,
    HeuristicPoliceAgent,
)
from pursuit_evasion_rl.training.algorithms import (
    MAPPOAlgorithm,
    MADDPGAlgorithm,
    _get_obs_dim,
)
from pursuit_evasion_rl.utils.config_manager import ConfigManager

logger = logging.getLogger(__name__)


class Evaluator:
    """학습된 모델의 성능을 평가하는 클래스.

    지정된 모델을 로드하여 에피소드를 실행하고 통계를 산출한다.
    또한 휴리스틱 에이전트와의 비교 평가를 지원한다.

    Attributes:
        _ready: 모델 로드 성공 여부
    """

    def __init__(self, model_path: str, config_path: str = "configs/default.yaml") -> None:
        """모델 및 환경을 로드한다.

        Args:
            model_path: 학습된 모델 체크포인트 파일 경로
            config_path: 환경 설정 YAML 파일 경로
        """
        self._ready = False
        self._model_path = model_path
        self._config_path = config_path

        # 설정 로드
        try:
            config_manager = ConfigManager(config_path)
            self._config = config_manager.load()
        except (SystemExit, Exception) as e:
            print(f"[ERROR] 설정 파일 로드 실패: {config_path} - {e}")
            return

        # 환경 설정 딕셔너리 구성
        env_config: dict[str, Any] = {
            "network_mode": self._config.network_mode,
            "num_nodes": self._config.num_nodes,
            "density": self._config.density,
            "fixed_edges": self._config.fixed_edges,
            "num_police": self._config.num_police,
            "max_steps": self._config.max_steps,
            "terminal_reward": self._config.terminal_reward,
            "shaping_scale": self._config.shaping_scale,
            "cooperation_bonus": self._config.cooperation_bonus,
            "invalid_action_penalty": self._config.invalid_action_penalty,
        }

        # 환경 생성
        self._env = PursuitEnvironment(env_config)

        # 알고리즘 인스턴스 생성
        obs_dim = _get_obs_dim(self._config.num_police, self._env.network.max_degree)
        action_dim = self._env.network.max_degree + 1  # 인접 노드 + stay

        if self._config.algorithm == "MADDPG":
            self._algorithm = MADDPGAlgorithm(
                obs_dim=obs_dim,
                action_dim=action_dim,
                num_police=self._config.num_police,
                learning_rate=self._config.learning_rate,
                discount_factor=self._config.discount_factor,
                epsilon=0.0,  # 평가 시에는 탐험 비활성화
            )
        else:
            self._algorithm = MAPPOAlgorithm(
                obs_dim=obs_dim,
                action_dim=action_dim,
                num_police=self._config.num_police,
                learning_rate=self._config.learning_rate,
                discount_factor=self._config.discount_factor,
                epsilon=self._config.epsilon,
            )

        # 모델 로드
        if not os.path.exists(model_path):
            print(f"[ERROR] 모델 파일이 존재하지 않습니다: {model_path}")
            return

        try:
            self._algorithm.load(model_path)
        except Exception as e:
            print(f"[ERROR] 모델 로드 실패: {model_path} - {e}")
            return

        self._ready = True

    def evaluate(self, num_episodes: int = 100) -> dict:
        """학습된 모델로 지정된 에피소드 수만큼 평가를 실행한다.

        Args:
            num_episodes: 평가할 에피소드 수 (최소 1, 최대 10000, 기본 100)

        Returns:
            평가 결과 딕셔너리:
                - police_win_rate: 경찰 승리율
                - fugitive_win_rate: 도망자 승리율
                - timeout_rate: 타임아웃 비율
                - avg_reward_police: 경찰 평균 보상
                - avg_reward_fugitive: 도망자 평균 보상
                - avg_episode_length: 평균 에피소드 길이
        """
        if not self._ready:
            print("[ERROR] 모델이 준비되지 않았습니다. 평가를 수행할 수 없습니다.")
            return {
                "police_win_rate": 0.0,
                "fugitive_win_rate": 0.0,
                "timeout_rate": 0.0,
                "avg_reward_police": 0.0,
                "avg_reward_fugitive": 0.0,
                "avg_episode_length": 0.0,
            }

        # 에피소드 수 클램핑
        num_episodes = max(1, min(10000, num_episodes))

        police_wins = 0
        fugitive_wins = 0
        timeouts = 0
        total_reward_police = 0.0
        total_reward_fugitive = 0.0
        total_steps = 0

        for ep in range(num_episodes):
            obs, _ = self._env.reset(seed=ep)
            done = False
            ep_reward_police = 0.0
            ep_reward_fugitive = 0.0
            ep_steps = 0

            while not done:
                # 각 에이전트의 행동 선택
                actions: dict[str, int] = {}
                for agent_id in obs:
                    actions[agent_id] = self._algorithm.get_action(agent_id, obs[agent_id])

                obs, rewards, terminated, truncated, info = self._env.step(actions)

                # 보상 누적
                for agent_id, reward in rewards.items():
                    if agent_id.startswith("police"):
                        ep_reward_police += reward
                    else:
                        ep_reward_fugitive += reward

                ep_steps += 1

                # 종료 판정 (모든 에이전트의 terminated/truncated 동일)
                any_agent = next(iter(terminated))
                done = terminated[any_agent] or truncated[any_agent]

            # 종료 원인 파악
            any_info = info[next(iter(info))]
            reason = any_info.get("termination_reason")
            if reason == "police_win":
                police_wins += 1
            elif reason == "fugitive_win":
                fugitive_wins += 1
            else:
                timeouts += 1

            total_reward_police += ep_reward_police
            total_reward_fugitive += ep_reward_fugitive
            total_steps += ep_steps

        results = {
            "police_win_rate": police_wins / num_episodes,
            "fugitive_win_rate": fugitive_wins / num_episodes,
            "timeout_rate": timeouts / num_episodes,
            "avg_reward_police": total_reward_police / num_episodes,
            "avg_reward_fugitive": total_reward_fugitive / num_episodes,
            "avg_episode_length": total_steps / num_episodes,
        }

        self._print_results(results, num_episodes)
        return results

    def compare_with_heuristic(self, num_episodes: int = 100) -> dict:
        """학습된 RL 에이전트와 휴리스틱 에이전트의 성능을 비교한다.

        동일 조건(에피소드 수)에서 RL 모델과 휴리스틱 에이전트를 각각 실행하고
        비교 테이블을 출력한다.

        Args:
            num_episodes: 비교 평가할 에피소드 수 (최소 1, 최대 10000, 기본 100)

        Returns:
            비교 결과 딕셔너리:
                - rl: RL 모델 평가 결과
                - heuristic: 휴리스틱 에이전트 평가 결과
        """
        num_episodes = max(1, min(10000, num_episodes))

        # RL 모델 평가
        print("=" * 60)
        print("  RL 모델 평가")
        print("=" * 60)
        rl_results = self.evaluate(num_episodes)

        # 휴리스틱 에이전트 평가
        print("\n" + "=" * 60)
        print("  휴리스틱 에이전트 평가")
        print("=" * 60)
        heuristic_results = self._evaluate_heuristic(num_episodes)

        # 비교 테이블 출력
        self._print_comparison_table(rl_results, heuristic_results)

        return {"rl": rl_results, "heuristic": heuristic_results}

    def _evaluate_heuristic(self, num_episodes: int) -> dict:
        """휴리스틱 에이전트로 평가를 실행한다.

        Args:
            num_episodes: 평가할 에피소드 수

        Returns:
            평가 결과 딕셔너리 (evaluate 메서드와 동일 키 구조)
        """
        police_agent = HeuristicPoliceAgent(self._env.network)
        fugitive_agent = HeuristicFugitiveAgent(self._env.network)

        police_wins = 0
        fugitive_wins = 0
        timeouts = 0
        total_reward_police = 0.0
        total_reward_fugitive = 0.0
        total_steps = 0

        for ep in range(num_episodes):
            obs, _ = self._env.reset(seed=ep)
            done = False
            ep_reward_police = 0.0
            ep_reward_fugitive = 0.0
            ep_steps = 0

            while not done:
                actions: dict[str, int] = {}

                # 도망자 위치 파악 (휴리스틱 경찰이 필요로 함)
                fugitive_pos = self._env._positions["fugitive"]

                # 경찰 에이전트 행동
                for agent_id in self._env._police_ids:
                    current_node = self._env._positions[agent_id]
                    actions[agent_id] = police_agent.act(current_node, fugitive_pos)

                # 도망자 에이전트 행동
                fugitive_current = self._env._positions["fugitive"]
                actions["fugitive"] = fugitive_agent.act(fugitive_current)

                obs, rewards, terminated, truncated, info = self._env.step(actions)

                # 보상 누적
                for agent_id, reward in rewards.items():
                    if agent_id.startswith("police"):
                        ep_reward_police += reward
                    else:
                        ep_reward_fugitive += reward

                ep_steps += 1

                any_agent = next(iter(terminated))
                done = terminated[any_agent] or truncated[any_agent]

            # 종료 원인 파악
            any_info = info[next(iter(info))]
            reason = any_info.get("termination_reason")
            if reason == "police_win":
                police_wins += 1
            elif reason == "fugitive_win":
                fugitive_wins += 1
            else:
                timeouts += 1

            total_reward_police += ep_reward_police
            total_reward_fugitive += ep_reward_fugitive
            total_steps += ep_steps

        results = {
            "police_win_rate": police_wins / num_episodes,
            "fugitive_win_rate": fugitive_wins / num_episodes,
            "timeout_rate": timeouts / num_episodes,
            "avg_reward_police": total_reward_police / num_episodes,
            "avg_reward_fugitive": total_reward_fugitive / num_episodes,
            "avg_episode_length": total_steps / num_episodes,
        }

        self._print_results(results, num_episodes)
        return results

    def _print_results(self, results: dict, num_episodes: int) -> None:
        """평가 결과를 테이블 형태로 출력한다."""
        print(f"\n{'─' * 40}")
        print(f"  평가 결과 ({num_episodes} 에피소드)")
        print(f"{'─' * 40}")
        print(f"  경찰 승리율:       {results['police_win_rate']:.2%}")
        print(f"  도망자 승리율:     {results['fugitive_win_rate']:.2%}")
        print(f"  타임아웃 비율:     {results['timeout_rate']:.2%}")
        print(f"  경찰 평균 보상:    {results['avg_reward_police']:.4f}")
        print(f"  도망자 평균 보상:  {results['avg_reward_fugitive']:.4f}")
        print(f"  평균 에피소드 길이: {results['avg_episode_length']:.1f}")
        print(f"{'─' * 40}\n")

    def _print_comparison_table(self, rl_results: dict, heuristic_results: dict) -> None:
        """RL 모델과 휴리스틱 에이전트의 비교 테이블을 출력한다."""
        print("\n" + "=" * 60)
        print("  비교 결과: RL 모델 vs 휴리스틱 에이전트")
        print("=" * 60)
        print(f"{'지표':<20} {'RL 모델':>15} {'휴리스틱':>15}")
        print(f"{'─' * 50}")
        print(
            f"{'경찰 승리율':<20} "
            f"{rl_results['police_win_rate']:>14.2%} "
            f"{heuristic_results['police_win_rate']:>14.2%}"
        )
        print(
            f"{'도망자 승리율':<20} "
            f"{rl_results['fugitive_win_rate']:>14.2%} "
            f"{heuristic_results['fugitive_win_rate']:>14.2%}"
        )
        print(
            f"{'타임아웃 비율':<20} "
            f"{rl_results['timeout_rate']:>14.2%} "
            f"{heuristic_results['timeout_rate']:>14.2%}"
        )
        print(
            f"{'경찰 평균 보상':<20} "
            f"{rl_results['avg_reward_police']:>14.4f} "
            f"{heuristic_results['avg_reward_police']:>14.4f}"
        )
        print(
            f"{'도망자 평균 보상':<20} "
            f"{rl_results['avg_reward_fugitive']:>14.4f} "
            f"{heuristic_results['avg_reward_fugitive']:>14.4f}"
        )
        print(
            f"{'평균 에피소드 길이':<20} "
            f"{rl_results['avg_episode_length']:>14.1f} "
            f"{heuristic_results['avg_episode_length']:>14.1f}"
        )
        print("=" * 60)
