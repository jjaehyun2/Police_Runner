"""학습 파이프라인 모듈.

TrainingPipeline 클래스를 제공한다.
YAML 설정 파일을 로드하고, 환경과 알고리즘을 생성하여
Self-Play 학습 루프를 실행한다.
"""

from __future__ import annotations

import logging
import os
import time
from typing import Any

logger = logging.getLogger(__name__)


class TrainingPipeline:
    """Self-Play 강화학습 파이프라인.

    설정 파일을 기반으로 환경과 알고리즘을 초기화하고,
    Self-Play 학습 루프를 실행한다. 에피소드별 보상, 승률,
    에피소드 길이를 로깅하며, 지정 간격으로 체크포인트를 저장한다.

    Attributes:
        config: SimulationConfig 설정 객체
        env: PursuitEnvironment 인스턴스
        police_algo: 경찰 팀 알고리즘
        fugitive_algo: 도망자 알고리즘
        self_play_manager: SelfPlayManager 인스턴스
    """

    def __init__(self, config_path: str = "configs/default.yaml") -> None:
        """TrainingPipeline 초기화.

        설정 파일을 로드하고 환경, 알고리즘, Self-Play 매니저를 생성한다.
        설정 로드 실패 시 에러 메시지를 출력하고 학습을 시작하지 않는다.

        Args:
            config_path: YAML 설정 파일 경로
        """
        self._initialized = False

        try:
            self._setup(config_path)
            self._initialized = True
        except Exception as e:
            print(f"[ERROR] 학습 파이프라인 초기화 실패: {e}")
            logger.error("학습 파이프라인 초기화 실패: %s", e)

    def _setup(self, config_path: str) -> None:
        """설정 로드 및 컴포넌트 초기화."""
        from pursuit_evasion_rl.env.pursuit_env import PursuitEnvironment
        from pursuit_evasion_rl.training.algorithms import (
            MADDPGAlgorithm,
            MAPPOAlgorithm,
            _get_obs_dim,
        )
        from pursuit_evasion_rl.training.self_play import SelfPlayManager
        from pursuit_evasion_rl.utils.config_manager import ConfigManager

        # 설정 로드
        config_manager = ConfigManager(config_path)
        self.config = config_manager.load()

        # 환경 생성
        env_config: dict[str, Any] = {
            "network_mode": self.config.network_mode,
            "num_nodes": self.config.num_nodes,
            "density": self.config.density,
            "fixed_edges": self.config.fixed_edges,
            "num_police": self.config.num_police,
            "max_steps": self.config.max_steps,
            "fixed_max_degree": 10,  # 고정 행동 공간 크기
            "terminal_reward": self.config.terminal_reward,
            "shaping_scale": self.config.shaping_scale,
            "cooperation_bonus": self.config.cooperation_bonus,
            "invalid_action_penalty": self.config.invalid_action_penalty,
        }
        self.env = PursuitEnvironment(env_config)

        # 관측/행동 차원 계산 (fixed_max_degree 기반)
        fixed_max_degree = env_config.get("fixed_max_degree", 10)
        obs_dim = _get_obs_dim(self.config.num_police, fixed_max_degree)
        action_dim = fixed_max_degree + 1  # 인접 노드 + stay

        # 알고리즘 생성
        self.police_algo, self.fugitive_algo = self._create_algorithms(
            obs_dim, action_dim
        )

        # Self-Play 매니저 생성
        sp_config = {
            "discount_factor": self.config.discount_factor,
            "batch_size": self.config.batch_size,
            "win_rate_window": 100,
        }
        self.self_play_manager = SelfPlayManager(
            env=self.env,
            police_algo=self.police_algo,
            fugitive_algo=self.fugitive_algo,
            config=sp_config,
        )

    def _create_algorithms(
        self, obs_dim: int, action_dim: int
    ) -> tuple[Any, Any]:
        """설정에 따라 알고리즘을 생성한다.

        Returns:
            (police_algo, fugitive_algo) 튜플.
            MAPPO의 경우 같은 인스턴스를 공유 (파라미터 공유).
            MADDPG의 경우 같은 인스턴스를 공유 (내부에서 경찰/도망자 분리).
        """
        from pursuit_evasion_rl.training.algorithms import (
            MADDPGAlgorithm,
            MAPPOAlgorithm,
        )

        if self.config.algorithm == "MAPPO":
            algo = MAPPOAlgorithm(
                obs_dim=obs_dim,
                action_dim=action_dim,
                num_police=self.config.num_police,
                learning_rate=self.config.learning_rate,
                discount_factor=self.config.discount_factor,
                epsilon=self.config.epsilon,
            )
            # MAPPO는 하나의 인스턴스에서 경찰/도망자 모두 처리
            return algo, algo

        elif self.config.algorithm == "MADDPG":
            algo = MADDPGAlgorithm(
                obs_dim=obs_dim,
                action_dim=action_dim,
                num_police=self.config.num_police,
                learning_rate=self.config.learning_rate,
                discount_factor=self.config.discount_factor,
                epsilon=self.config.epsilon,
                replay_buffer_size=self.config.replay_buffer_size,
                batch_size=self.config.batch_size,
            )
            # MADDPG도 하나의 인스턴스에서 경찰/도망자 모두 처리
            return algo, algo

        else:
            raise ValueError(
                f"지원하지 않는 알고리즘: {self.config.algorithm}. "
                f"'MAPPO' 또는 'MADDPG'를 사용하세요."
            )

    def train(self) -> None:
        """Self-Play 학습을 실행한다.

        에피소드별 보상, 승률, 에피소드 길이를 로깅하며,
        checkpoint_interval마다 모델을 저장한다.
        max_episodes에 도달하면 학습을 종료한다.

        초기화에 실패한 경우 학습을 시작하지 않는다.
        """
        if not self._initialized:
            print("[ERROR] 초기화에 실패하여 학습을 시작할 수 없습니다.")
            return

        max_episodes = self.config.max_episodes
        checkpoint_interval = self.config.checkpoint_interval

        logger.info(
            "학습 시작: algorithm=%s, max_episodes=%d, checkpoint_interval=%d",
            self.config.algorithm,
            max_episodes,
            checkpoint_interval,
        )
        print(f"[INFO] 학습 시작: {self.config.algorithm}, "
              f"max_episodes={max_episodes}")

        start_time = time.time()

        for episode in range(1, max_episodes + 1):
            # 에피소드 실행
            result = self.self_play_manager.run_episode()

            # 에이전트 업데이트
            train_metrics = self.self_play_manager.update_agents(
                result["trajectories"]
            )

            # 통계 조회
            stats = self.self_play_manager.get_stats()

            # 에피소드 로깅
            logger.info(
                "Episode %d/%d | "
                "Reward(P): %.3f | Reward(F): %.3f | "
                "Length: %d | Result: %s | "
                "WinRate(P): %.1f%% | WinRate(F): %.1f%%",
                episode,
                max_episodes,
                result["total_reward_police"],
                result["total_reward_fugitive"],
                result["episode_length"],
                result["termination_reason"],
                stats["police_win_rate"] * 100,
                stats["fugitive_win_rate"] * 100,
            )

            # 주기적 콘솔 출력 (100 에피소드마다)
            if episode % 100 == 0:
                elapsed = time.time() - start_time
                print(
                    f"[Episode {episode}/{max_episodes}] "
                    f"Police WR: {stats['police_win_rate']*100:.1f}% | "
                    f"Fugitive WR: {stats['fugitive_win_rate']*100:.1f}% | "
                    f"Avg Length: {stats['avg_episode_length']:.1f} | "
                    f"Elapsed: {elapsed:.1f}s"
                )

            # 체크포인트 저장
            if episode % checkpoint_interval == 0:
                self.save_checkpoint(episode)

        # 최종 체크포인트 저장
        self.save_checkpoint(max_episodes)

        total_time = time.time() - start_time
        print(f"[INFO] 학습 완료: {max_episodes} episodes, {total_time:.1f}s")
        logger.info("학습 완료: %d episodes, %.1f초", max_episodes, total_time)

    def save_checkpoint(self, episode: int) -> None:
        """모델 체크포인트를 저장한다.

        Args:
            episode: 현재 에피소드 번호
        """
        checkpoint_dir = "checkpoints"
        os.makedirs(checkpoint_dir, exist_ok=True)

        # 에피소드별 체크포인트
        path = os.path.join(checkpoint_dir, f"checkpoint_ep{episode}.pt")
        try:
            self.police_algo.save(path)
            logger.info("체크포인트 저장 완료: %s", path)
        except OSError as e:
            logger.warning("체크포인트 저장 실패 (%s): %s", path, e)
            # 대체 경로 시도
            alt_path = os.path.join(".", f"checkpoint_ep{episode}.pt")
            try:
                self.police_algo.save(alt_path)
                logger.info("대체 경로에 체크포인트 저장: %s", alt_path)
            except OSError as e2:
                logger.error("체크포인트 저장 완전 실패: %s", e2)

        # latest 심볼릭 저장
        latest_path = os.path.join(checkpoint_dir, "latest.pt")
        try:
            self.police_algo.save(latest_path)
        except OSError:
            pass


if __name__ == "__main__":
    import sys

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    config_path = sys.argv[1] if len(sys.argv) > 1 else "configs/default.yaml"
    pipeline = TrainingPipeline(config_path=config_path)
    pipeline.train()
