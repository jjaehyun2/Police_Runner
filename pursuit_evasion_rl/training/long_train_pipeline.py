"""장기 학습 파이프라인 모듈.

LongTrainPipeline 클래스를 제공한다.
10,000+ 에피소드의 안정적 학습 실행, 체크포인트 기반 재개,
커리큘럼 학습 통합, Self-Play 양팀 업데이트를 지원한다.
"""

from __future__ import annotations

import logging
import os
import time
from typing import Any

import torch

logger = logging.getLogger(__name__)


class LongTrainPipeline:
    """장기 학습 실행 및 체크포인트 관리 파이프라인.

    10,000+ 에피소드의 안정적 학습을 지원하며,
    체크포인트 저장/로드/재개, 커리큘럼 학습 통합,
    Self-Play 양팀 업데이트를 포함한다.

    Attributes:
        config: 학습 설정 딕셔너리
        env: PursuitEnvironment 인스턴스
        algorithm: 학습 알고리즘 (MAPPO 또는 MADDPG)
        self_play_manager: SelfPlayManager 인스턴스
        curriculum_manager: CurriculumManager 인스턴스 (선택적)
    """

    def __init__(self, config: dict) -> None:
        """LongTrainPipeline 초기화.

        Args:
            config: 학습 설정 딕셔너리. 키:
                - max_episodes: int (default 10000)
                - checkpoint_interval: int (default 100)
                - log_interval: int (default 50)
                - checkpoint_dir: str (default "checkpoints")
                - algorithm: str ("MAPPO" or "MADDPG")
                - learning_rate: float
                - discount_factor: float
                - epsilon: float
                - num_police: int
                - num_nodes: int (환경 파라미터)
                - density: float (환경 파라미터)
                - fixed_max_degree: int (default 10)
                - max_steps: int (default 100)
                - curriculum: dict | None (커리큘럼 설정)
                    - levels: list of level dicts
                    - win_rate_threshold: float (default 0.9)
                    - promotion_window: int (default 500)
        """
        self.config = config
        self.max_episodes: int = config.get("max_episodes", 10000)
        self.checkpoint_interval: int = config.get("checkpoint_interval", 100)
        self.log_interval: int = config.get("log_interval", 50)
        self.checkpoint_dir: str = config.get("checkpoint_dir", "checkpoints")

        # 커리큘럼 매니저 (선택적)
        self.curriculum_manager = None
        curriculum_config = config.get("curriculum")
        if curriculum_config is not None:
            self._init_curriculum(curriculum_config)

        # 환경 및 알고리즘 초기화
        self._init_environment()
        self._init_algorithm()
        self._init_self_play()

        # 로그 파일 설정
        self._setup_logging()

        # 학습 상태
        self._start_episode: int = 1

    def _init_curriculum(self, curriculum_config: dict) -> None:
        """커리큘럼 매니저를 초기화한다."""
        from pursuit_evasion_rl.curriculum.curriculum_manager import (
            CurriculumLevel,
            CurriculumManager,
        )

        levels_data = curriculum_config.get("levels", [])
        levels = [
            CurriculumLevel(
                level_id=lv.get("level_id", idx + 1),
                name=lv.get("name", f"level_{idx + 1}"),
                num_nodes=lv["num_nodes"],
                density=lv["density"],
                boundary_ratio=lv.get("boundary_ratio", 0.2),
                num_police=lv.get("num_police", self.config.get("num_police", 3)),
            )
            for idx, lv in enumerate(levels_data)
        ]

        cm_config = {
            "win_rate_threshold": curriculum_config.get("win_rate_threshold", 0.9),
            "promotion_window": curriculum_config.get("promotion_window", 500),
        }
        self.curriculum_manager = CurriculumManager(levels=levels, config=cm_config)

    def _get_env_config(self) -> dict[str, Any]:
        """현재 환경 설정을 반환한다 (커리큘럼 적용 시 레벨 기반)."""
        if self.curriculum_manager is not None:
            cm_env = self.curriculum_manager.get_env_config()
            return {
                "network_mode": "random",
                "num_nodes": cm_env["num_nodes"],
                "density": cm_env["density"],
                "num_police": cm_env["num_police"],
                "fixed_max_degree": self.config.get("fixed_max_degree", 10),
                "max_steps": self.config.get("max_steps", 100),
            }
        return {
            "network_mode": "random",
            "num_nodes": self.config.get("num_nodes", 20),
            "density": self.config.get("density", 0.3),
            "num_police": self.config.get("num_police", 3),
            "fixed_max_degree": self.config.get("fixed_max_degree", 10),
            "max_steps": self.config.get("max_steps", 100),
        }

    def _init_environment(self) -> None:
        """환경을 초기화한다."""
        from pursuit_evasion_rl.env.pursuit_env import PursuitEnvironment

        env_config = self._get_env_config()
        self.env = PursuitEnvironment(env_config)

    def _init_algorithm(self) -> None:
        """알고리즘을 초기화한다."""
        from pursuit_evasion_rl.training.algorithms import (
            MADDPGAlgorithm,
            MAPPOAlgorithm,
            _get_obs_dim,
        )

        fixed_max_degree = self.config.get("fixed_max_degree", 10)
        num_police = self.config.get("num_police", 3)
        obs_dim = _get_obs_dim(num_police, fixed_max_degree)
        action_dim = fixed_max_degree + 1  # 인접 노드 + stay

        algorithm_name = self.config.get("algorithm", "MAPPO")

        if algorithm_name == "MAPPO":
            self.algorithm = MAPPOAlgorithm(
                obs_dim=obs_dim,
                action_dim=action_dim,
                num_police=num_police,
                learning_rate=self.config.get("learning_rate", 3e-4),
                discount_factor=self.config.get("discount_factor", 0.99),
                epsilon=self.config.get("epsilon", 0.1),
                hidden_dims=self.config.get("hidden_dims", None),
            )
        elif algorithm_name == "MADDPG":
            self.algorithm = MADDPGAlgorithm(
                obs_dim=obs_dim,
                action_dim=action_dim,
                num_police=num_police,
                learning_rate=self.config.get("learning_rate", 3e-4),
                discount_factor=self.config.get("discount_factor", 0.99),
                epsilon=self.config.get("epsilon", 0.1),
            )
        else:
            raise ValueError(
                f"지원하지 않는 알고리즘: {algorithm_name}. "
                f"'MAPPO' 또는 'MADDPG'를 사용하세요."
            )

    def _init_self_play(self) -> None:
        """Self-Play 매니저를 초기화한다."""
        from pursuit_evasion_rl.training.self_play import SelfPlayManager

        sp_config = {
            "discount_factor": self.config.get("discount_factor", 0.99),
            "batch_size": self.config.get("batch_size", 64),
            "win_rate_window": 100,
        }
        self.self_play_manager = SelfPlayManager(
            env=self.env,
            police_algo=self.algorithm,
            fugitive_algo=self.algorithm,
            config=sp_config,
        )

    def _setup_logging(self) -> None:
        """로그 파일을 설정한다."""
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        log_path = os.path.join(self.checkpoint_dir, "training.log")
        file_handler = logging.FileHandler(log_path, encoding="utf-8")
        file_handler.setLevel(logging.INFO)
        file_handler.setFormatter(
            logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
        )
        logger.addHandler(file_handler)
        logger.setLevel(logging.INFO)

    def _rebuild_environment(self) -> None:
        """커리큘럼 승급 후 환경을 재구성한다 (모델 가중치 보존)."""
        from pursuit_evasion_rl.env.pursuit_env import PursuitEnvironment
        from pursuit_evasion_rl.training.self_play import SelfPlayManager

        env_config = self._get_env_config()
        self.env = PursuitEnvironment(env_config)

        sp_config = {
            "discount_factor": self.config.get("discount_factor", 0.99),
            "batch_size": self.config.get("batch_size", 64),
            "win_rate_window": 100,
        }
        self.self_play_manager = SelfPlayManager(
            env=self.env,
            police_algo=self.algorithm,
            fugitive_algo=self.algorithm,
            config=sp_config,
        )

    def train(self) -> None:
        """메인 학습 루프.

        max_episodes 동안 학습을 실행한다.
        매 에피소드마다 Self-Play로 양팀을 업데이트하고,
        log_interval마다 통계를 로깅하며,
        checkpoint_interval마다 체크포인트를 저장한다.
        커리큘럼이 활성화된 경우 승급 조건을 확인한다.
        """
        logger.info(
            "장기 학습 시작: algorithm=%s, max_episodes=%d, "
            "checkpoint_interval=%d, log_interval=%d",
            self.config.get("algorithm", "MAPPO"),
            self.max_episodes,
            self.checkpoint_interval,
            self.log_interval,
        )
        print(
            f"[INFO] 장기 학습 시작: {self.config.get('algorithm', 'MAPPO')}, "
            f"max_episodes={self.max_episodes}"
        )

        start_time = time.time()

        for episode in range(self._start_episode, self.max_episodes + 1):
            # 에피소드 실행
            result = self.self_play_manager.run_episode()

            # 양팀 정책 업데이트 (Self-Play)
            self.self_play_manager.update_agents(result["trajectories"])

            # 커리큘럼 결과 보고
            if self.curriculum_manager is not None:
                is_police_win = result["termination_reason"] == "police_win"
                self.curriculum_manager.report_episode_result(is_police_win)

                # 승급 조건 확인 (최종 레벨이 아닐 때만)
                if (
                    self.curriculum_manager.should_promote()
                    and not self.curriculum_manager.is_completed()
                ):
                    promoted = self.curriculum_manager.promote()
                    if promoted:
                        level = self.curriculum_manager.current_level
                        msg = (
                            f"[커리큘럼 승급] Episode {episode}: "
                            f"→ {level.name} (nodes={level.num_nodes}, "
                            f"density={level.density})"
                        )
                        print(msg)
                        logger.info(msg)
                        # 환경 재구성 (모델 가중치 보존)
                        self._rebuild_environment()

            # 주기적 로깅
            if episode % self.log_interval == 0:
                self._log_stats(episode, start_time)

            # 체크포인트 저장
            if episode % self.checkpoint_interval == 0:
                self.save_checkpoint(episode)

        # 최종 체크포인트 저장
        self.save_checkpoint(self.max_episodes)

        total_time = time.time() - start_time
        print(
            f"[INFO] 학습 완료: {self.max_episodes} episodes, "
            f"{total_time:.1f}s"
        )
        logger.info(
            "학습 완료: %d episodes, %.1f초", self.max_episodes, total_time
        )

    def _log_stats(self, episode: int, start_time: float) -> None:
        """학습 통계를 로깅한다."""
        stats = self.self_play_manager.get_stats()
        elapsed = time.time() - start_time

        # 커리큘럼 레벨 정보
        level_info = ""
        if self.curriculum_manager is not None:
            cm_stats = self.curriculum_manager.get_stats()
            level_info = f" | Level: {cm_stats['current_level']}"

        log_msg = (
            f"Episode {episode}/{self.max_episodes} | "
            f"Police WR: {stats['police_win_rate']*100:.1f}% | "
            f"Fugitive WR: {stats['fugitive_win_rate']*100:.1f}% | "
            f"Avg Length: {stats['avg_episode_length']:.1f}"
            f"{level_info} | "
            f"Elapsed: {elapsed:.1f}s"
        )

        print(f"[{log_msg}]")
        logger.info(log_msg)

    def save_checkpoint(self, episode: int) -> None:
        """전체 학습 상태를 체크포인트로 저장한다.

        저장 내용:
            - episode: 현재 에피소드 번호
            - model_state: 모델 파라미터 (state_dict)
            - optimizer_state: 옵티마이저 상태
            - curriculum_state: 커리큘럼 상태 (활성화 시)
            - stats: 학습 통계
            - config: 실행 설정

        Args:
            episode: 현재 에피소드 번호
        """
        os.makedirs(self.checkpoint_dir, exist_ok=True)

        # 모델 및 옵티마이저 상태 수집
        from pursuit_evasion_rl.training.algorithms import MAPPOAlgorithm

        if isinstance(self.algorithm, MAPPOAlgorithm):
            model_state = {
                "police_actor": self.algorithm.police_actor.state_dict(),
                "police_critic": self.algorithm.police_critic.state_dict(),
                "fugitive_actor": self.algorithm.fugitive_actor.state_dict(),
                "fugitive_critic": self.algorithm.fugitive_critic.state_dict(),
            }
            optimizer_state = {
                "police_optimizer": self.algorithm.police_optimizer.state_dict(),
                "fugitive_optimizer": self.algorithm.fugitive_optimizer.state_dict(),
            }
        else:
            # MADDPG
            model_state = {
                "police_actor": self.algorithm.police_actor.state_dict(),
                "police_critic": self.algorithm.police_critic.state_dict(),
                "fugitive_actor": self.algorithm.fugitive_actor.state_dict(),
                "fugitive_critic": self.algorithm.fugitive_critic.state_dict(),
                "police_actor_target": self.algorithm.police_actor_target.state_dict(),
                "fugitive_actor_target": self.algorithm.fugitive_actor_target.state_dict(),
                "police_critic_target": self.algorithm.police_critic_target.state_dict(),
                "fugitive_critic_target": self.algorithm.fugitive_critic_target.state_dict(),
            }
            optimizer_state = {
                "police_actor_optimizer": self.algorithm.police_actor_optimizer.state_dict(),
                "police_critic_optimizer": self.algorithm.police_critic_optimizer.state_dict(),
                "fugitive_actor_optimizer": self.algorithm.fugitive_actor_optimizer.state_dict(),
                "fugitive_critic_optimizer": self.algorithm.fugitive_critic_optimizer.state_dict(),
            }

        # 커리큘럼 상태
        curriculum_state = None
        if self.curriculum_manager is not None:
            curriculum_state = {
                "current_level_index": self.curriculum_manager._current_level_index,
                "episode_results": list(self.curriculum_manager._episode_results),
                "total_episodes": self.curriculum_manager._total_episodes,
            }

        # 학습 통계
        stats = self.self_play_manager.get_stats()

        checkpoint = {
            "episode": episode,
            "model_state": model_state,
            "optimizer_state": optimizer_state,
            "curriculum_state": curriculum_state,
            "stats": {
                "police_win_rate": stats["police_win_rate"],
                "fugitive_win_rate": stats["fugitive_win_rate"],
                "avg_episode_length": stats["avg_episode_length"],
                "avg_reward": stats.get("avg_reward", 0.0),
            },
            "config": self.config,
        }

        # 에피소드별 체크포인트 저장
        path = os.path.join(self.checkpoint_dir, f"checkpoint_ep{episode}.pt")
        try:
            torch.save(checkpoint, path)
            logger.info("체크포인트 저장: %s", path)
        except OSError as e:
            logger.error("체크포인트 저장 실패 (%s): %s", path, e)

        # latest 체크포인트 덮어쓰기
        latest_path = os.path.join(self.checkpoint_dir, "latest.pt")
        try:
            torch.save(checkpoint, latest_path)
        except OSError:
            pass

    def load_checkpoint(self, path: str) -> dict:
        """체크포인트를 로드하고 메타데이터를 반환한다.

        Args:
            path: 체크포인트 파일 경로

        Returns:
            체크포인트 메타데이터 딕셔너리:
                - episode: 저장 시 에피소드 번호
                - stats: 학습 통계
                - curriculum_state: 커리큘럼 상태 (있을 경우)

        Raises:
            FileNotFoundError: 파일이 존재하지 않을 때
        """
        if not os.path.exists(path):
            raise FileNotFoundError(f"체크포인트 파일 없음: {path}")

        checkpoint = torch.load(path, map_location="cpu")

        # 모델 상태 복원
        from pursuit_evasion_rl.training.algorithms import MAPPOAlgorithm

        model_state = checkpoint["model_state"]
        optimizer_state = checkpoint["optimizer_state"]

        if isinstance(self.algorithm, MAPPOAlgorithm):
            self.algorithm.police_actor.load_state_dict(model_state["police_actor"])
            self.algorithm.police_critic.load_state_dict(model_state["police_critic"])
            self.algorithm.fugitive_actor.load_state_dict(model_state["fugitive_actor"])
            self.algorithm.fugitive_critic.load_state_dict(model_state["fugitive_critic"])
            self.algorithm.police_optimizer.load_state_dict(
                optimizer_state["police_optimizer"]
            )
            self.algorithm.fugitive_optimizer.load_state_dict(
                optimizer_state["fugitive_optimizer"]
            )
        else:
            # MADDPG
            self.algorithm.police_actor.load_state_dict(model_state["police_actor"])
            self.algorithm.police_critic.load_state_dict(model_state["police_critic"])
            self.algorithm.fugitive_actor.load_state_dict(model_state["fugitive_actor"])
            self.algorithm.fugitive_critic.load_state_dict(model_state["fugitive_critic"])
            self.algorithm.police_actor_target.load_state_dict(
                model_state["police_actor_target"]
            )
            self.algorithm.fugitive_actor_target.load_state_dict(
                model_state["fugitive_actor_target"]
            )
            self.algorithm.police_critic_target.load_state_dict(
                model_state["police_critic_target"]
            )
            self.algorithm.fugitive_critic_target.load_state_dict(
                model_state["fugitive_critic_target"]
            )
            self.algorithm.police_actor_optimizer.load_state_dict(
                optimizer_state["police_actor_optimizer"]
            )
            self.algorithm.police_critic_optimizer.load_state_dict(
                optimizer_state["police_critic_optimizer"]
            )
            self.algorithm.fugitive_actor_optimizer.load_state_dict(
                optimizer_state["fugitive_actor_optimizer"]
            )
            self.algorithm.fugitive_critic_optimizer.load_state_dict(
                optimizer_state["fugitive_critic_optimizer"]
            )

        # 커리큘럼 상태 복원
        curriculum_state = checkpoint.get("curriculum_state")
        if curriculum_state is not None and self.curriculum_manager is not None:
            from collections import deque

            self.curriculum_manager._current_level_index = curriculum_state[
                "current_level_index"
            ]
            self.curriculum_manager._episode_results = deque(
                curriculum_state["episode_results"],
                maxlen=self.curriculum_manager.promotion_window,
            )
            self.curriculum_manager._total_episodes = curriculum_state[
                "total_episodes"
            ]
            # 환경을 복원된 커리큘럼 레벨에 맞게 재구성
            self._rebuild_environment()

        logger.info(
            "체크포인트 로드 완료: %s (episode=%d)", path, checkpoint["episode"]
        )

        return {
            "episode": checkpoint["episode"],
            "stats": checkpoint.get("stats", {}),
            "curriculum_state": curriculum_state,
        }

    def resume(self, checkpoint_path: str) -> None:
        """체크포인트에서 학습을 재개한다.

        모델, 옵티마이저, 에피소드 번호, 커리큘럼 상태를 복원하고
        이어서 학습을 계속한다.

        Args:
            checkpoint_path: 체크포인트 파일 경로
        """
        metadata = self.load_checkpoint(checkpoint_path)
        self._start_episode = metadata["episode"] + 1

        logger.info(
            "학습 재개: episode %d부터 시작", self._start_episode
        )
        print(
            f"[INFO] 학습 재개: episode {self._start_episode}부터 "
            f"(checkpoint: {checkpoint_path})"
        )

        self.train()


if __name__ == "__main__":
    import sys

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    # 기본 설정으로 실행
    config = {
        "max_episodes": 10000,
        "checkpoint_interval": 100,
        "log_interval": 50,
        "checkpoint_dir": "checkpoints",
        "algorithm": "MAPPO",
        "learning_rate": 3e-4,
        "discount_factor": 0.99,
        "epsilon": 0.1,
        "num_police": 3,
        "num_nodes": 20,
        "density": 0.3,
        "fixed_max_degree": 10,
        "max_steps": 100,
    }

    # 커리큘럼 설정 (옵션)
    if "--curriculum" in sys.argv:
        config["curriculum"] = {
            "levels": [
                {"level_id": 1, "name": "easy", "num_nodes": 10, "density": 0.5},
                {"level_id": 2, "name": "medium", "num_nodes": 20, "density": 0.3},
                {"level_id": 3, "name": "hard", "num_nodes": 30, "density": 0.25},
            ],
            "win_rate_threshold": 0.9,
            "promotion_window": 500,
        }

    pipeline = LongTrainPipeline(config)

    # 재개 모드
    if len(sys.argv) > 1 and sys.argv[1] == "--resume":
        ckpt_path = sys.argv[2] if len(sys.argv) > 2 else "checkpoints/latest.pt"
        pipeline.resume(ckpt_path)
    else:
        pipeline.train()
