"""연속 2D 환경 학습 스크립트."""
import sys
import os
import warnings
import logging
import time

sys.path.insert(0, ".")
warnings.filterwarnings("ignore")

logging.getLogger("pursuit_evasion_rl").setLevel(logging.ERROR)
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

import numpy as np
from collections import deque

from pursuit_evasion_rl.continuous_env import ContinuousPursuitEnv
from pursuit_evasion_rl.training.gaussian_mappo import GaussianMAPPOAlgorithm

# 설정
MAX_EPISODES = 50000
LOG_INTERVAL = 500
CHECKPOINT_INTERVAL = 5000
CHECKPOINT_DIR = "checkpoints/continuous"

env_config = {
    "map_width": 100.0,
    "map_height": 100.0,
    "num_police": 5,
    "max_speed_police": 10.0,
    "max_speed_fugitive": 10.0,
    "capture_radius": 8.0,
    "dt": 0.1,
    "max_steps": 300,
    "road_map_config": {
        "mode": "grid",
        "grid_rows": 5,
        "grid_cols": 5,
        "grid_spacing": 20.0,
        "segment_width": 3.0,
        "speed_limit": 10.0,
    },
}

print("=" * 60)
print("  연속 2D 환경 학습 시작")
print(f"  - {MAX_EPISODES} 에피소드")
print(f"  - 경찰 {env_config['num_police']}대 (속도 10) vs 도망자 (속도 10)")
print(f"  - 체포 반경: {env_config['capture_radius']}, max_steps: {env_config['max_steps']}")
print(f"  - GaussianMAPPO, 128x128")
print("=" * 60)

# 환경 생성
env = ContinuousPursuitEnv(env_config)

# 알고리즘 생성
algo = GaussianMAPPOAlgorithm(
    obs_dim=51,
    action_dim=2,
    num_police=env_config["num_police"],
    learning_rate=3e-4,
    discount_factor=0.99,
    clip_epsilon=0.2,
    entropy_coeff=0.01,
    hidden_dims=[128, 128],
    max_speed_police=env_config["max_speed_police"],
    max_speed_fugitive=env_config["max_speed_fugitive"],
)

# 통계 추적
win_rate_window = deque(maxlen=100)
episode_lengths = deque(maxlen=100)

os.makedirs(CHECKPOINT_DIR, exist_ok=True)
start_time = time.time()

for ep in range(1, MAX_EPISODES + 1):
    obs, _ = env.reset(seed=ep)

    # 경험 수집
    police_obs, police_actions, police_rewards, police_log_probs = [], [], [], []
    fugitive_obs, fugitive_actions, fugitive_rewards, fugitive_log_probs = [], [], [], []

    done = False
    step_count = 0

    while not done:
        actions = {}
        for aid in env.agent_ids:
            action = algo.get_action(aid, obs[aid])
            actions[aid] = action

            # 경험 저장
            if aid.startswith("police"):
                police_obs.append(obs[aid].copy())
                police_actions.append(action.copy())
                police_log_probs.append(algo.get_log_prob(aid, obs[aid], action))
            else:
                fugitive_obs.append(obs[aid].copy())
                fugitive_actions.append(action.copy())
                fugitive_log_probs.append(algo.get_log_prob(aid, obs[aid], action))

        obs, rewards, terminated, truncated, info = env.step(actions)

        # 보상 기록
        for aid in env.agent_ids:
            if aid.startswith("police"):
                police_rewards.append(rewards.get(aid, 0.0))
            else:
                fugitive_rewards.append(rewards.get(aid, 0.0))

        step_count += 1
        any_agent = next(iter(terminated))
        done = terminated[any_agent] or truncated[any_agent]

    # 학습 업데이트
    batch = {
        "police_obs": police_obs,
        "police_actions": police_actions,
        "police_rewards": police_rewards,
        "police_old_log_probs": police_log_probs,
        "fugitive_obs": fugitive_obs,
        "fugitive_actions": fugitive_actions,
        "fugitive_rewards": fugitive_rewards,
        "fugitive_old_log_probs": fugitive_log_probs,
    }
    algo.train_step(batch)

    # 통계
    termination = info.get("termination", "timeout")
    win_rate_window.append(1 if termination == "police_capture" else 0)
    episode_lengths.append(step_count)

    # 로깅
    if ep % LOG_INTERVAL == 0:
        police_wr = sum(win_rate_window) / max(len(win_rate_window), 1) * 100
        avg_len = sum(episode_lengths) / max(len(episode_lengths), 1)
        elapsed = time.time() - start_time
        fugitive_wr = sum(1 for r in win_rate_window if r == 0) / max(len(win_rate_window), 1) * 100

        # 도망자 승리 vs 타임아웃 구분은 info에서 확인
        print(
            f"[Ep {ep:5d}/{MAX_EPISODES}] "
            f"Police WR: {police_wr:5.1f}% | "
            f"Avg Length: {avg_len:6.1f} | "
            f"Elapsed: {elapsed:.1f}s"
        )

    # 체크포인트 저장
    if ep % CHECKPOINT_INTERVAL == 0:
        path = os.path.join(CHECKPOINT_DIR, f"continuous_ep{ep}.pt")
        algo.save(path)

# 최종 저장
algo.save(os.path.join(CHECKPOINT_DIR, "continuous_final.pt"))
elapsed = time.time() - start_time
print(f"\n[완료] {MAX_EPISODES} episodes, {elapsed:.1f}s")
print(f"최종 경찰 승률: {sum(win_rate_window)/max(len(win_rate_window),1)*100:.1f}%")
