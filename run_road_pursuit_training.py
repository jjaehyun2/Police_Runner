"""재설계된 도로 추격 환경 학습 스크립트.

이산 행동 공간 (교차로 방향 선택) + 연속 위치 (도로 위 자동 전진)
경찰만 RL 학습, 도주자는 휴리스틱 고정.
"""
import sys
import os
import warnings
import logging
import time
from collections import deque

sys.path.insert(0, ".")
warnings.filterwarnings("ignore")
logging.getLogger("pursuit_evasion_rl").setLevel(logging.ERROR)
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

import numpy as np

from pursuit_evasion_rl.road_pursuit import RoadPursuitEnv
from pursuit_evasion_rl.road_pursuit.heuristic_fugitive import HeuristicFugitive
from pursuit_evasion_rl.training.algorithms import MAPPOAlgorithm, _get_obs_dim, _flatten_observation

# ===== 설정 =====
MAX_EPISODES = 20000
LOG_INTERVAL = 500
CHECKPOINT_INTERVAL = 2000
CHECKPOINT_DIR = "checkpoints/road_pursuit"

FIXED_MAX_DEGREE = 5
NUM_POLICE = 6

env_config = {
    "grid_rows": 6,
    "grid_cols": 6,
    "block_size": 30.0,
    "num_police": NUM_POLICE,
    "speed": 50.0,
    "capture_radius": 10.0,
    "max_steps": 150,
    "dt": 0.1,
    "fixed_max_degree": FIXED_MAX_DEGREE,
    "road_width": 10.0,
    "boundary_ratio": 0.5,
    "num_dead_ends": 2,
    "seed": None,
    "cooperation_bonus": 0.05,
}

print("=" * 60)
print("  도로 추격 환경 학습 v10 (경찰 6대)")
print(f"  - {MAX_EPISODES} 에피소드")
print(f"  - 6x6 그리드 (블록 30m), 경찰 {NUM_POLICE}대")
print(f"  - 탈출구 50% (~10개), 체포 반경 10")
print(f"  - 탈출 판정 정상, 처음부터 새로 학습")
print("=" * 60)

# ===== 환경 생성 =====
env = RoadPursuitEnv(env_config)
print(f"\n네트워크: {env.network.num_intersections} 교차로, "
      f"{env.network.num_segments} 세그먼트")
print(f"탈출구: {len(env.network.get_boundary_intersections())}개")

# 도주자 휴리스틱 생성 (시야 2블록=60m, 랜덤 30%)
fugitive_heuristic = HeuristicFugitive(env.network, vision_range=60.0, random_rate=0.3)

# ===== 알고리즘 생성 (경찰만) =====
obs, _ = env.reset(seed=0)
sample_obs = obs["police_0"]
obs_dim = sum(np.asarray(v).flatten().shape[0] for v in sample_obs.values())
action_dim = FIXED_MAX_DEGREE + 1

print(f"obs_dim={obs_dim}, action_dim={action_dim}")

algo = MAPPOAlgorithm(
    obs_dim=obs_dim,
    action_dim=action_dim,
    num_police=NUM_POLICE,
    learning_rate=1e-4,
    discount_factor=0.99,
    epsilon=0.15,
    hidden_dims=[128, 128],
)

# 경찰 6대는 새로운 구성이므로 처음부터 학습
import os
print("경찰 6대 — 처음부터 새로 학습")

# ===== 학습 루프 (경찰만 학습) =====
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

win_rate_window = deque(maxlen=100)
episode_lengths = deque(maxlen=100)
start_time = time.time()

for ep in range(1, MAX_EPISODES + 1):
    obs, _ = env.reset(seed=ep)

    # 경찰 경험만 수집 (도주자는 학습 안 함)
    police_obs, police_actions, police_rewards, police_log_probs = [], [], [], []

    done = False
    step_count = 0

    while not done:
        masks = env.get_action_masks()
        actions = {}

        # 경찰: RL 에이전트 (행동 마스크 적용)
        for pid in env.police_ids:
            mask = masks.get(pid)
            action = algo.get_action(pid, obs[pid], action_mask=mask)
            actions[pid] = action

            obs_flat = _flatten_observation(obs[pid])
            log_prob = algo.get_log_prob(pid, obs[pid], action)
            police_obs.append(obs_flat)
            police_actions.append(action)
            police_log_probs.append(log_prob)

        # 도주자: 휴리스틱 (시야 제한)
        fugitive_state = env._vehicle_states[env.fugitive_id]
        police_vehicle_states = [env._vehicle_states[pid] for pid in env.police_ids]
        fugitive_action = fugitive_heuristic.act(
            fugitive_state, police_vehicle_states, FIXED_MAX_DEGREE
        )
        actions[env.fugitive_id] = fugitive_action

        obs, rewards, terminated, truncated, info = env.step(actions)

        # 경찰 보상만 기록
        for pid in env.police_ids:
            police_rewards.append(rewards.get(pid, 0.0))

        step_count += 1
        any_agent = next(iter(terminated))
        done = terminated[any_agent] or truncated[any_agent]

    # 경찰만 학습 업데이트
    if police_obs:
        batch = {
            "police_obs": police_obs,
            "police_actions": police_actions,
            "police_rewards": police_rewards,
            "police_old_log_probs": police_log_probs,
            "fugitive_obs": [],
            "fugitive_actions": [],
            "fugitive_rewards": [],
            "fugitive_old_log_probs": [],
        }
        algo.train_step(batch)

    # 통계
    termination = info[next(iter(info))].get("termination_reason", "timeout")
    win_rate_window.append(1 if termination == "police_win" else 0)
    episode_lengths.append(step_count)

    # 로깅
    if ep % LOG_INTERVAL == 0:
        police_wr = sum(win_rate_window) / max(len(win_rate_window), 1) * 100
        avg_len = sum(episode_lengths) / max(len(episode_lengths), 1)
        elapsed = time.time() - start_time
        print(
            f"[Ep {ep:5d}/{MAX_EPISODES}] "
            f"Police WR: {police_wr:5.1f}% | "
            f"Avg Length: {avg_len:6.1f} | "
            f"Elapsed: {elapsed:.1f}s"
        )

    # 체크포인트 저장
    if ep % CHECKPOINT_INTERVAL == 0:
        path = os.path.join(CHECKPOINT_DIR, f"road_pursuit_ep{ep}.pt")
        algo.save(path)

# 최종 저장
algo.save(os.path.join(CHECKPOINT_DIR, "road_pursuit_final.pt"))
elapsed = time.time() - start_time
print(f"\n[완료] {MAX_EPISODES} episodes, {elapsed:.1f}s")
print(f"최종 경찰 승률: {sum(win_rate_window)/max(len(win_rate_window),1)*100:.1f}%")
print(f"평균 에피소드 길이: {sum(episode_lengths)/max(len(episode_lengths),1):.1f}")
