"""최강 도주자로 모델 테스트 — 랜덤 0%, 시야 무제한."""
import sys
from pathlib import Path as _Path
sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))
import warnings; warnings.filterwarnings("ignore")
import logging; logging.disable(logging.WARNING)

import numpy as np
from pursuit_evasion_rl.road_pursuit import RoadPursuitEnv
from pursuit_evasion_rl.road_pursuit.heuristic_fugitive import HeuristicFugitive
from pursuit_evasion_rl.training.algorithms import MAPPOAlgorithm, _flatten_observation

NUM_POLICE = 4
FIXED_MAX_DEGREE = 5
NUM_TEST_EPISODES = 200

env_config = {
    "grid_rows": 6, "grid_cols": 6, "block_size": 30.0,
    "num_police": NUM_POLICE, "speed": 50.0, "capture_radius": 15.0,
    "max_steps": 150, "dt": 0.1, "fixed_max_degree": FIXED_MAX_DEGREE,
    "road_width": 10.0, "boundary_ratio": 0.75, "num_dead_ends": 2,
}

env = RoadPursuitEnv(env_config)

# 최강 도주자: 시야 무제한 (999m), 랜덤 0%
fugitive_hard = HeuristicFugitive(env.network, vision_range=999.0, random_rate=0.0)

# 학습된 모델 로드
obs, _ = env.reset(seed=0)
obs_dim = sum(np.asarray(v).flatten().shape[0] for v in obs["police_0"].values())
action_dim = FIXED_MAX_DEGREE + 1

algo = MAPPOAlgorithm(
    obs_dim=obs_dim, action_dim=action_dim, num_police=NUM_POLICE,
    learning_rate=1e-4, discount_factor=0.99, epsilon=0.15, hidden_dims=[128, 128],
)
algo.load("checkpoints/road_pursuit/road_pursuit_final.pt")
print("모델 로드 완료")

# 테스트
police_wins = 0
fugitive_wins = 0
timeouts = 0
total_length = 0

for ep in range(NUM_TEST_EPISODES):
    obs, _ = env.reset(seed=ep + 10000)
    done = False
    steps = 0

    while not done:
        masks = env.get_action_masks()
        actions = {}

        # 경찰: 학습된 모델
        for pid in env.police_ids:
            actions[pid] = algo.get_action(pid, obs[pid], action_mask=masks[pid])

        # 도주자: 최강 휴리스틱 (전역 시야, 랜덤 0%)
        fs = env._vehicle_states[env.fugitive_id]
        ps = [env._vehicle_states[p] for p in env.police_ids]
        actions[env.fugitive_id] = fugitive_hard.act(fs, ps, FIXED_MAX_DEGREE)

        obs, rewards, terminated, truncated, info = env.step(actions)
        steps += 1
        any_agent = next(iter(terminated))
        done = terminated[any_agent] or truncated[any_agent]

    reason = info[next(iter(info))].get("termination_reason", "timeout")
    if reason == "police_win":
        police_wins += 1
    elif reason == "fugitive_win":
        fugitive_wins += 1
    else:
        timeouts += 1
    total_length += steps

print(f"\n{'='*50}")
print(f"  테스트 결과: 최강 도주자 (전역 시야, 랜덤 0%)")
print(f"  {NUM_TEST_EPISODES} 에피소드")
print(f"{'='*50}")
print(f"  경찰 승리: {police_wins} ({police_wins/NUM_TEST_EPISODES*100:.1f}%)")
print(f"  도주자 탈출: {fugitive_wins} ({fugitive_wins/NUM_TEST_EPISODES*100:.1f}%)")
print(f"  타임아웃: {timeouts} ({timeouts/NUM_TEST_EPISODES*100:.1f}%)")
print(f"  평균 길이: {total_length/NUM_TEST_EPISODES:.1f}")
print(f"{'='*50}")
