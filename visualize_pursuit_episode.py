"""도로 추격 에피소드 시각화 — 학습된 모델로 한 에피소드를 프레임별 PNG 저장."""
import sys
import os
sys.path.insert(0, ".")
import warnings; warnings.filterwarnings("ignore")
import logging; logging.disable(logging.WARNING)

import matplotlib
matplotlib.use("Agg")
import numpy as np

from pursuit_evasion_rl.road_pursuit import RoadPursuitEnv
from pursuit_evasion_rl.road_pursuit.heuristic_fugitive import HeuristicFugitive
from pursuit_evasion_rl.road_pursuit.visualizer import RoadVisualizer
from pursuit_evasion_rl.training.algorithms import MAPPOAlgorithm, _flatten_observation

# 설정
NUM_POLICE = 4
FIXED_MAX_DEGREE = 5
OUTPUT_DIR = "pursuit_episode_frames"
os.makedirs(OUTPUT_DIR, exist_ok=True)

env_config = {
    "grid_rows": 8, "grid_cols": 8, "block_size": 30.0,
    "num_police": NUM_POLICE, "speed": 50.0, "capture_radius": 10.0,
    "max_steps": 200, "dt": 0.1, "fixed_max_degree": FIXED_MAX_DEGREE,
    "road_width": 10.0, "boundary_ratio": 0.75, "num_dead_ends": 3,
}

env = RoadPursuitEnv(env_config)

# 모델 로드
obs, _ = env.reset(seed=0)
obs_dim = sum(np.asarray(v).flatten().shape[0] for v in obs["police_0"].values())
action_dim = FIXED_MAX_DEGREE + 1

algo = MAPPOAlgorithm(
    obs_dim=obs_dim, action_dim=action_dim, num_police=NUM_POLICE,
    learning_rate=1e-4, discount_factor=0.99, epsilon=0.15, hidden_dims=[128, 128],
)
algo.load("checkpoints/road_pursuit/road_pursuit_final.pt")
print("모델 로드 완료")

# 도주자 휴리스틱
fugitive_heuristic = HeuristicFugitive(env.network, vision_range=60.0, random_rate=0.3)

# 시각화 도구
viz = RoadVisualizer(env.network, figsize=(10, 10))

# 에피소드 실행
obs, _ = env.reset(seed=77)
print(f"네트워크: {env.network.num_intersections}교차로, 탈출구 {len(env.network.get_boundary_intersections())}개")

# 초기 상태 저장
viz.render(env._vehicle_states, step=0, title="Step 0 - Start")
viz.save(os.path.join(OUTPUT_DIR, "step_000.png"), dpi=100, bbox_inches="tight")
print("  Step 0 saved")

done = False
step = 0

while not done:
    masks = env.get_action_masks()
    actions = {}

    # 경찰: 학습된 모델
    for pid in env.police_ids:
        actions[pid] = algo.get_action(pid, obs[pid], action_mask=masks[pid])

    # 도주자: 휴리스틱
    fs = env._vehicle_states[env.fugitive_id]
    ps = [env._vehicle_states[p] for p in env.police_ids]
    actions[env.fugitive_id] = fugitive_heuristic.act(fs, ps, FIXED_MAX_DEGREE)

    obs, rewards, terminated, truncated, info = env.step(actions)
    step += 1

    any_agent = next(iter(terminated))
    done = terminated[any_agent] or truncated[any_agent]

    # 5스텝마다 프레임 저장 (너무 많으면 보기 힘드니까)
    if step % 3 == 0 or done:
        reason = info[next(iter(info))].get("termination_reason", "")
        title = f"Step {step}"
        if done:
            if reason == "police_win":
                title += " - CAPTURED!"
            elif reason == "fugitive_win":
                title += " - ESCAPED!"
            else:
                title += " - TIMEOUT"

        viz.render(env._vehicle_states, step=step, title=title)
        viz.save(os.path.join(OUTPUT_DIR, f"step_{step:03d}.png"), dpi=100, bbox_inches="tight")
        print(f"  Step {step} saved ({reason if done else 'in progress'})")

viz.close()
print(f"\n완료! {OUTPUT_DIR}/ 폴더에 프레임 저장됨 (총 {step} 스텝)")
print(f"결과: {info[next(iter(info))].get('termination_reason', 'unknown')}")
