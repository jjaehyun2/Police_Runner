"""탈출 판정 버그 수정 검증."""
import sys
from pathlib import Path as _Path
sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))
import warnings; warnings.filterwarnings("ignore")
import logging; logging.disable(logging.WARNING)

import numpy as np
from pursuit_evasion_rl.road_pursuit import RoadPursuitEnv
from pursuit_evasion_rl.road_pursuit.heuristic_fugitive import HeuristicFugitive

env = RoadPursuitEnv({
    "grid_rows": 6, "grid_cols": 6, "block_size": 30,
    "num_police": 4, "speed": 50, "capture_radius": 10,
    "max_steps": 100, "dt": 0.1, "fixed_max_degree": 5,
    "boundary_ratio": 0.75, "num_dead_ends": 2,
})

# 완벽한 도주자 (시야 무제한, 랜덤 0%) vs 랜덤 경찰
hf = HeuristicFugitive(env.network, vision_range=999, random_rate=0.0)

police_wins = 0
fugitive_wins = 0
timeouts = 0

for ep in range(100):
    obs, _ = env.reset(seed=ep + 200)
    done = False
    while not done:
        actions = {}
        masks = env.get_action_masks()
        # 랜덤 경찰
        for pid in env.police_ids:
            valid = np.where(masks[pid])[0]
            actions[pid] = int(np.random.choice(valid))
        # 완벽 도주자
        fs = env._vehicle_states[env.fugitive_id]
        ps = [env._vehicle_states[p] for p in env.police_ids]
        actions[env.fugitive_id] = hf.act(fs, ps, 5)
        obs, rew, term, trunc, info = env.step(actions)
        done = term[next(iter(term))] or trunc[next(iter(trunc))]

    reason = info[next(iter(info))].get("termination_reason", "timeout")
    if reason == "police_win":
        police_wins += 1
    elif reason == "fugitive_win":
        fugitive_wins += 1
    else:
        timeouts += 1

print(f"랜덤 경찰 vs 완벽 도주자 (100 에피소드):")
print(f"  경찰 승: {police_wins}")
print(f"  도주자 탈출: {fugitive_wins}")
print(f"  타임아웃: {timeouts}")
print()
if fugitive_wins > 0:
    print("✅ 탈출 판정 정상 작동!")
else:
    print("❌ 도주자가 한 번도 못 도망감 → 여전히 버그")
