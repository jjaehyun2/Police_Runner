"""재설계 도로 환경 시각화 (탈출구 축소 + 막다른 길)."""
from pathlib import Path as _ImgPath
_IMAGE_DIR = _ImgPath(__file__).resolve().parents[1] / "docs" / "images"
import sys
from pathlib import Path as _Path
sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))
import warnings; warnings.filterwarnings("ignore")
import logging; logging.disable(logging.WARNING)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from pursuit_evasion_rl.road_pursuit import RoadPursuitEnv
from pursuit_evasion_rl.road_pursuit.visualizer import RoadVisualizer

env_config = {
    "grid_rows": 6, "grid_cols": 6, "block_size": 30.0,
    "num_police": 4, "speed": 50.0, "capture_radius": 15.0,
    "max_steps": 150, "dt": 0.1, "fixed_max_degree": 5,
    "road_width": 10.0, "boundary_ratio": 0.5, "num_dead_ends": 2, "seed": 7,
}

env = RoadPursuitEnv(env_config)
obs, _ = env.reset(seed=42)

print(f"교차로: {env.network.num_intersections}개")
print(f"세그먼트: {env.network.num_segments}개")
print(f"탈출 가능 경계: {env.network.get_boundary_intersections()}")
print(f"경계 수: {len(env.network.get_boundary_intersections())}개 (원래 20개에서 축소)")

# 막다른 길 교차로 찾기 (outgoing 1개인 내부 교차로)
boundary_set = set(env.network.get_boundary_intersections())
dead_ends = [
    iid for iid, inter in env.network.intersections.items()
    if iid not in boundary_set and len(inter.outgoing_segments) <= 1
]
print(f"막다른 길 교차로: {dead_ends}")

# 시각화
viz = RoadVisualizer(env.network, figsize=(12, 12))
fig = viz.render(env._vehicle_states, step=0, title="Road Pursuit: Reduced Exits + Dead Ends")

# 추가 표시: 막다른 길 강조
for de_id in dead_ends:
    pos = env.network.intersections[de_id].position
    viz.ax.scatter(pos[0], pos[1], c="orange", s=200, zorder=6,
                   marker="X", edgecolors="red", linewidths=2)
    viz.ax.annotate("DEAD\nEND", (pos[0], pos[1]),
                   textcoords="offset points", xytext=(-15, -20),
                   fontsize=7, color="red", fontweight="bold")

# 범례 추가
from matplotlib.lines import Line2D
extra_legend = [
    Line2D([0], [0], marker="X", color="w", markerfacecolor="orange",
           markersize=12, markeredgecolor="red", label="Dead End (blocked)"),
]
existing_handles = viz.ax.get_legend().legend_handles
viz.ax.legend(handles=existing_handles + extra_legend, loc="upper right", fontsize=9)

viz.fig.tight_layout()
viz.save(str(_IMAGE_DIR / "road_pursuit_with_deadends.png"), dpi=150, bbox_inches="tight")
print(f"\n저장 완료: {_IMAGE_DIR / 'road_pursuit_with_deadends.png'}")
