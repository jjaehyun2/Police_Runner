"""연속 2D 환경 구조 시각화 스크립트."""
import sys
import warnings
import logging

sys.path.insert(0, ".")
warnings.filterwarnings("ignore")
logging.disable(logging.WARNING)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

from pursuit_evasion_rl.continuous_env import ContinuousPursuitEnv

# 환경 생성 (학습과 동일 설정)
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

env = ContinuousPursuitEnv(env_config)
obs, _ = env.reset(seed=42)

fig, ax = plt.subplots(1, 1, figsize=(10, 10))

# 1. 도로 세그먼트 그리기
for seg in env._road_map.get_segments():
    sx, sy = seg.start_point
    ex, ey = seg.end_point
    ax.plot([sx, ex], [sy, ey], color="#888888", linewidth=seg.width * 0.8, solid_capstyle="round")

# 2. 교차점 그리기
for inter in env._road_map.get_intersections():
    ix, iy = inter.position
    ax.plot(ix, iy, "ko", markersize=4)

# 3. 경계 출구 표시
exits = env._road_map.get_boundary_exits()
for ex_pt in exits:
    ax.plot(ex_pt[0], ex_pt[1], "g^", markersize=12, markeredgecolor="darkgreen", markeredgewidth=1.5)

# 4. 맵 경계 표시
min_x, min_y, max_x, max_y = env._road_map.map_bounds
rect = plt.Rectangle((min_x, min_y), max_x - min_x, max_y - min_y,
                      fill=False, edgecolor="red", linewidth=2, linestyle="--")
ax.add_patch(rect)

# 5. 에이전트 위치 표시
for aid, state in env._states.items():
    x, y = state.position
    if state.agent_type == "police":
        ax.plot(x, y, "bs", markersize=14, markeredgecolor="navy", markeredgewidth=1.5)
        ax.annotate(aid.replace("police_", "P"), (x, y), fontsize=8,
                    ha="center", va="center", color="white", fontweight="bold")
    else:
        ax.plot(x, y, "ro", markersize=16, markeredgecolor="darkred", markeredgewidth=1.5)
        ax.annotate("F", (x, y), fontsize=9, ha="center", va="center",
                    color="white", fontweight="bold")

    # 체포 반경 표시 (경찰만)
    if state.agent_type == "police":
        circle = plt.Circle((x, y), env.capture_radius, fill=False,
                           edgecolor="blue", linestyle=":", linewidth=0.8, alpha=0.4)
        ax.add_patch(circle)

    # 방향 화살표
    dx = 5.0 * np.cos(state.heading)
    dy = 5.0 * np.sin(state.heading)
    ax.annotate("", xy=(x + dx, y + dy), xytext=(x, y),
                arrowprops=dict(arrowstyle="->", color="black", lw=1.5))

# 6. 범례
legend_elements = [
    mpatches.Patch(facecolor="blue", label="경찰 (Police)"),
    mpatches.Patch(facecolor="red", label="도망자 (Fugitive)"),
    plt.Line2D([0], [0], marker="^", color="w", markerfacecolor="green",
               markersize=12, label="경계 출구 (탈출 지점)"),
    plt.Line2D([0], [0], color="#888888", linewidth=3, label="도로 (Road)"),
    plt.Line2D([0], [0], color="red", linewidth=2, linestyle="--", label="맵 경계 (탈출선)"),
    plt.Line2D([0], [0], color="blue", linewidth=1, linestyle=":", label="체포 반경 (8.0)"),
]
ax.legend(handles=legend_elements, loc="upper right", fontsize=10)

# 설정
ax.set_xlim(-5, max_x + 5)
ax.set_ylim(-5, max_y + 5)
ax.set_aspect("equal")
ax.set_xlabel("X 좌표", fontsize=12)
ax.set_ylabel("Y 좌표", fontsize=12)
ax.set_title("연속 2D 추격-도주 환경 구조\n(5x5 그리드 도로, 경찰 5대 vs 도망자 1대)", fontsize=13)
ax.grid(True, alpha=0.2)

plt.tight_layout()
plt.savefig("continuous_env_structure.png", dpi=150, bbox_inches="tight")
print("저장 완료: continuous_env_structure.png")
plt.close()
