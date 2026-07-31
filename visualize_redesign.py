"""재설계된 도로 추격 환경 시각화."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
import numpy as np

fig, ax = plt.subplots(1, 1, figsize=(12, 10))

# 도로 네트워크 (4x4 블록, 실제 도로처럼 폭 있는 도로)
ROAD_WIDTH = 8
BLOCK_SIZE = 40
OFFSET = 20  # 맵 여백

# 도로 색상
road_color = "#444444"
lane_color = "#FFFFFF"
sidewalk_color = "#CCCCCC"

# 수평 도로 그리기 (4줄)
for row in range(4):
    y = OFFSET + row * BLOCK_SIZE
    # 도로 바닥
    road_rect = plt.Rectangle((OFFSET - 5, y - ROAD_WIDTH/2), 
                               3 * BLOCK_SIZE + 10, ROAD_WIDTH,
                               facecolor=road_color, edgecolor="none")
    ax.add_patch(road_rect)
    # 중앙선 (점선)
    ax.plot([OFFSET - 5, OFFSET + 3*BLOCK_SIZE + 5], [y, y], 
            color="yellow", linewidth=1.5, linestyle="--")
    # 방향 화살표 (도로 위)
    for col in range(3):
        x_mid = OFFSET + col * BLOCK_SIZE + BLOCK_SIZE/2
        ax.annotate("", xy=(x_mid + 12, y + 2), xytext=(x_mid - 12, y + 2),
                   arrowprops=dict(arrowstyle="->", color="white", lw=1.2))
        ax.annotate("", xy=(x_mid - 12, y - 2), xytext=(x_mid + 12, y - 2),
                   arrowprops=dict(arrowstyle="->", color="white", lw=1.2))

# 수직 도로 그리기 (4줄)
for col in range(4):
    x = OFFSET + col * BLOCK_SIZE
    road_rect = plt.Rectangle((x - ROAD_WIDTH/2, OFFSET - 5),
                               ROAD_WIDTH, 3 * BLOCK_SIZE + 10,
                               facecolor=road_color, edgecolor="none")
    ax.add_patch(road_rect)
    # 중앙선
    ax.plot([x, x], [OFFSET - 5, OFFSET + 3*BLOCK_SIZE + 5],
            color="yellow", linewidth=1.5, linestyle="--")

# 교차로 강조 (밝은 색 사각형)
for row in range(4):
    for col in range(4):
        x = OFFSET + col * BLOCK_SIZE
        y = OFFSET + row * BLOCK_SIZE
        inter_rect = plt.Rectangle((x - ROAD_WIDTH/2, y - ROAD_WIDTH/2),
                                    ROAD_WIDTH, ROAD_WIDTH,
                                    facecolor="#555555", edgecolor="#888888",
                                    linewidth=1.5, linestyle="-")
        ax.add_patch(inter_rect)

# 교차로에 방향 선택 표시 (하나만 예시)
example_x = OFFSET + 1 * BLOCK_SIZE
example_y = OFFSET + 2 * BLOCK_SIZE
# 방향 선택 화살표들
directions = [(15, 0, "직진"), (0, 15, "좌회전"), (-15, 0, "유턴"), (0, -15, "우회전")]
for dx, dy, label in directions:
    ax.annotate("", xy=(example_x + dx, example_y + dy),
               xytext=(example_x, example_y),
               arrowprops=dict(arrowstyle="-|>", color="orange", lw=2))

ax.annotate("교차로\n(방향 선택)", (example_x, example_y + 20),
           fontsize=9, ha="center", color="orange", fontweight="bold")

# 맵 경계 (빨간 점선)
boundary = plt.Rectangle((OFFSET - 10, OFFSET - 10),
                          3 * BLOCK_SIZE + 20, 3 * BLOCK_SIZE + 20,
                          fill=False, edgecolor="red", linewidth=2.5, linestyle="--")
ax.add_patch(boundary)

# 경계 출구 표시 (초록 삼각형)
exits = [
    (OFFSET - 10, OFFSET + 0 * BLOCK_SIZE),
    (OFFSET - 10, OFFSET + 1 * BLOCK_SIZE),
    (OFFSET - 10, OFFSET + 2 * BLOCK_SIZE),
    (OFFSET - 10, OFFSET + 3 * BLOCK_SIZE),
    (OFFSET + 3*BLOCK_SIZE + 10, OFFSET + 0 * BLOCK_SIZE),
    (OFFSET + 3*BLOCK_SIZE + 10, OFFSET + 2 * BLOCK_SIZE),
    (OFFSET + 1 * BLOCK_SIZE, OFFSET - 10),
    (OFFSET + 2 * BLOCK_SIZE, OFFSET + 3*BLOCK_SIZE + 10),
]
for ex, ey in exits:
    ax.plot(ex, ey, "g^", markersize=14, markeredgecolor="darkgreen", markeredgewidth=1.5)

# ========== 에이전트 배치 ==========

# 경찰 5대 (파란색 사각형, 도로 위에 배치)
police_positions = [
    (OFFSET + 0.3 * BLOCK_SIZE, OFFSET + 0 * BLOCK_SIZE + 2),   # P0: 수평 도로 위
    (OFFSET + 2 * BLOCK_SIZE + 2, OFFSET + 0.6 * BLOCK_SIZE),   # P1: 수직 도로 위
    (OFFSET + 3 * BLOCK_SIZE, OFFSET + 2 * BLOCK_SIZE + 2),     # P2: 수평 도로 위
    (OFFSET + 1 * BLOCK_SIZE - 2, OFFSET + 2.4 * BLOCK_SIZE),   # P3: 수직 도로 위
    (OFFSET + 2.7 * BLOCK_SIZE, OFFSET + 3 * BLOCK_SIZE - 2),   # P4: 수평 도로 위
]

for i, (px, py) in enumerate(police_positions):
    # 차량 바디
    car = FancyBboxPatch((px - 4, py - 2.5), 8, 5,
                          boxstyle="round,pad=0.3",
                          facecolor="#2196F3", edgecolor="navy", linewidth=1.5)
    ax.add_patch(car)
    ax.text(px, py, f"P{i}", fontsize=7, ha="center", va="center",
            color="white", fontweight="bold")
    # 체포 반경
    circle = plt.Circle((px, py), 8, fill=False, edgecolor="blue",
                        linestyle=":", linewidth=1, alpha=0.5)
    ax.add_patch(circle)

# 도망자 1대 (빨간색, 도로 위)
fugitive_pos = (OFFSET + 1.5 * BLOCK_SIZE, OFFSET + 1 * BLOCK_SIZE - 2)
fx, fy = fugitive_pos
car_f = FancyBboxPatch((fx - 4, fy - 2.5), 8, 5,
                        boxstyle="round,pad=0.3",
                        facecolor="#F44336", edgecolor="darkred", linewidth=1.5)
ax.add_patch(car_f)
ax.text(fx, fy, "F", fontsize=8, ha="center", va="center",
        color="white", fontweight="bold")

# ========== 설명 텍스트 ==========

# 도로 세그먼트 설명
ax.annotate("도로 세그먼트\n(폭 있음, 방향 고정)", 
           xy=(OFFSET + 0.5*BLOCK_SIZE, OFFSET + 0*BLOCK_SIZE + ROAD_WIDTH/2 + 3),
           fontsize=8, color="#333333", ha="center")

# 범례
legend_elements = [
    mpatches.Patch(facecolor="#2196F3", edgecolor="navy", label="경찰 차량 (Police)"),
    mpatches.Patch(facecolor="#F44336", edgecolor="darkred", label="도망자 차량 (Fugitive)"),
    plt.Line2D([0], [0], marker="^", color="w", markerfacecolor="green",
               markersize=12, label="경계 출구 (탈출 지점)"),
    mpatches.Patch(facecolor="#444444", label="도로 (폭 있음, 방향 구속)"),
    mpatches.Patch(facecolor="#555555", edgecolor="#888888", label="교차로 (방향 선택 가능)"),
    plt.Line2D([0], [0], color="red", linewidth=2, linestyle="--", label="맵 경계"),
    plt.Line2D([0], [0], color="blue", linewidth=1, linestyle=":", label="체포 반경 (8.0)"),
    plt.Line2D([0], [0], color="orange", linewidth=2, label="교차로 방향 선택지"),
]
ax.legend(handles=legend_elements, loc="upper right", fontsize=9, framealpha=0.9)

# 핵심 설명 박스
info_text = (
    "[ 재설계 환경 ]\n"
    "• 차량은 도로 위에서만 이동 (도로 방향으로만)\n"
    "• 교차로에서만 방향 전환 가능 (직진/좌/우/유턴)\n"
    "• 행동 = 교차로에서 '어느 도로로 갈지' 선택 (이산)\n"
    "• 도로 폭 내에서 연속 좌표 (progress 0~1)\n"
    "• 체포: 반경 8 이내 접근 시"
)
ax.text(0.02, 0.02, info_text, transform=ax.transAxes,
        fontsize=9, verticalalignment="bottom",
        bbox=dict(boxstyle="round", facecolor="lightyellow", alpha=0.9))

# 축 설정
ax.set_xlim(0, OFFSET + 3*BLOCK_SIZE + 30)
ax.set_ylim(0, OFFSET + 3*BLOCK_SIZE + 30)
ax.set_aspect("equal")
ax.set_xlabel("X (미터)", fontsize=11)
ax.set_ylabel("Y (미터)", fontsize=11)
ax.set_title("재설계: 도로 기반 추격-도주 환경\n(폭 있는 도로 + 교차로 방향 선택)", fontsize=13, fontweight="bold")
ax.set_facecolor("#F5F5F5")

plt.tight_layout()
plt.savefig("redesign_env_structure.png", dpi=150, bbox_inches="tight")
print("저장 완료: redesign_env_structure.png")
plt.close()
