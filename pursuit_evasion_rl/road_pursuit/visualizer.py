"""도로 추격 환경 시각화 모듈.

matplotlib 기반으로 2D 도로 네트워크와 차량 위치를 시각화한다.
- 도로: 두꺼운 회색 선
- 교차로: 원형 마커
- 경찰: 파란색 사각형
- 도망자: 빨간색 사각형
- 경계 교차로: 초록색 마커
"""

from __future__ import annotations

from typing import Any

import numpy as np

from pursuit_evasion_rl.road_pursuit.road_network_2d import RoadNetwork2D
from pursuit_evasion_rl.road_pursuit.vehicle_model import VehicleState


class RoadVisualizer:
    """도로 추격 환경 시각화 클래스.

    Attributes:
        network: 2D 도로 네트워크
        fig: matplotlib Figure
        ax: matplotlib Axes
    """

    def __init__(self, network: RoadNetwork2D, figsize: tuple[float, float] = (10, 10)) -> None:
        """초기화.

        Args:
            network: 2D 도로 네트워크
            figsize: 그림 크기 (인치)
        """
        self.network = network
        self._figsize = figsize
        self.fig = None
        self.ax = None

    def _ensure_fig(self) -> None:
        """matplotlib figure/axes가 없으면 생성한다."""
        import matplotlib.pyplot as plt

        if self.fig is None or self.ax is None:
            self.fig, self.ax = plt.subplots(1, 1, figsize=self._figsize)

    def render(
        self,
        vehicle_states: dict[str, VehicleState] | None = None,
        step: int = 0,
        title: str | None = None,
    ) -> Any:
        """현재 상태를 시각화한다.

        Args:
            vehicle_states: 차량 상태 딕셔너리 (None이면 네트워크만 표시)
            step: 현재 스텝 번호
            title: 그림 제목

        Returns:
            matplotlib Figure 객체
        """
        import matplotlib.pyplot as plt
        import matplotlib.patches as mpatches

        self._ensure_fig()
        self.ax.clear()

        # 1. 도로 세그먼트 그리기
        for seg in self.network.segments.values():
            self.ax.plot(
                [seg.start_pos[0], seg.end_pos[0]],
                [seg.start_pos[1], seg.end_pos[1]],
                color="gray",
                linewidth=max(1.0, seg.width * 0.3),
                alpha=0.6,
                solid_capstyle="round",
            )
            # 방향 화살표 (중간 지점에 작은 삼각형)
            mid_x = (seg.start_pos[0] + seg.end_pos[0]) / 2
            mid_y = (seg.start_pos[1] + seg.end_pos[1]) / 2
            dx, dy = seg.direction
            self.ax.annotate(
                "",
                xy=(mid_x + dx * 3, mid_y + dy * 3),
                xytext=(mid_x - dx * 3, mid_y - dy * 3),
                arrowprops=dict(arrowstyle="->", color="gray", lw=0.8),
            )

        # 2. 교차로 그리기
        boundary_set = set(self.network.get_boundary_intersections())
        for iid, inter in self.network.intersections.items():
            color = "green" if iid in boundary_set else "black"
            marker_size = 80 if iid in boundary_set else 40
            self.ax.scatter(
                inter.position[0],
                inter.position[1],
                c=color,
                s=marker_size,
                zorder=5,
                edgecolors="black",
                linewidths=0.5,
            )
            self.ax.annotate(
                str(iid),
                (inter.position[0], inter.position[1]),
                textcoords="offset points",
                xytext=(5, 5),
                fontsize=7,
                color="darkgray",
            )

        # 3. 차량 그리기
        if vehicle_states is not None:
            for aid, state in vehicle_states.items():
                pos = state.position_2d(self.network)

                if state.agent_type == "police":
                    color = "blue"
                    marker = "s"
                    size = 120
                elif state.agent_type == "fugitive":
                    color = "red"
                    marker = "s"
                    size = 150
                else:
                    color = "purple"
                    marker = "o"
                    size = 80

                self.ax.scatter(
                    pos[0], pos[1],
                    c=color, marker=marker, s=size,
                    zorder=10, edgecolors="black", linewidths=1.0,
                )
                self.ax.annotate(
                    aid.replace("police_", "P").replace("fugitive", "F"),
                    (pos[0], pos[1]),
                    textcoords="offset points",
                    xytext=(8, 8),
                    fontsize=8,
                    fontweight="bold",
                    color=color,
                )

        # 4. 범례 및 제목
        legend_elements = [
            mpatches.Patch(color="blue", label="Police"),
            mpatches.Patch(color="red", label="Fugitive"),
            mpatches.Patch(color="green", label="Boundary"),
            mpatches.Patch(color="gray", label="Road"),
        ]
        self.ax.legend(handles=legend_elements, loc="upper right", fontsize=9)

        if title:
            self.ax.set_title(title, fontsize=12)
        else:
            self.ax.set_title(f"Road Pursuit Environment - Step {step}", fontsize=12)

        self.ax.set_aspect("equal")
        self.ax.grid(True, alpha=0.2)
        self.ax.set_xlabel("X (meters)")
        self.ax.set_ylabel("Y (meters)")

        self.fig.tight_layout()
        return self.fig

    def render_to_array(
        self,
        vehicle_states: dict[str, VehicleState] | None = None,
        step: int = 0,
    ) -> np.ndarray:
        """현재 상태를 RGB numpy 배열로 렌더링한다.

        Args:
            vehicle_states: 차량 상태
            step: 현재 스텝

        Returns:
            (H, W, 3) uint8 배열
        """
        fig = self.render(vehicle_states, step)
        fig.canvas.draw()
        buf = fig.canvas.buffer_rgba()
        img = np.asarray(buf)[:, :, :3].copy()
        return img

    def close(self) -> None:
        """시각화 리소스를 해제한다."""
        import matplotlib.pyplot as plt

        if self.fig is not None:
            plt.close(self.fig)
            self.fig = None
            self.ax = None

    def save(self, filepath: str, **kwargs: Any) -> None:
        """현재 그림을 파일로 저장한다.

        Args:
            filepath: 저장 경로
            **kwargs: plt.savefig에 전달할 추가 인자
        """
        if self.fig is not None:
            self.fig.savefig(filepath, **kwargs)
