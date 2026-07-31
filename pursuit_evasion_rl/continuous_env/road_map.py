"""연속 2D 도로 네트워크 관리 모듈.

RoadSegment, Intersection 데이터클래스와 RoadMap 클래스를 제공한다.
직선 세그먼트 기반의 격자형/랜덤 도로 생성 및 공간 쿼리를 지원한다.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class RoadSegment:
    """도로 구간 정의."""

    segment_id: int
    start_point: tuple[float, float]
    end_point: tuple[float, float]
    control_points: list[tuple[float, float]] | None = None
    width: float = 3.0
    speed_limit: float = 10.0


@dataclass
class Intersection:
    """교차점 정의."""

    position: tuple[float, float]
    connected_segments: list[int] = field(default_factory=list)


class RoadMap:
    """연속 2D 도로 네트워크 관리."""

    def __init__(self, config: dict | None = None) -> None:
        self.segments: list[RoadSegment] = []
        self.intersections: list[Intersection] = []
        self._config = config or {}
        self.map_bounds: tuple[float, float, float, float] = (0.0, 0.0, 100.0, 100.0)

    def generate_grid(self, rows: int, cols: int, spacing: float) -> None:
        """격자형 도로 네트워크를 생성한다.

        수평 도로(rows+1개)와 수직 도로(cols+1개)를 spacing 간격으로 배치한다.
        """
        self.segments.clear()
        self.intersections.clear()

        width = self._config.get("segment_width", 3.0)
        speed_limit = self._config.get("speed_limit", 10.0)

        max_x = cols * spacing
        max_y = rows * spacing
        self.map_bounds = (0.0, 0.0, max_x, max_y)

        seg_id = 0

        # 수평 도로: 각 행에 대해 좌→우 세그먼트
        for r in range(rows + 1):
            y = r * spacing
            for c in range(cols):
                x_start = c * spacing
                x_end = (c + 1) * spacing
                self.segments.append(
                    RoadSegment(
                        segment_id=seg_id,
                        start_point=(x_start, y),
                        end_point=(x_end, y),
                        width=width,
                        speed_limit=speed_limit,
                    )
                )
                seg_id += 1

        # 수직 도로: 각 열에 대해 하→상 세그먼트
        for c in range(cols + 1):
            x = c * spacing
            for r in range(rows):
                y_start = r * spacing
                y_end = (r + 1) * spacing
                self.segments.append(
                    RoadSegment(
                        segment_id=seg_id,
                        start_point=(x, y_start),
                        end_point=(x, y_end),
                        width=width,
                        speed_limit=speed_limit,
                    )
                )
                seg_id += 1

        # 교차점 식별: 격자의 모든 교차 지점
        for r in range(rows + 1):
            for c in range(cols + 1):
                pos = (c * spacing, r * spacing)
                connected = self._find_connected_segments(pos)
                self.intersections.append(Intersection(position=pos, connected_segments=connected))

    def generate_random(self, num_segments: int, seed: int | None = None) -> None:
        """랜덤 도로 네트워크를 생성한다."""
        self.segments.clear()
        self.intersections.clear()

        rng = np.random.default_rng(seed)
        min_x, min_y, max_x, max_y = self.map_bounds
        width = self._config.get("segment_width", 3.0)
        speed_limit = self._config.get("speed_limit", 10.0)

        points: list[tuple[float, float]] = []
        # 맵 경계에 몇 개의 점을 추가
        num_points = num_segments + 1
        for _ in range(num_points):
            x = float(rng.uniform(min_x, max_x))
            y = float(rng.uniform(min_y, max_y))
            points.append((x, y))

        # 인접한 점들을 연결하여 세그먼트 생성
        for i in range(num_segments):
            j = (i + 1) % num_points
            self.segments.append(
                RoadSegment(
                    segment_id=i,
                    start_point=points[i],
                    end_point=points[j],
                    width=width,
                    speed_limit=speed_limit,
                )
            )

        # 교차점 식별
        self._identify_intersections()

    def is_on_road(self, position: tuple[float, float], tolerance: float = 3.0) -> bool:
        """주어진 위치가 도로 위에 있는지 확인한다."""
        px, py = position
        for seg in self.segments:
            dist = self._point_to_segment_distance(px, py, seg)
            if dist <= tolerance:
                return True
        return False

    def nearest_point_on_road(self, position: tuple[float, float]) -> tuple[float, float]:
        """주어진 위치에서 가장 가까운 도로 위 지점을 반환한다."""
        px, py = position
        best_point: tuple[float, float] = (px, py)
        best_dist = float("inf")

        for seg in self.segments:
            proj = self._project_point_on_segment(px, py, seg)
            dist = np.hypot(px - proj[0], py - proj[1])
            if dist < best_dist:
                best_dist = dist
                best_point = proj

        return best_point

    def get_segment_at(self, position: tuple[float, float]) -> RoadSegment | None:
        """주어진 위치를 포함하는 도로 세그먼트를 반환한다."""
        px, py = position
        best_seg: RoadSegment | None = None
        best_dist = float("inf")

        for seg in self.segments:
            dist = self._point_to_segment_distance(px, py, seg)
            if dist <= seg.width and dist < best_dist:
                best_dist = dist
                best_seg = seg

        return best_seg

    def get_boundary_exits(self) -> list[tuple[float, float]]:
        """맵 경계에 위치한 도로 끝점(출구)을 반환한다."""
        min_x, min_y, max_x, max_y = self.map_bounds
        exits: list[tuple[float, float]] = []
        tol = 1e-6

        for seg in self.segments:
            for point in [seg.start_point, seg.end_point]:
                px, py = point
                if (
                    abs(px - min_x) < tol
                    or abs(px - max_x) < tol
                    or abs(py - min_y) < tol
                    or abs(py - max_y) < tol
                ):
                    if point not in exits:
                        exits.append(point)

        return exits

    def get_segments(self) -> list[RoadSegment]:
        """모든 도로 세그먼트를 반환한다."""
        return self.segments

    def get_intersections(self) -> list[Intersection]:
        """모든 교차점을 반환한다."""
        return self.intersections

    def _find_connected_segments(self, pos: tuple[float, float], tol: float = 1e-6) -> list[int]:
        """주어진 위치에 연결된 세그먼트 ID 목록을 반환한다."""
        connected: list[int] = []
        for seg in self.segments:
            if (
                abs(seg.start_point[0] - pos[0]) < tol
                and abs(seg.start_point[1] - pos[1]) < tol
            ) or (
                abs(seg.end_point[0] - pos[0]) < tol
                and abs(seg.end_point[1] - pos[1]) < tol
            ):
                connected.append(seg.segment_id)
        return connected

    def _identify_intersections(self, tol: float = 1e-6) -> None:
        """세그먼트 끝점을 분석하여 교차점을 식별한다."""
        point_map: dict[tuple[float, float], list[int]] = {}

        for seg in self.segments:
            for pt in [seg.start_point, seg.end_point]:
                rounded = (round(pt[0], 4), round(pt[1], 4))
                if rounded not in point_map:
                    point_map[rounded] = []
                point_map[rounded].append(seg.segment_id)

        for pos, seg_ids in point_map.items():
            if len(seg_ids) >= 2:
                self.intersections.append(
                    Intersection(position=pos, connected_segments=seg_ids)
                )

    @staticmethod
    def _point_to_segment_distance(
        px: float, py: float, seg: RoadSegment
    ) -> float:
        """점에서 직선 세그먼트까지의 최단 거리를 계산한다."""
        sx, sy = seg.start_point
        ex, ey = seg.end_point

        dx = ex - sx
        dy = ey - sy
        seg_len_sq = dx * dx + dy * dy

        if seg_len_sq < 1e-12:
            return float(np.hypot(px - sx, py - sy))

        t = max(0.0, min(1.0, ((px - sx) * dx + (py - sy) * dy) / seg_len_sq))
        proj_x = sx + t * dx
        proj_y = sy + t * dy

        return float(np.hypot(px - proj_x, py - proj_y))

    @staticmethod
    def _project_point_on_segment(
        px: float, py: float, seg: RoadSegment
    ) -> tuple[float, float]:
        """점을 직선 세그먼트 위에 투영한다."""
        sx, sy = seg.start_point
        ex, ey = seg.end_point

        dx = ex - sx
        dy = ey - sy
        seg_len_sq = dx * dx + dy * dy

        if seg_len_sq < 1e-12:
            return (sx, sy)

        t = max(0.0, min(1.0, ((px - sx) * dx + (py - sy) * dy) / seg_len_sq))
        return (sx + t * dx, sy + t * dy)
