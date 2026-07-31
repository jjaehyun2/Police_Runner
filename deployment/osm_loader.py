"""OpenStreetMap 도로 데이터 로더.

OSM 도로 네트워크를 RoadNetwork2D 호환 형식으로 변환한다.
osmnx 라이브러리를 사용하여 실제 도로망을 로드하고,
교차로(intersection)와 세그먼트(segment) 형태로 변환한다.
"""

from __future__ import annotations

import json
import logging
import math
from pathlib import Path

logger = logging.getLogger(__name__)

try:
    import osmnx as ox
    import networkx as nx

    OSMNX_AVAILABLE = True
except ImportError:
    OSMNX_AVAILABLE = False
    logger.warning(
        "osmnx가 설치되지 않았습니다. OSM 도로 로딩 기능을 사용하려면 "
        "'pip install osmnx' 명령으로 설치하세요."
    )


def _latlon_to_meters(lat: float, lon: float, ref_lat: float, ref_lon: float) -> tuple[float, float]:
    """위경도를 미터 단위 로컬 좌표로 변환한다 (간이 UTM 변환).

    Args:
        lat: 위도
        lon: 경도
        ref_lat: 기준 위도 (네트워크 중심)
        ref_lon: 기준 경도 (네트워크 중심)

    Returns:
        (x_meters, y_meters) 튜플
    """
    lat_rad = math.radians(ref_lat)
    meters_per_deg_lat = 111320.0
    meters_per_deg_lon = 111320.0 * math.cos(lat_rad)

    x = (lon - ref_lon) * meters_per_deg_lon
    y = (lat - ref_lat) * meters_per_deg_lat
    return (x, y)


class OSMRoadLoader:
    """OpenStreetMap 도로 데이터를 로드하여 RoadNetwork2D 형식으로 변환."""

    def __init__(self) -> None:
        if not OSMNX_AVAILABLE:
            logger.warning(
                "osmnx를 사용할 수 없습니다. "
                "설치 명령: pip install osmnx"
            )

    def load_from_place(self, place_name: str, network_type: str = "drive") -> dict:
        """도시/구역명으로 도로 네트워크를 로드한다.

        Args:
            place_name: "Daejeon, South Korea" 등의 지역명
            network_type: "drive" (차도만), "walk", "bike", "all" 등

        Returns:
            {"intersections": [...], "segments": [...], "boundary_intersections": [...]}

        Raises:
            ImportError: osmnx가 설치되지 않은 경우
            ValueError: 네트워크를 로드할 수 없는 경우
        """
        if not OSMNX_AVAILABLE:
            raise ImportError(
                "osmnx 라이브러리가 필요합니다. "
                "설치: pip install osmnx"
            )

        logger.info("OSM에서 도로 네트워크 로딩 중: %s", place_name)
        try:
            G = ox.graph_from_place(place_name, network_type=network_type)
        except Exception as e:
            raise ValueError(f"도로 네트워크 로딩 실패 ({place_name}): {e}") from e

        return self._convert_osm_to_network(G)

    def load_from_bbox(
        self, north: float, south: float, east: float, west: float, network_type: str = "drive"
    ) -> dict:
        """위경도 범위(bounding box)로 도로 네트워크를 로드한다.

        Args:
            north: 북쪽 경계 위도
            south: 남쪽 경계 위도
            east: 동쪽 경계 경도
            west: 서쪽 경계 경도
            network_type: 네트워크 유형

        Returns:
            {"intersections": [...], "segments": [...], "boundary_intersections": [...]}

        Raises:
            ImportError: osmnx가 설치되지 않은 경우
        """
        if not OSMNX_AVAILABLE:
            raise ImportError(
                "osmnx 라이브러리가 필요합니다. "
                "설치: pip install osmnx"
            )

        logger.info(
            "OSM에서 도로 네트워크 로딩 중: bbox(N=%.4f, S=%.4f, E=%.4f, W=%.4f)",
            north, south, east, west,
        )
        try:
            G = ox.graph_from_bbox(north, south, east, west, network_type=network_type)
        except Exception as e:
            raise ValueError(f"도로 네트워크 로딩 실패 (bbox): {e}") from e

        return self._convert_osm_to_network(G)

    def _convert_osm_to_network(self, G) -> dict:
        """osmnx 그래프를 RoadNetwork2D 호환 형식으로 변환한다.

        변환 규칙:
        - OSM 노드 → 교차로 (intersection_id, position(x, y))
        - OSM 엣지 → 세그먼트 (segment_id, start/end intersection, start/end pos, length)
        - 네트워크 경계의 노드(degree 1) → boundary_intersections

        Args:
            G: osmnx/networkx 그래프

        Returns:
            {
                "intersections": [
                    {"intersection_id": int, "position": [x, y],
                     "outgoing_segments": [int, ...], "incoming_segments": [int, ...]},
                    ...
                ],
                "segments": [
                    {"segment_id": int, "start_intersection_id": int,
                     "end_intersection_id": int, "start_pos": [x, y],
                     "end_pos": [x, y], "length": float},
                    ...
                ],
                "boundary_intersections": [int, ...],
                "metadata": {"num_intersections": int, "num_segments": int, "source": str}
            }
        """
        # OSM 노드 ID를 0부터의 연속 정수 ID로 매핑
        osm_nodes = list(G.nodes())
        node_id_map: dict[int, int] = {osm_id: idx for idx, osm_id in enumerate(osm_nodes)}

        # 기준점 (중심 좌표) 계산
        lats = [G.nodes[n].get("y", 0.0) for n in osm_nodes]
        lons = [G.nodes[n].get("x", 0.0) for n in osm_nodes]
        ref_lat = sum(lats) / len(lats) if lats else 0.0
        ref_lon = sum(lons) / len(lons) if lons else 0.0

        # 교차로 변환
        intersections: list[dict] = []
        node_positions: dict[int, tuple[float, float]] = {}

        for osm_id in osm_nodes:
            new_id = node_id_map[osm_id]
            lat = G.nodes[osm_id].get("y", 0.0)
            lon = G.nodes[osm_id].get("x", 0.0)
            x, y = _latlon_to_meters(lat, lon, ref_lat, ref_lon)
            node_positions[new_id] = (x, y)

            intersections.append({
                "intersection_id": new_id,
                "position": [x, y],
                "outgoing_segments": [],
                "incoming_segments": [],
            })

        # 세그먼트 변환
        segments: list[dict] = []
        segment_id = 0

        for u, v, data in G.edges(data=True):
            start_id = node_id_map[u]
            end_id = node_id_map[v]
            start_pos = node_positions[start_id]
            end_pos = node_positions[end_id]

            # 길이: OSM 데이터에 있으면 사용, 없으면 유클리드 거리
            length = data.get("length", 0.0)
            if length == 0.0:
                dx = end_pos[0] - start_pos[0]
                dy = end_pos[1] - start_pos[1]
                length = math.sqrt(dx * dx + dy * dy)

            seg = {
                "segment_id": segment_id,
                "start_intersection_id": start_id,
                "end_intersection_id": end_id,
                "start_pos": list(start_pos),
                "end_pos": list(end_pos),
                "length": length,
            }
            segments.append(seg)

            # outgoing/incoming 업데이트
            intersections[start_id]["outgoing_segments"].append(segment_id)
            intersections[end_id]["incoming_segments"].append(segment_id)

            segment_id += 1

        # 경계 교차로 식별: degree가 1인 노드 (막다른 길 끝)
        # 또는 네트워크 가장자리에 있는 노드
        boundary_intersections: list[int] = []
        for osm_id in osm_nodes:
            new_id = node_id_map[osm_id]
            # undirected degree가 1이면 경계로 간주
            in_deg = G.in_degree(osm_id) if G.is_directed() else 0
            out_deg = G.out_degree(osm_id) if G.is_directed() else 0
            total_deg = in_deg + out_deg if G.is_directed() else G.degree(osm_id)

            if total_deg <= 2:
                boundary_intersections.append(new_id)

        # 경계가 없으면 외곽 노드들로 지정 (위치 기반)
        if not boundary_intersections and intersections:
            all_x = [pos[0] for pos in node_positions.values()]
            all_y = [pos[1] for pos in node_positions.values()]
            x_range = max(all_x) - min(all_x) if all_x else 1.0
            y_range = max(all_y) - min(all_y) if all_y else 1.0
            threshold = 0.1  # 경계 10% 범위

            for new_id, (x, y) in node_positions.items():
                rel_x = (x - min(all_x)) / x_range if x_range > 0 else 0.5
                rel_y = (y - min(all_y)) / y_range if y_range > 0 else 0.5
                if rel_x < threshold or rel_x > (1 - threshold) or rel_y < threshold or rel_y > (1 - threshold):
                    boundary_intersections.append(new_id)

        network_data = {
            "intersections": intersections,
            "segments": segments,
            "boundary_intersections": sorted(set(boundary_intersections)),
            "metadata": {
                "num_intersections": len(intersections),
                "num_segments": len(segments),
                "source": "OpenStreetMap via osmnx",
                "ref_lat": ref_lat,
                "ref_lon": ref_lon,
            },
        }

        logger.info(
            "네트워크 변환 완료: 교차로 %d개, 세그먼트 %d개, 경계 %d개",
            len(intersections), len(segments), len(boundary_intersections),
        )
        return network_data

    def save_network(self, network_data: dict, filepath: str) -> None:
        """변환된 네트워크를 JSON으로 저장한다.

        Args:
            network_data: _convert_osm_to_network 결과
            filepath: 저장 경로 (.json)
        """
        path = Path(filepath)
        path.parent.mkdir(parents=True, exist_ok=True)

        with open(path, "w", encoding="utf-8") as f:
            json.dump(network_data, f, ensure_ascii=False, indent=2)

        logger.info("네트워크 저장 완료: %s", filepath)

    def load_network(self, filepath: str) -> dict:
        """저장된 네트워크 JSON을 로드한다.

        Args:
            filepath: JSON 파일 경로

        Returns:
            네트워크 데이터 딕셔너리

        Raises:
            FileNotFoundError: 파일이 존재하지 않는 경우
        """
        path = Path(filepath)
        if not path.exists():
            raise FileNotFoundError(f"네트워크 파일을 찾을 수 없습니다: {filepath}")

        with open(path, "r", encoding="utf-8") as f:
            network_data = json.load(f)

        logger.info(
            "네트워크 로드 완료: 교차로 %d개, 세그먼트 %d개",
            len(network_data.get("intersections", [])),
            len(network_data.get("segments", [])),
        )
        return network_data
