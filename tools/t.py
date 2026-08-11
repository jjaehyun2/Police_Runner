# -*- coding: utf-8 -*-
"""실제 대전 OSM 도로망 기반 포위 작전 시각자료 생성 스크립트.

Gemini 생성 가상 지도 대신 실제 OpenStreetMap 대전 둔산동 도로망 위에
순찰차·도주차량·추천 경로를 그려 제안서 [그림]으로 사용한다.

실행 (팀 로컬 PC에서):
    pip install osmnx matplotlib
    python make_osm_figure.py
출력: osm_daejeon_pursuit.png
"""
from pathlib import Path as _ImgPath
_IMAGE_DIR = _ImgPath(__file__).resolve().parents[1] / "docs" / "images"
import osmnx as ox
import networkx as nx
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

# 한글 폰트 (Windows: 맑은 고딕 / 없으면 시스템에 맞게 교체)
matplotlib.rcParams["font.family"] = "Malgun Gothic"
matplotlib.rcParams["axes.unicode_minus"] = False

# 1) 대전시청(둔산동) 중심 반경 900m 차량 도로망 로드 — 실제 OSM 데이터
CENTER = (36.3504, 127.3845)  # 대전시청
G = ox.graph_from_point(CENTER, dist=900, network_type="drive")
G = ox.convert.to_undirected(G)

# 2) 도주차량 위치(시청네거리 인근)와 순찰차 5대 초기 위치 지정
fugitive_pt = (36.3512, 127.3860)
police_pts = [
    (36.3560, 127.3800),  # P1 북서
    (36.3555, 127.3915),  # P2 북동
    (36.3455, 127.3925),  # P3 남동
    (36.3450, 127.3790),  # P4 남서
    (36.3500, 127.3745),  # P5 서측
]
fug_node = ox.distance.nearest_nodes(G, fugitive_pt[1], fugitive_pt[0])
police_nodes = [ox.distance.nearest_nodes(G, p[1], p[0]) for p in police_pts]

# 3) 포위 지점: 도주차량 주변 주요 교차로(도주 가능 경로의 길목) — 각 순찰차의 목표
#    간단화를 위해 도주 노드에서 3~4홉 떨어진 서로 다른 방향의 교차로를 자동 선택
rings = nx.single_source_shortest_path_length(G, fug_node, cutoff=4)
targets = [n for n, d in rings.items() if d == 4][:5]
routes = [nx.shortest_path(G, p, t, weight="length") for p, t in zip(police_nodes, targets)]

# 4) 그리기
fig, ax = ox.plot_graph_routes(
    G, routes, route_colors=["#1f4fb0"] * len(routes), route_linewidth=3.5,
    node_size=0, edge_color="#c8ccd4", edge_linewidth=0.8,
    bgcolor="white", show=False, close=False, figsize=(13, 9),
)
# 도주차량 · 순찰차 마커
ax.scatter(G.nodes[fug_node]["x"], G.nodes[fug_node]["y"], s=340, c="#c1170e",
           marker="s", zorder=5, edgecolors="white", linewidths=1.5, label="도주차량")
for i, n in enumerate(police_nodes):
    ax.scatter(G.nodes[n]["x"], G.nodes[n]["y"], s=280, c="#1f4fb0", marker="s",
               zorder=5, edgecolors="white", linewidths=1.5,
               label="순찰차" if i == 0 else None)
for i, t in enumerate(targets):
    ax.scatter(G.nodes[t]["x"], G.nodes[t]["y"], s=200, c="#0f7b6c", marker="X",
               zorder=5, label="목표 차단 지점" if i == 0 else None)

fp = fm.FontProperties(family="Malgun Gothic")
ax.set_title("실제 대전 둔산동 OSM 도로망 — 엔진 추천 차단 지점·이동 경로 (osm_loader 실데이터)",
             fontproperties=fp, fontsize=15)
ax.legend(prop=fp, loc="lower right", fontsize=11)
plt.savefig(_IMAGE_DIR / "osm_daejeon_pursuit.png", dpi=200, bbox_inches="tight", facecolor="white")
print(f"saved: {_IMAGE_DIR / 'osm_daejeon_pursuit.png'}")