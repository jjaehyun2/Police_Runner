"""실제 OSM 도로 추격 데모 시각화 (LIVE, 외부 네트워크 접속).

OSMnx로 대전 중심부 실도로 bbox를 실제 조회 → 결정노드 코어싱/차수 분할 →
경찰 6대 협력 추격/차단 정책으로 에피소드 실행 → matplotlib 프레임/요약 저장.

시나리오 설계(포위가 실제로 보이도록):
- FIX 1: 도주자 시야(200m) >> 체포반경(40m) 이라 도주자가 실제로 회피한다.
- FIX 2: 협력 추격/차단 정책(InterceptPolicePolicy) — 경찰이 수렴하며 탈출로를 막는다.
- FIX 3: 필드 크기를 경찰 도달 범위에 맞추고(약 0.8km), 경찰을 도주자 '주위'에 배치하며,
         도주자 속도를 낮춰(8m/s) 즉시 탈출을 막는다. 그래야 포위가 성립한다.

주의: 실행 시 OpenStreetMap(Overpass)에 실제 요청을 보낸다(opt-in).
"""

from __future__ import annotations

import math
from pathlib import Path

from pursuit_evasion_rl.osm_demo.coarsening import coarsen_raw_graph, prepare_model_network
from pursuit_evasion_rl.osm_demo.metrics import placement_position, summarize_metrics
from pursuit_evasion_rl.osm_demo.models import BoundedArea, EpisodeConfig, VehiclePlacement
from pursuit_evasion_rl.osm_demo.osm_source import OSMnxSource
from pursuit_evasion_rl.osm_demo.rendering import FontChoice, OSMRenderer
from pursuit_evasion_rl.osm_demo.runner import (
    default_fugitive_factory,
    run_batch,
    run_single_episode,
)
from demo_intercept_policy import InterceptPolicePolicy

OUT = Path("out_osm_real")
FUGITIVE_VISION_M = 200.0


def surrounding_placement(net, band=(150.0, 500.0)):
    """도주자를 중앙 교차로에, 경찰 6대를 그 주위 6방향(60도 간격)으로 배치."""
    pos = {i.id: i.position_xy for i in net.intersections}
    drivable = [i.id for i in net.intersections if not i.virtual and i.outgoing_segment_ids]
    cx = sum(pos[i][0] for i in drivable) / len(drivable)
    cy = sum(pos[i][1] for i in drivable) / len(drivable)
    fug = min(drivable, key=lambda i: math.dist(pos[i], (cx, cy)))
    fx, fy = pos[fug]
    used = {fug}
    police = []
    for s in range(6):
        target = math.radians(60 * s)
        best, best_score = None, math.inf
        for i in drivable:
            if i in used:
                continue
            dx, dy = pos[i][0] - fx, pos[i][1] - fy
            dist = math.hypot(dx, dy)
            if not (band[0] <= dist <= band[1]):
                continue
            ang = math.atan2(dy, dx)
            diff = abs(math.atan2(math.sin(ang - target), math.cos(ang - target)))
            if diff < best_score:
                best_score, best = diff, i
        if best is None:
            best = min((i for i in drivable if i not in used), key=lambda i: math.dist(pos[i], (fx, fy)))
        used.add(best)
        police.append(VehiclePlacement(intersection_id=best))
    return police, VehiclePlacement(intersection_id=fug)


def main() -> None:
    OUT.mkdir(exist_ok=True)
    # FIX 3: 경찰 도달 범위에 맞춘 약 0.8km 중심부 영역
    area = BoundedArea(name="daejeon-core", north=36.359, south=36.351,
                       east=127.397, west=127.388, max_area_km2=10.0)
    print(f"[area] {area.name}  bbox N={area.north} S={area.south} E={area.east} W={area.west}")

    print("[fetch] querying real OpenStreetMap roads via OSMnx ...")
    acquisition = OSMnxSource().fetch_with_metadata(area, network_type="drive")
    raw = acquisition.graph
    print(f"[fetch] raw nodes={len(raw.nodes)} directed edges={len(raw.edges)} "
          f"status={acquisition.metadata.operation_status} crs={acquisition.metadata.metric_crs}")

    prepared = prepare_model_network(coarsen_raw_graph(raw))
    network = prepared.network
    print(f"[prepare] intersections={len(network.intersections)} segments={len(network.segments)} "
          f"max_out_degree={prepared.statistics.max_out_degree}")

    config = EpisodeConfig(dt_s=1.0, police_speed_mps=14.0, fugitive_speed_mps=8.0,
                           capture_radius_m=40.0, max_steps=120)
    policy = InterceptPolicePolicy(network)                       # FIX 2
    fugitive_factory = default_fugitive_factory(vision_range_m=FUGITIVE_VISION_M)  # FIX 1
    placement = surrounding_placement(network, band=(150.0, 500.0))  # FIX 3
    print(f"[scenario] police_speed=14 fugitive_speed=8 vision=200 capture=40 (surrounding placement)")

    renderer = OSMRenderer(network, config, font=FontChoice("DejaVu Sans", False))

    record = run_single_episode(network, config, policy, run_id="osm-real", run_seed=21,
                                fugitive_factory=fugitive_factory, placement=placement)
    print(f"[episode] outcome={record.outcome.value} steps={len(record.transitions)}")

    # 포위 여부를 수치로: 경찰-도주자 평균거리 시작 vs 끝
    sts = [record.initial_state] + [t.state for t in record.transitions]

    def avg_dist(st):
        f = placement_position(network, st.fugitive)
        return sum(math.dist(placement_position(network, p), f) for p in st.police) / 6

    print(f"[converge] avg police-fugitive distance: start={avg_dist(sts[0]):.0f}m end={avg_dist(sts[-1]):.0f}m")

    frames = renderer.render_episode_frames(record, max_frames=60)
    for i, fig in enumerate(frames):
        renderer.save_figure(fig, str(OUT / f"frame_{i:03d}.png"))
    print(f"[frames] {len(frames)} PNG saved")

    batch = run_batch(network, config, policy, run_id="osm-real-batch", run_seed=200,
                      episode_count=20, fugitive_factory=fugitive_factory, placement=placement)
    summary = summarize_metrics(batch, network, policy_kind="intercept", minimum_evaluation_episodes=1)
    print(f"[batch] outcomes={dict(summary.outcomes)} rates={dict(summary.rates)}")
    renderer.save_figure(renderer.render_summary(summary, representative_record=batch[0]),
                         str(OUT / "summary.png"))


if __name__ == "__main__":
    main()
