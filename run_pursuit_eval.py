"""포위 추격 알고리즘 자동 평가 하네스 (풀 2.2km 실도로, 내부 추격).

OSM을 한 번만 받아 캐시(pickle)하고, 여러 시드로 에피소드를 돌려
검거율 / 소요 스텝 / 도주자 진동 / 포위 품질(둘러싼 각도 커버리지)을 측정한다.

사용:
  python run_pursuit_eval.py            # 기본 파라미터로 평가
  python run_pursuit_eval.py --seeds 40 --max-steps 1000 --capture 20 \
      --fugitive-speed 10 --police-speed 14 --vision 120 --n-chase 2 --lead 18
"""

from __future__ import annotations

import argparse
import copyreg
import math
import pickle
import statistics
from pathlib import Path
from types import MappingProxyType

import numpy as np

# 프로즌 도메인 모델의 MappingProxyType 필드를 pickle 가능하게 등록
copyreg.pickle(MappingProxyType, lambda p: (MappingProxyType, (dict(p),)))

from pursuit_evasion_rl.osm_demo.coarsening import coarsen_raw_graph, prepare_model_network
from pursuit_evasion_rl.osm_demo.metrics import placement_position
from pursuit_evasion_rl.osm_demo.models import BoundedArea, EpisodeConfig, EpisodeOutcome, VehiclePlacement
from pursuit_evasion_rl.osm_demo.osm_source import OSMnxSource
from pursuit_evasion_rl.osm_demo.presets import DAEJEON_DRIVE_PRESET
from pursuit_evasion_rl.osm_demo.runner import run_single_episode

from demo_pursuit import EncirclementPolice, SmartEvader, _Graph, make_interior_network

CACHE = Path("out_pursuit/interior_net.pkl")


def load_network() -> object:
    CACHE.parent.mkdir(parents=True, exist_ok=True)
    if CACHE.exists():
        return pickle.loads(CACHE.read_bytes())
    area = DAEJEON_DRIVE_PRESET.area  # 풀 2.2km
    print(f"[fetch] live OSM {area.name} (한 번만) ...")
    raw = OSMnxSource().fetch_with_metadata(area, "drive").graph
    net = make_interior_network(prepare_model_network(coarsen_raw_graph(raw)).network)
    CACHE.write_bytes(pickle.dumps(net))
    print(f"[fetch] cached interior network -> {CACHE}")
    return net


def surround_placement(graph, rng, band=(400.0, 950.0)):
    """도주자를 (약간 무작위) 중앙 근처에, 경찰 6대를 그 주위 6방향 링에 배치(dragnet)."""
    pos = graph.pos
    drivable = [i for i in pos if graph.fwd.get(i)]
    cx = sum(pos[i][0] for i in drivable) / len(drivable)
    cy = sum(pos[i][1] for i in drivable) / len(drivable)
    # 중앙 근처 무작위 도주자
    center_pool = sorted(drivable, key=lambda i: math.dist(pos[i], (cx, cy)))[:20]
    fug = center_pool[int(rng.integers(0, len(center_pool)))]
    fx, fy = pos[fug]
    used = {fug}
    police = []
    for s in range(6):
        target = math.radians(60 * s + float(rng.uniform(-15, 15)))
        best, best_score = None, math.inf
        for i in drivable:
            if i in used:
                continue
            dx, dy = pos[i][0] - fx, pos[i][1] - fy
            d = math.hypot(dx, dy)
            if not (band[0] <= d <= band[1]):
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


def evaluate(args) -> dict:
    net = load_network()
    graph = _Graph(net)  # 시드 간 공유(거리 캐시 재사용)
    pos = graph.pos
    cfg = EpisodeConfig(dt_s=1.0, police_speed_mps=args.police_speed,
                        fugitive_speed_mps=args.fugitive_speed,
                        capture_radius_m=args.capture, max_steps=args.max_steps)

    captures = 0
    steps_to_capture: list[int] = []
    revisit_rates: list[float] = []
    coverage_at_end: list[float] = []   # 포위 각도 커버리지(1=완전 포위)
    near_counts: list[int] = []         # 종료 시 200m 이내 경찰 수

    for seed in range(args.seeds):
        policy = EncirclementPolice(net, graph=graph, fugitive_speed_mps=args.fugitive_speed,
                                    police_speed_mps=args.police_speed, lead_time_s=args.lead)
        placement = None
        if args.surround:
            placement = surround_placement(graph, np.random.default_rng(1000 + seed))
        rec = run_single_episode(
            net, cfg, policy, run_id=f"eval{seed}", run_seed=seed,
            fugitive_factory=lambda rng: SmartEvader(net, rng=rng, graph=graph, vision_range_m=args.vision),
            placement=placement,
        )
        sts = [rec.initial_state] + [t.state for t in rec.transitions]
        if rec.outcome is EpisodeOutcome.CAPTURE:
            captures += 1
            steps_to_capture.append(len(rec.transitions))

        # 도주자 진동율: 도주자가 밟은 세그먼트 시퀀스에서 직전 세그먼트의 역방향으로
        # 되돌아간(=왔던 길로 U턴) 비율
        segs = []
        for t in rec.transitions:
            for ev in t.events:
                if ev.startswith("fugitive:depart:"):
                    segs.append(int(ev.rsplit(":", 1)[1]))
        seg = {s.id: s for s in net.segments}
        back = 0
        for i in range(1, len(segs)):
            a, b = seg[segs[i - 1]], seg[segs[i]]
            if b.start_id == a.end_id and b.end_id == a.start_id:  # 역주행 U턴
                back += 1
        revisit_rates.append(back / max(1, len(segs) - 1))

        # 종료 시 포위 품질: 도주자 기준 경찰들의 방위각 분포(최대 각도 간극이 작을수록 포위)
        last = sts[-1]
        f = placement_position(net, last.fugitive)
        bearings = sorted(math.atan2(placement_position(net, p)[1] - f[1],
                                     placement_position(net, p)[0] - f[0]) for p in last.police)
        gaps = [(bearings[(i + 1) % 6] - bearings[i]) % (2 * math.pi) for i in range(6)]
        max_gap = max(gaps)
        coverage_at_end.append(1.0 - max_gap / (2 * math.pi))  # 1이면 완전 포위, 0.83이면 한 방향 뚫림
        near_counts.append(sum(1 for p in last.police
                               if math.dist(placement_position(net, p), f) <= 200.0))

    cap_rate = captures / args.seeds
    result = {
        "seeds": args.seeds,
        "capture_rate": round(cap_rate, 3),
        "median_steps": int(statistics.median(steps_to_capture)) if steps_to_capture else None,
        "mean_steps": round(statistics.mean(steps_to_capture), 1) if steps_to_capture else None,
        "mean_revisit_rate": round(statistics.mean(revisit_rates), 3),
        "mean_encircle_coverage": round(statistics.mean(coverage_at_end), 3),
        "mean_near_count(<=200m)": round(statistics.mean(near_counts), 2),
    }
    return result


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=30)
    ap.add_argument("--max-steps", type=int, default=1000)
    ap.add_argument("--capture", type=float, default=20.0)
    ap.add_argument("--fugitive-speed", type=float, default=10.0)
    ap.add_argument("--police-speed", type=float, default=14.0)
    ap.add_argument("--vision", type=float, default=120.0)
    ap.add_argument("--n-chase", type=int, default=2)
    ap.add_argument("--lead", type=float, default=18.0)
    ap.add_argument("--surround", action="store_true", help="경찰을 도주자 주위 링에 배치(dragnet)")
    args = ap.parse_args()

    print(f"[params] seeds={args.seeds} max_steps={args.max_steps} capture={args.capture} "
          f"fug_speed={args.fugitive_speed} pol_speed={args.police_speed} vision={args.vision} "
          f"n_chase={args.n_chase} lead={args.lead}")
    result = evaluate(args)
    print("[result]")
    for k, v in result.items():
        print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
