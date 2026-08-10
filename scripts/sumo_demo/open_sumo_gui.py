"""SUMO 자체 창(SUMO-GUI)으로 열기.

관제 화면(브라우저)과 달리 이것은 SUMO가 직접 그리는 교통공학용 뷰다.
차량이 실제 축척이라 도시 전체를 보면 점처럼 작지만, 확대하면 차선·신호
현시·차간거리 같은 물리를 그대로 볼 수 있다. 발표용으로는 관제 화면을,
"진짜 교통 시뮬레이터 위에서 돌고 있다"를 보여줄 때는 이쪽을 쓴다.

두 가지 모드가 있다.

  --live     추격까지 함께 재생한다(러너가 SUMO-GUI를 띄우고 제어).
  (기본)      배경 교통과 차단 구간만 있는 지도를 연다. 추격 차량은 없다.

사용:
    py -3.12 scripts\\sumo_demo\\open_sumo_gui.py
    py -3.12 scripts\\sumo_demo\\open_sumo_gui.py --live --barriers 8
"""
from __future__ import annotations

import argparse
from pathlib import Path
import subprocess
import sys

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from pursuit_evasion_rl.sumo_env.barriers import (  # noqa: E402
    BarrierConfig, choose_barriers, write_poi_additional,
)
from pursuit_evasion_rl.sumo_env.net_builder import (  # noqa: E402
    build_network, largest_cached_snapshot, sumo_binary,
)
from pursuit_evasion_rl.sumo_env.traffic import TrafficConfig, write_background_routes  # noqa: E402

OUT_DIR = REPO_ROOT / "scripts" / "sumo_demo" / "out"


def main() -> int:
    parser = argparse.ArgumentParser(description="SUMO-GUI로 대전 도로망 열기")
    parser.add_argument("--live", action="store_true",
                        help="추격까지 함께 재생 (run_demo --gui 와 동일)")
    parser.add_argument("--background", type=int, default=300)
    parser.add_argument("--barriers", type=int, default=8)
    parser.add_argument("--seed", type=int, default=12)
    parser.add_argument("--delay", type=int, default=150, help="스텝당 ms")
    parser.add_argument("--barrier-image", default=None, help="차단 아이콘 PNG")
    args = parser.parse_args()

    network = build_network(largest_cached_snapshot(REPO_ROOT / "cache"),
                            output_root=REPO_ROOT / "cache/sumo")
    print("network: %s" % network.net_path)
    print("  edges=%d nodes=%d traffic_lights=%d"
          % (network.edge_count, network.node_count, network.traffic_light_count))

    if args.live:
        # 추격까지 보려면 러너가 SUMO를 제어해야 한다. 여기서 직접 띄우면
        # TraCI 연결이 둘로 갈려 어느 쪽도 제대로 돌지 않는다.
        command = [sys.executable, str(REPO_ROOT / "scripts/sumo_demo/run_demo.py"),
                   "--gui", "--episodes", "1", "--realtime", str(args.delay / 1000.0),
                   "--background", str(args.background), "--barriers", str(args.barriers),
                   "--seed", str(args.seed)]
        print("추격 재생 모드 — run_demo 로 위임합니다")
        return subprocess.call(command, cwd=str(REPO_ROOT))

    import sumolib

    net = sumolib.net.readNet(str(network.net_path))
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    route_path = OUT_DIR / f"gui_background_{args.seed}.rou.xml"
    write_background_routes(net, route_path,
                            TrafficConfig(vehicle_count=args.background, seed=args.seed))
    barriers = choose_barriers(net, BarrierConfig(count=args.barriers,
                                                  enabled=args.barriers > 0),
                               seed=args.seed + 4231)
    poi_path = OUT_DIR / "gui_barriers.add.xml"
    write_poi_additional(barriers, poi_path, image_file=args.barrier_image)
    print("background=%d  barriers=%d" % (args.background, len(barriers)))

    command = [sumo_binary("sumo-gui"),
               "-n", str(network.net_path),
               "-r", str(route_path),
               "-a", str(poi_path),
               "--delay", str(args.delay),
               "--start", "true",
               "--no-warnings", "true",
               "--ignore-route-errors", "true",
               "--window-size", "1400,900"]
    print("여는 중… 창이 뜨면 마우스 휠로 확대하세요.")
    return subprocess.call(command, cwd=str(REPO_ROOT))


if __name__ == "__main__":
    raise SystemExit(main())