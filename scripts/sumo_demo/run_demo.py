"""Run the SUMO pursuit demo and stream state to the control-room dashboard.

Usage:
    py -3.12 scripts/sumo_demo/run_demo.py                 # headless
    py -3.12 scripts/sumo_demo/run_demo.py --gui           # with SUMO-GUI
    py -3.12 scripts/sumo_demo/run_demo.py --episodes 5    # batch, prints a summary

The dashboard reads ``scripts/sumo_demo/out/state.json``; this writer is the
only producer, so the two processes stay decoupled.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import statistics
import sys
import time

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from pursuit_evasion_rl.sumo_env.environment import (  # noqa: E402
    POLICE_IDS, EpisodeOutcome, SumoEpisodeConfig, SumoPursuitEnv,
)
from pursuit_evasion_rl.sumo_env.net_builder import (  # noqa: E402
    build_network, largest_cached_snapshot,
)
from pursuit_evasion_rl.sumo_env.policies import EncirclementPolicy  # noqa: E402
from pursuit_evasion_rl.sumo_env.barriers import write_poi_additional  # noqa: E402
from pursuit_evasion_rl.sumo_env.scene import export_network, scene_frame  # noqa: E402
from pursuit_evasion_rl.sumo_env.barriers import BarrierConfig  # noqa: E402
from pursuit_evasion_rl.sumo_env.traffic import TrafficConfig  # noqa: E402

OUT_DIR = Path(__file__).resolve().parent / "out"
STATE_PATH = OUT_DIR / "state.json"
NETWORK_PATH = OUT_DIR / "network.json"
BARRIER_POI_PATH = OUT_DIR / "barriers.add.xml"
SCENE_PATH = OUT_DIR / "scene.json"

OUTCOME_LABEL = {
    EpisodeOutcome.CAPTURE: "captured",
    EpisodeOutcome.ESCAPE: "escaped",
    EpisodeOutcome.TIMEOUT: "timeout",
    EpisodeOutcome.VOID: "void",
}
ACTION_LABEL = {True: "이동", False: "대기"}

#: An officer within this distance of an exit is treated as blocking it.
BLOCK_RADIUS_M = 80.0


def _publish(path: Path, payload: str, attempts: int = 6) -> None:
    """Replace a published file as atomically as Windows allows.

    POSIX lets you rename over an open file; Windows refuses with
    PermissionError while the dashboard has the target open for reading.
    With the runner writing every frame and the dashboard polling twice a
    second, that collision is routine rather than exceptional, so retry
    briefly and then fall back to writing in place.  A torn read is
    recoverable -- the dashboard already treats invalid JSON as "no update
    this frame" -- but a crashed episode is not.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(payload, encoding="utf-8")
    for attempt in range(attempts):
        try:
            temporary.replace(path)
            return
        except PermissionError:
            time.sleep(0.02 * (attempt + 1))
    try:
        path.write_text(payload, encoding="utf-8")
    finally:
        temporary.unlink(missing_ok=True)


def _elapsed(seconds: float) -> str:
    return "%02d:%02d" % (int(seconds) // 60, int(seconds) % 60)


def _containment_pct(env: SumoPursuitEnv, state) -> float:
    """Share of the fugitive's escape routes that an officer already controls.

    An exit counts as controlled when an officer is either nearer to it than
    the fugitive is (it will get there first) or simply parked on top of it.
    The second clause matters: once officers close in, they stop being "nearer
    than" a fugitive that is metres from its own exits, and a purely
    comparative measure would read 0% at the moment of capture.
    """
    if state.fugitive is None or not state.police:
        return 0.0
    if state.outcome == EpisodeOutcome.CAPTURE:
        return 100.0
    exits = env.legal_targets("F0")
    if not exits:
        return 100.0
    covered = 0
    for edge_id in exits:
        try:
            shape = env.sumolib_net.getEdge(edge_id).getShape()
        except Exception:
            continue
        midpoint = (
            sum(point[0] for point in shape) / len(shape),
            sum(point[1] for point in shape) / len(shape),
        )
        fugitive_gap = _distance(midpoint, state.fugitive.position_xy)
        officer_gaps = [_distance(midpoint, officer.position_xy) for officer in state.police]
        if min(officer_gaps) < fugitive_gap or min(officer_gaps) <= BLOCK_RADIUS_M:
            covered += 1
    return 100.0 * covered / len(exits)


def _distance(a, b) -> float:
    return ((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2) ** 0.5


def _write_state(env, state, latencies, tick_s, events, scene=None) -> None:
    fugitive = state.fugitive
    officers = []
    nearest = {"officer": None, "distance_m": None}
    best = None
    for officer in state.police:
        distance = _distance(officer.position_xy, fugitive.position_xy) if fugitive else None
        officers.append({
            "id": officer.vehicle_id,
            "edge": officer.edge_id,
            "action": ACTION_LABEL[officer.speed_mps > 0.5],
            "distance_m": round(distance, 1) if distance is not None else None,
            "state": "차단" if officer.speed_mps < 0.5 else "이동",
        })
        if distance is not None and (best is None or distance < best):
            best, nearest = distance, {"officer": officer.vehicle_id, "distance_m": round(distance, 1)}
    ordered = sorted(latencies) if latencies else [0.0]
    payload = {
        "status": OUTCOME_LABEL.get(state.outcome, "running"),
        "step": state.step,
        "sim_time_s": state.sim_time_s,
        "elapsed_str": _elapsed(state.sim_time_s),
        "background_vehicles": state.background_count,
        "containment_pct": round(_containment_pct(env, state), 1),
        "nearest": nearest,
        "officers": officers,
        "fugitive": {
            "edge": fugitive.edge_id if fugitive else None,
            "speed_kph": round(fugitive.speed_mps * 3.6, 1) if fugitive else None,
        },
        "latency": {
            "p50_ms": round(ordered[len(ordered) // 2], 2),
            "p99_ms": round(ordered[min(len(ordered) - 1, int(len(ordered) * 0.99))], 2),
        },
        "tick_s": tick_s,
        "events": events[-12:],
    }
    if scene is not None:
        # The map view is written separately: it changes every frame and is an
        # order of magnitude larger than the panel data, so a dashboard that
        # only wants the numbers never has to download it.
        _publish(SCENE_PATH, json.dumps(scene, separators=(",", ":")))
    _publish(STATE_PATH, json.dumps(payload, ensure_ascii=False, indent=2))


def run_episode(env, seed, *, live, realtime_s, quiet=False, barrier_image=None):
    state = env.reset(seed=seed)
    policy = EncirclementPolicy(env)
    latencies, events = [], []
    origin = (0.0, 0.0)
    if live:
        OUT_DIR.mkdir(parents=True, exist_ok=True)
        origin = tuple(export_network(env, NETWORK_PATH)["origin"])
        # SUMO 자체 창을 함께 띄울 때 차단 지점을 보여주기 위한 additional 파일.
        # 우리 관제 화면은 이 파일 없이도 그린다.
        write_poi_additional(env.barriers, BARRIER_POI_PATH, image_file=barrier_image)
    while state.outcome is None:
        started = time.perf_counter()
        targets = policy.dispatch(state)
        latencies.append((time.perf_counter() - started) * 1000.0)
        state = env.step_targets(targets)
        for raw in state.events[-2:]:
            entry = {"t": _elapsed(state.sim_time_s), "msg": raw}
            if not events or events[-1] != entry:
                events.append(entry)
        if live:
            frame = scene_frame(env, origin, assignments=policy._assignments)
            _write_state(env, state, latencies, realtime_s, events, scene=frame)
            if realtime_s:
                time.sleep(realtime_s)
    if live:
        frame = scene_frame(env, origin, assignments=policy._assignments)
        _write_state(env, state, latencies, realtime_s, events, scene=frame)
    separation = env.min_separation_m()
    if not quiet:
        print("seed=%-3d outcome=%-8s steps=%-4d background=%-4d min_sep=%s p50=%.2fms"
              % (seed, state.outcome, state.step, state.background_count,
                 ("%.1fm" % separation) if separation else "n/a",
                 statistics.median(latencies) if latencies else 0.0))
    return state, latencies


def main() -> int:
    parser = argparse.ArgumentParser(description="SUMO pursuit demo")
    parser.add_argument("--gui", action="store_true", help="launch sumo-gui")
    parser.add_argument("--episodes", type=int, default=1)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--max-steps", type=int, default=450)
    parser.add_argument("--background", type=int, default=300)
    parser.add_argument("--realtime", type=float, default=0.0,
                        help="seconds to sleep per step (use 0.1 with --gui)")
    parser.add_argument("--no-live", action="store_true", help="do not write state.json")
    parser.add_argument("--barriers", type=int, default=6,
                        help="에피소드당 무작위 도로 차단 수 (0이면 끔)")
    parser.add_argument("--barrier-image", default=None,
                        help="SUMO additional 파일에 쓸 아이콘 PNG 경로 (선택)")
    args = parser.parse_args()

    network = build_network(largest_cached_snapshot(REPO_ROOT / "cache"),
                            output_root=REPO_ROOT / "cache/sumo")
    print("network: edges=%d nodes=%d traffic_lights=%d  barriers/episode=%d"
          % (network.edge_count, network.node_count, network.traffic_light_count, args.barriers))

    outcomes, all_latencies = [], []
    for index in range(args.episodes):
        seed = args.seed + index
        config = SumoEpisodeConfig(
            max_steps=args.max_steps, gui=args.gui, seed=seed,
            traffic=TrafficConfig(vehicle_count=args.background, seed=seed),
            barriers=BarrierConfig(count=args.barriers, enabled=args.barriers > 0),
        )
        env = SumoPursuitEnv(network, config, work_dir=REPO_ROOT / "cache/sumo/runs")
        try:
            state, latencies = run_episode(
                env, seed, live=not args.no_live, realtime_s=args.realtime,
                barrier_image=args.barrier_image,
            )
            outcomes.append(state.outcome)
            all_latencies.extend(latencies)
        finally:
            env.close()

    voided = sum(1 for item in outcomes if item == EpisodeOutcome.VOID)
    scored = [item for item in outcomes if item != EpisodeOutcome.VOID]
    captures = sum(1 for item in scored if item == EpisodeOutcome.CAPTURE)
    ordered = sorted(all_latencies) or [0.0]
    print("-" * 68)
    print("scored=%d capture=%d (%.0f%%) escape=%d timeout=%d  [void=%d excluded]"
          % (len(scored), captures, 100.0 * captures / max(1, len(scored)),
             sum(1 for item in scored if item == EpisodeOutcome.ESCAPE),
             sum(1 for item in scored if item == EpisodeOutcome.TIMEOUT), voided))
    print("decision latency: p50=%.2fms p99=%.2fms n=%d"
          % (ordered[len(ordered) // 2],
             ordered[min(len(ordered) - 1, int(len(ordered) * 0.99))], len(ordered)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())