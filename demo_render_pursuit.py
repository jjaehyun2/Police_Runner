"""개선된 포위 추격을 실제 OSM(풀 2.2km, 내부 추격) 위에 렌더링.

검거로 끝나는 시드를 찾아 프레임 시퀀스로 저장하고, 경찰-도주자 평균거리 추이를 출력한다.
"""
from __future__ import annotations
import math, pickle, copyreg
from types import MappingProxyType
copyreg.pickle(MappingProxyType, lambda p: (MappingProxyType, (dict(p),)))
from pathlib import Path

from pursuit_evasion_rl.osm_demo.metrics import placement_position
from pursuit_evasion_rl.osm_demo.models import EpisodeConfig, EpisodeOutcome
from pursuit_evasion_rl.osm_demo.rendering import FontChoice, OSMRenderer
from pursuit_evasion_rl.osm_demo.runner import run_single_episode
from demo_pursuit import EncirclementPolice, GoalEvader, _Graph
import numpy as np
from run_pursuit_eval import surround_placement

OUT = Path("out_osm_real")
NET = Path("out_pursuit/interior_net.pkl")

POLICE_SPEED, FUG_SPEED, VISION, CAPTURE, MAX_STEPS = 18.0, 8.0, 150.0, 20.0, 1500


def main() -> None:
    OUT.mkdir(exist_ok=True)
    net = pickle.loads(NET.read_bytes())
    g = _Graph(net)
    cfg = EpisodeConfig(dt_s=1.0, police_speed_mps=POLICE_SPEED, fugitive_speed_mps=FUG_SPEED,
                        capture_radius_m=CAPTURE, max_steps=MAX_STEPS)
    print(f"[net] intersections={len(net.intersections)} segments={len(net.segments)} (full 2.2km, interior)")

    # 검거로 끝나는 시드 탐색
    chosen = None
    for seed in range(40):
        pol = EncirclementPolice(net, graph=g, fugitive_speed_mps=FUG_SPEED, police_speed_mps=POLICE_SPEED)
        placement = surround_placement(g, np.random.default_rng(1000 + seed))  # dragnet 포위 배치
        rec = run_single_episode(net, cfg, pol, run_id="render", run_seed=seed,
                                 fugitive_factory=lambda rng: GoalEvader(net, rng=rng, graph=g, vision_range_m=VISION),
                                 placement=placement)
        # 포위가 잘 보이도록 20스텝 이상 지속된 검거 에피소드 선택
        if rec.outcome is EpisodeOutcome.CAPTURE and len(rec.transitions) >= 20:
            chosen = (seed, rec)
            break
    if chosen is None:
        print("no capture in scanned seeds")
        return
    seed, rec = chosen
    sts = [rec.initial_state] + [t.state for t in rec.transitions]

    def p(pl):
        return placement_position(net, pl)

    def avg(st):
        f = p(st.fugitive)
        return sum(math.dist(p(x), f) for x in st.police) / 6

    print(f"[episode] seed={seed} outcome={rec.outcome.value} steps={len(rec.transitions)}")
    print(f"[converge] avg police-fugitive distance start={avg(sts[0]):.0f}m -> end={avg(sts[-1]):.0f}m")

    for i in range(0, len(sts), max(1, len(sts) // 10)):
        print(f"  step{i:>4} avg={avg(sts[i]):>5.0f}m  nearest={min(math.dist(p(x), p(sts[i].fugitive)) for x in sts[i].police):>4.0f}m")

    renderer = OSMRenderer(net, cfg, font=FontChoice("DejaVu Sans", False))
    frames = renderer.render_episode_frames(rec, max_frames=80)
    for i, fig in enumerate(frames):
        renderer.save_figure(fig, str(OUT / f"frame_{i:03d}.png"))
    print(f"[frames] {len(frames)} PNG saved to {OUT}/")


if __name__ == "__main__":
    main()
