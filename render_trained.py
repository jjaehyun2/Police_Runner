"""학습된 best.pt 정책의 실제 OSM 검거 장면 렌더링(포위 배치)."""
from __future__ import annotations
import argparse, math, pickle, copyreg
from types import MappingProxyType
from pathlib import Path
import numpy as np

copyreg.pickle(MappingProxyType, lambda p: (MappingProxyType, (dict(p),)))

from pursuit_evasion_rl.osm_demo.metrics import placement_position
from pursuit_evasion_rl.osm_demo.models import EpisodeConfig, EpisodeOutcome
from pursuit_evasion_rl.osm_demo.rendering import FontChoice, OSMRenderer
from pursuit_evasion_rl.osm_demo.runner import run_single_episode
from demo_pursuit import GoalEvader, _Graph
from eval_trained import LearnedPolice
from run_pursuit_eval import surround_placement

OUT = Path("out_osm_real")
NET = Path("out_pursuit/interior_net.pkl")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="checkpoints/osm_mappo/best.pt")
    ap.add_argument("--police-speed", type=float, default=16.0)
    ap.add_argument("--fugitive-speed", type=float, default=9.0)
    ap.add_argument("--capture", type=float, default=25.0)
    ap.add_argument("--vision", type=float, default=150.0)
    ap.add_argument("--max-steps", type=int, default=450)
    ap.add_argument("--mode", default="surround")
    args = ap.parse_args()

    OUT.mkdir(exist_ok=True)
    net = pickle.loads(NET.read_bytes())
    g = _Graph(net)
    cfg = EpisodeConfig(dt_s=1.0, police_speed_mps=args.police_speed, fugitive_speed_mps=args.fugitive_speed,
                        capture_radius_m=args.capture, max_steps=args.max_steps)

    def p(pl): return placement_position(net, pl)
    def avg(st):
        f = p(st.fugitive)
        return sum(math.dist(p(x), f) for x in st.police) / 6

    chosen = None
    for seed in range(40):
        pol = LearnedPolice(net, g, args.ckpt)
        placement = surround_placement(g, np.random.default_rng(5000 + seed)) if args.mode == "surround" else None
        rec = run_single_episode(net, cfg, pol, run_id="rt", run_seed=10000 + seed,
                                 fugitive_factory=lambda rng: GoalEvader(net, rng=rng, graph=g, vision_range_m=args.vision),
                                 placement=placement)
        if rec.outcome is EpisodeOutcome.CAPTURE and len(rec.transitions) >= 20:
            chosen = (seed, rec); break
    if chosen is None:
        print("no suitable capture episode found"); return
    seed, rec = chosen
    sts = [rec.initial_state] + [t.state for t in rec.transitions]
    print(f"[episode] ckpt={args.ckpt} seed={seed} outcome={rec.outcome.value} steps={len(rec.transitions)}")
    print(f"[converge] avg police-fugitive distance start={avg(sts[0]):.0f}m -> end={avg(sts[-1]):.0f}m")
    for i in range(0, len(sts), max(1, len(sts)//10)):
        print(f"  step{i:>4} avg={avg(sts[i]):>5.0f}m nearest={min(math.dist(p(x),p(sts[i].fugitive)) for x in sts[i].police):>4.0f}m")

    renderer = OSMRenderer(net, cfg, font=FontChoice("DejaVu Sans", False))
    for f in OUT.glob("frame_*.png"):
        f.unlink()
    frames = renderer.render_episode_frames(rec, max_frames=80)
    for i, fig in enumerate(frames):
        renderer.save_figure(fig, str(OUT / f"frame_{i:03d}.png"))
    print(f"[frames] {len(frames)} PNG saved to {OUT}/")


if __name__ == "__main__":
    main()
