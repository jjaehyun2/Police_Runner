"""Watch one pursuit in sumo-gui (needs a machine with a display).

  python sumo/demo_gui.py [--map daejeon_demo] [--n 6] [--idx 0] [--spec rmacro:runs/ms_ga/best.pt]
                          [--profile ideal|real] [--traffic 2] [--ignore-red]
Env SUMO_GUI_DELAY_MS slows playback (default 100 ms per 0.25 s step).
Police are blue, the fugitive red; the episode starts with officers surrounding the fugitive.
"""
import argparse
import os
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(HERE.parent))  # noqa: E702
os.environ.setdefault("ME_PLACEMENT", "ring")

import common  # noqa: E402
import multi_sim  # noqa: E402
from sumo_sim import SumoPursuit  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--map", default="daejeon_demo")
ap.add_argument("--n", type=int, default=6)
ap.add_argument("--idx", type=int, default=0)
ap.add_argument("--spec", default="rmacro:" + str(HERE.parent / "runs/ms_ga/best.pt"))
ap.add_argument("--profile", default="ideal")
ap.add_argument("--traffic", type=float, default=None)
ap.add_argument("--ignore-red", action="store_true")
a = ap.parse_args()
net = common.load_network(a.map)
case = multi_sim.make_ring_case(net, f"sumo-{a.map}", a.idx, a.n, max_steps=600)
r = SumoPursuit(a.map, case, a.spec, profile=a.profile, traffic=a.traffic, ignore_red=a.ignore_red, gui=True).run()
print(r.outcome, f"{r.seconds:.1f}s")
