"""Random search over role-scorer initializations (neuroevolution, generation 0).

Each seed gives an untrained CandidateScorer used with argmax; seeds are ranked
on sealed solvable validation cases (vf=16) and the top ones re-checked.
Usage: python seed_search.py <n_seeds> <val_episodes>
"""
from __future__ import annotations
import json, os, sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
os.environ.setdefault("BG_SOLVABLE", "1"); os.environ.setdefault("BG_FUGITIVE_SPEED", "16")
from common import capture_rate, evaluate

n, eps = int(sys.argv[1]), int(sys.argv[2])
out = Path(__file__).resolve().parent / "results" / "seed_search.jsonl"
for seed in range(n):
    t = time.time()
    r = capture_rate(evaluate(f"macro:init{seed}", "daejeon_validation", "beatgreedy-daejeon_validation", eps))
    rec = dict(seed=seed, val16=r, episodes=eps)
    out.open("a").write(json.dumps(rec) + "\n")
    print(json.dumps(rec), f"{time.time()-t:.0f}s", flush=True)
