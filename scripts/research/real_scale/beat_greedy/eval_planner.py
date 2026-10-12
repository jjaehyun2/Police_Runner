"""Capture rate of the planning teacher itself on the validation cases."""
import sys, json, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np
from multiprocessing import get_context
from engine import run_episode
from common import load_network
R = dict(time_penalty=0.01, capture=10, escape=10, shaping=0.0)

def job(args):
    i, vf, teacher = args
    import torch; torch.set_num_threads(1)
    net = load_network("daejeon_validation")
    e = run_episode(net, tag="beatgreedy-daejeon_validation", index=i, vp=16.0, vf=vf, actor=None,
                    mode="teacher", teacher_name=teacher, beta=1.0, rng=np.random.default_rng(i), reward_cfg=R)
    return i, e.outcome, e.steps

if __name__ == "__main__":
    vf, n, workers, teacher = float(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3]), sys.argv[4]
    t = time.time()
    with get_context("fork").Pool(workers) as p:
        res = p.map(job, [(i, vf, teacher) for i in range(n)], chunksize=1)
    print(teacher, "vf", vf, "capture", np.mean([o == "capture" for _, o, _ in res]), "sec", round(time.time() - t))
    Path("results").mkdir(exist_ok=True)
    Path(f"results/planner_{teacher.replace(':','_')}_vf{vf}.json").write_text(json.dumps(res))
