#!/bin/bash
# Is the SUMO traffic inflation a density effect or "one slow car blocks the fast fugitive"?
# Tongyeong large, ring 1500-2500, N=3, 16/19 m/s, no yielding, same 100 cases (9000+); trip period 10 s and 30 s
# (3 s gave encircle 97 / ours 96 vs 72 / 89 without traffic).
cd "$(dirname "$0")/../../.."
export RT_CLASS_FIX=1
for p in 30 10; do
  env SE_WORKERS=10 SE_IGNORE_RED=1 SE_OFFSET=9000 ME_MAX_STEPS=1200 SE_DIRECT_SPEED=1 ME_PLACEMENT=ring ME_RING=1500,2500 \
      ME_VP=16 ME_VF=19 SE_TRAFFIC=$p /workspace/cluster/.venv/bin/python sumo/eval_sumo.py tongyeong_large_dev 100 3 real \
      encircle rmacro:runs/nvar_ga/best.pt > runs/sumo_v2fix/density/tr$p.log 2>&1
  ls -t sumo/results | head -1 > runs/sumo_v2fix/density/tr$p.file
done
touch runs/sumo_v2fix/density/DONE
