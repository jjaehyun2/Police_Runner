#!/bin/bash
# Part 2 setting: background traffic WITH opposite-direction overtaking for every vehicle (SE_OPPOSITES=1), no yielding.
# Without overtaking, one slow civilian car on a single-lane road erased the fast fugitive's speed edge and lifted every
# policy to 88-97%. Queued after the density test. Tongyeong / Yeosu large, ring 1500-2500 m, N=3, 16/19 m/s.
cd "$(dirname "$0")/../.."
PY=/workspace/cluster/.venv/bin/python
while [ ! -f runs/sumo_v2fix/density/DONE ]; do sleep 120; done
export RT_CLASS_FIX=1 SE_OPPOSITES=1
for m in tongyeong_large_dev yeosu_large_dev; do $PY sumo/build_net.py $m real >> runs/sumo_v2fix/overtake.out 2>&1; done
P="rmacro:runs/nvar_ga/best.pt cmacro:runs/nvar_ga/best.pt encircle greedy"
b="SE_WORKERS=16 SE_IGNORE_RED=1 SE_OFFSET=9000 ME_MAX_STEPS=1200 SE_DIRECT_SPEED=1 ME_PLACEMENT=ring ME_RING=1500,2500 ME_VP=16 ME_VF=19"
for spec in "tongyeong_large_dev 3" "yeosu_large_dev 3"; do
  set -- $spec
  env $b SE_TRAFFIC=$2 $PY sumo/eval_sumo.py $1 200 3 real $P >> runs/sumo_v2fix/overtake.log 2>&1
  ls -t sumo/results | head -1 >> runs/sumo_v2fix/overtake.files
done
touch runs/sumo_v2fix/OVERTAKE_DONE
