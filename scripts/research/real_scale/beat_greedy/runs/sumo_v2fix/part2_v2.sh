#!/bin/bash
# Part 2: does the traffic-aware cost (v2, cmacro) beat road distance (rmacro) when there is traffic, and does the
# gain grow with density? Light traffic keeps the policies separable (Tongyeong 16/19: ours - encircle +16 at 30 s,
# +9 at 10 s, -1 at 3 s). Trip period 10 s and 6 s, no yielding, default nets (no overtaking); large maps, ring
# 1500-2500 m, N=3, 16/19 and 22/26 m/s; same cases 9000+; 200 cases. Queued after the overtaking run.
cd "$(dirname "$0")/../.."
PY=/workspace/cluster/.venv/bin/python
while [ ! -f runs/sumo_v2fix/OVERTAKE_DONE ]; do sleep 120; done
export RT_CLASS_FIX=1
for m in tongyeong_large_dev yeosu_large_dev; do for p in 10 6; do
  [ -f sumo/nets/${m}_real_traffic$p.rou.xml ] || $PY -c "
import sys; sys.path[:0]=['sumo','.']; from build_net import traffic; traffic('$m','real',$p,end_s=1800.0)" >> runs/sumo_v2fix/part2_v2.out 2>&1
done; done
P="rmacro:runs/nvar_ga/best.pt cmacro:runs/nvar_ga/best.pt encircle"
b="SE_WORKERS=16 SE_IGNORE_RED=1 SE_OFFSET=9000 ME_MAX_STEPS=1200 SE_DIRECT_SPEED=1 ME_PLACEMENT=ring ME_RING=1500,2500"
for m in tongyeong_large_dev yeosu_large_dev; do
  for p in 10 6; do
    for sp in "16 19" "22 26"; do
      set -- $sp
      env $b SE_TRAFFIC=$p ME_VP=$1 ME_VF=$2 $PY sumo/eval_sumo.py $m 200 3 real $P >> runs/sumo_v2fix/part2_v2.log 2>&1
      ls -t sumo/results | head -1 >> runs/sumo_v2fix/part2_v2.files
    done
  done
done
touch runs/sumo_v2fix/PART2_DONE
