#!/bin/bash
# Part 1 (policy comparison): corrected SUMO without background traffic, where SUMO and the point-mass simulator
# agree (Tongyeong 100-case check: ours 89/88, encircle 72/71, greedy 57/54). Large maps, ring 1500-2500 m, N=3,
# speeds 16/19, 22/26, 16/16; direct target speeds; road-class fix; fresh cases 9000+; 200 cases.
cd "$(dirname "$0")/../.."
PY=/workspace/cluster/.venv/bin/python
export RT_CLASS_FIX=1
P="rmacro:runs/nvar_ga/best.pt encircle greedy"
b="SE_WORKERS=16 SE_IGNORE_RED=1 SE_OFFSET=9000 ME_MAX_STEPS=1200 SE_DIRECT_SPEED=1 ME_PLACEMENT=ring ME_RING=1500,2500"
for m in tongyeong_large_dev yeosu_large_dev busan_gamcheon_large_dev; do
  for sp in "16 19" "22 26" "16 16"; do
    set -- $sp
    env $b ME_VP=$1 ME_VF=$2 $PY sumo/eval_sumo.py $m 200 3 real $P >> runs/sumo_v2fix/notraffic_grid.log 2>&1
    ls -t sumo/results | head -1 >> runs/sumo_v2fix/notraffic_grid.files
  done
done
touch runs/sumo_v2fix/NOTRAFFIC_DONE
