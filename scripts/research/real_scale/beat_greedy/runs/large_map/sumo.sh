#!/bin/bash
# SUMO check of the large-map point-mass result: normal density (Tongyeong 3 s, Gamcheon 2 s trip period, ~30% halting,
# same vehicles/road-km as the 3 km maps at 5 s), emergency yielding 5.6 m/s, both ignore red, 1200 s horizon,
# fresh cases (offset 9000). Policies: ours (road / v2 traffic) and encircle.
cd "$(dirname "$0")/../.."
PY=/workspace/cluster/.venv/bin/python
L=runs/large_map/sumo.log
P="rmacro:runs/nvar_ga/best.pt cmacro:runs/nvar_ga/best.pt encircle"
b="SE_WORKERS=12 SE_IGNORE_RED=1 SE_YIELD_V=5.6 SE_OFFSET=9000 ME_MAX_STEPS=1200"
for spec in "tongyeong_large_dev 3" "busan_gamcheon_large_dev 2"; do
  set -- $spec; m=$1; tr=$2
  env $b SE_TRAFFIC=$tr ME_PLACEMENT=random ME_RING=400,700 ME_VF=16 $PY sumo/eval_sumo.py $m 200 3 real $P >> $L 2>&1
  env $b SE_TRAFFIC=$tr ME_PLACEMENT=ring ME_RING=1500,2500 ME_VF=16 $PY sumo/eval_sumo.py $m 200 3 real $P >> $L 2>&1
  env $b SE_TRAFFIC=$tr ME_PLACEMENT=random ME_RING=400,700 ME_VF=19 $PY sumo/eval_sumo.py $m 200 6 real $P >> $L 2>&1
done
touch runs/large_map/SUMO_DONE
