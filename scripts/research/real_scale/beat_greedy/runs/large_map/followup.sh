#!/bin/bash
# After runs/large_map/sumo.sh (old nets, original road-class lookup):
#  1. rebuild the Tongyeong / Gamcheon large nets with RT_CLASS_FIX=1 (living_street 20 km/h); every run below sets
#     RT_CLASS_FIX=1 so the policies' speed belief matches the net
#  2. fast chases (ring 1500-2500 m, N=3; 16/19 and 22/26 m/s; speed factor 3) on Tongyeong, Gamcheon, Yeosu large
#  3. Yeosu large at normal density (3 s): random N3, ring N3
#  4. yield-speed sensitivity on Tongyeong large (random N3, 16/16): no yielding, 3, 5.6, 8 m/s
cd "$(dirname "$0")/../.."
PY=/workspace/cluster/.venv/bin/python
while [ ! -f runs/large_map/SUMO_DONE ] && pgrep -f 'large_map/sumo[.]sh' >/dev/null; do sleep 120; done
export RT_CLASS_FIX=1
for m in tongyeong_large_dev busan_gamcheon_large_dev; do $PY sumo/build_net.py $m real >> runs/large_map/followup.out 2>&1; done
echo "nets rebuilt $(date)" >> runs/large_map/followup.out
P="rmacro:runs/nvar_ga/best.pt cmacro:runs/nvar_ga/best.pt encircle"
b="SE_WORKERS=12 SE_IGNORE_RED=1 SE_OFFSET=9000 ME_MAX_STEPS=1200"
F=runs/large_map/sumo_fast.log
for spec in "tongyeong_large_dev 3" "busan_gamcheon_large_dev 2" "yeosu_large_dev 3"; do
  set -- $spec
  for sp in "16 19" "22 26"; do
    set -- $1 $2 $sp
    env $b SE_YIELD_V=5.6 SE_TRAFFIC=$2 SE_SPEED_FACTOR=3 ME_PLACEMENT=ring ME_RING=1500,2500 ME_VP=$3 ME_VF=$4 \
      $PY sumo/eval_sumo.py $1 200 3 real $P >> $F 2>&1
  done
done
Y=runs/large_map/sumo_yeosu.log
env $b SE_YIELD_V=5.6 SE_TRAFFIC=3 ME_PLACEMENT=random ME_RING=400,700 ME_VF=16 $PY sumo/eval_sumo.py yeosu_large_dev 200 3 real $P >> $Y 2>&1
env $b SE_YIELD_V=5.6 SE_TRAFFIC=3 ME_PLACEMENT=ring ME_RING=1500,2500 ME_VF=16 $PY sumo/eval_sumo.py yeosu_large_dev 200 3 real $P >> $Y 2>&1
S=runs/large_map/yield_sens.log
for yv in none 3 5.6 8; do
  y=""; [ $yv != none ] && y="SE_YIELD_V=$yv"
  echo "== yield $yv" >> $S
  env $b $y SE_TRAFFIC=3 ME_PLACEMENT=random ME_RING=400,700 ME_VF=16 $PY sumo/eval_sumo.py tongyeong_large_dev 200 3 real \
    rmacro:runs/nvar_ga/best.pt encircle >> $S 2>&1
done
touch runs/large_map/FOLLOWUP_DONE
