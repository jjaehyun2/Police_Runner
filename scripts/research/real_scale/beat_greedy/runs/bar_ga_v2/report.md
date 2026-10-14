# Barricade GA report (/workspace/cluster/scripts/research/real_scale/beat_greedy/runs/bar_ga_v2)

generated 2026-10-13 11:51

## Validation maps (selection)

**daejeon_validation**

| policy | SR | capture s | barricades/ep | W/L vs never | escape/timeout | by N |
|---|---|---|---|---|---|---|
| never | 97.0 | 31.1 | 0.0 | –/– | 18/0 | {'3': 0.85, '4': 1.0, '5': 1.0, '6': 1.0, '8': 1.0} |
| role | 100.0 | 30.3 | 0.0 | 18/0 | 0/0 | {'3': 1.0, '4': 1.0, '5': 1.0, '6': 1.0, '8': 1.0} |
| learned | 100.0 | 30.3 | 0.0 | 18/0 | 0/0 | {'3': 1.0, '4': 1.0, '5': 1.0, '6': 1.0, '8': 1.0} |

**incheon_validation**

| policy | SR | capture s | barricades/ep | W/L vs never | escape/timeout | by N |
|---|---|---|---|---|---|---|
| never | 92.3 | 42.3 | 0.0 | –/– | 23/0 | {'3': 0.833, '4': 0.967, '5': 0.917, '6': 0.933, '8': 0.967} |
| role | 92.7 | 44.0 | 0.0 | 1/0 | 22/0 | {'3': 0.85, '4': 0.967, '5': 0.917, '6': 0.933, '8': 0.967} |
| learned | 92.7 | 44.0 | 0.0 | 1/0 | 22/0 | {'3': 0.85, '4': 0.967, '5': 0.917, '6': 0.933, '8': 0.967} |

**suwon2_validation**

| policy | SR | capture s | barricades/ep | W/L vs never | escape/timeout | by N |
|---|---|---|---|---|---|---|
| never | 85.3 | 27.4 | 0.0 | –/– | 44/0 | {'3': 0.7, '4': 0.833, '5': 0.9, '6': 0.933, '8': 0.9} |
| role | 87.0 | 27.3 | 0.0 | 6/1 | 39/0 | {'3': 0.717, '4': 0.817, '5': 0.917, '6': 0.95, '8': 0.95} |
| learned | 87.0 | 27.3 | 0.0 | 6/1 | 39/0 | {'3': 0.717, '4': 0.817, '5': 0.917, '6': 0.95, '8': 0.95} |

## Held-out maps (one shot)

**daejeon_test**

| policy | SR | capture s | barricades/ep | W/L vs never | escape/timeout | by N |
|---|---|---|---|---|---|---|
| never | 89.3 | 46.6 | 0.0 | –/– | 64/0 | {'3': 0.767, '4': 0.892, '5': 0.925, '6': 0.908, '8': 0.975} |
| learned | 88.3 | 47.1 | 0.0 | 10/16 | 70/0 | {'3': 0.767, '4': 0.867, '5': 0.933, '6': 0.875, '8': 0.975} |

**seoul_zero_shot**

| policy | SR | capture s | barricades/ep | W/L vs never | escape/timeout | by N |
|---|---|---|---|---|---|---|
| never | 70.8 | 53.9 | 0.0 | –/– | 116/1 | {'3': 0.525, '4': 0.588, '5': 0.775, '6': 0.838, '8': 0.812} |
| learned | 71.0 | 54.8 | 0.0 | 5/4 | 116/0 | {'3': 0.525, '4': 0.575, '5': 0.8, '6': 0.85, '8': 0.8} |

**busan_zero_shot**

| policy | SR | capture s | barricades/ep | W/L vs never | escape/timeout | by N |
|---|---|---|---|---|---|---|
| never | 65.0 | 66.9 | 0.0 | –/– | 140/0 | {'3': 0.525, '4': 0.575, '5': 0.725, '6': 0.75, '8': 0.675} |
| learned | 46.5 | 50.7 | 0.0 | 6/80 | 214/0 | {'3': 0.438, '4': 0.325, '5': 0.4, '6': 0.625, '8': 0.537} |

## SUMO daejeon_demo (team size)

**3**

| policy | SR | capture s | barricades/ep | W/L vs never | escape/timeout | by N |
|---|---|---|---|---|---|---|
| never | 88.3 | 41.1 | 0.0 | –/– | 7/0 | {'3': 0.883} |
| role | 83.3 | 44.4 | 0.0 | 2/5 | 10/0 | {'3': 0.833} |
| rules | 85.0 | 52.1 | 0.48 | 2/4 | 9/0 | {'3': 0.85} |
| learned | 83.3 | 44.4 | 0.0 | 2/5 | 10/0 | {'3': 0.833} |

**4**

| policy | SR | capture s | barricades/ep | W/L vs never | escape/timeout | by N |
|---|---|---|---|---|---|---|
| never | 88.0 | 43.6 | 0.0 | –/– | 12/0 | {'4': 0.88} |
| role | 89.0 | 44.5 | 0.0 | 3/2 | 11/0 | {'4': 0.89} |
| rules | 88.0 | 40.8 | 0.72 | 4/4 | 11/1 | {'4': 0.88} |
| learned | 89.0 | 44.5 | 0.0 | 3/2 | 11/0 | {'4': 0.89} |

**6**

| policy | SR | capture s | barricades/ep | W/L vs never | escape/timeout | by N |
|---|---|---|---|---|---|---|
| never | 100.0 | 23.3 | 0.0 | –/– | 0/0 | {'6': 1.0} |
| role | 100.0 | 23.8 | 0.0 | 0/0 | 0/0 | {'6': 1.0} |
| rules | 96.7 | 28.5 | 1.18 | 0/2 | 2/0 | {'6': 0.967} |
| learned | 100.0 | 23.8 | 0.0 | 0/0 | 0/0 | {'6': 1.0} |
