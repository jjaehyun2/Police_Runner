"""Point-mass screen on the large (~6 km) development maps: speed x start distance x team size.
Usage: python runs/large_map/pm_screen.py <map> [<map> ...]   (paired: same cases for every policy)"""
import os, subprocess, json, sys, math
PY = "/workspace/cluster/.venv/bin/python"
EPS = int(os.environ.get("LM_EPS", "200"))
CODE = '''
import sys, json, multi_eval as me
m, n, spec, eps = sys.argv[1], int(sys.argv[2]), sys.argv[3], int(sys.argv[4])
r = me.evaluate(m, eps, n, 1, spec, workers=12)
print(json.dumps({str(k): [v[0], v[1]] for k, v in r.items()}))
'''
POL = ("rmacro:runs/nvar_ga/best.pt", "encircle", "greedy")
SPEEDS = ((16, 16), (16, 19), (22, 22), (22, 26))
PLACE = (("random", "400,700"), ("ring", "1500,2500"))
NS = (3, 6)
def run(m, spec, vp, vf, pl, ring, n):
    env = {**os.environ, "ME_PLACEMENT": pl, "ME_RING": ring, "ME_VP": str(vp), "ME_VF": str(vf), "ME_MAX_STEPS": "1200"}
    r = subprocess.run([PY, "-c", CODE, m, str(n), spec, str(EPS)], env=env, capture_output=True, text=True)
    if not r.stdout.strip():
        return None, r.stderr[-300:]
    return json.loads(r.stdout.strip().splitlines()[-1]), None
def sr(a): return sum(v[0] == "success" for v in a.values()) / len(a)
def ats(a): s = [v[1] for v in a.values() if v[0] == "success"]; return sum(s) / max(1, len(s))
def cmp(a, b):
    n = len(a); w = sum(a[k][0] == "success" and b[k][0] != "success" for k in a); l = sum(b[k][0] == "success" and a[k][0] != "success" for k in a)
    ci = 1.96 * math.sqrt(max(w + l - (w - l) ** 2 / n, 0)) / n
    return f"{100*(w-l)/n:+5.1f}±{100*ci:4.1f}"
for m in sys.argv[1:]:
    for vp, vf in SPEEDS:
        for pl, ring in PLACE:
            for n in NS:
                res, err = {}, None
                for p in POL:
                    res[p], err = run(m, p, vp, vf, pl, ring, n)
                    if res[p] is None:
                        break
                if err:
                    print(f"{m} vp{vp} vf{vf} {pl} N{n} ERR {err}", flush=True); continue
                o, e, g = (res[p] for p in POL)
                print(f"{m:26s} vp={vp} vf={vf} {pl:6s} {ring:9s} N={n} | ours {sr(o):.3f} ({ats(o):.0f}st) enc {sr(e):.3f} ({ats(e):.0f}st) "
                      f"greedy {sr(g):.3f} | ours-enc {cmp(o, e)}  ours-greedy {cmp(o, g)}", flush=True)
