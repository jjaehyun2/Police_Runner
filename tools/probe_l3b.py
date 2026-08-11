# L3b: decision-local probe. At intersections, for a given fugitive position:
#  - STAY-TRAP: every outgoing 16m move increases euclidean distance (all-regress local minimum)
#  - GRADIENT-FLIP: the road-optimal first hop regresses euclid, while some other hop progresses
#    (euclid reward points AWAY from the road-optimal action)
# Also: net own-reward comparison of STAY vs road-optimal move at flip nodes.
import json, math, heapq, random, os
from probe_l3 import load_graph, dijkstra

ROOT = os.path.dirname(os.path.abspath(__file__))

def first_16m_point(xy, u, v, step=16.0):
    a, b = xy[u], xy[v]
    L = math.dist(a, b)
    t = min(1.0, step / L) if L > 0 else 1.0
    return (a[0]+t*(b[0]-a[0]), a[1]+t*(b[1]-a[1]))

def main():
    random.seed(11)
    files = sorted(f for f in os.listdir(os.path.join(ROOT, "cache")) if f.endswith(".json"))
    for f in files:
        xy, adj, bbox, _ = load_graph(os.path.join(ROOT, "cache", f))
        nodes = [n for n in xy if n in adj]
        # reverse graph for road distance TO target
        radj = {}
        for u, lst in adj.items():
            for v, w, _ in lst:
                radj.setdefault(v, []).append((u, w, None))
        stay_trap = flip = ok = checked = 0
        decision_nodes = [n for n in nodes if len(adj[n]) >= 2]  # real decisions
        for _ in range(40):
            t = random.choice(nodes)
            dist_to_t, _ = dijkstra(radj, t)   # road distance from any node TO t
            txy = xy[t]
            for u in decision_nodes:
                if u == t or u not in dist_to_t: continue
                d_u = dist_to_t[u]
                if d_u < 100 or d_u > 3000: continue
                du_e = math.dist(xy[u], txy)
                outs = []
                for v, w, _ in adj[u]:
                    de = math.dist(first_16m_point(xy, u, v), txy) - du_e   # euclid delta of first step
                    droad = (w + dist_to_t.get(v, math.inf))                # road dist via v
                    outs.append((droad, de))
                if not outs or all(math.isinf(o[0]) for o in outs): continue
                checked += 1
                best_road = min(outs, key=lambda o: o[0])
                if all(o[1] > 1e-9 for o in outs):
                    stay_trap += 1
                elif best_road[1] > 1e-9 and any(o[1] < -1e-9 for o in outs):
                    flip += 1
                else:
                    ok += 1
        print("="*70)
        print(f"MAP {f[:12]} decision-intersections checked={checked}")
        print(f"  STAY-TRAP (all moves regress euclid): {stay_trap} = {100*stay_trap/checked:.2f}%")
        print(f"  GRADIENT-FLIP (road-optimal regresses, other move progresses): {flip} = {100*flip/checked:.2f}%")
        print(f"  aligned: {ok} = {100*ok/checked:.2f}%")

main()
