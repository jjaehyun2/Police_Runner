# L3 probe: how often does road-optimal motion INCREASE euclidean distance (regress penalty)?
# Dependency-free: json, math, heapq, random only.
import json, math, heapq, random, sys, os

ROOT = os.path.dirname(os.path.abspath(__file__))

DRIVE_EXCLUDE = {
    "footway","cycleway","path","pedestrian","steps","track","corridor","elevator",
    "escalator","proposed","construction","bridleway","abandoned","platform","raceway",
    "service","busway","bus_guideway",
}

def load_graph(path):
    data = json.load(open(path, encoding="utf-8"))
    nodes = {}
    for el in data["elements"]:
        if el["type"] == "node":
            nodes[el["id"]] = (el["lon"], el["lat"])
    # project: equirectangular around mean lat
    lats = [p[1] for p in nodes.values()]
    lat0 = sum(lats)/len(lats)
    kx = 111320.0*math.cos(math.radians(lat0)); ky = 110540.0
    xy = {nid:(lon*kx, lat*ky) for nid,(lon,lat) in nodes.items()}
    adj = {}   # node -> list of (neighbor, length, polyline)
    def add_edge(a, b, poly):
        L = sum(math.dist(poly[i], poly[i+1]) for i in range(len(poly)-1))
        if L <= 0: return
        adj.setdefault(a, []).append((b, L, poly))
    nways = 0
    for el in data["elements"]:
        if el["type"] != "way": continue
        tags = el.get("tags", {})
        hw = tags.get("highway")
        if hw is None or hw.split("_link")[0] in DRIVE_EXCLUDE or hw in DRIVE_EXCLUDE:
            continue
        nds = [n for n in el["nodes"] if n in xy]
        if len(nds) < 2: continue
        nways += 1
        oneway = tags.get("oneway", "no")
        implied = hw.startswith("motorway") or tags.get("junction") in ("roundabout","circular")
        fwd = True; bwd = not (oneway in ("yes","true","1") or implied)
        if oneway == "-1": fwd, bwd = False, True
        poly = [xy[n] for n in nds]
        if fwd:
            for i in range(len(nds)-1):
                add_edge(nds[i], nds[i+1], [poly[i], poly[i+1]])
        if bwd:
            for i in range(len(nds)-1):
                add_edge(nds[i+1], nds[i], [poly[i+1], poly[i]])
    lons = [p[0] for p in nodes.values()]
    bbox = (min(lons), max(lons), min(lats), max(lats))
    return xy, adj, bbox, nways

def dijkstra(adj, src):
    dist = {src: 0.0}; prev = {}
    q = [(0.0, src)]
    while q:
        d, u = heapq.heappop(q)
        if d > dist.get(u, math.inf): continue
        for v, w, _ in adj.get(u, ()):
            nd = d + w
            if nd < dist.get(v, math.inf):
                dist[v] = nd; prev[v] = u
                heapq.heappush(q, (nd, v))
    return dist, prev

def path_nodes(prev, src, dst):
    p = [dst]
    while p[-1] != src:
        if p[-1] not in prev: return None
        p.append(prev[p[-1]])
    return p[::-1]

def walk_metrics(xy, path, target_xy, step_m=16.0):
    """Walk polyline of node path in step_m increments; per-step euclid delta to target."""
    pts = [xy[n] for n in path]
    # densify
    samples = [pts[0]]
    for a, b in zip(pts, pts[1:]):
        L = math.dist(a, b)
        if L == 0: continue
        n = max(1, int(L // step_m))
        # walk in exact step_m chunks across segment boundaries handled below
        samples.append(b)
    # simpler: resample whole polyline at step_m arc-length
    arc = [0.0]
    for a, b in zip(pts, pts[1:]):
        arc.append(arc[-1] + math.dist(a, b))
    total = arc[-1]
    if total == 0: return None
    def point_at(s):
        for i in range(len(arc)-1):
            if arc[i+1] >= s:
                seg = arc[i+1]-arc[i]
                t = 0 if seg == 0 else (s-arc[i])/seg
                return (pts[i][0]+t*(pts[i+1][0]-pts[i][0]), pts[i][1]+t*(pts[i+1][1]-pts[i][1]))
        return pts[-1]
    n_steps = int(total // step_m)
    if n_steps < 1: return None
    regress = 0; total_steps = 0; worst = 0.0; reward_move = 0.0
    d_prev = math.dist(point_at(0), target_xy)
    for k in range(1, n_steps+1):
        d_now = math.dist(point_at(k*step_m), target_xy)
        delta = (d_prev - d_now)/50.0          # trainer scale
        own = 0.6*delta if delta >= 0 else 0.6*1.6*delta
        reward_move += own - 0.02
        if d_now > d_prev + 1e-9:
            regress += 1; worst = max(worst, d_now-d_prev)
        total_steps += 1
        d_prev = d_now
    return regress, total_steps, worst, reward_move, total

def main():
    random.seed(7)
    files = [f for f in os.listdir(os.path.join(ROOT, "cache")) if f.endswith(".json")]
    results = {}
    for f in files:
        xy, adj, bbox, nways = load_graph(os.path.join(ROOT, "cache", f))
        results[f] = (xy, adj, bbox, nways)
        print(f"{f[:8]}: nodes={len(xy)} ways={nways} bbox_lon=({bbox[0]:.3f},{bbox[1]:.3f}) lat=({bbox[2]:.3f},{bbox[3]:.3f})")
    # identify train bbox 127.368-127.382 / 36.334-36.346
    for f,(xy,adj,bbox,_) in results.items():
        tag = "?"
        if bbox[0] <= 127.383 and bbox[1] >= 127.367 and bbox[2] <= 36.347 and bbox[3] >= 36.333 and bbox[1] < 127.39:
            tag = "TRAIN"
        print(f, "->", tag)

    for f,(xy,adj,bbox,nways) in sorted(results.items()):
        nodes = [n for n in xy if n in adj]
        if len(nodes) < 10: continue
        print("="*70); print("MAP", f[:12])
        pair_stats = []; step_regress = 0; step_total = 0
        per_pair_regress = 0; pairs = 0
        rew_move_worse_than_stay = 0
        sampled = 0; attempts = 0
        while sampled < 300 and attempts < 5000:
            attempts += 1
            s = random.choice(nodes); t = random.choice(list(xy))
            if s == t: continue
            dist, prev = dijkstra(adj, s)
            if t not in dist or dist[t] < 100 or dist[t] > 3000: continue
            pn = path_nodes(prev, s, t)
            if pn is None: continue
            m = walk_metrics(xy, pn, xy[t])
            if m is None: continue
            regress, tot, worst, reward_move, road_len = m
            eu = math.dist(xy[s], xy[t])
            sampled += 1; pairs += 1
            step_regress += regress; step_total += tot
            if regress > 0: per_pair_regress += 1
            # stay reward for same duration (static target): tot * -0.02
            if reward_move < tot * -0.02: rew_move_worse_than_stay += 1
            pair_stats.append((regress/tot, worst, road_len/eu))
        if not pairs: continue
        pair_stats.sort()
        frac = [p[0] for p in pair_stats]
        detour = sorted(p[2] for p in pair_stats)
        print(f"pairs={pairs} (road dist 100-3000m, police step 16m)")
        print(f"steps regressing euclid: {step_regress}/{step_total} = {100*step_regress/step_total:.1f}%")
        print(f"pairs with >=1 regress step: {per_pair_regress}/{pairs} = {100*per_pair_regress/pairs:.1f}%")
        print(f"pairs where TOTAL reward(move road-optimally, static fugitive) < reward(stay): {rew_move_worse_than_stay}/{pairs}")
        print(f"median/regress-step-frac p50={frac[len(frac)//2]:.3f} p90={frac[int(len(frac)*0.9)]:.3f} max={frac[-1]:.3f}")
        print(f"detour ratio road/euclid p50={detour[len(detour)//2]:.2f} p90={detour[int(len(detour)*0.9)]:.2f} max={detour[-1]:.2f}")
        # euclid-greedy 2-cycles: for random targets, count nodes whose greedy neighbor's greedy neighbor is itself
        cyc = 0; checked = 0
        for _ in range(30):
            t = random.choice(nodes); txy = xy[t]
            g = {}
            for u in nodes:
                nbrs = [v for v,_,_ in adj.get(u,())]
                if not nbrs: continue
                g[u] = min(nbrs, key=lambda v: math.dist(xy[v], txy))
            for u in nodes:
                v = g.get(u)
                if v is not None and v != t and g.get(v) == u and math.dist(xy[u], txy) > 50:
                    cyc += 1
                checked += 1
        print(f"euclid-greedy 2-cycle traps: {cyc}/{checked} node-target checks = {100*cyc/checked:.2f}%")

if __name__ == "__main__":
    main()
