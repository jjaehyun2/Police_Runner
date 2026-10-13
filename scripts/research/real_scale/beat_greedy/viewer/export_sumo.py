"""Export SUMO (TraCI) replays on the Daejeon demo map for the viewer: positions every second
from SUMO, officers' targets, and the fugitive's escape region from the SUMO placements."""
import base64, heapq, json, os, sys
BG = "/workspace/cluster/scripts/research/real_scale/beat_greedy/"
sys.path.insert(0, BG + "sumo"); sys.path.insert(0, BG)
os.environ["ME_PLACEMENT"] = "ring"; os.environ["ME_MAX_STEPS"] = "600"
import common, multi_sim
from obs_ext import exit_nodes
from sumo_sim import SumoPursuit

SPECS = {"ga": "rmacro:" + BG + "runs/ms_ga/best.pt", "enc": "encircle"}
MARGIN_S = 25.0 / 16.0


def geometry(net):
    index, roads = {}, []
    for s in net.segments:
        key = tuple(sorted((s.start_id, s.end_id)))
        if key in index:
            continue
        index[key] = len(roads)
        roads.append([[round(x, 1), round(y, 1)] for x, y in s.geometry_xy])
    return roads, index


def region(g, reach, vp, vf, exits):
    tp = {}
    for n0, off in reach[:-1]:
        for node, d in g.dist_from(n0).items():
            t = (off + d) / vp
            if t < tp.get(node, 1e18):
                tp[node] = t
    f0, foff = reach[-1]
    best, heap, reg = {f0: foff / vf}, [(foff / vf, f0)], set()
    while heap:
        t, u = heapq.heappop(heap)
        if t > best.get(u, 1e18) or t >= tp.get(u, 1e18) - MARGIN_S:
            continue
        reg.add(u)
        for nb, w in g.fwd[u]:
            if t + w / vf < best.get(nb, 1e18):
                best[nb] = t + w / vf; heapq.heappush(heap, (t + w / vf, nb))
    return reg, [e for e in exits if e in reg]


def bits(idx, n):
    b = bytearray((n + 7) // 8)
    for i in idx:
        b[i >> 3] |= 1 << (i & 7)
    return base64.b64encode(bytes(b)).decode()


def convert(res, net, g, roads, index, exits, pos):
    P = lambda xy: [round(xy[0], 1), round(xy[1], 1)]
    frames = []
    for fr in res.trajectory:
        reg, ox = region(g, [tuple(r) for r in fr["reach"]], 16.0, 16.0, exits)
        segs = [i for (a, b), i in index.items() if a in reg and b in reg]
        goals = fr["goals"]
        frames.append(dict(p=[P(q) for q in fr["p"]], f=P(fr["f"]), reg=bits(segs, len(roads)),
                           ox=[P(pos[e]) for e in ox], rn=len(reg),
                           role=[0 if gk[0] == "pursue" else 1 for gk in goals],
                           tgt=[None if gk[0] == "pursue" else P(pos[gk[1]]) for gk in goals]))
    return dict(outcome=res.outcome, steps=round(res.seconds), frames=frames)


def main():
    m = "daejeon_demo"
    net = common.load_network(m)
    g = multi_sim.graph_of(net)
    roads, index = geometry(net)
    exits = exit_nodes(net)
    pos = {i.id: i.position_xy for i in net.intersections}
    out = dict(maps={m: dict(roads=roads, exits=[[round(pos[e][0], 1), round(pos[e][1], 1)] for e in exits])},
               episodes=[])
    for n, want in [(int(a), b) for a, b in (x.split(":") for x in sys.argv[2:])]:
        for idx in range(200):
            case = multi_sim.make_ring_case(net, f"sumo-{m}", idx, n, max_steps=600)
            a = SumoPursuit(m, case, SPECS["ga"], record=True).run()
            if (a.outcome == "success") != (want in ("win", "both")) or (want == "both" and a.seconds < 40):
                continue
            b = SumoPursuit(m, case, SPECS["enc"], record=True).run()
            if {"win": b.outcome != "success", "both": b.outcome == "success", "lose": b.outcome == "success"}[want]:
                out["episodes"].append(dict(map=m, idx=idx, n=n, kind=want,
                                            runs={"ga": convert(a, net, g, roads, index, exits, pos),
                                                  "enc": convert(b, net, g, roads, index, exits, pos)}))
                print(n, want, idx, a.outcome, a.seconds, b.outcome, b.seconds, flush=True)
                break
    json.dump(out, open(sys.argv[1], "w"), separators=(",", ":"))


main()
