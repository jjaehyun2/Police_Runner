"""Export encirclement-start replays (fugitive in the middle, officers surrounding) with the
fugitive's escape region per step: the roads it can reach strictly before every officer
(Dijkstra from the fugitive that may not pass a node an officer reaches first)."""
import base64, heapq, json, os, sys
sys.path.insert(0, "/workspace/cluster/scripts/research/real_scale/beat_greedy")
os.environ["ME_MAX_STEPS"] = "600"
os.environ["ME_PLACEMENT"] = "ring"
import multi_eval as me, common, multi_sim
from obs_ext import exit_nodes
import macro

BG = "/workspace/cluster/scripts/research/real_scale/beat_greedy/"
SPECS = {"ga": "rmacro:" + BG + "runs/ms_ga/best.pt", "enc": "encircle"}
MARGIN_S = 25.0 / 16.0          # an officer within capture radius of a node blocks it


def geometry(net):
    index, roads = {}, []
    for s in net.segments:
        if s.virtual:
            continue
        key = tuple(sorted((s.start_id, s.end_id)))
        if key in index:
            continue
        index[key] = len(roads)
        roads.append([[round(x, 1), round(y, 1)] for x, y in s.geometry_xy])
    pos = {i.id: i.position_xy for i in net.intersections}
    return roads, index, pos


def escape_region(sim, exits):
    g, vp, vf = sim.g, sim.case.vp, sim.case.vf
    tp = {}
    for v in sim.police:
        n0, off = sim.reach(v)
        for node, d in g.dist_from(n0).items():
            t = (off + d) / vp
            if t < tp.get(node, 1e18):
                tp[node] = t
    f0, foff = sim.reach(sim.fugs[0])
    best, heap, region = {f0: foff / vf}, [(foff / vf, f0)], set()
    while heap:
        t, u = heapq.heappop(heap)
        if t > best.get(u, 1e18) or t >= tp.get(u, 1e18) - MARGIN_S:
            continue
        region.add(u)
        for nb, w in g.fwd[u]:
            nt = t + w / vf
            if nt < best.get(nb, 1e18):
                best[nb] = nt; heapq.heappush(heap, (nt, nb))
    return region, sorted(e for e in exits if e in region)


def bits(idx, n):
    b = bytearray((n + 7) // 8)
    for i in idx:
        b[i >> 3] |= 1 << (i & 7)
    return base64.b64encode(bytes(b)).decode()


def run(net, case, key, n, geo, exits):
    roads, index, pos = geo
    team = me.team_for(SPECS[key], net, n)
    pol, stash = team.policy, {}
    if isinstance(pol, macro.MacroTeamPolicy):
        orig = pol._round_setup
        def rs(police, fugitive):
            out = orig(police, fugitive); stash["cands"] = out[0]; return out
        pol._round_setup = rs
    else:
        orig = pol._assign
        def asg(decisions, fnode, *a):
            t = orig(decisions, fnode, *a); stash["targets"], stash["fnode"] = t, fnode; return t
        pol._assign = asg
    sim = multi_sim.MultiPursuitSim(net, case)
    P = lambda xy: [round(xy[0], 1), round(xy[1], 1)]
    frames = []

    def snap():
        region, open_exits = escape_region(sim, exits)
        segs = [i for (a, b), i in index.items() if a in region and b in region]
        f = dict(p=[P(sim.position(v)) for v in sim.police], f=P(sim.position(sim.fugs[0])),
                 reg=bits(segs, len(roads)), ox=[P(pos[e]) for e in open_exits], rn=len(region))
        if isinstance(pol, macro.MacroTeamPolicy):
            f["role"] = list(pol.roles)
            f["tgt"] = [None if (t is None or r == 0) else P(pos[t]) for r, t in zip(pol.roles, pol.targets)]
        else:
            t, fn = stash.get("targets"), stash.get("fnode")
            if t is not None:
                f["role"] = [0 if x == fn else 1 for x in t]
                f["tgt"] = [None if x == fn else P(pos[x]) for x in t]
        return f
    while sim.outcome is None:
        provs = team.act(sim)
        frames.append(snap())
        sim.step(provs)
    frames.append(snap())
    return dict(outcome=sim.outcome, steps=sim.step_count, frames=frames)


def main():
    picks = [tuple(x.split(":")) for x in sys.argv[2:]]      # map:n:kind
    out = dict(maps={}, episodes=[])
    for m in dict.fromkeys(p[0] for p in picks):
        net = common.load_network(m)
        geo = geometry(net)
        exits = exit_nodes(net)
        pos = geo[2]
        out["maps"][m] = dict(roads=geo[0], exits=[[round(pos[e][0], 1), round(pos[e][1], 1)] for e in exits])
        for mm, n, want in picks:
            if mm != m:
                continue
            n = int(n)
            for idx in range(300):
                case = multi_sim.make_ring_case(net, f"ringview-{m}", idx, n, max_steps=600)
                a = run(net, case, "ga", n, geo, exits)
                ok_a = a["outcome"] == "success"
                if want in ("win", "both") and not ok_a or want == "lose" and ok_a:
                    continue
                b = run(net, case, "enc", n, geo, exits)
                ok_b = b["outcome"] == "success"
                if {"win": not ok_b, "both": ok_b and a["steps"] > 60, "lose": ok_b}[want]:
                    out["episodes"].append(dict(map=m, idx=idx, n=n, kind=want, runs={"ga": a, "enc": b}))
                    print(m, n, want, idx, a["outcome"], a["steps"], b["outcome"], b["steps"], flush=True)
                    break
    json.dump(out, open(sys.argv[1], "w"), separators=(",", ":"))


main()
