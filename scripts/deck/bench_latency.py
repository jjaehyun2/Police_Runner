"""HTTP /recommend latency benchmark against the real deployment/api_server.py.

Starts the FastAPI app as a subprocess (uvicorn), sets PURSUIT_NUM_POLICE,
PURSUIT_MAX_DEGREE and the new PURSUIT_HIDDEN_DIMS env var so the lazily
constructed engine matches checkpoints/road_pursuit/road_pursuit_final.pt
(6 officers, max_degree 5, hidden_dims [128, 128] -- see the fix in
deployment/api_server.py's _get_engine), loads a synthetic 10x10 road grid via
/load_network_file, runs 50 warmup + 1000 timed /recommend calls, and writes
scripts/deck/out/latency_stats.json.

This is adapted from the working probe at
C:\\Users\\dmsak\\.claude\\jobs\\468d69b5\\tmp\\probe\\{bench_latency,serve_probe}.py,
which proved the end-to-end path (subprocess server + preloaded network +
inline hidden_dims override) works; this version launches api_server.py
through its normal env-var-driven lazy engine init instead of poking
internal module globals, so it also exercises the PURSUIT_HIDDEN_DIMS fix.

Run:
    py -3.12 scripts/deck/bench_latency.py
"""
from __future__ import annotations

import json
import os
import platform
import random
import statistics
import subprocess
import sys
import time
from pathlib import Path

import httpx

REPO_ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = Path(__file__).resolve().parent / "out"

HOST, PORT = "127.0.0.1", 8932
BASE_URL = f"http://{HOST}:{PORT}"
N, WARMUP = 1000, 50
ROWS = COLS = 10
SPACING = 100.0
CHECKPOINT_LABEL = "demo-layer road_pursuit_final.pt, 6 officers, 100-intersection synthetic grid"


def build_grid_network() -> dict:
    """A 10x10 bidirectional grid: 100 intersections, matches the deck's demo scale."""
    inters, segs = [], []
    nid = lambda r, c: r * COLS + c
    pos = {}
    for r in range(ROWS):
        for c in range(COLS):
            pos[nid(r, c)] = (c * SPACING, r * SPACING)
            inters.append(
                {
                    "intersection_id": nid(r, c),
                    "position": [c * SPACING, r * SPACING],
                    "outgoing_segments": [],
                    "incoming_segments": [],
                }
            )
    sid = 0
    for r in range(ROWS):
        for c in range(COLS):
            here = nid(r, c)
            for dr, dc in ((0, 1), (1, 0)):
                nr, nc = r + dr, c + dc
                if nr >= ROWS or nc >= COLS:
                    continue
                there = nid(nr, nc)
                for a, b in ((here, there), (there, here)):
                    segs.append(
                        {
                            "segment_id": sid,
                            "start_intersection_id": a,
                            "end_intersection_id": b,
                            "start_pos": list(pos[a]),
                            "end_pos": list(pos[b]),
                            "length": SPACING,
                        }
                    )
                    inters[a]["outgoing_segments"].append(sid)
                    inters[b]["incoming_segments"].append(sid)
                    sid += 1
    boundary = [
        nid(r, c) for r in range(ROWS) for c in range(COLS) if r in (0, ROWS - 1) or c in (0, COLS - 1)
    ]
    return {
        "intersections": inters,
        "segments": segs,
        "boundary_intersections": sorted(boundary),
        "metadata": {
            "num_intersections": len(inters),
            "num_segments": len(segs),
            "source": "synthetic grid, scripts/deck/bench_latency.py",
        },
    }


def _percentile(ordered: list[float], fraction: float) -> float:
    rank = fraction * (len(ordered) - 1)
    lo, hi = int(rank), min(int(rank) + 1, len(ordered) - 1)
    weight = rank - lo
    return ordered[lo] * (1 - weight) + ordered[hi] * weight


def _hardware_info() -> dict:
    import torch

    return {
        "processor": platform.processor(),
        "cpu_count": os.cpu_count(),
        "platform": platform.platform(),
        "python_version": platform.python_version(),
        "torch_version": torch.__version__,
    }


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    network_path = OUT_DIR / "bench_grid_network.json"
    network_path.write_text(json.dumps(build_grid_network()), encoding="utf-8")

    env = os.environ.copy()
    env["PURSUIT_MODEL_PATH"] = str(REPO_ROOT / "checkpoints" / "road_pursuit" / "road_pursuit_final.pt")
    env["PURSUIT_NUM_POLICE"] = "6"
    env["PURSUIT_MAX_DEGREE"] = "5"
    env["PURSUIT_HIDDEN_DIMS"] = "128,128"

    server = subprocess.Popen(
        [
            sys.executable,
            "-m",
            "uvicorn",
            "deployment.api_server:app",
            "--host",
            HOST,
            "--port",
            str(PORT),
            "--log-level",
            "warning",
        ],
        cwd=str(REPO_ROOT),
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )

    try:
        client = httpx.Client(timeout=30.0)
        health = None
        for _ in range(60):
            try:
                health = client.get(f"{BASE_URL}/health").json()
                break
            except httpx.ConnectError:
                if server.poll() is not None:
                    out = server.stdout.read() if server.stdout else ""
                    raise RuntimeError(f"server exited early:\n{out}")
                time.sleep(0.5)
        if health is None:
            raise RuntimeError("server never became reachable on /health")
        print("health:", health, flush=True)

        load_resp = client.post(
            f"{BASE_URL}/load_network_file",
            json={"filepath": str(network_path), "network_id": "default"},
        )
        load_resp.raise_for_status()
        print("network loaded:", load_resp.json(), flush=True)

        rng = random.Random(20260810)

        def one_call():
            police = rng.sample(range(ROWS * COLS), 6)
            fug = rng.choice([i for i in range(ROWS * COLS) if i not in police])
            resp = client.post(
                f"{BASE_URL}/recommend",
                json={
                    "police_positions": police,
                    "fugitive_position": fug,
                    "network_id": "default",
                    "current_step": rng.randint(0, 100),
                },
            )
            resp.raise_for_status()
            return resp

        first = one_call()
        recs = first.json()["recommendations"]
        print(f"sample response recs: {len(recs)}", flush=True)
        if len(recs) != 6:
            raise RuntimeError(f"expected 6 recommendations (num_police=6), got {len(recs)}")

        for _ in range(WARMUP):
            one_call()

        samples_ms: list[float] = []
        wall_start = time.time()
        for _ in range(N):
            start = time.perf_counter_ns()
            one_call()
            samples_ms.append((time.perf_counter_ns() - start) / 1e6)
        wall_s = time.time() - wall_start

        ordered = sorted(samples_ms)
        stats = {
            "n": N,
            "warmup": WARMUP,
            "mean_ms": statistics.fmean(samples_ms),
            "p50_ms": _percentile(ordered, 0.50),
            "p95_ms": _percentile(ordered, 0.95),
            "p99_ms": _percentile(ordered, 0.99),
            "max_ms": ordered[-1],
            "wall_seconds": wall_s,
            "checkpoint_label": CHECKPOINT_LABEL,
            "checkpoint_path": env["PURSUIT_MODEL_PATH"],
            "num_police": 6,
            "network": "synthetic 10x10 grid, 100 intersections",
            "hardware": _hardware_info(),
            "samples_ms": samples_ms,
        }
        out_path = OUT_DIR / "latency_stats.json"
        out_path.write_text(json.dumps(stats, indent=2), encoding="utf-8")
        print(f"wrote {out_path}")
        print(
            f"mean={stats['mean_ms']:.3f}ms p50={stats['p50_ms']:.3f}ms "
            f"p95={stats['p95_ms']:.3f}ms p99={stats['p99_ms']:.3f}ms max={stats['max_ms']:.3f}ms"
        )
        client.close()
    finally:
        server.terminate()
        try:
            server.wait(timeout=10)
        except subprocess.TimeoutExpired:
            server.kill()
            server.wait(timeout=10)


if __name__ == "__main__":
    main()
