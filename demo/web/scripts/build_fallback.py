"""Convert one recorded SUMO episode to the dashboard's offline replay format.

Run with the project venv: .venv/bin/python demo/web/scripts/build_fallback.py
"""

import base64
import json
from pathlib import Path

from pyproj import Transformer


ROOT = Path(__file__).resolve().parents[3]
SOURCE = ROOT / "scripts/research/real_scale/beat_greedy/viewer/sumo_dj.json"
DEST = ROOT / "demo/web/public/fallback.json"
TO_LL = Transformer.from_crs("EPSG:32652", "EPSG:4326", always_xy=True)


def ll(xy):
    lon, lat = TO_LL.transform(*xy)
    return [round(lat, 6), round(lon, 6)]


def main():
    data = json.loads(SOURCE.read_text())
    episode = next(e for e in data["episodes"] if e["map"] == "daejeon_demo" and e["idx"] == 0 and e["n"] == 6)
    original = episode["runs"]["ga"]
    geometry = data["maps"]["daejeon_demo"]
    all_roads = [[ll(p) for p in road] for road in geometry["roads"]]
    points = [p for road in all_roads for p in road]
    bounds = [[min(p[i] for p in points) for i in range(2)], [max(p[i] for p in points) for i in range(2)]]
    raw_frames = original["frames"]
    active = sorted({i for fr in raw_frames for i in range(len(all_roads))
                     if base64.b64decode(fr["reg"])[i >> 3] & (1 << (i & 7))})
    roads = [all_roads[i] for i in active]
    frames = []
    for t, fr in enumerate(raw_frames):
        old_bits = base64.b64decode(fr["reg"])
        bits = bytearray((len(active) + 7) // 8)
        for j, i in enumerate(active):
            if old_bits[i >> 3] & (1 << (i & 7)):
                bits[j >> 3] |= 1 << (j & 7)
        frames.append({
            "t": t, "p": [ll(p) for p in fr["p"]], "f": ll(fr["f"]),
            "role": fr["role"], "tgt": [ll(p) if p else None for p in fr["tgt"]],
            "reg": base64.b64encode(bytes(bits)).decode(), "open": len(fr["ox"]),
            "ox": [ll(p) for p in fr["ox"]],
        })
    payload = {
        "map": {"name": "daejeon_demo", "label": "대전 둔산 · 녹화", "roads": roads,
                "exits": [ll(p) for p in geometry["exits"]], "bounds": bounds},
        "setup": {"police": frames[0]["p"], "fugitive": frames[0]["f"],
                  "policy": "GA 역할 정책", "engine": "sumo", "profile": "ideal", "traffic": None},
        "frames": frames, "outcome": original["outcome"], "seconds": original["steps"],
    }
    DEST.parent.mkdir(parents=True, exist_ok=True)
    DEST.write_text(json.dumps(payload, ensure_ascii=False, separators=(",", ":")) + "\n")
    print(f"{DEST}: {len(frames)} frames, {len(roads)} roads")


if __name__ == "__main__":
    main()
