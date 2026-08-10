"""Render the latency distribution chart from scripts/deck/out/latency_stats.json.

Histogram + CDF of the 1000 timed /recommend calls, with p50/p95/p99 markers,
written to scripts/deck/out/latency_dist.png.

Run:
    py -3.12 scripts/deck/make_latency_chart.py
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.rcParams["font.family"] = ["Malgun Gothic", "DejaVu Sans", "sans-serif"]
matplotlib.rcParams["axes.unicode_minus"] = False
import matplotlib.pyplot as plt
import numpy as np

OUT_DIR = Path(__file__).resolve().parent / "out"
STATS_PATH = OUT_DIR / "latency_stats.json"

HIST_COLOR = "#2E5EAA"
CDF_COLOR = "#4E9E63"
MARKER_COLORS = {"p50_ms": "#D97D3D", "p95_ms": "#B0413E", "p99_ms": "#7A1F2B"}


def main() -> None:
    with STATS_PATH.open(encoding="utf-8") as fh:
        stats = json.load(fh)

    samples = np.asarray(stats["samples_ms"], dtype=float)
    ordered = np.sort(samples)
    cdf = np.arange(1, len(ordered) + 1) / len(ordered)

    fig, (ax_hist, ax_cdf) = plt.subplots(1, 2, figsize=(11, 5))

    ax_hist.hist(samples, bins=40, color=HIST_COLOR, alpha=0.85, edgecolor="white", linewidth=0.4)
    for key in ("p50_ms", "p95_ms", "p99_ms"):
        ax_hist.axvline(stats[key], color=MARKER_COLORS[key], linestyle="--", linewidth=1.4)
        ax_hist.annotate(
            f"{key.replace('_ms', '')}={stats[key]:.2f}ms",
            xy=(stats[key], ax_hist.get_ylim()[1] * 0.9),
            rotation=90,
            va="top",
            ha="right",
            fontsize=8,
            color=MARKER_COLORS[key],
        )
    ax_hist.set_xlabel("latency (ms)")
    ax_hist.set_ylabel("count")
    ax_hist.set_title(f"/recommend latency histogram (n={stats['n']}, warmup={stats['warmup']})")
    ax_hist.spines["top"].set_visible(False)
    ax_hist.spines["right"].set_visible(False)

    ax_cdf.plot(ordered, cdf, color=CDF_COLOR, linewidth=1.8)
    for key in ("p50_ms", "p95_ms", "p99_ms"):
        ax_cdf.axvline(stats[key], color=MARKER_COLORS[key], linestyle="--", linewidth=1.2)
        frac = {"p50_ms": 0.50, "p95_ms": 0.95, "p99_ms": 0.99}[key]
        ax_cdf.plot(stats[key], frac, "o", color=MARKER_COLORS[key])
        ax_cdf.annotate(
            f"{key.replace('_ms', '')}",
            xy=(stats[key], frac),
            xytext=(5, -3),
            textcoords="offset points",
            fontsize=8,
            color=MARKER_COLORS[key],
        )
    ax_cdf.set_xlabel("latency (ms)")
    ax_cdf.set_ylabel("cumulative fraction")
    ax_cdf.set_title("/recommend latency CDF")
    ax_cdf.set_ylim(0, 1.02)
    ax_cdf.spines["top"].set_visible(False)
    ax_cdf.spines["right"].set_visible(False)

    fig.suptitle(f"{stats['checkpoint_label']} -- {stats['network']}", fontsize=9, y=1.02)
    fig.tight_layout()

    out_path = OUT_DIR / "latency_dist.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out_path} ({out_path.stat().st_size} bytes)")


if __name__ == "__main__":
    main()
