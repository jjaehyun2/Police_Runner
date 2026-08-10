"""Render the two capture-rate deck charts from scripts/deck/out/capture_stats.json.

Two charts, two baseline families, never mixed in one figure:

* capture_vs_trivial.png -- MAPPO vs lowest-legal-action, per seed + pooled,
  with 95% CI error bars (poolable family, n=500/seed, n=1500 pooled).
* capture_vs_smart.png -- MAPPO vs greedy_intercept vs directed_shortest_path,
  per seed only (non-poolable, different eval vintage), with an honest
  caption stating MAPPO currently trails these baselines.

Run:
    py -3.12 scripts/deck/make_capture_chart.py
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.rcParams["font.family"] = ["Malgun Gothic", "DejaVu Sans", "sans-serif"]
matplotlib.rcParams["axes.unicode_minus"] = False
import matplotlib.pyplot as plt

OUT_DIR = Path(__file__).resolve().parent / "out"
STATS_PATH = OUT_DIR / "capture_stats.json"

MAPPO_COLOR = "#2E5EAA"
BASELINE_COLOR = "#B0B0B0"
GREEDY_COLOR = "#D97D3D"
DSP_COLOR = "#4E9E63"


def _load_stats() -> dict:
    with STATS_PATH.open(encoding="utf-8") as fh:
        return json.load(fh)


def make_capture_vs_trivial(stats: dict) -> Path:
    vs = stats["vs_lowest_legal_action"]
    seeds = [0, 1, 4]
    labels = [f"seed {s}" for s in seeds] + ["pooled (n=1500)"]
    rows = [vs["per_seed"][str(s)] for s in seeds] + [vs["pooled"]]

    mappo_rates = [row["proposed_capture_rate"] for row in rows]
    baseline_rates = [row["baseline_capture_rate"] for row in rows]

    def _yerr(rate_key: str, ci_key: str) -> list[list[float]]:
        lowers, uppers = [], []
        for row in rows:
            rate = row[rate_key]
            ci = row[ci_key]
            lowers.append(max(0.0, rate - ci["lower"]))
            uppers.append(max(0.0, ci["upper"] - rate))
        return [lowers, uppers]

    mappo_yerr = _yerr("proposed_capture_rate", "proposed_capture_rate_ci95")
    baseline_yerr = _yerr("baseline_capture_rate", "baseline_capture_rate_ci95")

    x = range(len(labels))
    width = 0.35
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(
        [i - width / 2 for i in x], mappo_rates, width, label="MAPPO (proposed)", color=MAPPO_COLOR,
        yerr=mappo_yerr, capsize=4, error_kw={"ecolor": "#1A1A1A", "elinewidth": 1.2},
    )
    ax.bar(
        [i + width / 2 for i in x], baseline_rates, width, label="lowest-legal-action (baseline)",
        color=BASELINE_COLOR, yerr=baseline_yerr, capsize=4, error_kw={"ecolor": "#1A1A1A", "elinewidth": 1.2},
    )

    # Risk-difference annotation (the paired-comparison quantity, distinct from
    # the two marginal rate CIs drawn as error bars above) shown above each pair.
    for i, row in enumerate(rows):
        rd = row["risk_difference_proposed_minus_baseline"]
        rd_ci = row["risk_difference_ci95_proposed_minus_baseline"]
        y = max(mappo_rates[i], baseline_rates[i]) + 0.08
        ax.annotate(
            f"RD +{rd:.3f}\n[{rd_ci['lower']:.3f}, {rd_ci['upper']:.3f}]",
            xy=(i, y),
            ha="center",
            fontsize=7.5,
            color="#333333",
        )

    ax.set_ylim(0, 1.05)
    ax.set_ylabel("capture rate")
    ax.set_title("MAPPO vs. lowest-legal-action baseline (paired, per seed + pooled)")
    ax.set_xticks(list(x))
    ax.set_xticklabels(labels)
    ax.legend(loc="lower right")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()

    out_path = OUT_DIR / "capture_vs_trivial.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


def make_capture_vs_smart(stats: dict) -> Path:
    vs = stats["vs_smart_baselines"]
    seeds = [0, 1, 4]
    baselines = ["greedy_intercept", "directed_shortest_path"]
    baseline_colors = {"greedy_intercept": GREEDY_COLOR, "directed_shortest_path": DSP_COLOR}

    def _wilson_yerr(rate: float, ci: dict) -> tuple[float, float]:
        return max(0.0, rate - ci["lower"]), max(0.0, ci["upper"] - rate)

    # MAPPO's rate/CI are read from the greedy_intercept row per seed (both rows
    # in a seed's file evaluate the same checkpoint, so the two MAPPO CIs are
    # nearly identical up to Wilson-interval rounding on n=500; picking one
    # baseline's row consistently keeps the bar's error bar internally
    # attributable to a single stated table).
    mappo_rates, mappo_yerr = [], [[], []]
    baseline_rate_series: dict[str, list[float]] = {b: [] for b in baselines}
    baseline_yerr_series: dict[str, list[list[float]]] = {b: [[], []] for b in baselines}
    for s in seeds:
        results = vs["per_seed"][str(s)]["results"]
        anchor = results[baselines[0]]
        mappo_rates.append(anchor["proposed_capture_rate"])
        lo, hi = _wilson_yerr(anchor["proposed_capture_rate"], anchor["proposed_capture_rate_ci95_wilson_marginal"])
        mappo_yerr[0].append(lo)
        mappo_yerr[1].append(hi)
        for b in baselines:
            stats = results[b]
            baseline_rate_series[b].append(stats["baseline_capture_rate"])
            lo, hi = _wilson_yerr(stats["baseline_capture_rate"], stats["baseline_capture_rate_ci95_wilson_marginal"])
            baseline_yerr_series[b][0].append(lo)
            baseline_yerr_series[b][1].append(hi)

    labels = [f"seed {s}" for s in seeds]
    x = range(len(labels))
    width = 0.25
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(
        [i - width for i in x], mappo_rates, width, label="MAPPO (proposed)", color=MAPPO_COLOR,
        yerr=mappo_yerr, capsize=4, error_kw={"ecolor": "#1A1A1A", "elinewidth": 1.2},
    )
    for offset, b in zip((0, width), baselines):
        ax.bar(
            [i + offset for i in x],
            baseline_rate_series[b],
            width,
            label=b,
            color=baseline_colors[b],
            yerr=baseline_yerr_series[b],
            capsize=4,
            error_kw={"ecolor": "#1A1A1A", "elinewidth": 1.2},
        )

    ax.set_ylim(0, 1.05)
    ax.set_ylabel("capture rate")
    ax.set_title("MAPPO vs. smart baselines (per-seed, not poolable)")
    ax.set_xticks(list(x))
    ax.set_xticklabels(labels)
    ax.legend(loc="lower right")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    caption = (
        "MAPPO currently trails both smart baselines on every seed shown here "
        "(risk difference is negative in all six comparisons). Different eval "
        "vintage from the lowest-legal-action chart -- do not compare directly."
    )
    fig.text(0.5, 0.01, caption, ha="center", fontsize=8, color="#333333", wrap=True)
    fig.tight_layout(rect=(0, 0.06, 1, 1))

    out_path = OUT_DIR / "capture_vs_smart.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


def main() -> None:
    stats = _load_stats()
    p1 = make_capture_vs_trivial(stats)
    p2 = make_capture_vs_smart(stats)
    print(f"wrote {p1} ({p1.stat().st_size} bytes)")
    print(f"wrote {p2} ({p2.stat().st_size} bytes)")


if __name__ == "__main__":
    main()
