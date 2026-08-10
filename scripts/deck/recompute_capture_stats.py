"""Recompute capture-rate statistics for the mentoring deck from committed artifacts.

Two baseline families are kept STRICTLY SEPARATE and never pooled together:

* ``vs_lowest_legal_action`` -- rebuilt from the 500-case-per-seed paired
  outcome files (``real_scale_seed{0,1,4}_outcome_FIXED.json``). Each file's
  ``paired_cases`` list is turned into :class:`PairedCase` objects
  (outcome_a = lowest-legal-action baseline, outcome_b = proposed MAPPO
  policy) and run through :func:`compare_paired_binary`, per-seed and pooled
  across all three seeds (1500 cases). These cases were captured before the
  placement fix landed (see the ``_FIXED`` suffix and the note below).

* ``vs_smart_baselines`` -- read verbatim from the ``smart_baseline_eval_seed*``
  summary files (greedy_intercept, directed_shortest_path). These are
  summary-only (no per-case records survive), from a *different, later* eval
  vintage (post-placement-fix), and are NOT poolable across seeds or with the
  lowest-legal-action family. They are reported per-seed only, exactly as
  produced by that eval run.

Run:
    py -3.12 scripts/deck/recompute_capture_stats.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from pursuit_evasion_rl.osm_demo.metrics import wilson_interval  # noqa: E402
from pursuit_evasion_rl.research.statistics.paired import (  # noqa: E402
    BootstrapPlan,
    EffectDirection,
    PairedCase,
    PracticalThreshold,
    compare_paired_binary,
)

ARTIFACT_DIR = REPO_ROOT / "artifacts" / "research" / "reports" / "real_scale_2026-08"
OUT_DIR = Path(__file__).resolve().parent / "out"
SEEDS = (0, 1, 4)

RATIONALE = "mentoring-deck capture-rate recompute, 2026-08-10, fixed n_bootstrap"


def _load_outcome_fixed(seed: int) -> dict:
    path = ARTIFACT_DIR / f"real_scale_seed{seed}_outcome_FIXED.json"
    with path.open(encoding="utf-8") as fh:
        return json.load(fh)


def _load_smart_baseline(seed: int) -> dict:
    path = ARTIFACT_DIR / f"smart_baseline_eval_seed{seed}.json"
    with path.open(encoding="utf-8") as fh:
        return json.load(fh)


def _cases_from_outcome_fixed(payload: dict) -> list[PairedCase]:
    """Build PairedCase objects with A=baseline, B=proposed (verified against the file's own rates).

    The raw ``paired_cases`` records label fields ``outcome_a``/``outcome_b``,
    but empirically ``mean(outcome_a) == payload["proposed_capture_rate"]`` and
    ``mean(outcome_b) == payload["baseline_capture_rate"]`` for every seed here
    -- i.e. the raw labels are the reverse of "outcome_a=baseline,
    outcome_b=proposed". This function swaps them on the way in so every
    PairedCase built here (and everything downstream: risk-difference sign,
    the LESS_IS_BETTER gate direction) consistently means A=baseline,
    B=proposed.
    """
    seed = payload["training_seed"]
    return [
        PairedCase(
            case_id=case["case_id"],
            training_seed=seed,
            outcome_a=case["outcome_b"],  # baseline, despite the raw field name
            outcome_b=case["outcome_a"],  # proposed, despite the raw field name
        )
        for case in payload["paired_cases"]
    ]


def _bootstrap_plan(seed_label: str) -> BootstrapPlan:
    return BootstrapPlan(rng_seed=20260810, rationale=RATIONALE)


def _threshold() -> PracticalThreshold:
    # effect_size fed to the gate is primary.risk_difference = p_a - p_b, with
    # A=baseline and B=proposed. A meaningful proposed-policy win therefore
    # shows up as a negative effect, so LESS_IS_BETTER is the direction that
    # asks "is B better than A", not "is A better than B".
    return PracticalThreshold(
        metric="capture_rate",
        unit="probability",
        minimum_effect=0.05,
        direction=EffectDirection.LESS_IS_BETTER,
    )


def _ci_to_dict(ci) -> dict:
    return {"lower": ci.lower, "upper": ci.upper, "confidence_level": ci.confidence_level, "method": ci.method}


def _analysis_to_dict(result, *, proposed_rate: float, baseline_rate: float) -> dict:
    """Summarize a PairedComparisonResult with an honest, deck-friendly sign convention.

    The library's ``risk_difference`` on the primary analysis is p_a - p_b
    (outcome_a=baseline minus outcome_b=proposed), i.e. negative when the
    proposed policy captures more often. The deck reports the difference the
    other way around (proposed minus baseline) so a positive number always
    means "the proposed policy is ahead" -- both numbers are included below
    so nothing is hidden by the sign flip.
    """
    primary = result.primary
    table = primary.table
    return {
        "n_pairs": primary.n_pairs,
        "n_seeds": primary.n_seeds,
        "proposed_capture_rate": proposed_rate,
        "proposed_capture_rate_ci95": _ci_to_dict(primary.proportion_b_interval),
        "baseline_capture_rate": baseline_rate,
        "baseline_capture_rate_ci95": _ci_to_dict(primary.proportion_a_interval),
        "risk_difference_proposed_minus_baseline": -primary.risk_difference,
        "risk_difference_ci95_proposed_minus_baseline": {
            "lower": -primary.risk_difference_interval.upper,
            "upper": -primary.risk_difference_interval.lower,
        },
        "library_risk_difference_a_minus_b": primary.risk_difference,
        "library_risk_difference_ci95_a_minus_b": _ci_to_dict(primary.risk_difference_interval),
        "table": {"n11": table.n11, "n10": table.n10, "n01": table.n01, "n00": table.n00},
        "p_value_raw": primary.p_value_raw,
        "gate": None
        if result.gate is None
        else {
            "statistically_significant": result.gate.statistically_significant,
            "practically_significant": result.gate.practically_significant,
            "superiority": result.gate.superiority,
            "non_inferiority": result.gate.non_inferiority,
            "reasons": list(result.gate.reasons),
        },
    }


def build_vs_lowest_legal_action() -> dict:
    per_seed_cases: dict[int, list[PairedCase]] = {}
    per_seed_payload: dict[int, dict] = {}
    for seed in SEEDS:
        payload = _load_outcome_fixed(seed)
        per_seed_payload[seed] = payload
        per_seed_cases[seed] = _cases_from_outcome_fixed(payload)

    out: dict = {"per_seed": {}, "pooled": None}
    for seed in SEEDS:
        cases = per_seed_cases[seed]
        payload = per_seed_payload[seed]
        result = compare_paired_binary(
            cases,
            comparison_id=f"deck_capture_vs_lowest_legal_seed{seed}",
            metric="capture_rate",
            plan=_bootstrap_plan(f"seed{seed}"),
            threshold=_threshold(),
        )
        out["per_seed"][str(seed)] = _analysis_to_dict(
            result,
            proposed_rate=payload["proposed_capture_rate"],
            baseline_rate=payload["baseline_capture_rate"],
        )
        out["per_seed"][str(seed)]["source_file"] = f"real_scale_seed{seed}_outcome_FIXED.json"
        out["per_seed"][str(seed)]["checkpoint"] = payload["checkpoint"]
        out["per_seed"][str(seed)]["n_episodes_reported"] = payload["n_episodes"]

    pooled_cases = [case for seed in SEEDS for case in per_seed_cases[seed]]
    pooled_result = compare_paired_binary(
        pooled_cases,
        comparison_id="deck_capture_vs_lowest_legal_pooled",
        metric="capture_rate",
        plan=_bootstrap_plan("pooled"),
        threshold=_threshold(),
    )
    pooled_n = len(pooled_cases)
    pooled_proposed = sum(1 for c in pooled_cases if c.outcome_b) / pooled_n
    pooled_baseline = sum(1 for c in pooled_cases if c.outcome_a) / pooled_n
    out["pooled"] = _analysis_to_dict(
        pooled_result, proposed_rate=pooled_proposed, baseline_rate=pooled_baseline
    )
    out["pooled"]["source_files"] = [f"real_scale_seed{s}_outcome_FIXED.json" for s in SEEDS]
    out["pooled"]["seeds"] = list(SEEDS)
    out["provenance"] = {
        "outcome_a": "lowest-legal-action baseline",
        "outcome_b": "proposed MAPPO policy",
        "n_per_seed": 500,
        "seeds": list(SEEDS),
        "caveat": (
            "These paired cases (the _FIXED files) predate the placement fix "
            "that also produced the smart_baseline_eval_seed*.json files under "
            "vs_smart_baselines. Do not mix the two families -- see that key's "
            "own provenance note."
        ),
    }
    return out


def _smart_baseline_wilson_cis(results: dict) -> dict:
    """Per-baseline Wilson CIs for the proposed/baseline marginal rates.

    These summary files carry only a 2x2 table (n11/n10/n01/n00), not
    per-case records, so the paired hierarchical-bootstrap CI used for
    vs_lowest_legal_action is not reproducible here. As an honest
    approximation for chart error bars, each rate's own Wilson interval is
    computed from its marginal count via the table -- note this ignores the
    pairing (it's a single-arm CI, not a paired-difference CI). The table's
    "a" index in these files matches the *proposed* policy, not baseline
    (verified against the file's own baseline_capture_rate/proposed_capture_rate
    fields, same reversal as vs_lowest_legal_action).
    """
    cis = {}
    for name, stats in results.items():
        table = stats["table"]
        n_pairs = stats["n_pairs"]
        successes_proposed = table["n11"] + table["n10"]
        successes_baseline = table["n11"] + table["n01"]
        p_lo, p_hi = wilson_interval(successes_proposed, n_pairs)
        b_lo, b_hi = wilson_interval(successes_baseline, n_pairs)
        cis[name] = {
            "proposed_capture_rate_ci95_wilson_marginal": {"lower": p_lo, "upper": p_hi},
            "baseline_capture_rate_ci95_wilson_marginal": {"lower": b_lo, "upper": b_hi},
        }
    return cis


def build_vs_smart_baselines() -> dict:
    per_seed: dict[str, dict] = {}
    for seed in SEEDS:
        payload = _load_smart_baseline(seed)
        wilson_cis = _smart_baseline_wilson_cis(payload["results"])
        results_with_ci = {
            name: {**stats, **wilson_cis[name]} for name, stats in payload["results"].items()
        }
        per_seed[str(seed)] = {
            "checkpoint": payload["checkpoint"],
            "checkpoint_update_index": payload["checkpoint_update_index"],
            "network": payload["network"],
            "episodes_per_baseline": payload["episodes_per_baseline"],
            "results": results_with_ci,
            "source_file": f"smart_baseline_eval_seed{seed}.json",
        }
    return {
        "per_seed": per_seed,
        "provenance": {
            "baselines": ["greedy_intercept", "directed_shortest_path"],
            "vintage": "post-placement-fix (later than the vs_lowest_legal_action _FIXED cases)",
            "poolable": False,
            "caveat": (
                "Summary-only: no per-case PairedCase records survive in these files, so "
                "compare_paired_binary cannot be re-run on them here. Numbers below are the "
                "risk_difference / CI / table already computed by that eval run and are copied "
                "through verbatim, per seed. Never pool across seeds or mix with "
                "vs_lowest_legal_action -- different baselines, different eval vintage."
            ),
        },
    }


def main() -> None:
    vs_lowest = build_vs_lowest_legal_action()
    vs_smart = build_vs_smart_baselines()

    out = {
        "vs_lowest_legal_action": vs_lowest,
        "vs_smart_baselines": vs_smart,
    }
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUT_DIR / "capture_stats.json"
    out_path.write_text(json.dumps(out, indent=2), encoding="utf-8")

    print(f"wrote {out_path}")
    print("\n=== vs_lowest_legal_action ===")
    for seed in SEEDS:
        row = vs_lowest["per_seed"][str(seed)]
        print(
            f"  seed{seed}: proposed={row['proposed_capture_rate']:.3f} "
            f"baseline={row['baseline_capture_rate']:.3f} "
            f"RD(+)={row['risk_difference_proposed_minus_baseline']:.3f}"
        )
    pooled = vs_lowest["pooled"]
    print(
        f"  pooled: proposed={pooled['proposed_capture_rate']:.3f} "
        f"baseline={pooled['baseline_capture_rate']:.3f} "
        f"RD(+)={pooled['risk_difference_proposed_minus_baseline']:.3f} n={pooled['n_pairs']}"
    )
    print("\n=== vs_smart_baselines (per-seed, non-poolable) ===")
    for seed in SEEDS:
        row = vs_smart["per_seed"][str(seed)]
        for baseline_name, stats in row["results"].items():
            print(
                f"  seed{seed} vs {baseline_name}: proposed={stats['proposed_capture_rate']:.3f} "
                f"baseline={stats['baseline_capture_rate']:.3f} RD={stats['risk_difference']:.3f}"
            )


if __name__ == "__main__":
    main()
