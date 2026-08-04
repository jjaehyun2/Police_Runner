# Real-scale Daejeon main study — 2026-08 status

5 independent seeds, `updates=35000` each, `episodes_per_update=1`, `max_steps=450`,
trained on the real (live-fetched OSMnx) Daejeon `train` network, evaluated on the
real Daejeon `validation` network. Launcher: `scripts/research/real_scale/run_seed.py`.

As of this report: **seeds 0, 1, 2, 4 finished training (35000/35000)**; **seed 3 still
training** (~27000/35000). Everything below covers seeds 0/1/4, whose evaluation has
been redone with the fixes described below.

## Bugs found and fixed during this run

Three real, independently-discovered bugs affected this run. None required
retraining — training itself was unaffected by all three; only the final
evaluation step (and, for the third, scenario generation) needed to be redone.

### 1. Checkpoint-selection freeze (`training/trainer.py`)

`validation_capture_rate` is a single-episode 0/1 score by default. The
checkpoint-selection comparison used strict `>` ("improvement"), so once a run
first hit the ceiling (often within the first few hundred updates), every
later update tied it and none counted as "better" — the saved checkpoint froze
near initialization while the rest of training ran for nothing. Fixed by
treating a tie as an improvement (`>=`), so the checkpoint keeps moving with
training. Regression test:
`tests/research/test_checkpoint_selection_prefers_recent_ties.py`.

### 2. `EpisodeOutcome` enum-identity bug (`experiments/main_study.py`)

`replay_episode()` (`evaluation/paired.py`) returns `research.domain.EpisodeOutcome`.
`main_study.py` imported a *different* class with identical member names from
`osm_demo.models`, so `outcome is EpisodeOutcome.CAPTURE` was silently always
`False` — every seed's final 500-episode paired evaluation recorded a
fabricated **0% capture rate for both the proposed policy and the baseline**,
regardless of what actually happened during replay. Fixed the import; pinned
with `tests/research/test_main_study_execution.py::test_main_study_episode_outcome_is_the_same_class_replay_episode_returns`.
`scripts/research/real_scale/reeval_seed_fixed.py` redoes the evaluation step
against each seed's already-completed final checkpoint.

### 3. Officers placeable at dead-end intersections (`variants/placement.py`)

`generate_placement()` already excluded boundary/no-outgoing-segment nodes
from the **fugitive**'s candidate pool, but not from the **officers**' pool.
An officer could be drawn onto a pure sink intersection
(`outgoing_segment_ids == ()`, e.g. a bbox-edge dead end) and then have
exactly one legal action (`STAY`) for the rest of the episode — physically
unable to ever move again, indistinguishable from a policy that "gave up."
Fixed by requiring `outgoing_segment_ids` non-empty for officer candidates
too. Measured effect on seed 4 vs `greedy_intercept`: capture rate 74.6% →
76.4%, gap -18.6pp → -17.4pp — real but small; most of the gap has a
different cause (see below).

All three fixes verified with the full offline suite (`832 passed, 1 skipped`,
zero regressions) after each change.

## Headline results (seeds 0, 1, 4; post-fix)

**vs. `lowest-legal-action` baseline** (trivial: always picks the lowest legal
action index), 500 paired episodes/seed, real Daejeon validation network:

| seed | our model | baseline |
|---|---|---|
| 0 | 75.6% | 48.6% |
| 1 | 78.4% | 45.0% |
| 4 | 77.0% | 47.2% |

**vs. the pre-existing, audited numeric baselines** (`greedy_intercept_policy`,
`directed_shortest_path_policy` — see `policies/baselines.py`; not written this
session), 500 paired episodes/seed/baseline, post placement-fix:

| seed | our model | greedy_intercept | directed_shortest_path |
|---|---|---|---|
| 0 | 81.2% | 93.4% (−12.2pp, CI −15.2…−9.2) | 88.4% (−7.2pp) |
| 1 | 80.4% | 94.2% (−13.8pp, CI −16.8…−10.8) | 87.4% (−7.0pp) |
| 4 | 76.4% | 93.8% (−17.4pp, CI −20.6…−14.2) | 85.2% |

**We currently lose to both audited baselines, consistently across all three
finished seeds, with confidence intervals that exclude zero.** This is the
main open problem — see the diagnosis below.

## Visual diagnosis (`visualizations/`)

- `failure_episode.gif` / `_contact_sheet.png` — a real episode our model
  loses (`escape`, seed 4's checkpoint). 3 of 6 officers never move: two are
  placed at genuine dead-end intersections (bug #3, now fixed), but one
  (officer 2, at a real 3-way junction) sits with high-confidence `STAY`
  logits (9–12 vs. −10…+1 for the real alternatives) for the entire episode —
  a genuine learned behavior, not a physical constraint. See
  `diagnose_frozen_officers.py`'s output for the full per-step logit trace
  that established this.
- `greedy_intercept_episode.gif` / `_contact_sheet.png` — one `greedy_intercept`
  capture, for reference (note: the specific episode picked here happens to
  start with an officer already adjacent to the fugitive, so it's not the
  most representative "active chase" example — see `paired_*` below for a
  fairer comparison).
- `paired_ours.gif` + `paired_greedy.gif` (and matching contact sheets) — **the
  same sealed case** (identical initial placement for all 6 officers and the
  fugitive, identical fugitive RNG stream) replayed under both policies:
  `greedy_intercept` captures at step 79; our model is still chasing at step
  99 (visually right on top of the fugitive), then **fails to seal the same
  boundary-curve exit route** the fugitive escapes through, and times out to
  `escape` at step 182. Both policies converge the group up the same central
  corridor almost identically — the divergence is specifically in the final
  seal, not the overall chase. This matches the earlier finding
  (`containment_triptych.png`) that the fugitive's BFS-reachable region stays
  near the network's full size until a single chokepoint block collapses it
  to ~0 in one step: our model's chokepoint-blocking timing is less reliable
  than `greedy_intercept`'s always-reactive rule.
- `containment_triptych.png` — three-stage real angular-coverage figure
  (19%→45%→51%, monotone) built for a presentation slide; shows the gradual
  encirclement-forming side of the same behavior.
- `pursuit_encirclement_figure.png` — single-episode capture figure with
  per-officer real direction/distance callouts, built for a presentation
  slide.

## Open problem for whoever picks this up

The dead-end placement bug (#3) is fixed and measured (~1-2pp). The
**dominant** remaining gap vs. `greedy_intercept`/`directed_shortest_path` (still
7-17pp after the fix) looks behaviorally like two related issues, both visible
in the artifacts above:

1. Officers far from the fugitive sometimes settle into a confident,
   high-margin `STAY` even with real legal alternatives (see
   `diagnose_frozen_officers.py`'s trace for officer 2). Possibly weak reward
   signal / undertrained regime for officers that start far away.
2. Even when the group correctly converges (which it does, visibly, in the
   paired comparison), the final chokepoint-sealing move is less reliable
   than `greedy_intercept`'s simple always-reactive rule — timing/coordination
   at the last moment before capture/escape resolves.

Reproduce the diagnostics: `scripts/research/real_scale/diagnose_frozen_officers.py`
(per-step legal actions + policy logits for a real failure episode) and
`scripts/research/real_scale/render_paired_comparison_frames.py` (same-case
paired video). Both take a checkpoint path as configured constants at the top
of the file (update the `CHECKPOINT` path to point at whichever seed you want
to inspect).

## Reproducing / continuing

- `scripts/research/real_scale/run_seed.py <seed>` — the launcher used for all
  5 seeds (needs an admitted `MainStudyAdmissionToken`: a sealed protocol +
  completed pilot report + passing quality attestation; see `main_study.py`).
- `scripts/research/real_scale/reeval_seed_fixed.py <seed> <checkpoint_path>
  [output_path]` — redo just the final paired evaluation (vs.
  lowest-legal-action) against an already-trained checkpoint with the enum
  fix applied.
- `scripts/research/real_scale/smart_baseline_eval.py <seed> <checkpoint_path>`
  — evaluate a checkpoint against both audited numeric baselines, 500
  episodes each, with full `compare_paired_binary` statistics.
- `scripts/research/real_scale/aggregate_real_scale_seeds.py` — pool
  independently-run seed outcomes into one `ConditionStudyResult` via
  `pool_seed_outcomes()`, matching a single-process run's statistics exactly.

Seed 3 is still training; once it finishes, redo its evaluation the same way
before including it in any pooled/final statistics.
