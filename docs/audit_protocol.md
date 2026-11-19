# Audit protocol record

**Status.** This is a reconstruction of the fixed protocol for the deciding run,
generated from the frozen run configurations and per-cell records in the
artifact. The original timestamped pre-registration file was not retained.
This document therefore describes what was fixed and executed; it is not a
substitute for a timestamped pre-registration.

## Question and primary outcome

Does strategy-conditioned partitioned replay improve mid-encounter adaptation
relative to a single replay buffer at a calibrated, non-saturated difficulty?
The primary outcome is the paired per-seed difference (partitioned minus single)
in offline win rate in the deployment-relevant condition with the oracle label
withheld (`cond-off`). Secondary outcomes are online win rate, adaptation time,
the quarter-capacity control, switch reachability, and the oracle-label critic
scale.

## Design

- 2 label conditions: ground-truth conditioning on / off.
- 3 replay arms: single buffer at 200,000 transitions; partitioned replay at
  50,000 per partition (200,000 total); single buffer at 50,000 (quarter-capacity
  control).
- 6 paired seeds: 0, 1, 2, 3, 4, 5.
- 36 cells total; 400 training episodes and 12 evaluation episodes per cell.
- Player HP 60, NPC HP 20; episode cap 400 steps.
- Pairwise switches: aggressive <-> defensive, evasive <-> balanced.
- One switch per episode at step 40.
- Replay seed offset 0.
- Primary comparison: partitioned minus single200k, paired by seed.
- Null or negative outcomes were to be reported regardless of direction.

## Difficulty selection

The original HP 20 task was saturated. The calibration sweep used single200k,
seed 0, 100 training and 12 evaluation episodes at HP 20, 40, 60, and 80. The
selection rule was the lowest HP with an offline win rate below 1.000 and a
non-degenerate median adaptation time. HP 40 was 11/12 with median adaptation
0; HP 60 was 9/12 with median adaptation 1.5 and was selected. The full sweep is
in `results/audit/calibration/calibration.json`.

## Statistics

- Paired differences by seed; exact 95% bootstrap intervals by full enumeration
  of the 6^6 paired resamples.
- Exact two-sided Wilcoxon signed-rank test by enumeration of sign assignments;
  statistic is the positive-rank sum; exact zeros dropped; midranks for ties.
- Holm correction across the four Table II rows.
- TOST equivalence check at +/-0.05.
- MDE80 computed from the observed paired variance at two-sided alpha 0.05.
- No adaptation-time test was pre-specified; adaptation time is descriptive.

## Episode statuses

Evaluation episodes are classified as `ok`, `censored`, `no-post`, or
`short-base`. Only `ok` and `censored` enter the adaptation average; excluded
counts are reported. The corrected grid has 864 episodes: 856 `ok`, 7
`censored`, 1 `no-post`, 0 `short-base`.

## Deviations and archival material

- The submitted code used a four-cycle switch ring and replay seed offset +11;
  the corrected run uses the pairwise schedule and offset 0.
- The exact Wilcoxon implementation was corrected from a signed-rank sum to the
  positive-rank sum; all Table II p-values were recomputed.
- The oracle-label collapse (F5) moved from seed 5/all arms to seed 2/single
  buffers under the corrected schedule. The three-arm seed-5 claim is withdrawn.
- Detector-labelled cells and classifier retraining remain archival; they were
  not rerun under the corrected schedule.
- Table I repeated-checkpoint values are printed records; their per-repeat logs
  do not survive.
