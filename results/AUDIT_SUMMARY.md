# Audit results summary (corrected protocol)

Protocol: pairwise switch (`aggressive↔defensive`, `evasive↔balanced`),
`switch_step=40`, replay seed offset 0, player HP 60, 400 training / 12
evaluation episodes, paired seeds 0–5.

The pre-audit positive claim is in `FINAL_SUMMARY.md`; that file is retained as
historical record and is superseded by this one.

## Deciding grid (36 cells)

Source: `results/audit/analysis_groundtruth.json`.

| Cond. | Metric | Single | Partitioned | Δ (part − single) | 95% CI | exact p |
|---|---|---:|---:|---:|---|---:|
| off | offline | 0.931 | 0.806 | −0.125 | [−0.194, −0.056] | 0.125 |
| off | online | 0.764 | 0.736 | −0.028 | [−0.222, +0.153] | 1.000 |
| on | offline | 0.694 | 0.806 | +0.111 | [−0.042, +0.319] | 0.500 |
| on | online | 0.528 | 0.639 | +0.111 | [−0.264, +0.444] | 0.563 |

No row survives Holm correction. The cond-on rows include the single-buffer F5
collapse on seed 2.

## Capacity check (F4)

Source: `results/audit/analysis_groundtruth.json`.

| Cond. | Metric | Single 200k | Single 50k | Δ (50k − 200k) | 95% CI | exact p |
|---|---|---:|---:|---:|---|---:|
| off | offline | 0.931 | 0.917 | −0.014 | [−0.153, +0.097] | 0.750 |
| off | online | 0.764 | 0.653 | −0.111 | [−0.236, +0.028] | 0.156 |
| on | offline | 0.694 | 0.681 | −0.014 | [−0.056, +0.028] | 1.000 |
| on | online | 0.528 | 0.653 | +0.125 | [0.000, +0.347] | 0.500 |

## Switch reachability (F6)

Source: `results/audit/analysis_groundtruth.json`.

- 864 evaluation episodes over 72 cell-evaluations.
- 1 `no-post`, 0 `short-base`, 7 `censored`, 856 `ok`.
- 863/864 episodes reach the step-40 switch.

## Oracle-label collapse (F5)

Corrected protocol, cond-on seed 2:

| Arm | Final catch | First-block mean \|Q\| | Max mean \|Q\| |
|---|---:|---:|---:|
| single200k | 0.02 | 663 | 5.3×10^5 |
| single50k | 0.12 | 663 | 2.1×10^5 |
| partitioned | 0.80 | 136 | 739 |

Cond-off seed 2 is healthy in all arms (final catch 0.88–0.90, max mean |Q|
391–428). Under the original four-cycle schedule, single200k collapsed at seed 5
and partitioned stayed healthy; the submitted claim that all three arms
collapsed is not reproduced.

## Calibration

Source: `results/audit/calibration/calibration.json` (single200k, seed 0,
100 training / 12 evaluation episodes).

| HP | Offline win rate | Adaptation mean | Adaptation median |
|---:|---:|---:|---:|
| 20 | 1.000 | 3.9 | 2.0 |
| 40 | 0.917 | 4.4 | 0.0 |
| 60 | 0.750 | 14.7 | 1.5 |
| 80 | 0.750 | 4.8 | 0.0 |

HP 60 is the lowest setting with both a non-ceiling win rate and non-degenerate
adaptation.

## Artifacts

- `results/audit/cells/` — 36 cell JSONs.
- `results/audit/train/` and `results/audit/eval/` — per-episode records.
- `results/audit/analysis_groundtruth.json` — aggregated results.
- `results/audit/archive_offset11/` — archived four-cycle + replay offset 11.
- `results/audit/archive_4cycle_offset0/` — archived four-cycle + replay offset 0.
- `data/audit/checkpoints/` — corrected-grid checkpoints.
- `data/audit/detector_hp60_randomwalk.pt` — detector checkpoint.
