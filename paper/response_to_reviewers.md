# Response to Reviewers

> **SUPERSEDED — do not upload this file.** The final response is
> `response_to_reviewers.html` and `response_to_reviewers.pdf`, built from the
> actual reviewer comments. This markdown file retains the earlier template only.

## Summary of changes

1. Rebuilt the deciding grid under the protocol the paper specifies: pairwise strategy switches (`aggressive↔defensive`, `evasive↔balanced`) and a zero replay-seed offset. The prior code used a four-cycle ring and an arbitrary `+11` replay offset.
2. Fixed the single-buffer RNG stream and verified deterministic reproduction.
3. Fixed an exact Wilcoxon implementation error: the sign-flip enumeration now computes the positive-rank sum, not a signed sum. This changes several exact p-values.
4. Re-ran all 36 ground-truth cells at HP 60 and recomputed Table II, F4, F6, and F5.
5. Corrected F5's scope: the oracle-label collapse reproduces in the single-buffer arms on seed 2, not in all three arms on seed 5. Partitioned replay remains healthy.
6. Replaced Figs. 3 and 4 with records from the corrected grid and stated the provenance of retained archival results.
7. Updated the manuscript's claims from "the effect is absent" to the measured result: no benefit, with the deployment-relevant offline point estimate against partitioned replay and no significant effect after correction.

## Point-by-point responses

### [R1.1 — insert reviewer comment on baseline reproducibility / noise floor]

**Response.** We agree that the baseline must be seeded as carefully as the treatment. The original single-buffer sampler drew indices from the global environment RNG stream; partitioned replay's per-partition storage incidentally isolated its sampling. We gave each replay variant a dedicated RNG stream and verified determinism by exact reproduction of training curves and statistics. Table I is retained as the printed record because its per-repetition logs no longer exist; the manuscript now labels those values as non-regenerable rather than presenting them as fresh measurements.

**Changes.** Section IV-B ("Explicit seeding"); Section V-A; Table I caption.

### [R1.2 — insert reviewer comment on the deciding run / multiple comparisons]

**Response.** The deciding run is a 36-cell grid at six paired seeds. We report exact 95% bootstrap intervals by full enumeration (6^6 = 46,656 paired resamples) and exact two-sided Wilcoxon signed-rank tests by sign-flip enumeration, with Holm correction across the four Table II rows. We have corrected the sign-flip implementation: the previous code formed a signed sum of ranks, which inflated p-values; the corrected test uses the positive-rank sum. All Table II p-values and the Holm-adjusted values have been recomputed.

**Changes.** Section V-C; Table II; Fig. 3; `src/audit/stats.py`.

### [R2.1 — insert reviewer comment on the switch schedule / F5]

**Response.** This was the most consequential correction. The paper specifies pairwise partners, but the audited code cycled through a four-strategy ring. Under the specified pairwise schedule, F5 does not behave as originally reported. The collapse moves from seed 5 to seed 2 and is confined to the single-buffer arms: single200k falls to final catch 0.02 with mean |Q| reaching 5.0×10^5, single50k falls to 0.12 with a peak of 2.1×10^5, and partitioned replay remains healthy at 0.80 with a peak of 739. Seed 5 no longer collapses. The revised manuscript reports the defect in corrected scope and explicitly withdraws the three-arm claim. We retain the archived four-cycle curve as provenance, not as the deciding-run result.

**Changes.** Abstract; Section III-E; Section IV-B ("Corrected switch schedule"); Section V-E; Fig. 4; Limitations; Conclusion.

### [R2.2 — insert reviewer comment on cond-on rows in Table II]

**Response.** The cond-on rows are now labelled as contaminated by the single-buffer F5 collapse on seed 2 and are shown for completeness only. The deployment-relevant cond-off rows do not contain that collapse. We no longer describe the cond-on offline interval as a partitioning effect.

**Changes.** Table II caption; Section V-C; Section V-E.

### [R3.1 — insert reviewer comment on the statistical test / exact enumeration]

**Response.** We fixed the exact Wilcoxon enumeration and re-derived every p-value from the per-seed records. The corrected Table II results are: cond-off offline −0.125, 95% CI [−0.194, −0.056], p = 0.125; cond-off online −0.028, CI [−0.222, +0.153], p = 1.000; cond-on offline +0.111, CI [−0.042, +0.319], p = 0.500; cond-on online +0.111, CI [−0.264, +0.444], p = 0.563. No row survives Holm correction. The offline cond-off bootstrap interval excludes zero but the exact signed-rank test does not; we report both and do not claim a significant harm.

**Changes.** Section V-C; Table II; Fig. 3; `experiments/analyze_audit.py`.

### [R4.1 — insert reviewer comment on the capacity ablation]

**Response.** We accept that "matched total capacity" is not established by assertion. The 50k arm is a separate quarter-capacity single buffer (50k total), while partitioned uses 50k per partition (200k total). At HP 60, reducing single-buffer capacity changes cond-off offline win rate by −0.014 (p = 0.75) and online by −0.111 (p = 0.156), against partitioning differences of −0.125 and −0.028. The capacity manipulation is therefore of the same magnitude as the treatment, and the revised text says a capacity ablation controls for capacity only when that effect is measured.

**Changes.** Section III-D; Section V-D.

### [R5.1 — insert reviewer comment on classifier calibration / detector-labelled cells]

**Response.** The deployed classifier is calibrated to the random-walk opponent, not the competent chaser. The revised manuscript keeps the measured gap (0.836 → 0.306 episode-level accuracy; medians 0.969 → 0.043) and the local-retrain result (window accuracy 0.267 → 0.802; episode accuracy 0.306 → 0.441; random-walk accuracy 0.836 → 0.250). The detector-labelled grid was not re-run under the corrected switch schedule; the manuscript now says the detector-labelled rerun is retained as an archival result, not as a fresh replication.

**Changes.** Section V-G; Limitations; Future work.

### [R5.2 — insert reviewer comment on the calibration / HP selection]

**Response.** The final calibration sweep at 100 training episodes and 12 evaluation episodes gives offline win rates of 1.000 (HP 20), 0.917 (HP 40), 0.750 (HP 60), and 0.750 (HP 80). HP 40 is off the ceiling but its median adaptation is 0; HP 60 is the lowest swept value with both a non-ceiling win rate and non-degenerate adaptation. We state this selection explicitly and retain HP as a limitation.

**Changes.** Section IV-D; Fig. 2; Limitations.

### [R6.1 — insert reviewer comment on artifacts / reproducibility package]

**Response.** The revised artifact tree contains the pre-registered protocol, run-naming scheme, per-episode records, checkpoints, raw cell JSONs, the exact analysis script, and the figure-generation script. Archived four-cycle and offset-11 runs are retained under `results/audit/archive_*` so the provenance of every retained number is visible. The manuscript's claims are limited to the corrected pairwise/offset-0 grid; archival values are labelled as records.

**Changes.** Section VI-C; artifact README; `results/audit/analysis_groundtruth.json`; `experiments/analyze_audit.py`; `experiments/make_paper_figures.py`.

## Residual limitations stated in the revision

- Six paired seeds per cell; wide intervals; the null result is a failure to detect a benefit, not proof of absence.
- The detector-labelled grid and the F7 retraining grid were not re-run under the corrected switch schedule.
- The original ground-truth/cond-on checkpoints were overwritten; F5 is characterized only where artifacts survive.
- The HP sweep is a researcher degree of freedom; the selection rule is reported, but future work should vary HP.
