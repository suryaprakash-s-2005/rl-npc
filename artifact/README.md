# Audit artifact

This artifact accompanies the revised manuscript. It contains the corrected
pairwise/offset-0 ground-truth grid, the per-episode training and evaluation
records, checkpoints, analysis scripts, figure scripts, and the reconstructed
protocol record. It does not contain the original submitted PDF or any
`project2` material.

## Protocol

- `protocol/audit_protocol.md`: reconstructed fixed protocol record.
- Corrected run: pairwise switch, `switch_step=40`, replay seed offset 0,
  player HP 60, 400 training / 12 evaluation episodes, seeds 0-5.
- Primary comparison: partitioned minus single200k, paired by seed, cond-off
  offline win rate. Negative/null outcomes were to be reported.

## Provenance classes

1. **Corrected-grid (reproduced):** `results/audit/cells/`,
   `results/audit/train/`, `results/audit/eval/`,
   `results/audit/analysis_groundtruth.json`, `results/audit/f5_diagnostics.json`,
   and `results/audit/f7/`.
2. **Archival:** `results/audit/archive_offset11/` (four-cycle, replay offset
   +11) and `results/audit/archive_4cycle_offset0/` (four-cycle, offset 0).
   These are retained for provenance and are not the deciding run.
3. **Printed-only/non-regenerable:** Table I repeated-checkpoint rates and the
   original 0.915 detector log. Their per-repeat records do not survive.

The original timestamped pre-registration file was not retained. The protocol
record above is reconstructed from the frozen configs; this is stated in the
manuscript.

## Layout

- `code/src/`, `code/experiments/`: audited system and analysis code.
- `configs/`: per-run configs copied from the training records.
- `checkpoints/`: corrected-grid policy checkpoints and the random-walk detector.
- `results/audit/cells/`: one JSON per cell.
- `results/audit/train/`, `results/audit/eval/`: per-episode JSONL records.
- `results/audit/calibration/`: final HP sweep.
- `results/audit/f7/`: inference-only classifier diagnostics (no retraining).
- `results/audit/analysis_groundtruth.json`: Table II, F4, F5, F6 values.
- `results/audit/f5_diagnostics.json`: critic-scale onset and catch-rate data.
- `results/AUDIT_SUMMARY.md`: corrected results summary.

## Reproduce

```powershell
& 'D:\torch_venv\Scripts\python.exe' -m experiments.analyze_audit
& 'D:\torch_venv\Scripts\python.exe' -m experiments.analyze_f5_diagnostics
& 'D:\torch_venv\Scripts\python.exe' -m experiments.make_paper_figures
& 'D:\torch_venv\Scripts\python.exe' -m experiments.run_audit_f7 --skip-retrain
```

Check `MANIFEST.sha256` for file hashes.
