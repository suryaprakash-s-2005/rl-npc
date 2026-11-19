# Revised manuscript source

`main.html` is the canonical editable source. The submitted PDF had no surviving
LaTeX/Word source, so the manuscript was reconstructed from
`submitted_text.txt` and the embedded figures extracted from the submitted PDF.

## Build

```powershell
& 'C:\Program Files\Google\Chrome\Application\chrome.exe' `
  --headless=new --disable-gpu --no-pdf-header-footer `
  --print-to-pdf='D:\IEEE\project3\paper\revised_draft.pdf' `
  'file:///D:/IEEE/project3/paper/main.html'

& 'D:\torch_venv\Scripts\python.exe' -m experiments.build_highlighted_manuscript
& 'C:\Program Files\Google\Chrome\Application\chrome.exe' `
  --headless=new --disable-gpu --no-pdf-header-footer `
  --print-to-pdf='D:\IEEE\project3\paper\highlighted_manuscript.pdf' `
  'file:///D:/IEEE/project3/paper/highlighted.html'

& 'C:\Program Files\Google\Chrome\Application\chrome.exe' `
  --headless=new --disable-gpu --no-pdf-header-footer `
  --print-to-pdf='D:\IEEE\project3\paper\response_to_reviewers.pdf' `
  'file:///D:/IEEE/project3/paper/response_to_reviewers.html'
```

The clean and highlighted manuscripts render to six US-Letter pages; the
response sheet renders to four pages. Final upload copies are in
`D:\IEEE\project3\submission\`.

## Numeric sources

- `results/audit/analysis_groundtruth.json` — Table II, F4, F5, F6 from the corrected
  pairwise/offset-0 36-cell grid.
- `results/audit/calibration/calibration.json` — final HP calibration sweep.
- `results/audit/train/*/episodes.jsonl` — per-episode training records used for Fig. 4.
- `results/audit/archive_offset11/` — archived four-cycle + replay-offset-11 run.
- `results/audit/archive_4cycle_offset0/` — archived four-cycle + replay-offset-0 run.

Regenerate the analysis and figures with:

```powershell
& 'D:\torch_venv\Scripts\python.exe' -m experiments.analyze_audit
& 'D:\torch_venv\Scripts\python.exe' -m experiments.make_paper_figures
```

## Provenance note

`submitted_text.txt` is the text extracted from `D:\IEEE\WcAISR2027_Paper.pdf`.
The four archived submitted figures are `figures/p2_x171.png`, `p3_x203.png`,
`p4_x217.png`, and `p4_x220.png`. Figs. 3 and 4 in `main.html` are regenerated from
the corrected records; Fig. 1 remains the submitted pipeline diagram. Fig. 2 is
regenerated from the final calibration sweep.
