"""Produce final paper figures from the compact within-episode-switch grids.

Reads:
  results/grid_full.json          (GT on/off seeds 0-2, DET on seeds 0-2)
  results/grid_gt_cond_off.json   (GT cond-off seeds 3-5, if present)
  results/grid_det_cond_on.json   (DET cond-on seeds 0-5)
  results/grid_det_cond_off.json  (DET cond-off seeds 0-5)

Writes:
  results/final_win_adapt.png     grouped bar chart: win rate + adaptation time
  results/final_summary.md        consolidated table (overwritten by hand doc too)

Usage:
    python experiments/plot_final.py
"""

import json
import os

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results")
COLORS = {"single": "#c44e52", "partitioned": "#55a868", "partitioned_noprior": "#4c72b0"}


def load(path):
    fp = os.path.join(OUT, path)
    if os.path.exists(fp):
        return json.load(open(fp)).get("rows", [])
    return []


def collect():
    """Return dict config -> {variant: list_of_rows}."""
    full = load("grid_full.json")
    gt_off_ext = load("grid_gt_cond_off.json")
    det_on = load("grid_det_cond_on.json")
    det_off = load("grid_det_cond_off.json")

    data = {("gt", "on"): {}, ("gt", "off"): {}, ("det", "on"): {}, ("det", "off"): {}}
    seen = set()

    def put(config, row):
        v = row["variant"]
        key = (config, v, row["seed"])
        if key in seen:
            return
        seen.add(key)
        data[config].setdefault(v, []).append(row)

    for r in full:
        if r["tagging"] == "gt" and r["cond"] == "on":
            put(("gt", "on"), r)
        elif r["tagging"] == "gt" and r["cond"] == "off":
            put(("gt", "off"), r)
        elif r["tagging"] == "det" and r["cond"] == "on":
            put(("det", "on"), r)
    for r in gt_off_ext:
        put(("gt", "off"), r)
    for r in det_on:
        put(("det", "on"), r)
    for r in det_off:
        put(("det", "off"), r)
    return data


def avg(rows):
    if not rows:
        return 0.0, 0.0, 0
    wins = [r["win_rate"] for r in rows]
    adapts = [r["adapt_time"] for r in rows]
    # ignore sentinel -2.0 for adaptation mean
    as_ = [a for a in adapts if a >= 0]
    m_w, s_w = float(np.mean(wins)), float(np.std(wins))
    m_a = float(np.mean(as_)) if as_ else float("nan")
    s_a = float(np.std(as_)) if as_ else float("nan")
    return m_w, m_a, len(wins)


def fmt_row(config, v):
    rows = data[config].get(v, [])
    mw, ma, n = avg(rows)
    return mw, ma, n


def main():
    global data
    data = collect()

    order = [
        ("GT / cond-on", "gt", "on"),
        ("GT / cond-off", "gt", "off"),
        ("DET / cond-on", "det", "on"),
        ("DET / cond-off", "det", "off"),
    ]
    variants = ["single", "partitioned"]

    print(f"{'config':16s} {'variant':12s} {'n':>2s} {'win':>7s} {'adapt':>8s}")
    table = {}
    for label, tag, cond in order:
        for v in variants:
            mw, ma, n = fmt_row((tag, cond), v)
            table[(tag, cond, v)] = (mw, ma, n)
            print(f"{label:16s} {v:12s} {n:>2d} {mw:7.3f} {ma:8.1f}")

    # grouped bar chart: 4 configs x 2 variants, two panels (win, adapt)
    x = np.arange(len(order))
    w = 0.36
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.6))

    for key, ylabel, ylim, title in (
        ("win", "win rate", (0, 1.12), "Adaptive win (catch) rate"),
        ("adapt", "steps after switch", None, "Mean adaptation time"),
    ):
        ax = axes[0] if key == "win" else axes[1]
        for j, v in enumerate(variants):
            means = []
            stds = []
            for i, (label, tag, cond) in enumerate(order):
                mw, ma, n = table[(tag, cond, v)]
                rows = data[(tag, cond)].get(v, [])
                vals = [r["win_rate"] for r in rows] if key == "win" else [r["adapt_time"] for r in rows if r["adapt_time"] >= 0]
                std = float(np.std(vals)) if vals else 0.0
                val = mw if key == "win" else ma
                means.append(val)
                stds.append(std)
            ax.bar(x + (j - 0.5) * w, means, w, yerr=stds, capsize=3,
                   color=COLORS[v], label=v, edgecolor="black", linewidth=0.5)
        ax.set_xticks(x)
        ax.set_xticklabels([o[0].replace(" / ", "/") for o in order])
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        if ylim:
            ax.set_ylim(*ylim)
        for i, (label, tag, cond) in enumerate(order):
            pass
    axes[0].legend(title="replay")
    fig.tight_layout()
    fp = os.path.join(OUT, "final_win_adapt.png")
    fig.savefig(fp, dpi=150)
    print("\nsaved", fp)

    # also emit a compact markdown table
    lines = ["# Grid summary (within-episode personality switch)\n",
             "| condition | variant | n | win_rate | adapt_time |",
             "|---|---|---:|---:|---:|"]
    for label, tag, cond in order:
        for v in variants:
            mw, ma, n = table[(tag, cond, v)]
            ma_s = f"{ma:.1f}" if ma == ma else "n/a"
            lines.append(f"| {label} | {v} | {n} | {mw:.3f} | {ma_s} |")
    with open(os.path.join(OUT, "grid_summary.md"), "w") as f:
        f.write("\n".join(lines))
    print("saved", os.path.join(OUT, "grid_summary.md"))


if __name__ == "__main__":
    main()
