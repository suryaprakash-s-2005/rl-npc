"""Plot results from results/run_all_{gt,det}.json and results/train_*.json.

Usage:
    python -m experiments.plotting [--out results] [--show]

Produces:
    results/reward_curves.png        training reward (moving avg) per run
    results/summary_bars_gt.png      win rate + adaptation time, GT mode
    results/summary_bars_det.png     win rate + adaptation time, detector mode
    results/summary_bars_gt_det.png  paired GT vs detector (adaptation time)
"""

import argparse
import glob
import json
import os
import sys

import numpy as np

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results")
COLORS = {"single": "#c44e52", "partitioned": "#55a868", "partitioned_noprior": "#4c72b0"}


def load_rows(tagging, cond=None):
    if cond in ("on", "off"):
        fp = os.path.join(OUT, f"run_all_{tagging}_{cond}.json")
    else:
        fp = os.path.join(OUT, f"run_all_{tagging}.json")
    if not os.path.exists(fp):
        print(f"no {fp} found; run `python -m experiments.run_all --tagging {tagging} --cond {cond or 'on'}` first")
        return []
    return json.load(open(fp)).get("rows", [])


def plot_reward_curves():
    files = sorted(glob.glob(os.path.join(OUT, "train_*.json")))
    if not files:
        print("no results/train_*.json found")
        return
    fig, ax = plt.subplots(figsize=(8, 5))
    for fp in files:
        data = json.load(open(fp))
        hist = data["history"]
        eps = [h["episode"] for h in hist]
        rew = [h["reward"] for h in hist]
        win = 10
        smooth = np.convolve(rew, np.ones(win) / win, mode="valid") if len(rew) >= win else np.array(rew)
        tag = os.path.basename(fp).replace("train_", "").replace(".json", "")
        ax.plot(eps[: len(smooth)], smooth, label=tag, linewidth=1.0, alpha=0.7)
    ax.set_xlabel("episode")
    ax.set_ylabel("average reward (moving avg)")
    ax.legend(fontsize=7, ncol=2)
    ax.set_title("NPC training reward: strategy-aware vs baseline")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "reward_curves.png"), dpi=150)
    print("saved", os.path.join(OUT, "reward_curves.png"))


def _summarise(rows):
    groups = {v: {"win": [], "adapt": []} for v in COLORS}
    for r in rows:
        v = r.get("variant", r.get("replay"))
        if v not in groups:
            continue
        groups[v]["win"].append(r["win_rate"])
        groups[v]["adapt"].append(r["adapt_time"])
    variants = [v for v in groups if groups[v]["win"]]
    return variants, groups


def _bar(ax, variants, groups, key, title, ylabel, cmap, ylim=None):
    means = [float(np.mean(groups[v][key])) for v in variants]
    stds = [float(np.std(groups[v][key])) for v in variants]
    ax.bar(variants, means, yerr=stds, color=cmap, capsize=4, alpha=0.85)
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    if ylim:
        ax.set_ylim(*ylim)
    for i, (m, s) in enumerate(zip(means, stds)):
        ax.text(i, m + s + 0.02 * max(1, max(means)), f"{m:.1f}", ha="center", fontsize=9)


def plot_summary_bars():
    # per tagging: grouped bars for {variant x cond} on win rate + adaptation
    for tagging in ("gt", "det"):
        rows_on = load_rows(tagging, "on")
        rows_off = load_rows(tagging, "off")
        if not rows_on and not rows_off:
            rows_all = load_rows(tagging)
            if rows_all:
                rows_on = [r for r in rows_all if r.get("cond")]
                rows_off = [r for r in rows_all if not r.get("cond")]
        if not rows_on and not rows_off:
            continue
        variants, gon = _summarise(rows_on)
        _, goff = _summarise(rows_off)
        vlist = [v for v in variants if (gon[v]["win"] or (v in goff and goff[v]["win"]))]
        x = np.arange(len(vlist))
        w = 0.4
        fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
        for key, ylabel, ylim, color, title in (
            ("win", "win rate", (0, 1.05), "#4c72b0", "Win (catch) rate"),
            ("adapt", "steps after switch", None, "#dd8452", "Mean adaptation time"),
        ):
            ax = axes[0] if key == "win" else axes[1]
            a = [np.mean(gon[v][key]) for v in vlist]
            b = [np.mean(goff[v][key]) if v in goff else 0 for v in vlist]
            sa = [np.std(gon[v][key]) for v in vlist]
            sb = [np.std(goff[v][key]) if v in goff else 0 for v in vlist]
            ax.bar(x - w / 2, a, w, yerr=sa, label="cond on", color=color, capsize=3, alpha=0.9)
            ax.bar(x + w / 2, b, w, yerr=sb, label="cond off", color=color, capsize=3, alpha=0.4,
                   hatch="//")
            ax.set_title(title)
            ax.set_ylabel(ylabel)
            if ylim:
                ax.set_ylim(*ylim)
            ax.set_xticks(x)
            ax.set_xticklabels(vlist)
        axes[0].legend()
        fig.suptitle(f"Replay strategies :: policy conditioning [{tagging}]")
        fig.tight_layout()
        fp = os.path.join(OUT, f"summary_bars_{tagging}.png")
        fig.savefig(fp, dpi=150)
        print("saved", fp)

    # paired adaptation-time comparison (GT vs detector)
    rows_gt = load_rows("gt")
    rows_det = load_rows("det")
    if rows_gt and rows_det:
        variants, ggt = _summarise(rows_gt)
        _, gdet = _summarise(rows_det)
        vlist = [v for v in variants if v in gdet]
        x = np.arange(len(vlist))
        w = 0.35
        fig, ax = plt.subplots(figsize=(8, 5))
        a = [np.mean(ggt[v]["adapt"]) for v in vlist]
        b = [np.mean(gdet[v]["adapt"]) for v in vlist]
        sa = [np.std(ggt[v]["adapt"]) for v in vlist]
        sb = [np.std(gdet[v]["adapt"]) for v in vlist]
        ax.bar(x - w / 2, a, w, yerr=sa, label="groundtruth", color="#4c72b0", capsize=4)
        ax.bar(x + w / 2, b, w, yerr=sb, label="online detector", color="#dd8452", capsize=4)
        ax.set_xticks(x)
        ax.set_xticklabels(vlist)
        ax.set_ylabel("mean adaptation time (steps)")
        ax.set_title("Adaptation time: groundtruth vs online detector")
        ax.legend()
        fig.tight_layout()
        fp = os.path.join(OUT, "summary_bars_gt_det.png")
        fig.savefig(fp, dpi=150)
        print("saved", fp)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default=None)
    args = parser.parse_args()
    global OUT
    if args.out:
        OUT = args.out
    os.makedirs(OUT, exist_ok=True)
    plot_reward_curves()
    plot_summary_bars()


if __name__ == "__main__":
    main()