"""Regenerate the paper figures from audit records."""

import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.audit.config import AUDIT_ROOT

OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                   "paper", "figures")


def fig2():
    fp = os.path.join(AUDIT_ROOT, "calibration", "calibration.json")
    with open(fp) as f:
        rows = json.load(f)["rows"]
    rows = sorted(rows, key=lambda r: r["hp"])
    hps = [r["hp"] for r in rows]
    wr = [r["offline_win_rate"] for r in rows]
    fig, ax = plt.subplots(figsize=(3.5, 2.15), dpi=300)
    ax.axhline(1.0, color="0.65", linewidth=0.7, linestyle="--")
    ax.plot(hps, wr, color="#1f4e79", linewidth=1.2, marker="o",
            markersize=4.5)
    for hp, w in zip(hps, wr):
        ax.text(hp, w + 0.025, f"{w:.3f}", ha="center", fontsize=6.2)
    ax.scatter([60], [dict(zip(hps, wr))[60]], s=90, facecolors="none",
               edgecolors="#b23a48", linewidths=1.2, zorder=5)
    ax.annotate("selected", xy=(60, 0.75), xytext=(64, 0.86),
                fontsize=6.4, color="#b23a48",
                arrowprops=dict(arrowstyle="-", color="#b23a48", lw=0.7))
    ax.set_xlabel("Player HP", fontsize=7)
    ax.set_ylabel("Offline win rate", fontsize=7)
    ax.set_xlim(14, 86)
    ax.set_ylim(0.58, 1.06)
    ax.tick_params(labelsize=6.5)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    fig.tight_layout(pad=0.25)
    fig.savefig(os.path.join(OUT, "fig2_calibration.png"))
    fig.savefig(os.path.join(OUT, "fig2_calibration.pdf"))
    plt.close(fig)


def fig3(analysis):
    rows = analysis["table2"]
    labels = [f"{r['cond']}-{r['metric']}" for r in rows]
    y = np.arange(len(rows))[::-1]
    delta = np.array([r["delta"] for r in rows])
    lo = np.array([r["ci_low"] for r in rows])
    hi = np.array([r["ci_high"] for r in rows])
    p = [r["p_wilcoxon"] for r in rows]

    fig, ax = plt.subplots(figsize=(3.5, 2.25), dpi=300)
    ax.axvline(0.0, color="0.55", linewidth=0.7, linestyle="--")
    for yy, d, l, h, pp in zip(y, delta, lo, hi, p):
        ax.plot([l, h], [yy, yy], color="0.25", linewidth=1.1)
        ax.plot([l, l], [yy - 0.07, yy + 0.07], color="0.25", linewidth=0.9)
        ax.plot([h, h], [yy - 0.07, yy + 0.07], color="0.25", linewidth=0.9)
        ax.plot(d, yy, "o", color="#1f4e79", markersize=4.5)
        ax.text(0.53, yy, f"p={pp:.3g}", va="center", fontsize=6.3,
                transform=ax.get_yaxis_transform())
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=7)
    ax.set_xlabel("Paired difference (partitioned $-$ single)", fontsize=7)
    ax.set_xlim(-0.30, 0.50)
    ax.tick_params(axis="x", labelsize=6.5)
    ax.tick_params(axis="y", length=0)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    fig.tight_layout(pad=0.25)
    fig.savefig(os.path.join(OUT, "fig3_effects.png"))
    fig.savefig(os.path.join(OUT, "fig3_effects.pdf"))
    plt.close(fig)


def blocks(history, width=50):
    out = []
    for start in range(0, len(history), width):
        chunk = history[start:start + width]
        if not chunk:
            continue
        out.append((start, float(np.mean([e["caught"] for e in chunk]))))
    return out


def fig4():
    arms = [("single200k", "#b23a48"), ("single50k", "#d98c3f"),
            ("partitioned", "#1f4e79")]
    fig, ax = plt.subplots(figsize=(3.5, 2.2), dpi=300)
    for arm, color in arms:
        tag = f"{arm}-gt-on-ep400-sw40-seed2"
        fp = os.path.join(AUDIT_ROOT, "train", tag, "episodes.jsonl")
        with open(fp) as f:
            hist = [json.loads(line) for line in f]
        b = blocks(hist)
        xs = [x + 25 for x, _ in b]
        ys = [v for _, v in b]
        ax.plot(xs, ys, marker="o", markersize=3, linewidth=1.2,
                color=color, label=arm)
    ax.set_xlabel("Training episode", fontsize=7)
    ax.set_ylabel("Catch rate (50-episode blocks)", fontsize=7)
    ax.set_ylim(-0.03, 1.03)
    ax.tick_params(labelsize=6.5)
    ax.legend(fontsize=6.2, frameon=False, loc="upper left")
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    fig.tight_layout(pad=0.25)
    fig.savefig(os.path.join(OUT, "fig4_collapse.png"))
    fig.savefig(os.path.join(OUT, "fig4_collapse.pdf"))
    plt.close(fig)


def main():
    os.makedirs(OUT, exist_ok=True)
    with open(os.path.join(AUDIT_ROOT, "analysis_groundtruth.json")) as f:
        analysis = json.load(f)
    fig2()
    fig3(analysis)
    fig4()
    print(f"wrote figures to {OUT}")


if __name__ == "__main__":
    main()
