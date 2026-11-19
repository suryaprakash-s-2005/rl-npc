"""Small-sample paired statistics for the audit.

Implements the paper's exact choices where they are feasible:
  * paired bootstrap over all n^n index tuples for n <= 6 (6^6 = 46,656)
  * exact sign-flip Wilcoxon signed-rank enumeration for up to 20 non-zero pairs
  * Holm correction, minimum detectable effect, and TOST equivalence tests

For n > 6 the bootstrap uses a fixed-seed Monte Carlo approximation; the exact
Wilcoxon enumeration remains available through n=20.
"""

import itertools
import math

import numpy as np
from scipy import optimize, stats


def _as_arrays(a, b):
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    if a.shape != b.shape:
        raise ValueError(f"paired arrays must have equal shape, got {a.shape} and {b.shape}")
    return a, b


def paired_bootstrap(diff, alpha=0.05, n_boot=200000, seed=0):
    """Percentile bootstrap CI for the mean paired difference.

    If n <= 6, enumerates every n^n resample exactly (the paper's method).
    """
    diff = np.asarray(diff, dtype=np.float64)
    n = len(diff)
    if n == 0:
        return {"mean": float("nan"), "ci_low": float("nan"), "ci_high": float("nan"),
                "n_boot": 0, "method": "empty"}
    if n <= 6:
        means = np.empty(n ** n, dtype=np.float64)
        for i, idx in enumerate(itertools.product(range(n), repeat=n)):
            means[i] = diff[list(idx)].mean()
        method = f"exact-enumeration-{n}^{n}"
        n_boot = len(means)
    else:
        rng = np.random.default_rng(int(seed))
        idx = rng.integers(0, n, size=(int(n_boot), n))
        means = diff[idx].mean(axis=1)
        method = f"monte-carlo-{n_boot}"
    lo, hi = np.percentile(means, [100.0 * alpha / 2.0, 100.0 * (1.0 - alpha / 2.0)])
    # Two-sided bootstrap p-value for H0: mean(diff)=0.
    p = 2.0 * min(float((means <= 0).mean()), float((means >= 0).mean()))
    return {
        "mean": float(diff.mean()),
        "ci_low": float(lo),
        "ci_high": float(hi),
        "p_bootstrap": float(min(1.0, p)),
        "n_boot": int(n_boot),
        "method": method,
    }


def wilcoxon_exact(diff, max_exact=20):
    """Two-sided Wilcoxon signed-rank p-value; exact sign-flip enumeration."""
    diff = np.asarray(diff, dtype=np.float64)
    diff = diff[np.abs(diff) > 1e-15]
    n = len(diff)
    if n == 0:
        return {"statistic": 0.0, "p_value": 1.0, "method": "empty", "n_nonzero": 0}
    ranks = stats.rankdata(np.abs(diff), method="average")
    observed = float(ranks[diff > 0].sum())
    if n <= max_exact:
        wplus = np.empty(2 ** n, dtype=np.float64)
        for i, signs in enumerate(itertools.product((0.0, 1.0), repeat=n)):
            wplus[i] = float(np.sum(ranks * np.asarray(signs)))
        p_low = float((wplus <= observed + 1e-12).mean())
        p_high = float((wplus >= observed - 1e-12).mean())
        p = min(1.0, 2.0 * min(p_low, p_high))
        return {"statistic": observed, "p_value": p,
                "method": f"exact-signflip-{n}", "n_nonzero": n}
    # Normal approximation with tie correction for larger samples.
    se = math.sqrt(n * (n + 1) * (2 * n + 1) / 6.0)
    z = (observed - n * (n + 1) / 4.0) / se
    p = 2.0 * float(stats.norm.sf(abs(z)))
    return {"statistic": observed, "p_value": min(1.0, p),
            "method": "normal-approx", "n_nonzero": n}


def holm(p_values):
    p = np.asarray(p_values, dtype=np.float64)
    m = len(p)
    order = np.argsort(p)
    adjusted = np.empty(m, dtype=np.float64)
    running = 0.0
    for rank, idx in enumerate(order):
        val = min(1.0, (m - rank) * p[idx])
        running = max(running, val)
        adjusted[idx] = running
    return adjusted.tolist()


def mde_paired(diff, alpha=0.05, power=0.8):
    """Minimum detectable mean paired difference (two-sided t, power)."""
    diff = np.asarray(diff, dtype=np.float64)
    n = len(diff)
    if n < 2:
        return float("nan")
    sd = float(diff.std(ddof=1))
    if sd <= 1e-15:
        return 0.0
    se = sd / math.sqrt(n)
    df = n - 1
    tcrit = float(stats.t.ppf(1.0 - alpha / 2.0, df))

    def achieved(delta):
        ncp = delta / se
        return float(stats.nct.sf(tcrit, df, ncp) + stats.nct.cdf(-tcrit, df, ncp))

    try:
        hi = max(10.0 * sd, 1e-6)
        while achieved(hi) < power and hi < 1e9:
            hi *= 2.0
        delta = optimize.brentq(lambda x: achieved(x) - power, 0.0, hi)
        return float(delta)
    except Exception:
        z_alpha = float(stats.norm.ppf(1.0 - alpha / 2.0))
        z_power = float(stats.norm.ppf(power))
        return float((z_alpha + z_power) * se)


def tost(diff, margin, alpha=0.05):
    """Two one-sided tests for equivalence within +/- margin."""
    diff = np.asarray(diff, dtype=np.float64)
    n = len(diff)
    if n < 2 or margin <= 0:
        return {"p_value": float("nan"), "equivalent": False, "margin": margin}
    _, p_low = stats.ttest_1samp(diff, margin, alternative="less")
    _, p_high = stats.ttest_1samp(diff, -margin, alternative="greater")
    p = float(max(p_low, p_high))
    return {"p_value": p, "equivalent": bool(p < alpha), "margin": float(margin)}


def paired_stats(treatment, baseline, margin=None, alpha=0.05, n_boot=200000, seed=0):
    treatment, baseline = _as_arrays(treatment, baseline)
    diff = treatment - baseline
    boot = paired_bootstrap(diff, alpha=alpha, n_boot=n_boot, seed=seed)
    wil = wilcoxon_exact(diff)
    out = {
        "n": int(len(diff)),
        "mean_treatment": float(treatment.mean()),
        "mean_baseline": float(baseline.mean()),
        "mean_diff": float(diff.mean()),
        "median_diff": float(np.median(diff)),
        "sd_diff": float(diff.std(ddof=1)) if len(diff) > 1 else float("nan"),
        "cohen_dz": float(diff.mean() / diff.std(ddof=1))
        if len(diff) > 1 and diff.std(ddof=1) > 1e-15 else float("nan"),
        "ci_low": boot["ci_low"],
        "ci_high": boot["ci_high"],
        "p_bootstrap": boot["p_bootstrap"],
        "p_wilcoxon": wil["p_value"],
        "wilcoxon_method": wil["method"],
        "bootstrap_method": boot["method"],
        "mde_80": mde_paired(diff, alpha=alpha, power=0.8),
        "diff": diff.tolist(),
    }
    if margin is not None:
        out["tost"] = tost(diff, margin, alpha=alpha)
    return out
