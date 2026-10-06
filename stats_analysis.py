"""
stats_analysis.py — the inferential analyses, done properly
"""
from __future__ import annotations

import glob
import json
import os

import numpy as np


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------
def load_scores(results_dir, pattern="*_scores.json"):
    """Load every saved score file, keyed by (source, condition)."""
    out = {}
    for path in sorted(glob.glob(os.path.join(results_dir, pattern))):
        src = os.path.basename(path).replace("_scores.json", "")
        try:
            data = json.load(open(path))
        except (OSError, ValueError):
            continue
        for eid, item in data.items():
            out[(src, eid)] = item
    return out


def per_seed_metric(results, eid, metric="auprc"):
    """The per-seed values of one metric for one condition, in seed order."""
    if eid not in results:
        return None
    return np.array([d[metric] for d in results[eid]["per_seed"] if metric in d],
                    dtype=float)


# 1. Paired analysis across seeds
def paired_comparison(results, eid_a, eid_b, metric="auprc"):
    """Compare two conditions seed by seed.

    Returns the five paired differences together with a paired t-test and a
    paired effect size. The differences themselves are the primary output: with
    five pairs they can simply be read, and reading them is more informative than
    any single summary statistic computed from them.
    """
    a = per_seed_metric(results, eid_a, metric)
    b = per_seed_metric(results, eid_b, metric)
    if a is None or b is None or len(a) != len(b) or len(a) < 2:
        return None

    d = a - b
    n = len(d)
    mean_d = float(d.mean())
    sd_d = float(d.std(ddof=1))
    se = sd_d / np.sqrt(n) if sd_d > 0 else 0.0

    out = {
        "condition_a": eid_a, "condition_b": eid_b, "metric": metric,
        "n_pairs": n,
        "a_values": a.tolist(), "b_values": b.tolist(),
        "differences": d.tolist(),
        "mean_difference": mean_d,
        "sd_difference": sd_d,
        "a_mean": float(a.mean()), "b_mean": float(b.mean()),
        "n_favouring_a": int((d > 0).sum()),
        "ratio_of_means": float(a.mean() / b.mean()) if b.mean() else float("nan"),
    }

    if se > 0:
        from scipy import stats
        t_stat = mean_d / se
        p = float(2 * stats.t.sf(abs(t_stat), df=n - 1))
        ci = stats.t.interval(0.95, df=n - 1, loc=mean_d, scale=se)
        out.update({
            "paired_t": float(t_stat),
            "paired_p": p,
            "ci95_low": float(ci[0]), "ci95_high": float(ci[1]),
            # Cohen's d for paired samples: the mean difference in units of the
            # SD of the differences.
            "cohens_d": float(mean_d / sd_d),
        })
    else:
        out.update({"paired_t": None, "paired_p": None,
                    "ci95_low": None, "ci95_high": None, "cohens_d": None})

    # A distribution-free complement. With five pairs the smallest attainable
    # two-sided p is 1/16 = 0.0625, so this can never reach 0.05 however large the
    # effect -- which is itself worth reporting, because it shows the limit is the
    # sample size rather than the evidence.
    k = int((d > 0).sum())
    from math import comb
    tail = sum(comb(n, i) for i in range(k, n + 1)) / 2 ** n
    out["sign_test_p"] = float(min(1.0, 2 * tail))
    out["sign_test_floor"] = float(2 / 2 ** n)
    return out


# 2. Bootstrap over test transactions
def paired_bootstrap(scores_a, labels, scores_b, n_boot=2000, metric="auprc",
                     seed=0):
    """Resample test transactions to estimate uncertainty in the difference.

    This addresses a source of uncertainty the seed analysis cannot: which
    transactions happened to fall in the test period. The same resampled indices
    are applied to both conditions, so the comparison stays paired.

    It remains an analysis within ONE test period. It does not speak to other
    splits or other datasets.
    """
    from sklearn.metrics import average_precision_score
    rng = np.random.default_rng(seed)
    a, b, y = np.asarray(scores_a), np.asarray(scores_b), np.asarray(labels)
    n = len(y)
    diffs = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        yy = y[idx]
        if yy.sum() == 0 or yy.sum() == len(yy):
            continue                      # a resample with one class is undefined
        diffs.append(average_precision_score(yy, a[idx])
                     - average_precision_score(yy, b[idx]))
    if not diffs:
        return None
    diffs = np.array(diffs)
    return {
        "n_boot": len(diffs),
        "mean_difference": float(diffs.mean()),
        "ci95_low": float(np.percentile(diffs, 2.5)),
        "ci95_high": float(np.percentile(diffs, 97.5)),
        "fraction_favouring_a": float((diffs > 0).mean()),
        "differences": diffs.tolist(),
    }


# 3. Maximum attainable F1
def max_f1(scores, labels, n_thresholds=500):
    """The best F1 achievable by any threshold on these scores.

    The reported F1 depends on the operating threshold, which was selected on
    validation to reach a target precision of 0.10. A low reported F1 therefore
    has two possible causes: the ranking is poor, or the ranking is fine and the
    threshold was chosen for a different objective. Separating them requires
    sweeping the threshold, which needs nothing but the stored scores.
    """
    from sklearn.metrics import precision_recall_curve
    s, y = np.asarray(scores), np.asarray(labels)
    if y.sum() == 0:
        return None
    precision, recall, thresholds = precision_recall_curve(y, s)
    with np.errstate(divide="ignore", invalid="ignore"):
        f1 = 2 * precision * recall / (precision + recall)
    f1 = np.nan_to_num(f1)
    i = int(np.argmax(f1))
    return {
        "max_f1": float(f1[i]),
        "precision_at_max": float(precision[i]),
        "recall_at_max": float(recall[i]),
        "threshold_at_max": float(thresholds[min(i, len(thresholds) - 1)]),
    }


# 4. Seed-level variability
def seed_variability(results, conditions, metric="auprc"):
    """Per-seed values for several conditions, for the seed-comparison figure."""
    table = {}
    for eid in conditions:
        v = per_seed_metric(results, eid, metric)
        if v is not None:
            table[eid] = {
                "values": v.tolist(),
                "mean": float(v.mean()), "sd": float(v.std(ddof=1)) if len(v) > 1 else 0.0,
                "min": float(v.min()), "max": float(v.max()),
                # How large is the spread relative to the signal? A ratio near or
                # above 1 means seed noise is comparable to the effect being
                # measured, which bounds what any comparison can establish.
                "cv": float(v.std(ddof=1) / v.mean()) if v.mean() else float("nan"),
                "range_ratio": float(v.max() / v.min()) if v.min() > 0 else float("nan"),
            }
    return table
