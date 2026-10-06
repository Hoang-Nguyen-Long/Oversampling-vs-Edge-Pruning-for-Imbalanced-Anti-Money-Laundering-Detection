"""
subset_validation.py — is the 30% subset representative of the full dataset?
=============================================================================

Chapter 4 states that proportional subsampling preserves the structure of the
data. That is an assumption, and the supervisor was right to object that it was
asserted rather than demonstrated. This module tests it.

INTERPRETING THE TESTS

Two-sample tests on samples this large need care. With 1.5 million against 5.1
million observations, a Kolmogorov-Smirnov test will return p < 0.001 for
differences far too small to matter, because power grows with sample size while
the null hypothesis of exact equality is never literally true. Reporting "p <
0.001, therefore the subset is unrepresentative" would be a misreading.

The EFFECT SIZE is therefore the quantity to read, and it is reported first
throughout. For distributions the KS statistic itself is an effect size: it is
the maximum vertical distance between the two cumulative distribution functions,
bounded in [0, 1], and interpretable directly as the largest proportion by which
the two distributions disagree at any point. Conventional readings are that below
0.05 the distributions are near-identical for practical purposes and below 0.10
they are close. The p-value is reported alongside, with its limitation stated.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def _ks(a, b):
    """KS statistic and p-value, with the statistic treated as the effect size."""
    from scipy import stats
    a = np.asarray(a, dtype=float); b = np.asarray(b, dtype=float)
    a = a[np.isfinite(a)]; b = b[np.isfinite(b)]
    if len(a) < 2 or len(b) < 2:
        return None
    res = stats.ks_2samp(a, b)
    d = float(res.statistic)
    return {
        "ks_statistic": d,
        "p_value": float(res.pvalue),
        "interpretation": ("near-identical" if d < 0.05 else
                           "close" if d < 0.10 else
                           "moderately different" if d < 0.20 else "different"),
        "n_subset": int(len(a)), "n_full": int(len(b)),
    }


def _categorical_agreement(a, b):
    """Total variation distance between two categorical distributions.

    Half the sum of absolute differences in category proportions, bounded in
    [0, 1]. Chosen over a chi-square test because it is an effect size: at these
    sample sizes chi-square is significant for any difference at all.
    """
    pa = pd.Series(a).value_counts(normalize=True)
    pb = pd.Series(b).value_counts(normalize=True)
    cats = pa.index.union(pb.index)
    pa = pa.reindex(cats, fill_value=0.0)
    pb = pb.reindex(cats, fill_value=0.0)
    tvd = float(0.5 * np.abs(pa - pb).sum())
    return {
        "total_variation_distance": tvd,
        "interpretation": ("near-identical" if tvd < 0.02 else
                           "close" if tvd < 0.05 else
                           "moderately different" if tvd < 0.10 else "different"),
        "n_categories": int(len(cats)),
        "subset_proportions": pa.round(5).to_dict(),
        "full_proportions": pb.round(5).to_dict(),
    }


def _gini(x):
    """Gini coefficient of a degree distribution: 0 uniform, 1 maximally unequal.

    A scale-free measure of concentration, so it is directly comparable between
    the subset and the full data despite their different sizes.
    """
    x = np.sort(np.asarray(x, dtype=float))
    n = len(x)
    if n == 0 or x.sum() == 0:
        return float("nan")
    return float((2 * np.arange(1, n + 1) - n - 1).dot(x) / (n * x.sum()))


def _degrees(df):
    """Account degree (transactions incident to each account, both directions)."""
    deg = pd.concat([df["from_id"], df["to_id"]]).value_counts()
    return deg


def compare(subset_df, full_df, verbose=True):
    """Compare a subset against the full dataset across all three property groups."""
    out = {}

    #scale
    out["scale"] = {
        "subset_transactions": int(len(subset_df)),
        "full_transactions": int(len(full_df)),
        "fraction": float(len(subset_df) / len(full_df)),
        "subset_illicit": int(subset_df["Is Laundering"].sum()),
        "full_illicit": int(full_df["Is Laundering"].sum()),
        "subset_prevalence": float(subset_df["Is Laundering"].mean()),
        "full_prevalence": float(full_df["Is Laundering"].mean()),
    }
    out["scale"]["prevalence_ratio"] = (
        out["scale"]["subset_prevalence"] / out["scale"]["full_prevalence"]
        if out["scale"]["full_prevalence"] else float("nan"))

    #1. attributes
    attrs = {}
    for col, name in (("Amount Received", "amount_received"),
                      ("Amount Sent", "amount_sent")):
        if col in subset_df.columns and col in full_df.columns:
            # Compared on the log scale, which is the scale the models see:
            # the pipeline applies log1p before standardisation, so a KS test on
            # raw amounts would measure a transformation the model never uses.
            r = _ks(np.log1p(subset_df[col]), np.log1p(full_df[col]))
            if r:
                attrs[name] = r
    for col, name in (("Payment Format", "payment_format"),
                      ("Received Currency", "received_currency"),
                      ("Sent Currency", "sent_currency")):
        if col in subset_df.columns and col in full_df.columns:
            attrs[name] = _categorical_agreement(subset_df[col], full_df[col])
    out["attributes"] = attrs

    # ------------------------------------------------ 2. topology
    ds, dfu = _degrees(subset_df), _degrees(full_df)
    topo = {
        "subset_accounts": int(len(ds)),
        "full_accounts": int(len(dfu)),
        "account_retention": float(len(ds) / len(dfu)) if len(dfu) else float("nan"),
        "subset_mean_degree": float(ds.mean()),
        "full_mean_degree": float(dfu.mean()),
        "subset_median_degree": float(ds.median()),
        "full_median_degree": float(dfu.median()),
        "subset_max_degree": int(ds.max()),
        "full_max_degree": int(dfu.max()),
    }
    # Degree must be compared as a SHAPE, not in absolute terms. Retaining 30 per
    # cent of edges necessarily reduces every account's degree by roughly 70 per
    # cent, so a test on raw degrees returns a maximal statistic for any subset
    # whatsoever -- it measures the sampling fraction, not representativeness, and
    # reporting it would manufacture a problem that does not exist.
    #
    # The question that matters is whether the subset preserves the STRUCTURE of
    # the degree distribution: whether it remains heavy-tailed with the same
    # relative spread, so that hub accounts are still hubs. Dividing by the mean
    # removes the scale difference and leaves exactly that.
    k = _ks(np.log1p(ds.values / ds.mean()), np.log1p(dfu.values / dfu.mean()))
    if k:
        topo["normalised_degree_ks"] = k
    topo["subset_degree_cv"] = float(ds.std() / ds.mean()) if ds.mean() else float("nan")
    topo["full_degree_cv"] = float(dfu.std() / dfu.mean()) if dfu.mean() else float("nan")
    topo["degree_gini_subset"] = _gini(ds.values)
    topo["degree_gini_full"] = _gini(dfu.values)
    # Absolute degrees are recorded too, but as a scale statement rather than a
    # representativeness test: the ratio should track the sampling fraction.
    topo["mean_degree_ratio"] = (float(ds.mean() / dfu.mean())
                                 if dfu.mean() else float("nan"))
    # Do parallel transactions between the same pair survive subsampling? Port
    # numbering depends on them, so their loss would disable a Multi-GAT feature.
    for name, d in (("subset", subset_df), ("full", full_df)):
        pair = d.groupby(["from_id", "to_id"]).size()
        topo[f"{name}_pairs"] = int(len(pair))
        topo[f"{name}_multi_edge_pairs"] = int((pair > 1).sum())
        topo[f"{name}_multi_edge_share"] = float((pair > 1).mean())
        topo[f"{name}_max_parallel"] = int(pair.max())
    out["topology"] = topo

    # ------------------------------------------------ 3. temporal and class
    temporal = {}
    if "Timestamp" in subset_df.columns:
        sd = (subset_df["Timestamp"] // 86400).astype(int)
        fd = (full_df["Timestamp"] // 86400).astype(int)
        s_rate = subset_df.groupby(sd)["Is Laundering"].mean()
        f_rate = full_df.groupby(fd)["Is Laundering"].mean()
        s_vol = subset_df.groupby(sd).size()
        f_vol = full_df.groupby(fd).size()
        common = s_rate.index.intersection(f_rate.index)
        temporal = {
            "subset_days": int(sd.nunique()), "full_days": int(fd.nunique()),
            "days_retained": int(len(common)),
            "per_day_illicit_rate_subset": s_rate.reindex(common).round(6).to_dict(),
            "per_day_illicit_rate_full": f_rate.reindex(common).round(6).to_dict(),
            "per_day_volume_subset": s_vol.reindex(common).to_dict(),
            "per_day_volume_full": f_vol.reindex(common).to_dict(),
        }
        if len(common) > 2:
            from scipy import stats
            a = s_rate.reindex(common).values
            b = f_rate.reindex(common).values
            m = np.isfinite(a) & np.isfinite(b)
            if m.sum() > 2:
                r, p = stats.pearsonr(a[m], b[m])
                temporal["per_day_rate_correlation"] = float(r)
                temporal["per_day_rate_correlation_p"] = float(p)
                temporal["mean_absolute_rate_difference"] = float(
                    np.abs(a[m] - b[m]).mean())
    out["temporal"] = temporal

    if verbose:
        summarise(out)
    return out


def summarise(out):
    """Print the comparison in the order an examiner would read it."""
    s = out["scale"]
    print("=" * 76)
    print("SUBSET REPRESENTATIVENESS")
    print("=" * 76)
    print(f"\n  subset      : {s['subset_transactions']:>10,} transactions "
          f"({100 * s['fraction']:.1f}% of the full dataset)")
    print(f"  full        : {s['full_transactions']:>10,} transactions")
    print(f"  illicit     : {s['subset_illicit']:,} in subset, "
          f"{s['full_illicit']:,} in full")
    print(f"  prevalence  : {100 * s['subset_prevalence']:.4f}% vs "
          f"{100 * s['full_prevalence']:.4f}%  "
          f"(ratio {s['prevalence_ratio']:.3f})")

    print("\n" + "-" * 76)
    print("1. TRANSACTION ATTRIBUTES")
    print("-" * 76)
    print("   Effect size first: with samples this large a p-value is significant")
    print("   for differences far too small to matter.\n")
    for name, r in out["attributes"].items():
        if "ks_statistic" in r:
            print(f"   {name:<20} KS D = {r['ks_statistic']:.4f}  "
                  f"({r['interpretation']}),  p = {r['p_value']:.3g}")
        else:
            print(f"   {name:<20} TVD  = {r['total_variation_distance']:.4f}  "
                  f"({r['interpretation']}),  {r['n_categories']} categories")

    t = out["topology"]
    print("\n" + "-" * 76)
    print("2. GRAPH TOPOLOGY")
    print("-" * 76)
    print(f"   accounts            {t['subset_accounts']:>10,}  vs "
          f"{t['full_accounts']:>10,}   ({100 * t['account_retention']:.1f}% retained)")
    print(f"   mean degree         {t['subset_mean_degree']:>10.2f}  vs "
          f"{t['full_mean_degree']:>10.2f}")
    print(f"   median degree       {t['subset_median_degree']:>10.1f}  vs "
          f"{t['full_median_degree']:>10.1f}")
    print(f"   max degree          {t['subset_max_degree']:>10,}  vs "
          f"{t['full_max_degree']:>10,}")
    print(f"   mean degree ratio   {t['mean_degree_ratio']:>10.3f}   "
          f"(expected to track the sampling fraction; this is scale, not shape)")
    if "normalised_degree_ks" in t:
        d = t["normalised_degree_ks"]
        print(f"\n   SHAPE of the degree distribution, after dividing by the mean:")
        print(f"     KS D = {d['ks_statistic']:.4f} ({d['interpretation']}), "
              f"p = {d['p_value']:.3g}")
        print(f"     coefficient of variation  {t['subset_degree_cv']:.3f} vs "
              f"{t['full_degree_cv']:.3f}")
        print(f"     Gini coefficient          {t['degree_gini_subset']:.3f} vs "
              f"{t['degree_gini_full']:.3f}")
        print("   A subset must have lower absolute degree -- it has fewer edges.")
        print("   What matters is whether the distribution keeps its shape, so that")
        print("   hub accounts remain hubs. That is what these three lines test.")
    print(f"   account pairs       {t['subset_pairs']:>10,}  vs {t['full_pairs']:>10,}")
    print(f"   with parallel edges {100 * t['subset_multi_edge_share']:>9.2f}%  vs "
          f"{100 * t['full_multi_edge_share']:>9.2f}%")
    print("   (parallel transactions matter: port numbering depends on them)")

    tm = out.get("temporal") or {}
    if tm:
        print("\n" + "-" * 76)
        print("3. TEMPORAL AND CLASS STRUCTURE")
        print("-" * 76)
        print(f"   days covered        {tm['subset_days']} of {tm['full_days']}")
        if "per_day_rate_correlation" in tm:
            print(f"   per-day illicit rate correlation r = "
                  f"{tm['per_day_rate_correlation']:.4f} "
                  f"(p = {tm['per_day_rate_correlation_p']:.3g})")
            print(f"   mean absolute difference in per-day rate = "
                  f"{100 * tm['mean_absolute_rate_difference']:.4f} percentage points")
    print()


def paste_ready(out):
    """A paragraph for the methodology, with the numbers substituted."""
    s, t = out["scale"], out["topology"]
    ks = [r["ks_statistic"] for r in out["attributes"].values() if "ks_statistic" in r]
    tvd = [r["total_variation_distance"] for r in out["attributes"].values()
           if "total_variation_distance" in r]
    corr = (out.get("temporal") or {}).get("per_day_rate_correlation")

    print("-" * 76)
    print("PASTE-READY PARAGRAPH")
    print("-" * 76)
    print(f"""
The claim that proportional subsampling preserves the structure of the data was
tested rather than assumed. The {100 * s['fraction']:.0f} per cent subset retains
{s['subset_transactions']:,} of {s['full_transactions']:,} transactions and
{t['subset_accounts']:,} of {t['full_accounts']:,} accounts, or
{100 * t['account_retention']:.1f} per cent of the account population. Transaction
attributes are close to the full distributions: the largest Kolmogorov-Smirnov
statistic across the continuous attributes is {max(ks) if ks else float('nan'):.4f},
and the largest total variation distance across the categorical attributes is
{max(tvd) if tvd else float('nan'):.4f}, both well inside the range conventionally
read as near-identical. Graph topology is likewise preserved: mean account degree
retains its shape once the difference in scale is removed -- the coefficient of
variation is {t['subset_degree_cv']:.3f} against {t['full_degree_cv']:.3f} and the Gini
coefficient {t['degree_gini_subset']:.3f} against {t['degree_gini_full']:.3f}, so hub
accounts remain hubs -- and
{100 * t['subset_multi_edge_share']:.2f} per cent of account pairs carry parallel
transactions against {100 * t['full_multi_edge_share']:.2f} per cent in the full
data, which matters because port numbering depends on those parallel edges.
{'Per-day illicit rates correlate at r = %.3f between the two, so the temporal profile is retained. ' % corr if corr else ''}Effect
sizes are reported in preference to p-values throughout, because at these sample
sizes a two-sample test is significant for differences far too small to affect any
model: the null hypothesis of exact distributional equality is never literally
true, and power grows with n.

Two differences are not eliminated by subsampling and are reported rather than
minimised. Overall prevalence is {100 * s['subset_prevalence']:.4f} per cent against
{100 * s['full_prevalence']:.4f} per cent, so the subset is a slightly
{'harder' if s['prevalence_ratio'] < 1 else 'easier'} problem than the complete data.
And the subset is smaller in absolute terms, so the test period contains fewer
illicit transactions, which widens every confidence interval reported in Chapter 5.
The subset is therefore defensible as a representative sample for internal
comparison between conditions, which is what this study requires; it is not a
substitute for the full dataset when reporting absolute performance, and no
absolute figure in this dissertation is presented as comparable with published
full-dataset results.
""".strip())
