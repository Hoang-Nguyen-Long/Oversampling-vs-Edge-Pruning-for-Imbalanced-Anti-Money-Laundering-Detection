"""
figures.py
==========

All figures and tables for Chapters 4-5, in one place so the notebook and the
command-line runner produce IDENTICAL output (the notebook imports these same
functions, so the professor sees the generating code inside the document).

Every figure is written as 300-dpi PNG (Word) and vector PDF (LaTeX). Tables are
written as CSV and returned as DataFrames for inline display.

Style choices: the Okabe-Ito palette is colourblind-safe and prints legibly in
greyscale; error bars are always +/- 1 SD across seeds; and any AUPRC axis carries
the prevalence floor as a reference line, because an AUPRC of 0.006 is
meaningless until the reader knows chance is 0.0007.
"""
from __future__ import annotations
import glob, json, os, re

import numpy as np
import json
import pandas as pd
import matplotlib
# Force the non-interactive Agg backend ONLY when running headless (a plain
# script). Inside Jupyter it must NOT be forced, or plt.show() renders nothing
# inline -- and the whole point of the notebook is that the figures appear next
# to the code that generated them.
try:
    get_ipython()          # only defined inside IPython / Jupyter
    _IN_NOTEBOOK = True
except NameError:
    _IN_NOTEBOOK = False
if not _IN_NOTEBOOK:
    matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

plt.rcParams.update({
    "figure.dpi": 110, "savefig.dpi": 300, "savefig.bbox": "tight",
    "font.size": 11, "axes.titlesize": 12, "axes.titleweight": "bold",
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.alpha": 0.25, "grid.linewidth": 0.6,
    "figure.facecolor": "white", "legend.frameon": False,
})
GREY_TXT = "#555555"
PAL = ["#0072B2", "#E69F00", "#009E73", "#D55E00", "#CC79A7",
       "#56B4E9", "#F0E442", "#666666", "#000000"]


def _save(fig, outdir, name, show=False):
    os.makedirs(outdir, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(outdir, f"{name}.{ext}"))
    print("figure:", name)
    if show:
        plt.show()
    else:
        plt.close(fig)
    return fig


# DATA UNDERSTANDING

def data_understanding_figures(csv_path, outdir, show=False):
    """Figures 4.1-4.5: imbalance, temporal behaviour, amounts, split prevalence."""
    df = pd.read_csv(csv_path)
    ill = df["Is Laundering"].astype(int)
    prev = ill.mean()
    t0 = df["Timestamp"].min()
    day = ((df["Timestamp"] - t0) // 86400).astype(int)
    figs = {}

    # 4.1 class imbalance (log scale: linear would render the minority invisible)
    fig, ax = plt.subplots(figsize=(4.8, 3.4))
    c = ill.value_counts().sort_index()
    ax.bar(["Licit", "Illicit"], [c.get(0, 0), c.get(1, 0)], color=[PAL[0], PAL[3]])
    ax.set_yscale("log"); ax.set_ylabel("Transactions (log scale)")
    ax.set_title(f"Class imbalance: {prev*100:.4f}% illicit")
    for i, v in enumerate([c.get(0, 0), c.get(1, 0)]):
        ax.text(i, v, f"{v:,}", ha="center", va="bottom", fontsize=9)
    figs["4.1"] = _save(fig, outdir, "fig4_1_class_imbalance", show)

    # 4.2 illicit rate per day
    g = pd.DataFrame({"day": day, "ill": ill}).groupby("day")["ill"].agg(["mean", "size"])
    sparse = g["size"] < 0.01 * len(df)          # <1% of all transactions
    fig, ax = plt.subplots(figsize=(6.8, 3.6))
    ax.bar(g.index[~sparse], g["mean"][~sparse] * 100, color=PAL[3], alpha=.9,
           label="days with >=1% of transactions")
    if sparse.any():
        ax.bar(g.index[sparse], g["mean"][sparse] * 100, color="#CCCCCC",
               edgecolor=PAL[3], hatch="//", label="sparse days (<1% of transactions)")
        for dday, row in g[sparse].iterrows():
            ax.text(dday, row["mean"] * 100, f"n={int(row['size']):,}",
                    ha="center", va="bottom", fontsize=7, rotation=90, color=GREY_TXT)
    ax.axhline(prev * 100, color=PAL[0], ls="--", lw=1.4, label=f"overall {prev*100:.4f}%")
    ax.set_xlabel("Day"); ax.set_ylabel("Illicit rate (%)")
    ax.set_title("Illicit rate over time"); ax.legend(fontsize=8)
    figs["4.2"] = _save(fig, outdir, "fig4_2_illicit_rate_per_day", show)

    # 4.3 volume per day 
    fig, ax = plt.subplots(figsize=(6.8, 3.6))
    ax.bar(g.index, g["size"], color=PAL[0], alpha=.9)
    ax.set_yscale("log")
    ax.set_xlabel("Day"); ax.set_ylabel("Transactions (log scale)")
    ax.set_title("Transaction volume over time")
    figs["4.3"] = _save(fig, outdir, "fig4_3_volume_per_day", show)

    # 4.3b share of ALL illicit transactions per day 
    fig, ax = plt.subplots(figsize=(6.8, 3.6))
    share = 100 * g["size"] * g["mean"] / max(int(ill.sum()), 1)
    ax.bar(g.index, share, color=PAL[4], alpha=.9)
    ax.set_xlabel("Day"); ax.set_ylabel("% of all illicit transactions")
    ax.set_title("Where the illicit transactions actually are")
    figs["4.3b"] = _save(fig, outdir, "fig4_3b_illicit_share_per_day", show)

    # 4.4 amount distribution by class (log1p: raw amounts are strongly skewed)
    fig, ax = plt.subplots(figsize=(6.0, 3.4))
    amt = np.log1p(df["Amount Received"].astype(float))
    bins = np.linspace(amt.min(), amt.max(), 60)
    ax.hist(amt[ill == 0], bins=bins, color=PAL[0], alpha=.6, density=True, label="Licit")
    ax.hist(amt[ill == 1], bins=bins, color=PAL[3], alpha=.6, density=True, label="Illicit")
    ax.set_xlabel("log(1 + amount received)"); ax.set_ylabel("Density")
    ax.set_title("Amount distribution by class"); ax.legend()
    figs["4.4"] = _save(fig, outdir, "fig4_4_amount_distribution", show)

    # 4.5 prevalence across the chronological splits -- the temporal-bias check
    d = df.sort_values("Timestamp"); n = len(d); s = d["Is Laundering"].to_numpy()
    b1, b2 = int(.6 * n), int(.8 * n)
    rates = [s[:b1].mean()*100, s[b1:b2].mean()*100, s[b2:].mean()*100]
    fig, ax = plt.subplots(figsize=(5.0, 3.4))
    ax.bar(["Train\n(0-60%)", "Val\n(60-80%)", "Test\n(80-100%)"], rates,
           color=[PAL[0], PAL[1], PAL[2]])
    ax.axhline(prev*100, color="grey", ls="--", lw=1.2, label=f"overall {prev*100:.4f}%")
    ax.set_ylabel("Illicit rate (%)"); ax.set_title("Illicit rate across temporal splits")
    ax.legend()
    for i, r in enumerate(rates):
        ax.text(i, r, f"{r:.4f}%", ha="center", va="bottom", fontsize=8)
    figs["4.5"] = _save(fig, outdir, "fig4_5_split_prevalence", show)
    return figs



# helpers for reading results JSON
def _flatten(results):
    rows = []
    for eid, d in results.items():
        if eid.startswith('_'):
            continue   # metadata key, not an experiment
        a, sp = d.get("aggregate", {}), d.get("spec", {})
        rows.append(dict(
            exp=eid, model=sp.get("model"), interv=sp.get("intervention"), loss=sp.get("loss"),
            auprc=a.get("auprc_mean", np.nan), auprc_sd=a.get("auprc_std", 0),
            recall=a.get("recall_mean", np.nan), recall_sd=a.get("recall_std", 0),
            precision=a.get("precision_mean", np.nan), f1=a.get("f1_mean", np.nan),
            p_at_100=a.get("precision@100_mean", np.nan), r_at_100=a.get("recall@100_mean", np.nan),
            fpr=a.get("false_positive_rate_mean", np.nan),
            prevalence=a.get("prevalence_mean", a.get("prevalence", np.nan)),
            train_s=a.get("train_seconds_mean", np.nan),
            peak_mb=a.get("peak_memory_mb_mean", np.nan),
            retained=a.get("retained_train_edges_mean", np.nan),
            n_params=a.get("n_params_mean", np.nan),
            smoothing=d.get("smoothing")))
    return pd.DataFrame(rows).sort_values("exp").reset_index(drop=True)


def _find(results_dir, patterns):
    for p in patterns:
        hits = sorted(glob.glob(os.path.join(results_dir, p)))
        if hits:
            return hits[0]
    return None


# MAIN MATRIX (E0-E8)
def matrix_figures(results_dir, outdir, depth=2, show=False):
    """Figures 5.1-5.4: AUPRC, lift, recall/precision, efficiency for E0-E8."""
    path = _find(results_dir, [f"exp3_main_depth{depth}.json", f"exp4_depth{depth}.json"])
    if not path:
        print(f"[matrix_figures] no results for depth {depth}; skipping"); return {}
    df = _flatten(json.load(open(path)))
    lab = [f"{e}\n{m}" for e, m in zip(df.exp, df.model)]
    prev = np.nanmedian(df.prevalence.replace(0, np.nan))
    figs = {}

    # 5.1 AUPRC with the chance floor drawn in
    fig, ax = plt.subplots(figsize=(7.4, 4.0))
    ax.bar(lab, df.auprc, yerr=df.auprc_sd, capsize=4, color=PAL[0], alpha=.9)
    if np.isfinite(prev):
        ax.axhline(prev, color=PAL[3], ls="--", lw=1.4, label=f"chance = prevalence ({prev:.5f})")
        ax.legend()
    ax.set_ylabel("AUPRC (mean ± SD)"); ax.set_title(f"Detection performance, depth {depth}")
    figs["5.1"] = _save(fig, outdir, f"fig5_1_auprc_depth{depth}", show)

    # 5.2 lift over chance 
    if np.isfinite(prev) and prev > 0:
        fig, ax = plt.subplots(figsize=(7.4, 4.0))
        ax.bar(lab, df.auprc/prev, yerr=df.auprc_sd/prev, capsize=4, color=PAL[2], alpha=.9)
        ax.axhline(1, color="grey", ls="--", lw=1.2, label="chance (1×)")
        ax.set_ylabel("AUPRC lift (×chance)"); ax.set_title(f"AUPRC lift over chance, depth {depth}")
        ax.legend()
        figs["5.2"] = _save(fig, outdir, f"fig5_2_auprc_lift_depth{depth}", show)

    # 5.3 recall and precision side by side (the operational trade-off)
    fig, ax = plt.subplots(figsize=(7.4, 4.0))
    x = np.arange(len(df)); w = .38
    ax.bar(x - w/2, df.recall, w, yerr=df.recall_sd, capsize=3, color=PAL[1], label="Recall")
    ax.bar(x + w/2, df.precision, w, color=PAL[4], label="Precision")
    ax.set_xticks(x); ax.set_xticklabels(lab)
    ax.set_ylabel("Score"); ax.set_title(f"Recall vs precision at the operating threshold, depth {depth}")
    ax.legend()
    figs["5.3"] = _save(fig, outdir, f"fig5_3_recall_precision_depth{depth}", show)

    # 5.4 efficiency: training time and retained graph size (matched-budget evidence)
    fig, axes = plt.subplots(1, 2, figsize=(10.4, 3.8))
    axes[0].bar(lab, df.train_s, color=PAL[5], alpha=.9)
    axes[0].set_ylabel("Training time (s)"); axes[0].set_title("Wall-clock training cost")
    axes[1].bar(lab, df.retained, color=PAL[7], alpha=.9)
    axes[1].set_ylabel("Edges in training graph"); axes[1].set_title("Retained graph size")
    for a in axes: a.tick_params(axis="x", labelsize=8)
    fig.tight_layout()
    figs["5.4"] = _save(fig, outdir, f"fig5_4_efficiency_depth{depth}", show)
    return figs


# DEPTH SWEEP + OVER-SMOOTHING
def depth_figures(results_dir, outdir, depths=(2, 4, 8, 16), show=False):
    """Figures 5.5-5.7: AUPRC vs depth, and Dirichlet/MAD vs depth (H2, H4)."""
    recs, smooth = [], {}
    for d in depths:
        p = _find(results_dir, [f"exp4_depth{d}.json", f"exp3_main_depth{d}.json"])
        if not p:
            continue
        res = json.load(open(p))
        for eid, item in res.items():
            if eid.startswith('_'):
                continue
            a = item.get("aggregate", {})
            recs.append(dict(depth=d, exp=eid, model=item["spec"]["model"],
                             interv=item["spec"]["intervention"],
                             auprc=a.get("auprc_mean", np.nan), sd=a.get("auprc_std", 0),
                             recall=a.get("recall_mean", np.nan)))
            sm = item.get("smoothing")
            if sm:
                smooth[(d, eid)] = sm
    if not recs:
        print("[depth_figures] no depth results; skipping"); return {}
    df = pd.DataFrame(recs); figs = {}

    # 5.5 AUPRC vs depth -- THE over-smoothing performance figure
    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    for i, (exp, g) in enumerate(df.groupby("exp")):
        g = g.sort_values("depth")
        ax.errorbar(g.depth, g.auprc, yerr=g.sd, marker="o", capsize=3,
                    color=PAL[i % len(PAL)],
                    label=f"{exp} {g.model.iloc[0]}/{g.interv.iloc[0]}")
    ax.set_xscale("log", base=2); ax.set_xticks(sorted(df.depth.unique()))
    ax.get_xaxis().set_major_formatter(mticker.ScalarFormatter())
    ax.set_xlabel("Depth (layers)"); ax.set_ylabel("AUPRC (mean ± SD)")
    ax.set_title("Effect of depth on detection performance"); ax.legend(fontsize=8, ncol=2)
    figs["5.5"] = _save(fig, outdir, "fig5_5_auprc_vs_depth", show)

    # 5.6 final-layer Dirichlet energy vs depth (global and minority-incident)
    for scope, fname, num in [("dirichlet", "fig5_6_dirichlet_vs_depth", "5.6"),
                              ("mad", "fig5_7_mad_vs_depth", "5.7")]:
        rows = []
        for (d, eid), sm in smooth.items():
            # profile_layers_full returns FLAT keys: 'dirichlet' / 'dirichlet_minority'
            glob_v = sm.get(scope)
            min_v = sm.get(f"{scope}_minority")
            if glob_v:
                rows.append(dict(depth=d, exp=eid, final=glob_v[-1],
                                 final_min=(min_v[-1] if min_v else np.nan)))
        if not rows:
            continue
        sd = pd.DataFrame(rows)
        fig, axes = plt.subplots(1, 2, figsize=(10.6, 4.0), sharex=True)
        for i, (exp, g) in enumerate(sd.groupby("exp")):
            g = g.sort_values("depth")
            axes[0].plot(g.depth, g.final, marker="o", color=PAL[i % len(PAL)], label=exp)
            if g.final_min.notna().any():
                axes[1].plot(g.depth, g.final_min, marker="s", color=PAL[i % len(PAL)], label=exp)
        for a, t in zip(axes, ["All accounts", "Minority-incident accounts"]):
            a.set_xscale("log", base=2); a.set_xticks(sorted(sd.depth.unique()))
            a.get_xaxis().set_major_formatter(mticker.ScalarFormatter())
            a.set_xlabel("Depth (layers)"); a.set_title(t)
        axes[0].set_ylabel(f"Final-layer {scope}")
        axes[0].legend(fontsize=8, ncol=2)
        fig.suptitle(f"{scope.capitalize()} at the final layer as depth increases",
                     fontweight="bold")
        fig.tight_layout()
        figs[num] = _save(fig, outdir, fname, show)

    # 5.8 per-layer profiles at the deepest available depth
    deepest = max(d for d, _ in smooth) if smooth else None
    if deepest is not None:
        fig, axes = plt.subplots(1, 2, figsize=(10.6, 4.0))
        for i, ((d, eid), sm) in enumerate((k, v) for k, v in smooth.items() if k[0] == deepest):
            de = sm.get("dirichlet")
            ma = sm.get("mad")
            if de: axes[0].plot(range(len(de)), de, marker="o", color=PAL[i % len(PAL)], label=eid)
            if ma: axes[1].plot(range(len(ma)), ma, marker="o", color=PAL[i % len(PAL)], label=eid)
        axes[0].set_ylabel("Dirichlet energy"); axes[1].set_ylabel("MAD")
        for a in axes: a.set_xlabel("Layer")
        axes[0].legend(fontsize=8, ncol=2)
        fig.suptitle(f"Layer-wise representation collapse at depth {deepest}", fontweight="bold")
        fig.tight_layout()
        figs["5.8"] = _save(fig, outdir, f"fig5_8_layerwise_depth{deepest}", show)
    return figs


# INTERVENTION STRENGTH (RQ5)
def strength_figures(results_dir, outdir, show=False):
    """Figures 5.9-5.10: performance across DropEdge rate and GraphSMOTE-Edge ratio."""
    path = os.path.join(results_dir, "exp5_intervention_strength.json")
    if not os.path.exists(path):
        print("[strength_figures] no strength sweep; skipping"); return {}
    data = json.load(open(path)); figs = {}

    drop, smote = [], []
    for key, item in data.items():
        for eid, d in item.get("results", {}).items():
            if eid.startswith('_'):
                continue
            a = d.get("aggregate", {})
            row = dict(exp=eid, auprc=a.get("auprc_mean", np.nan), sd=a.get("auprc_std", 0),
                       recall=a.get("recall_mean", np.nan), recall_sd=a.get("recall_std", 0))
            if key.startswith("dropedge"):
                drop.append({**row, "p": item["drop_rate"]})
            else:
                smote.append({**row, "r": item["smote_ratio"], "k": item["smote_k"]})

    if drop:
        dd = pd.DataFrame(drop)
        fig, axes = plt.subplots(1, 2, figsize=(10.4, 4.0))
        for i, (exp, g) in enumerate(dd.groupby("exp")):
            g = g.sort_values("p")
            axes[0].errorbar(g.p, g.auprc, yerr=g.sd, marker="o", capsize=3, color=PAL[i], label=exp)
            axes[1].errorbar(g.p, g.recall, yerr=g.recall_sd, marker="o", capsize=3, color=PAL[i], label=exp)
        axes[0].set_ylabel("AUPRC"); axes[1].set_ylabel("Recall")
        for a in axes:
            a.set_xlabel("DropEdge rate p"); a.legend(fontsize=9)
        fig.suptitle("Sensitivity to DropEdge strength (H3, RQ5)", fontweight="bold")
        fig.tight_layout()
        figs["5.9"] = _save(fig, outdir, "fig5_9_dropedge_strength", show)

    if smote:
        ss = pd.DataFrame(smote)
        fig, axes = plt.subplots(1, 2, figsize=(10.4, 4.0))
        for i, ((exp, k), g) in enumerate(ss.groupby(["exp", "k"])):
            g = g.sort_values("r")
            axes[0].errorbar(g.r, g.auprc, yerr=g.sd, marker="o", capsize=3,
                             color=PAL[i % len(PAL)], label=f"{exp} K={k}")
            axes[1].errorbar(g.r, g.recall, yerr=g.recall_sd, marker="o", capsize=3,
                             color=PAL[i % len(PAL)], label=f"{exp} K={k}")
        axes[0].set_ylabel("AUPRC"); axes[1].set_ylabel("Recall")
        for a in axes:
            a.set_xlabel("Oversampling ratio r"); a.legend(fontsize=8)
        fig.suptitle("Sensitivity to GraphSMOTE-Edge strength (RQ5)", fontweight="bold")
        fig.tight_layout()
        figs["5.10"] = _save(fig, outdir, "fig5_10_smote_strength", show)
    return figs


# TABLES
def results_tables(results_dir, outdir, depths=(2, 4, 8, 16), reference_depth=2):
    """Table 5.1 (main matrix) and Table 5.2 (depth x diagnostics) as CSV + DataFrame."""
    os.makedirs(outdir, exist_ok=True)
    out = {}

    p = _find(results_dir, [f"exp3_main_depth{reference_depth}.json",
                            f"exp4_depth{reference_depth}.json"])
    if p:
        df = _flatten(json.load(open(p)))
        t1 = pd.DataFrame({
            "Exp": df.exp, "Backbone": df.model, "Intervention": df.interv, "Loss": df.loss,
            "AUPRC": [f"{a:.4f} ± {s:.4f}" for a, s in zip(df.auprc, df.auprc_sd)],
            "Lift": [f"{a/pv:.1f}×" if pv and np.isfinite(pv) and pv > 0 else "—"
                     for a, pv in zip(df.auprc, df.prevalence)],
            "Recall": [f"{r:.3f} ± {s:.3f}" for r, s in zip(df.recall, df.recall_sd)],
            "Precision": [f"{v:.3f}" for v in df.precision],
            "F1": [f"{v:.3f}" for v in df.f1],
            "P@100": [f"{v:.3f}" for v in df.p_at_100],
            "R@100": [f"{v:.3f}" for v in df.r_at_100],
            "Train s": [f"{v:.1f}" for v in df.train_s],
            "Peak MB": [f"{v:.0f}" if np.isfinite(v) else "—" for v in df.peak_mb],
            "Edges": [f"{v:,.0f}" if np.isfinite(v) else "—" for v in df.retained],
        }).set_index("Exp")
        t1.to_csv(os.path.join(outdir, "table5_1_main_matrix.csv"))
        print("table: table5_1_main_matrix.csv"); out["table5_1"] = t1

    rows = []
    for d in depths:
        pth = _find(results_dir, [f"exp4_depth{d}.json", f"exp3_main_depth{d}.json"])
        if not pth:
            continue
        for eid, item in json.load(open(pth)).items():
            if eid.startswith('_'):
                continue
            a, sm = item.get("aggregate", {}), item.get("smoothing")
            de = sm.get("dirichlet") if sm else None
            ma = sm.get("mad") if sm else None
            dmin = sm.get("dirichlet_minority") if sm else None
            rows.append({
                "Depth": d, "Exp": eid, "Backbone": item["spec"]["model"],
                "Intervention": item["spec"]["intervention"],
                "AUPRC": f"{a.get('auprc_mean',float('nan')):.4f} ± {a.get('auprc_std',0):.4f}",
                "Recall": f"{a.get('recall_mean',float('nan')):.3f}",
                "Final Dirichlet": f"{de[-1]:.3f}" if de else "—",
                "Final MAD": f"{ma[-1]:.4f}" if ma else "—",
                "Final Dirichlet (minority)": f"{dmin[-1]:.3f}" if dmin else "—",
            })
    if rows:
        t2 = pd.DataFrame(rows).sort_values(["Depth", "Exp"]).set_index(["Depth", "Exp"])
        t2.to_csv(os.path.join(outdir, "table5_2_depth_diagnostics.csv"))
        print("table: table5_2_depth_diagnostics.csv"); out["table5_2"] = t2

    # Table 5.3: intervention strength 
    # Reports the sweep over DropEdge rate and oversampling ratio. Its absence
    # was why H3 could not be assessed from the tables alone: the figures showed
    # the sweep but no table gave the numbers to quote.
    strength = None
    for name in ("exp5_intervention_strength", "exp3_fanout_sensitivity"):
        p = os.path.join(results_dir, f"{name}.json")
        if os.path.exists(p):
            try:
                strength = json.load(open(p))
            except (OSError, ValueError):
                strength = None
            break
    if strength:
        strength.pop("_config", None)
        rows = []
        for key, item in sorted(strength.items()):
            setting = (item.get("drop_rate") if item.get("drop_rate") is not None
                       else item.get("smote_ratio"))
            kind = ("DropEdge rate" if item.get("drop_rate") is not None
                    else "Oversampling ratio")
            for eid, d in (item.get("results") or {}).items():
                if eid.startswith("_"):
                    continue
                a = d.get("aggregate", {})
                rows.append({
                    "Intervention": kind,
                    "Setting": setting,
                    "Exp": eid,
                    "AUPRC": f"{a.get('auprc_mean', float('nan')):.4f} ± "
                             f"{a.get('auprc_std', float('nan')):.4f}",
                    "Recall": f"{a.get('recall_mean', float('nan')):.3f}",
                    "Precision": f"{a.get('precision_mean', float('nan')):.3f}",
                    "MP edges": f"{a.get('retained_train_edges_mean', 0):,.0f}",
                    "Seeds": a.get("n_seeds"),
                })
        if rows:
            t3 = pd.DataFrame(rows).sort_values(["Intervention", "Setting", "Exp"])
            t3.to_csv(os.path.join(outdir, "table5_3_intervention_strength.csv"),
                      index=False)
            print("table: table5_3_intervention_strength.csv"); out["table5_3"] = t3
    return out


# COMPREHENSIVE COMPARISON FIGURES
# Cover every criterion in Chapter 3, Section 3.10. All read saved JSON, so
# they can be regenerated freely without retraining.
def _wide(results_dir):
    """Deduplicated wide table: one row per (depth, experiment).

    The main-matrix and depth-sweep result files overlap at the reference depth,
    so the raw export contains each shared condition twice. Every figure below
    uses this deduplicated view.
    """
    from results_export import export_wide_dedup
    return export_wide_dedup(results_dir)


# Compact names keep ten conditions legible on one axis; the full specification
# is in the exported tables, so nothing is lost by abbreviating here.
_MODEL_SHORT = {"mlp": "MLP", "gat": "GAT", "multi_gat": "MGAT"}
_INTERV_SHORT = {"none": "-", "dropedge": "Drop", "edge_smote": "SMOTE",
                 "duplicate": "Dup"}


def _label(row):
    m = _MODEL_SHORT.get(row["model"], str(row["model"]))
    i = _INTERV_SHORT.get(row["intervention"], str(row["intervention"]))
    return f"{row['experiment']}\n{m}/{i}"


def efficiency_figures(results_dir, outdir, depth=2, show=False):
    """Figures 5.11-5.13: the full efficiency picture (RQ3, Section 3.10.4).

    Efficiency is reported as four separate quantities rather than one number,
    because the interventions load different stages: expansion pays at training
    time and enlarges the graph, contraction saves there, and neither materially
    changes inference cost.
    """
    df = _wide(results_dir)
    df = df[(df.depth == depth)]
    if df.empty:
        print(f"[efficiency_figures] no results at depth {depth}"); return {}
    df = df.sort_values("experiment")
    lab = [_label(r) for _, r in df.iterrows()]
    figs = {}

    # 5.11 four-panel efficiency comparison
    panels = [("train_seconds_mean", "Training time (s)", PAL[0], "train_seconds_std"),
              ("inference_seconds_mean", "Inference time (s)", PAL[5], None),
              ("peak_memory_mb_mean", "Peak GPU memory (MB)", PAL[3], None),
              ("retained_train_edges_mean", "Edges in training graph", PAL[7], None)]
    fig, axes = plt.subplots(2, 2, figsize=(12.5, 7.6))
    for ax, (col, ylab, colour, errcol) in zip(axes.ravel(), panels):
        if col not in df:
            ax.set_visible(False); continue
        if df[col].fillna(0).abs().sum() == 0:
            # Peak GPU memory is zero when the runs were executed on CPU; showing
            # an empty axis would imply the models used no memory at all.
            ax.text(0.5, 0.5, f"{ylab}\nnot recorded (CPU run)", ha="center",
                    va="center", transform=ax.transAxes, fontsize=10, color=GREY_TXT)
            ax.set_xticks([]); ax.set_yticks([]); ax.set_title(ylab); continue
        err = df[errcol] if (errcol and errcol in df) else None
        ax.bar(lab, df[col], yerr=err, capsize=3, color=colour, alpha=.9)
        ax.set_ylabel(ylab); ax.set_title(ylab)
        ax.tick_params(axis="x", labelsize=7)
        for t in ax.get_xticklabels():
            t.set_rotation(45); t.set_ha("right")
    fig.suptitle(f"Computational cost by configuration, depth {depth}", fontweight="bold")
    fig.tight_layout()
    figs["5.11"] = _save(fig, outdir, f"fig5_11_efficiency_depth{depth}", show)

    # 5.12 performance against cost -- the trade-off an institution actually faces
    if {"auprc_mean", "train_seconds_mean"} <= set(df.columns):
        fig, ax = plt.subplots(figsize=(7.0, 4.6))
        for i, (_, r) in enumerate(df.iterrows()):
            ax.scatter(r["train_seconds_mean"], r["auprc_mean"], s=90,
                       color=PAL[i % len(PAL)], zorder=3)
            ax.annotate(r["experiment"], (r["train_seconds_mean"], r["auprc_mean"]),
                        textcoords="offset points", xytext=(6, 4), fontsize=9)
        prev = np.nanmedian(df.get("prevalence_mean", pd.Series([np.nan])))
        if np.isfinite(prev):
            ax.axhline(prev, color=PAL[3], ls="--", lw=1.2, label=f"chance ({prev:.5f})")
            ax.legend()
        ax.set_xlabel("Training time (s)"); ax.set_ylabel("AUPRC")
        ax.set_title(f"Detection performance against training cost, depth {depth}")
        figs["5.12"] = _save(fig, outdir, f"fig5_12_performance_vs_cost_depth{depth}", show)

    # 5.13 cost across depth -- how each intervention scales
    dfa = _wide(results_dir)
    if dfa.depth.nunique() > 1 and "train_seconds_mean" in dfa:
        fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.2))
        for i, (exp, g) in enumerate(dfa.groupby("experiment")):
            g = g.sort_values("depth")
            axes[0].plot(g.depth, g.train_seconds_mean, marker="o",
                         color=PAL[i % len(PAL)], label=exp)
            if "peak_memory_mb_mean" in g:
                axes[1].plot(g.depth, g.peak_memory_mb_mean, marker="s",
                             color=PAL[i % len(PAL)], label=exp)
        axes[0].set_ylabel("Training time (s)"); axes[1].set_ylabel("Peak GPU memory (MB)")
        for a in axes:
            a.set_xlabel("Depth (layers)"); a.set_xscale("log", base=2)
            a.set_xticks(sorted(dfa.depth.dropna().unique()))
            a.get_xaxis().set_major_formatter(mticker.ScalarFormatter())
        axes[0].legend(fontsize=8, ncol=2)
        fig.suptitle("Computational cost as depth increases", fontweight="bold")
        fig.tight_layout()
        figs["5.13"] = _save(fig, outdir, "fig5_13_cost_vs_depth", show)
    return figs


def operational_figures(results_dir, outdir, depth=2, show=False):
    """Figures 5.14-5.15: alert-budget and threshold behaviour (RQ2, Section 3.10.3)."""
    df = _wide(results_dir)
    df = df[df.depth == depth].sort_values("experiment")
    if df.empty:
        print(f"[operational_figures] no results at depth {depth}"); return {}
    lab = [_label(r) for _, r in df.iterrows()]
    figs = {}

    # 5.14 top-k alert performance: what a fixed investigation budget recovers
    ks = [c for c in ("precision@100_mean", "recall@100_mean",
                      "precision@500_mean", "recall@500_mean") if c in df]
    if ks:
        fig, ax = plt.subplots(figsize=(9.0, 4.2))
        x = np.arange(len(df)); w = 0.8 / len(ks)
        for i, col in enumerate(ks):
            ax.bar(x + i * w - 0.4 + w / 2, df[col], w,
                   color=PAL[i % len(PAL)], label=col.replace("_mean", ""))
        ax.set_xticks(x); ax.set_xticklabels(lab, fontsize=7, rotation=45, ha="right")
        ax.set_ylabel("Score"); ax.legend(fontsize=8, ncol=2)
        ax.set_title(f"Top-k alert performance, depth {depth}")
        figs["5.14"] = _save(fig, outdir, f"fig5_14_topk_depth{depth}", show)

    # 5.15 confusion-matrix composition and false-positive rate
    if {"tp_mean", "fp_mean", "fn_mean"} <= set(df.columns):
        fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.2))
        x = np.arange(len(df))
        axes[0].bar(x, df.tp_mean, color=PAL[2], label="True positives")
        axes[0].bar(x, df.fn_mean, bottom=df.tp_mean, color=PAL[3], label="False negatives")
        axes[0].set_xticks(x); axes[0].set_xticklabels(lab, fontsize=7, rotation=45, ha="right")
        axes[0].set_ylabel("Illicit transactions"); axes[0].legend(fontsize=8)
        axes[0].set_title("Detected vs missed")
        if "false_positive_rate_mean" in df:
            axes[1].bar(x, df.false_positive_rate_mean * 100, color=PAL[1])
            axes[1].set_xticks(x); axes[1].set_xticklabels(lab, fontsize=7, rotation=45, ha="right")
            axes[1].set_ylabel("False-positive rate (%)")
            axes[1].set_title("False alarms as a share of licit transactions")
        fig.suptitle(f"Operational outcome at the selected threshold, depth {depth}",
                     fontweight="bold")
        fig.tight_layout()
        figs["5.15"] = _save(fig, outdir, f"fig5_15_confusion_depth{depth}", show)
    return figs


def master_comparison(results_dir, outdir, depth=2, show=False):
    """Figure 5.16: every headline criterion on one page, for the summary section.

    Values are normalised per metric to a 0-1 scale so that quantities on very
    different scales (AUPRC ~0.006 against training time ~200 s) can be read side
    by side. The underlying numbers are in the exported tables; this figure is for
    seeing the overall shape, not for reading values off.
    """
    df = _wide(results_dir)
    df = df[df.depth == depth].sort_values("experiment")
    if df.empty:
        print(f"[master_comparison] no results at depth {depth}"); return {}

    metrics = [("auprc_mean", "AUPRC", True), ("recall_mean", "Recall", True),
               ("precision_mean", "Precision", True), ("f1_mean", "F1", True),
               ("recall@100_mean", "Recall@100", True),
               ("false_positive_rate_mean", "FPR (lower better)", False),
               ("train_seconds_mean", "Train time (lower better)", False)]
    metrics = [m for m in metrics if m[0] in df.columns]
    mat, labels = [], []
    for col, name, higher_better in metrics:
        v = df[col].to_numpy(dtype=float)
        rng = np.nanmax(v) - np.nanmin(v)
        norm = (v - np.nanmin(v)) / rng if rng > 0 else np.zeros_like(v)
        mat.append(norm if higher_better else 1 - norm)
        labels.append(name)

    fig, ax = plt.subplots(figsize=(1.15 * len(df) + 4, 0.62 * len(labels) + 2.4))
    im = ax.imshow(np.array(mat), cmap="RdYlGn", aspect="auto", vmin=0, vmax=1)
    ax.set_xticks(range(len(df)))
    ax.set_xticklabels([_label({"experiment": r.experiment, "model": r.model,
                                "intervention": r.intervention})
                        for r in df.itertuples()], fontsize=8)
    ax.set_yticks(range(len(labels))); ax.set_yticklabels(labels, fontsize=9)
    for i in range(len(labels)):
        for j in range(len(df)):
            raw = df.iloc[j][metrics[i][0]]
            ax.text(j, i, f"{raw:.3g}", ha="center", va="center", fontsize=6.5)
    ax.set_title(f"All criteria compared, depth {depth} (green = better; cell shows the raw value)",
                 fontweight="bold", fontsize=11)
    fig.colorbar(im, ax=ax, shrink=0.7, label="normalised (green = better)")
    fig.tight_layout()
    return {"5.16": _save(fig, outdir, f"fig5_16_master_comparison_depth{depth}", show)}


# STATISTICAL FIGURES (Section 5.8)
def seed_comparison_figures(results, outdir, conditions=None, metric="auprc",
                            depth=None, show=False):
    """Figures 5.17-5.18: what the seeds actually did.

    A mean and a standard deviation compress five numbers into two and hide
    whether a condition is consistently better or merely better on average. With
    five paired runs the individual values can simply be plotted, and doing so is
    the most honest summary available.
    """
    from stats_analysis import seed_variability
    conditions = conditions or sorted(k for k in results if not k.startswith("_"))
    table = seed_variability(results, conditions, metric)
    if not table:
        print("[seed_comparison_figures] no per-seed values found"); return {}
    figs = {}

    # 5.17 every seed, every condition
    fig, ax = plt.subplots(figsize=(1.05 * len(table) + 3, 4.6))
    for i, (eid, d) in enumerate(table.items()):
        vals = d["values"]
        ax.scatter([i] * len(vals), vals, s=52, color=PAL[i % len(PAL)],
                   zorder=3, alpha=.85, edgecolors="white", linewidths=.8)
        ax.hlines(d["mean"], i - 0.28, i + 0.28, color=GREY_TXT, lw=2, zorder=4)
        for j, v in enumerate(vals):
            ax.annotate(str(j), (i, v), fontsize=6, color=GREY_TXT,
                        textcoords="offset points", xytext=(7, -2))
    ax.set_xticks(range(len(table)))
    ax.set_xticklabels(list(table.keys()), fontsize=9)
    ax.set_ylabel(metric.upper())
    ax.set_title(f"Per-seed {metric.upper()}"
                 + (f" at depth {depth}" if depth else "")
                 + "  (bar = mean, digits = seed index)")
    figs["5.17"] = _save(fig, outdir, f"fig5_17_seed_scatter{'_d%d' % depth if depth else ''}", show)

    # 5.18 how large is seed noise relative to the signal
    fig, ax = plt.subplots(figsize=(1.05 * len(table) + 3, 4.0))
    names = list(table.keys())
    cvs = [table[e]["cv"] for e in names]
    ax.bar(names, cvs, color=[PAL[i % len(PAL)] for i in range(len(names))], alpha=.9)
    ax.axhline(1.0, color=PAL[3], ls="--", lw=1.3,
               label="SD equals the mean")
    ax.set_ylabel("coefficient of variation (SD / mean)")
    ax.set_title("Seed-to-seed dispersion relative to the effect being measured")
    ax.legend(fontsize=8)
    figs["5.18"] = _save(fig, outdir, f"fig5_18_seed_dispersion{'_d%d' % depth if depth else ''}", show)
    return figs


def paired_difference_figure(comparison, outdir, show=False):
    """Figure 5.19: the five paired differences, plotted rather than summarised.

    A slope plot shows both the level of each condition and, crucially, whether
    every seed moved in the same direction -- which a mean and SD cannot convey.
    """
    if not comparison:
        return {}
    a, b = comparison["a_values"], comparison["b_values"]
    ea, eb = comparison["condition_a"], comparison["condition_b"]

    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.4))

    # left: slope plot, one line per seed
    for i, (va, vb) in enumerate(zip(a, b)):
        colour = PAL[2] if va > vb else PAL[3]
        axes[0].plot([0, 1], [vb, va], marker="o", color=colour, alpha=.85, lw=1.8)
        axes[0].annotate(f"seed {i}", (1, va), fontsize=7, color=GREY_TXT,
                         textcoords="offset points", xytext=(8, -2))
    axes[0].set_xticks([0, 1]); axes[0].set_xticklabels([eb, ea])
    axes[0].set_ylabel("AUPRC")
    axes[0].set_title(f"Matched seeds: {eb} to {ea}\n"
                      f"{comparison['n_favouring_a']} of {comparison['n_pairs']} favour {ea}")

    # right: the differences themselves, against zero
    d = comparison["differences"]
    axes[1].axvline(0, color=GREY_TXT, lw=1.2)
    axes[1].scatter(d, range(len(d)), s=70, color=PAL[0], zorder=3)
    axes[1].hlines(range(len(d)), 0, d, color=PAL[0], alpha=.5, lw=2)
    if comparison.get("ci95_low") is not None:
        axes[1].axvspan(comparison["ci95_low"], comparison["ci95_high"],
                        color=PAL[0], alpha=.12,
                        label="95% CI of the mean difference")
        axes[1].axvline(comparison["mean_difference"], color=PAL[0], ls="--", lw=1.5)
        axes[1].legend(fontsize=8, loc="lower right")
    axes[1].set_yticks(range(len(d)))
    axes[1].set_yticklabels([f"seed {i}" for i in range(len(d))], fontsize=8)
    axes[1].set_xlabel(f"AUPRC difference ({ea} minus {eb})")
    axes[1].set_title("Paired differences")
    fig.tight_layout()
    return {"5.19": _save(fig, outdir, f"fig5_19_paired_{ea}_vs_{eb}", show)}


def bootstrap_figure(boot, outdir, label="", show=False):
    """Figure 5.20: uncertainty from which test transactions were drawn.

    Complements the seed analysis, which holds the test set fixed and varies the
    training run. Neither speaks to other datasets or other splits.
    """
    if not boot:
        return {}
    d = np.array(boot["differences"])
    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    ax.hist(d, bins=45, color=PAL[0], alpha=.75, edgecolor="white")
    ax.axvline(0, color=GREY_TXT, lw=1.4)
    ax.axvline(boot["mean_difference"], color=PAL[3], lw=1.8, label="mean difference")
    ax.axvspan(boot["ci95_low"], boot["ci95_high"], color=PAL[3], alpha=.12,
               label="95% percentile interval")
    ax.set_xlabel("AUPRC difference on a resampled test set")
    ax.set_ylabel("bootstrap resamples")
    ax.set_title(f"Bootstrap over test transactions{': ' + label if label else ''}\n"
                 f"{100 * boot['fraction_favouring_a']:.1f}% of resamples favour the intervention")
    ax.legend(fontsize=8)
    return {"5.20": _save(fig, outdir, f"fig5_20_bootstrap_{label.replace(' ', '_')}", show)}


def threshold_figure(scores, labels, outdir, chosen_threshold=None, label="",
                     show=False):
    """Figure 5.21: F1 across all thresholds, against the one actually used.

    Separates two explanations for a low reported F1 -- a poor ranking, or a good
    ranking read at a threshold chosen for a different objective (here, a target
    precision of 0.10).
    """
    from sklearn.metrics import precision_recall_curve
    s, y = np.asarray(scores), np.asarray(labels)
    if y.sum() == 0:
        return {}
    precision, recall, thr = precision_recall_curve(y, s)
    with np.errstate(divide="ignore", invalid="ignore"):
        f1 = np.nan_to_num(2 * precision * recall / (precision + recall))
    best = int(np.argmax(f1))

    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.2))
    axes[0].plot(thr, f1[:-1], color=PAL[0], lw=1.8)
    axes[0].scatter([thr[min(best, len(thr) - 1)]], [f1[best]], s=80, color=PAL[3],
                    zorder=3, label=f"max F1 = {f1[best]:.4f}")
    if chosen_threshold is not None:
        axes[0].axvline(chosen_threshold, color=GREY_TXT, ls="--", lw=1.4,
                        label="threshold used (target precision 0.10)")
    axes[0].set_xlabel("threshold"); axes[0].set_ylabel("F1")
    axes[0].set_title("F1 across thresholds"); axes[0].legend(fontsize=8)

    axes[1].plot(recall, precision, color=PAL[0], lw=1.8)
    axes[1].scatter([recall[best]], [precision[best]], s=80, color=PAL[3], zorder=3)
    axes[1].axhline(y.mean(), color=PAL[3], ls=":", lw=1.2,
                    label=f"chance ({y.mean():.5f})")
    axes[1].set_xlabel("recall"); axes[1].set_ylabel("precision")
    axes[1].set_title("Precision-recall curve"); axes[1].legend(fontsize=8)
    fig.suptitle(f"Threshold analysis{': ' + label if label else ''}", fontweight="bold")
    fig.tight_layout()
    return {"5.21": _save(fig, outdir, f"fig5_21_threshold_{label.replace(' ', '_')}", show)}


# SUBSET REPRESENTATIVENESS (Section 4.3.1)
# Visual evidence that the subset preserves what the models depend on.
def subset_validation_figures(subset_df, full_df, outdir, show=False):
    """Figures 4.16-4.18: subset against full dataset.

    Distributions are plotted on the scales the models actually see -- log for
    amount and degree, both of which are heavy-tailed -- because a linear plot
    would be dominated by a few extreme values and would hide agreement across
    the bulk of the data.
    """
    import pandas as pd
    figs = {}
    S, F = "30% subset", "full dataset"

    # 4.16 attribute distributions
    fig, axes = plt.subplots(1, 3, figsize=(14.0, 4.2))
    if "Amount Received" in subset_df.columns:
        bins = np.linspace(0, np.log1p(full_df["Amount Received"]).max(), 60)
        axes[0].hist(np.log1p(full_df["Amount Received"]), bins=bins, density=True,
                     color=PAL[1], alpha=.55, label=F)
        axes[0].hist(np.log1p(subset_df["Amount Received"]), bins=bins, density=True,
                     histtype="step", lw=2, color=PAL[0], label=S)
        axes[0].set_xlabel("log(1 + amount received)"); axes[0].set_ylabel("density")
        axes[0].set_title("Transaction amount"); axes[0].legend(fontsize=8)

    deg_s = pd.concat([subset_df["from_id"], subset_df["to_id"]]).value_counts()
    deg_f = pd.concat([full_df["from_id"], full_df["to_id"]]).value_counts()
    ns = deg_s.values / deg_s.mean()
    nf = deg_f.values / deg_f.mean()
    bins = np.linspace(0, np.log1p(max(ns.max(), nf.max())), 60)
    axes[1].hist(np.log1p(nf), bins=bins, density=True,
                 color=PAL[1], alpha=.55, label=F)
    axes[1].hist(np.log1p(ns), bins=bins, density=True,
                 histtype="step", lw=2, color=PAL[0], label=S)
    axes[1].set_xlabel("log(1 + degree / mean degree)"); axes[1].set_ylabel("density")
    axes[1].set_title("Account degree (normalised)"); axes[1].legend(fontsize=8)

    if "Payment Format" in subset_df.columns:
        ps = subset_df["Payment Format"].value_counts(normalize=True)
        pf = full_df["Payment Format"].value_counts(normalize=True)
        cats = pf.index.union(ps.index)
        x = np.arange(len(cats)); w = 0.38
        axes[2].bar(x - w/2, pf.reindex(cats, fill_value=0), w, color=PAL[1],
                    alpha=.85, label=F)
        axes[2].bar(x + w/2, ps.reindex(cats, fill_value=0), w, color=PAL[0],
                    alpha=.85, label=S)
        axes[2].set_xticks(x); axes[2].set_xticklabels(cats, fontsize=7)
        axes[2].set_ylabel("proportion"); axes[2].set_title("Payment format")
        axes[2].legend(fontsize=8)
    fig.suptitle("Attribute and topology distributions: subset against full dataset",
                 fontweight="bold")
    fig.tight_layout()
    figs["4.16"] = _save(fig, outdir, "fig4_16_subset_distributions", show)

    # 4.17 empirical CDFs -- the KS statistic is the largest vertical gap here,
    # so plotting the CDFs shows directly what the statistic summarises
    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.2))
    for ax, (a, b, lab) in zip(axes, [
            (np.log1p(subset_df.get("Amount Received", pd.Series(dtype=float))),
             np.log1p(full_df.get("Amount Received", pd.Series(dtype=float))),
             "log(1 + amount received)"),
            # Degree is normalised by its mean before plotting, for the same
            # reason the statistic is: a subset must have lower absolute degree
            # because it has fewer edges, so an unnormalised plot shows two
            # separated curves for ANY subset and says nothing about whether the
            # structure was preserved. Dividing by the mean removes the scale and
            # leaves the shape, which is the thing in question.
            (np.log1p(deg_s.values / deg_s.mean()),
             np.log1p(deg_f.values / deg_f.mean()),
             "log(1 + degree / mean degree)")]):
        a = np.sort(np.asarray(a, dtype=float)); b = np.sort(np.asarray(b, dtype=float))
        if len(a) < 2 or len(b) < 2:
            ax.set_visible(False); continue
        ax.plot(b, np.linspace(0, 1, len(b)), color=PAL[1], lw=2.2, label=F)
        ax.plot(a, np.linspace(0, 1, len(a)), color=PAL[0], lw=1.6, ls="--", label=S)
        ax.set_xlabel(lab); ax.set_ylabel("cumulative proportion")
        ax.legend(fontsize=8)
    fig.suptitle("Empirical cumulative distributions "
                 "(the KS statistic is the largest vertical gap)", fontweight="bold")
    axes[1].text(0.02, 0.97, "degree normalised by its mean:\nabsolute degree must fall\nwith the sampling fraction",
                 transform=axes[1].transAxes, fontsize=7, va="top", color=GREY_TXT)
    fig.tight_layout()
    figs["4.17"] = _save(fig, outdir, "fig4_17_subset_cdf", show)

    # 4.18 temporal profile
    if "Timestamp" in subset_df.columns:
        sd = (subset_df["Timestamp"] // 86400).astype(int)
        fd = (full_df["Timestamp"] // 86400).astype(int)
        s_rate = subset_df.groupby(sd)["Is Laundering"].mean()
        f_rate = full_df.groupby(fd)["Is Laundering"].mean()
        s_vol = subset_df.groupby(sd).size(); f_vol = full_df.groupby(fd).size()
        common = sorted(set(s_rate.index) & set(f_rate.index))

        fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.2))
        axes[0].plot(common, [100 * f_rate[d] for d in common], marker="o",
                     color=PAL[1], lw=2, label=F)
        axes[0].plot(common, [100 * s_rate[d] for d in common], marker="s",
                     color=PAL[0], lw=1.6, ls="--", label=S)
        axes[0].set_xlabel("day"); axes[0].set_ylabel("illicit rate (%)")
        axes[0].set_title("Illicit rate per day"); axes[0].legend(fontsize=8)

        axes[1].plot(common, [f_vol[d] for d in common], marker="o",
                     color=PAL[1], lw=2, label=F)
        axes[1].plot(common, [s_vol[d] for d in common], marker="s",
                     color=PAL[0], lw=1.6, ls="--", label=S)
        axes[1].set_yscale("log")
        axes[1].set_xlabel("day"); axes[1].set_ylabel("transactions (log scale)")
        axes[1].set_title("Volume per day"); axes[1].legend(fontsize=8)
        fig.suptitle("Temporal profile is preserved by proportional subsampling",
                     fontweight="bold")
        fig.tight_layout()
        figs["4.18"] = _save(fig, outdir, "fig4_18_subset_temporal", show)
    return figs
