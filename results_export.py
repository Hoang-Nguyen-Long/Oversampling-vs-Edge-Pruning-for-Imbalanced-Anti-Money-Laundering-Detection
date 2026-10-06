"""
results_export.py
Converts the raw experiment JSON into tidy tables and comparison figures
covering the evaluation criteria specified in Chapter 3, Section 3.10.

Three levels of output:

  1. export_long()   -- every metric, every seed, every depth. The master record.
  2. export_wide()   -- one row per (experiment, depth), mean +/- SD. For tables.
  3. figures         -- the comparison charts, grouped by criterion.

Metrics covered (Section 3.10):
  Detection : auprc, auprc_lift, recall, precision, f1
  Threshold : recall_at_target_precision, achieved_precision, threshold
  Operational: precision@100/500, recall@100/500, false_positive_rate, tp/fp/fn/tn
  Efficiency: train_seconds, inference_seconds, peak_memory_mb,
              retained_train_edges, n_params, epochs_run
  Smoothing : dirichlet, mad, norm (global and minority-incident), per layer
"""
from __future__ import annotations

import glob
import json
import os
import re

import numpy as np
import pandas as pd


def _depth_from_name(path):
    m = re.search(r"depth(\d+)", os.path.basename(path))
    return int(m.group(1)) if m else None


def discover(results_dir):
    """Find every result file and label it with its depth and edge-update setting."""
    found = []
    for path in sorted(glob.glob(os.path.join(results_dir, "*.json"))):
        name = os.path.basename(path)
        if not (name.startswith("exp3_main") or name.startswith("exp4_depth")):
            continue
        found.append({"path": path,
                      "depth": _depth_from_name(path),
                      "edge_updates": name.endswith("_eu.json"),
                      "source": name})
    return found


# 1. LONG EXPORT -- the master record
def export_long(results_dir, out_csv=None):
    """One row per (source, depth, experiment, seed, metric, value).

    This is the format to retain: any subsequent figure can be built from it
    without modifying the experiment code.
    """
    rows = []
    for f in discover(results_dir):
        data = json.load(open(f["path"]))
        for eid, item in data.items():
            if eid.startswith('_'):
                continue   # metadata key, not an experiment
            spec = item.get("spec", {})
            base = {"source": f["source"], "depth": f["depth"],
                    "edge_updates": f["edge_updates"], "experiment": eid,
                    "model": spec.get("model"), "intervention": spec.get("intervention"),
                    "loss": spec.get("loss")}
            # per-seed values: the honest record, from which mean/SD are derived
            for i, per in enumerate(item.get("per_seed", [])):
                for k, v in per.items():
                    if isinstance(v, (int, float)):
                        rows.append({**base, "seed_index": i, "metric": k, "value": float(v)})
            # layer-wise smoothing diagnostics (one row per layer)
            sm = item.get("smoothing") or {}
            for key, series in sm.items():
                if isinstance(series, list):
                    for layer, v in enumerate(series):
                        if v is not None and np.isfinite(v):
                            rows.append({**base, "seed_index": None,
                                         "metric": f"{key}_layer{layer}", "value": float(v)})
    df = pd.DataFrame(rows)
    if out_csv:
        os.makedirs(os.path.dirname(out_csv), exist_ok=True)
        df.to_csv(out_csv, index=False)
        print(f"exported {len(df):,} rows -> {out_csv}")
    return df


# 2. WIDE EXPORT -- one row per condition, mean and SD
def export_wide(results_dir, out_csv=None):
    """One row per (source, depth, experiment); every metric as mean and SD columns."""
    rows = []
    for f in discover(results_dir):
        data = json.load(open(f["path"]))
        for eid, item in data.items():
            if eid.startswith('_'):
                continue   # metadata key, not an experiment
            spec, agg = item.get("spec", {}), item.get("aggregate", {})
            row = {"source": f["source"], "depth": f["depth"],
                   "edge_updates": f["edge_updates"], "experiment": eid,
                   "model": spec.get("model"), "intervention": spec.get("intervention"),
                   "loss": spec.get("loss"), "n_seeds": agg.get("n_seeds")}
            row.update({k: v for k, v in agg.items()
                        if k.endswith("_mean") or k.endswith("_std")})
            sm = item.get("smoothing") or {}
            for key, series in sm.items():
                if isinstance(series, list) and series:
                    row[f"{key}_final"] = series[-1]
                    row[f"{key}_first"] = series[0]
                    if series[0]:
                        row[f"{key}_retention"] = series[-1] / series[0]
            rows.append(row)
    df = pd.DataFrame(rows)
    if len(df):
        df = df.sort_values(["depth", "experiment"])
    if out_csv:
        os.makedirs(os.path.dirname(out_csv), exist_ok=True)
        df.to_csv(out_csv, index=False)
        print(f"exported {len(df)} rows x {len(df.columns)} columns -> {out_csv}")
    return df


def export_all(results_dir, out_dir):
    """Write both exports plus a small manifest describing what was found."""
    os.makedirs(out_dir, exist_ok=True)
    long_df = export_long(results_dir, os.path.join(out_dir, "results_long.csv"))
    wide_df = export_wide(results_dir, os.path.join(out_dir, "results_wide.csv"))
    manifest = {
        "files_found": [f["source"] for f in discover(results_dir)],
        "depths": sorted({f["depth"] for f in discover(results_dir) if f["depth"]}),
        "experiments": sorted(wide_df["experiment"].unique().tolist()) if len(wide_df) else [],
        "metrics": sorted(long_df["metric"].unique().tolist()) if len(long_df) else [],
        "n_long_rows": int(len(long_df)), "n_wide_rows": int(len(wide_df)),
    }
    with open(os.path.join(out_dir, "results_manifest.json"), "w") as fh:
        json.dump(manifest, fh, indent=2)
    print(f"manifest -> {os.path.join(out_dir, 'results_manifest.json')}")
    print(f"  {len(manifest['metrics'])} distinct metrics across depths {manifest['depths']}")
    return long_df, wide_df


def export_wide_dedup(results_dir, prefer="exp3_main"):
    """Wide export with ONE row per (depth, experiment, edge_updates).

    The main-matrix and depth-sweep files overlap at the reference depth: both
    contain results for the shared conditions. Plotting the raw table therefore
    shows each of those conditions twice. This keeps a single row per condition,
    preferring the file whose name starts with ``prefer`` (the main matrix, which
    covers all ten conditions) and falling back to the depth-sweep file.
    """
    df = export_wide(results_dir)
    if df.empty:
        return df
    df = df.copy()
    df["_priority"] = (~df["source"].str.startswith(prefer)).astype(int)
    df = (df.sort_values(["depth", "experiment", "_priority"])
            .drop_duplicates(subset=["depth", "experiment", "edge_updates"], keep="first")
            .drop(columns="_priority")
            .reset_index(drop=True))
    return df
