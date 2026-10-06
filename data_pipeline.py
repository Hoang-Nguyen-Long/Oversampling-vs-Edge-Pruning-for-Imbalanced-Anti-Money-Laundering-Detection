"""
data_pipeline.py

Builds leakage-safe PyG graph objects from the formatted IBM AML CSV
(Chapter 4, Sections C.1-C.5). This is the "Phase 1: Data Ingestion and
Multigraph Mapping" stage, but only AFTER data understanding has run.

It combines:
-aggregated node features and edge preprocessing (features.py), replacing
the reference repo's placeholder all-ones node features;

-the reference repo's proven temporal-split logic and Multi-GAT structural
enhancements (ports, ego IDs, time-deltas, reverse MP), reused so results
stay comparable with published Multi-GNN numbers.

The temporal split follows the reference implementation: transactions are
bucketed by day, and the day boundaries closest to a 60/20/20 split are chosen.
The three leakage controls are enforced explicitly and documented inline.
"""

from __future__ import annotations

import itertools
import logging

import numpy as np
import pandas as pd
import torch
from torch_geometric.data import Data

from features import NodeFeatureBuilder, build_edge_features, EdgeFeatureScaler

logger = logging.getLogger(__name__)

COL_FROM, COL_TO, COL_TIME, COL_LABEL = "from_id", "to_id", "Timestamp", "Is Laundering"


def _compute_daily_split(timestamps: np.ndarray, y: np.ndarray,
                         split_per=(0.6, 0.2, 0.2)):
    """Pick day-boundaries (i, j) whose cumulative sizes best match split_per.

    Reproduces the reference repo's split search so our splits coincide with the
    published Multi-GNN setup. Returns (train_days, val_days, test_days) as day
    index ranges.
    """
    n_days = int(timestamps.max() / (3600 * 24) + 1)
    daily_inds = []
    daily_totals = []
    for day in range(n_days):
        lo, hi = day * 24 * 3600, (day + 1) * 24 * 3600
        di = np.where((timestamps >= lo) & (timestamps < hi))[0]
        daily_inds.append(di)
        daily_totals.append(di.shape[0])
    d_ts = np.array(daily_totals)

    scores = {}
    idxs = list(range(len(d_ts)))
    for i, j in itertools.combinations(idxs, 2):
        totals = [d_ts[:i].sum(), d_ts[i:j].sum(), d_ts[j:].sum()]
        s = sum(totals)
        props = [v / s for v in totals]
        scores[(i, j)] = max(abs(p - t) / t for p, t in zip(props, split_per))
    i, j = min(scores, key=scores.get)
    split_days = [list(range(i)), list(range(i, j)), list(range(j, len(d_ts)))]
    return split_days, daily_inds


def build_graphs(df: pd.DataFrame, edge_feature_cols=None, add_reverse=False):
    """Construct leakage-safe train/val/test PyG ``Data`` objects.

    Returns a dict with the three graphs plus the seed-edge index tensors that
    tell the loaders which edges belong to each split's evaluation window.
    """
    df = df.sort_values(COL_TIME).reset_index(drop=True)
    timestamps = df[COL_TIME].to_numpy(dtype=float)
    y = df[COL_LABEL].to_numpy(dtype=int)
    n_accounts = int(pd.unique(df[[COL_FROM, COL_TO]].values.ravel()).max() + 1)

    # ---- Temporal split (day-bucketed 60/20/20) ----------------------------
    # The day-granularity split needs at least 3 distinct non-empty days. Very
    # small subsets (e.g. a fractional 2-day sample) may not have that, which
    # would make one period empty. In that case the split falls back to a strictly
    # chronological split by transaction COUNT (60/20/20). This is still fully
    # leakage-safe -- every training transaction precedes every validation one,
    # which precedes every test one -- it simply places the boundaries mid-day
    # instead of on day lines. The fallback is only used when a clean day split
    # is impossible, and the choice is logged so it can be reported.
    split_days, daily_inds = _compute_daily_split(timestamps, y)
    non_empty = [len(daily_inds[d]) > 0 for grp in split_days for d in grp]
    day_split_ok = all(
        sum(len(daily_inds[d]) for d in grp) > 0 for grp in split_days
    ) and len(split_days[0]) and len(split_days[1]) and len(split_days[2])

    if day_split_ok:
        tr_inds = np.concatenate([daily_inds[d] for d in split_days[0]])
        val_inds = np.concatenate([daily_inds[d] for d in split_days[1]])
        te_inds = np.concatenate([daily_inds[d] for d in split_days[2]])
    else:
        n = len(timestamps)
        order = np.argsort(timestamps, kind="stable")  # chronological order
        b1, b2 = int(0.6 * n), int(0.8 * n)
        tr_inds, val_inds, te_inds = order[:b1], order[b1:b2], order[b2:]
        logger.warning(
            "Too few distinct days for a day-granularity split; falling back to a "
            "chronological 60/20/20 split by transaction count (still leakage-safe).")

    train_end = timestamps[tr_inds].max()
    val_end = timestamps[val_inds].max()
    for name, inds in [("train", tr_inds), ("val", val_inds), ("test", te_inds)]:
        ir = y[inds].mean() * 100
        logger.info("%s: %d edges (%.2f%% of total), IR=%.4f%%",
                    name, len(inds), 100 * len(inds) / len(y), ir)

    # ---- Edge features (fit scaler on TRAIN only -- leakage control) -------
    raw_edges = build_edge_features(df)
    scaler = EdgeFeatureScaler().fit(raw_edges.iloc[tr_inds])
    edges_scaled = scaler.transform(raw_edges)
    if edge_feature_cols is None:
        # Exclude the raw "timestamp" column (index 0) from MODEL inputs: it is
        # kept in the Data object as an ordering key (for time-deltas / ports and
        # seed-edge lookup, mirroring the reference repo, which strips column 0
        # before the model), but feeding raw seconds (~1e6) to the network would
        # explode activations. Bounded cyclical hour/day features already encode
        # time. This mirrors the reference repo's `edge_attr[:, 1:]` stripping.
        edge_feature_cols = [c for c in edges_scaled.columns if c != "timestamp"]
    edge_attr_all = torch.tensor(edges_scaled[edge_feature_cols].to_numpy(), dtype=torch.float)

    # ---- Node features (fit on TRAIN transactions only -- leakage control) -
    nfb = NodeFeatureBuilder(n_accounts=n_accounts)
    x_train, _ = nfb.fit_transform_train(df.iloc[tr_inds])
    # For val/test graphs, node features may summarise up to the end of that
    # period (a bank knows history) but never use training-fitted stats from the
    # future. Message passing leakage is handled by the edge visibility rule below.
    x_val = nfb.transform(df.iloc[np.concatenate([tr_inds, val_inds])])
    x_test = nfb.transform(df)
    x_train_t = torch.tensor(x_train, dtype=torch.float)
    x_val_t = torch.tensor(x_val, dtype=torch.float)
    x_test_t = torch.tensor(x_test, dtype=torch.float)

    edge_index_all = torch.tensor(df[[COL_FROM, COL_TO]].to_numpy().T, dtype=torch.long)
    y_all = torch.tensor(y, dtype=torch.long)
    ts_all = torch.tensor(timestamps, dtype=torch.float)

    # ---- Edge VISIBILITY rule (leakage control #1) -------------------------
    e_tr = tr_inds
    e_val = np.concatenate([tr_inds, val_inds])
    e_te = np.arange(len(y))

    def make(edge_cols, node_x):
        return dict(edge_index=edge_index_all[:, edge_cols],
                    edge_attr=edge_attr_all[edge_cols],
                    y=y_all[edge_cols], ts=ts_all[edge_cols], x=node_x)

    tr = make(e_tr, x_train_t)
    va = make(e_val, x_val_t)
    te = make(e_te, x_test_t)

    tr_data = Data(x=tr["x"], edge_index=tr["edge_index"], edge_attr=tr["edge_attr"], y=tr["y"])
    val_data = Data(x=va["x"], edge_index=va["edge_index"], edge_attr=va["edge_attr"], y=va["y"])
    te_data = Data(x=te["x"], edge_index=te["edge_index"], edge_attr=te["edge_attr"], y=te["y"])
    tr_data.timestamps, val_data.timestamps, te_data.timestamps = tr["ts"], va["ts"], te["ts"]

    # Seed-edge indices: which rows of each graph are scored for that split.
    # For train it is all rows; for val it is the val block; for test the test block.
    seed = {
        "train": torch.arange(len(e_tr)),
        "val": torch.arange(len(tr_inds), len(e_val)),
        "test": torch.arange(len(tr_inds) + len(val_inds), len(e_te)),
    }
    return {"train": tr_data, "val": val_data, "test": te_data, "seed": seed,
            "n_node_features": nfb.n_node_features, "n_edge_features": edge_attr_all.shape[1]}
