"""
features.py
===========

Feature engineering for the AML multigraph (Chapter 4, Section C.3).
Implements the two feature-related improvements the supervisor asked for:
* Point 2 -- replace the placeholder identity / all-ones node features used in
  the reference implementation (Egressy et al., 2024) with aggregated
  node-level features that are meaningful and scalable for inductive learning:
  account activity, monetary behaviour, temporal behaviour, directional behaviour.
* Point 3 -- proper edge preprocessing: log-transform + standardisation of
  skewed amounts, timestamp-derived cyclical features, categorical encoding.

LEAKAGE CONTROL (supervisor point 4): every statistic used to build a feature is
computed on the TRAINING period only and applied unchanged to validation/test.
NodeFeatureBuilder and EdgeFeatureScaler are fitted on training transactions and
then transform later periods, so no training-time node vector can see a future
edge.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

COL_FROM, COL_TO, COL_TIME = "from_id", "to_id", "Timestamp"
COL_AMT_RECV, COL_CUR_RECV, COL_FORMAT = "Amount Received", "Received Currency", "Payment Format"

# NODE FEATURES
@dataclass
class NodeFeatureBuilder:
    """Aggregated per-account node features, fitted on training data only.

    Feature groups (mirroring the supervisor's table):
      * Account activity  : in/out transaction counts, total degree
      * Monetary behaviour: mean/max/total sent & received amount
      * Directional       : in/out amount ratio
      * Temporal          : activity window (last - first)
    """

    n_accounts: int

    def _aggregate(self, df: pd.DataFrame) -> np.ndarray:
        n = self.n_accounts
        out_cnt = np.zeros(n); in_cnt = np.zeros(n)
        out_sum = np.zeros(n); in_sum = np.zeros(n)
        out_max = np.zeros(n); in_max = np.zeros(n)
        first_t = np.full(n, np.nan); last_t = np.full(n, np.nan)

        frm = df[COL_FROM].to_numpy()
        to = df[COL_TO].to_numpy()
        amt = df[COL_AMT_RECV].to_numpy(dtype=float)
        t = df[COL_TIME].to_numpy(dtype=float)

        # np.add.at is an unbuffered scatter-add so repeated indices accumulate.
        np.add.at(out_cnt, frm, 1.0); np.add.at(in_cnt, to, 1.0)
        np.add.at(out_sum, frm, amt); np.add.at(in_sum, to, amt)
        np.maximum.at(out_max, frm, amt); np.maximum.at(in_max, to, amt)

        tmm = (pd.concat([pd.DataFrame({"id": frm, "t": t}),
                          pd.DataFrame({"id": to, "t": t})])
               .groupby("id")["t"].agg(["min", "max"]))
        first_t[tmm.index.to_numpy()] = tmm["min"].to_numpy()
        last_t[tmm.index.to_numpy()] = tmm["max"].to_numpy()
        activity_window = np.nan_to_num(last_t - first_t)

        total_deg = in_cnt + out_cnt
        with np.errstate(divide="ignore", invalid="ignore"):
            io_ratio = np.where(in_sum > 0, out_sum / (in_sum + 1e-6), 0.0)
        out_mean = np.where(out_cnt > 0, out_sum / np.maximum(out_cnt, 1), 0.0)
        in_mean = np.where(in_cnt > 0, in_sum / np.maximum(in_cnt, 1), 0.0)

        X = np.stack([
            out_cnt, in_cnt, total_deg,           # activity
            out_sum, in_sum, out_mean, in_mean,   # monetary
            out_max, in_max,                      # monetary extremes
            io_ratio,                             # directional
            activity_window,                      # temporal
        ], axis=1)
        return X.astype(np.float32)

    def fit_transform_train(self, train_df: pd.DataFrame):
        X = self._aggregate(train_df)
        mean = X.mean(axis=0, keepdims=True)
        std = X.std(axis=0, keepdims=True); std[std == 0] = 1.0
        self._mean, self._std = mean, std
        return (X - mean) / std, (mean, std)

    def transform(self, df: pd.DataFrame) -> np.ndarray:
        X = self._aggregate(df)
        return (X - self._mean) / self._std

    @property
    def n_node_features(self) -> int:
        return 11



# EDGE FEATURES

def build_edge_features(df: pd.DataFrame) -> pd.DataFrame:
    """Derive raw (unscaled) edge features from the transaction table.

    Cyclical time features encode hour-of-day and day-of-week as sine/cosine
    pairs so 23:00 and 00:00 are adjacent. Amounts are log1p-transformed here;
    standardisation happens in EdgeFeatureScaler (fitted on train only).
    Column 0 is kept as the raw timestamp because downstream Multi-GNN utilities
    (ports, time-deltas, seed lookup) rely on edge_attr[:, 0] as an ordering key.
    """
    out = pd.DataFrame(index=df.index)
    out["log_amount"] = np.log1p(df[COL_AMT_RECV].astype(float))
    seconds = df[COL_TIME].astype(float)
    hour = (seconds / 3600.0) % 24.0
    dow = (seconds / (3600.0 * 24.0)) % 7.0
    out["hour_sin"] = np.sin(2 * np.pi * hour / 24.0)
    out["hour_cos"] = np.cos(2 * np.pi * hour / 24.0)
    out["dow_sin"] = np.sin(2 * np.pi * dow / 7.0)
    out["dow_cos"] = np.cos(2 * np.pi * dow / 7.0)
    out["currency"] = df[COL_CUR_RECV].astype(float) if COL_CUR_RECV in df else 0.0
    out["payment_format"] = df[COL_FORMAT].astype(float) if COL_FORMAT in df else 0.0
    out.insert(0, "timestamp", seconds)
    return out


@dataclass
class EdgeFeatureScaler:
    """Standardises continuous edge features using training-only statistics.

    Only continuous columns are standardised. The timestamp column (index 0) is
    left untouched (downstream ordering), as are categorical code columns
    (standardising an arbitrary integer code is meaningless).
    """

    continuous_cols: tuple = ("log_amount", "hour_sin", "hour_cos", "dow_sin", "dow_cos")

    def fit(self, train_edges: pd.DataFrame):
        sub = train_edges[list(self.continuous_cols)]
        self._mean = sub.mean()
        self._std = sub.std().replace(0, 1.0)
        return self

    def transform(self, edges: pd.DataFrame) -> pd.DataFrame:
        edges = edges.copy()
        edges[list(self.continuous_cols)] = (
            edges[list(self.continuous_cols)] - self._mean) / self._std
        return edges



# COLUMN ROLES -- consumed by GraphSMOTE-Edge so that it interpolates only what
# is semantically interpolatable (Chapter 3, Section 3.6.1).
# Column order produced by build_edge_features():
#   0 timestamp | 1 log_amount | 2 hour_sin | 3 hour_cos | 4 dow_sin | 5 dow_cos
#   6 currency  | 7 payment_format
# NOTE: data_pipeline drops column 0 before the model sees it, so the indices
# used at intervention time are these MINUS ONE. get_column_roles() returns the
# post-drop indices, which is what the trainer passes through.

EDGE_COLUMN_ORDER = ["timestamp", "log_amount", "hour_sin", "hour_cos",
                     "dow_sin", "dow_cos", "currency", "payment_format"]


def get_column_roles(include_timestamp: bool = False):
    """Return (continuous_cols, categorical_cols, cyclical_spec) as column indices.

    ``include_timestamp=False`` matches the model-facing matrix, in which the raw
    timestamp column has been removed.
    cyclical_spec entries are (col, period_seconds, kind); they are recomputed
    from the timestamp rather than interpolated, so they stay consistent with the
    inherited time. When the timestamp column is absent the cyclical features
    cannot be recomputed and are simply inherited from the anchor instead, which
    is likewise consistent.
    """
    off = 0 if include_timestamp else 1
    idx = {name: i - off for i, name in enumerate(EDGE_COLUMN_ORDER)}
    categorical = [idx["currency"], idx["payment_format"]]
    continuous = [idx["log_amount"], idx["hour_sin"], idx["hour_cos"],
                  idx["dow_sin"], idx["dow_cos"]]
    if include_timestamp:
        cyclical = [(idx["hour_sin"], 86400.0, "sin"), (idx["hour_cos"], 86400.0, "cos"),
                    (idx["dow_sin"], 604800.0, "sin"), (idx["dow_cos"], 604800.0, "cos")]
        continuous = [idx["log_amount"]]     # cyclical handled by recomputation
    else:
        cyclical = []
        # Without the timestamp they cannot be recomputed; inherit from the anchor
        # so that the four cyclical values remain mutually consistent.
        continuous = [idx["log_amount"]]
        categorical = categorical + [idx["hour_sin"], idx["hour_cos"],
                                     idx["dow_sin"], idx["dow_cos"]]
    return continuous, categorical, cyclical
