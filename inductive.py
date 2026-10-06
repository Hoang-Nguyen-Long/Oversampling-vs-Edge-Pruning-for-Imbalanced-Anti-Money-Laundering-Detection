"""
inductive.py — performance on accounts unseen during training
Chapter 3 records that node features are frozen from the training period, so an
account first appearing in the test period carries all-zero aggregates (the
training mean after standardisation). That is a deliberate leakage control, and
it has a consequence the dissertation promised to measure but did not: those
accounts are harder, because the model has no history for them.
This module answers the question from stored predictions alone. It partitions the
test transactions by whether their endpoints were seen during training and
reports detection performance separately for each partition.

Why it matters operationally: new accounts are exactly where laundering networks
place mule activity, so a model that performs well overall but poorly on unseen
accounts is weaker than its headline figure suggests. Reporting the split makes
that visible rather than leaving it averaged away.
"""
from __future__ import annotations

import numpy as np


def account_partition(graphs):
    """Which test transactions touch an account absent from the training period.

    Returns a boolean array over test transactions: True where at least one
    endpoint is inductive (unseen in training).
    """
    tr, te = graphs["train"], graphs["test"]
    seed_idx = graphs["seed"]

    # n_real exists only on graphs prepared for Multi-GAT; the raw training
    # graph contains real transactions only, so its full edge set is correct.
    n_real = getattr(tr, "n_real", tr.edge_index.shape[1])
    train_accounts = set(tr.edge_index[:, :n_real].reshape(-1).tolist())
    te_idx = seed_idx["test"]
    src = te.edge_index[0, te_idx].tolist()
    dst = te.edge_index[1, te_idx].tolist()

    unseen = np.array([(s not in train_accounts) or (d not in train_accounts)
                       for s, d in zip(src, dst)], dtype=bool)
    return unseen


def evaluate_by_partition(scores, labels, unseen_mask, threshold=None):
    """Detection performance on transductive and inductive test transactions.

    Both partitions are evaluated with the SAME threshold, because in deployment
    a single threshold serves all transactions; using a partition-specific
    threshold would flatter the harder partition and describe a system nobody
    would build.
    """
    from sklearn.metrics import average_precision_score
    s, y = np.asarray(scores), np.asarray(labels)
    out = {}
    for name, mask in (("transductive", ~unseen_mask), ("inductive", unseen_mask)):
        if mask.sum() == 0 or y[mask].sum() == 0:
            out[name] = {"n": int(mask.sum()), "n_positive": int(y[mask].sum()),
                         "auprc": None, "prevalence": None, "note":
                         "too few positives in this partition to evaluate"}
            continue
        yy, ss = y[mask], s[mask]
        prev = float(yy.mean())
        entry = {
            "n": int(mask.sum()),
            "n_positive": int(yy.sum()),
            "prevalence": prev,
            "auprc": float(average_precision_score(yy, ss)),
        }
        entry["auprc_lift"] = entry["auprc"] / prev if prev else None
        if threshold is not None:
            pred = ss >= threshold
            tp = int((pred & (yy == 1)).sum()); fp = int((pred & (yy == 0)).sum())
            fn = int((~pred & (yy == 1)).sum())
            entry.update({
                "tp": tp, "fp": fp, "fn": fn,
                "recall": tp / (tp + fn) if (tp + fn) else 0.0,
                "precision": tp / (tp + fp) if (tp + fp) else 0.0,
            })
        out[name] = entry

    # The comparison that matters: lift rather than raw AUPRC, because the two
    # partitions have different prevalences and raw AUPRC is not comparable
    # across them.
    t, i = out.get("transductive", {}), out.get("inductive", {})
    if t.get("auprc_lift") and i.get("auprc_lift"):
        out["lift_ratio_transductive_over_inductive"] = (
            t["auprc_lift"] / i["auprc_lift"])
    return out
