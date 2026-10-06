"""
trainer.py
Full-graph training and evaluation loop (Chapter 4, Section D/E).

This uses full-batch training on the (small) IBM AML graph for clarity and
exact reproducibility. The reference Multi-GNN repo uses LinkNeighborLoader
mini-batching for the medium/large datasets; a note in the documentation
explains how to switch to that loader for scale. For the Small_HI / Small_LI
variants targeted by this dissertation, full-batch fits comfortably in memory
and removes mini-batch sampling as a confound between arms.

The loop applies the topological intervention INSIDE the training epoch and on
the TRAINING graph only:
  * DropEdge     -> re-sample a sparsified training graph each epoch;
  * Edge-SMOTE   -> augment the training graph once before training.
Validation and test always use the full, un-intervened graph, so the operating
threshold and reported metrics reflect predictions on the real topology.
"""

from __future__ import annotations

import logging
import time

import numpy as np
import torch

from interventions import drop_edge, edge_smote, minority_duplicate, build_loss
from features import get_column_roles
from evaluation import evaluate_scores, select_threshold_for_precision
from multigat import add_reverse_and_ports
import graph_cache

logger = logging.getLogger(__name__)


def _run_model(model, g, edge_index=None, edge_attr=None, sup=None):
    """Forward pass on a PreparedGraph (already on device, already augmented).

    ``edge_index``/``edge_attr`` override the graph's own tensors, allowing
    DropEdge to supply a sparsified message-passing graph without rebuilding it. ``sup`` is an optional (index, attr) pair for the supervision edges
    when they differ from the message-passing edges.
    """
    ei = g.edge_index if edge_index is None else edge_index
    ea = g.edge_attr if edge_attr is None else edge_attr
    if sup is not None:
        return model(g.x, ei, ea, sup_edge_index=sup[0], sup_edge_attr=sup[1])
    return model(g.x, ei, ea, n_classify=g.n_real)


def _scores(model, g):
    """Positive-class probabilities for the real edges of a PreparedGraph."""
    model.eval()
    with torch.no_grad():
        out = _run_model(model, g)
        return torch.softmax(out, dim=1)[:, 1].cpu().numpy()


def train_and_evaluate(model, graphs, config, device, seed=0, intervention="none",
                       loss_name="ce", target_precision=None, multi_gat=False):
    """Train one arm and return its test metrics + timing + smoothing profile.

    Parameters
    ----------
    intervention : {"none", "dropedge", "edge_smote"}
    loss_name    : {"ce", "weighted_ce", "focal"}
    """
    # Chapter 3, Section 3.10.2. The target precision is PRE-SPECIFIED, not chosen
    # after seeing test results. 0.10 is used because at a prevalence near 0.1% a
    # precision of 10% already means roughly one in ten alerts is genuine -- about
    # a hundredfold improvement on random inspection, and a plausible workload for
    # a review team. Recall is additionally reported at several fixed precisions
    # so the operating point is not a single arbitrary choice.
    if target_precision is None:
        target_precision = config.get("target_precision", 0.10)

    torch.manual_seed(seed)
    np.random.seed(seed)
    gen = torch.Generator().manual_seed(seed)

    tr_raw, va_raw, te_raw = graphs["train"], graphs["val"], graphs["test"]
    seed_idx = graphs["seed"]
    model = model.to(device)

    # Prepare each graph ONCE: move to the device and apply the Multi-GAT
    # augmentation. Previously this happened inside every forward pass, which
    # rebuilt the largest tensor in the program hundreds of times per run and
    # fragmented the allocator badly enough to cause spurious out-of-memory
    # failures.
    tr = graph_cache.prepare(tr_raw.x, tr_raw.edge_index, tr_raw.edge_attr,
                             tr_raw.y, device, multi_gat)
    va = graph_cache.prepare(va_raw.x, va_raw.edge_index, va_raw.edge_attr,
                             va_raw.y, device, multi_gat)
    te = graph_cache.prepare(te_raw.x, te_raw.edge_index, te_raw.edge_attr,
                             te_raw.y, device, multi_gat)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.get("lr", 0.006),
                                 weight_decay=config.get("weight_decay", 0.0))

    # Class counts (training) drive the imbalance-aware losses.
    class_counts = torch.bincount(tr_raw.y, minlength=2)
    loss_fn = build_loss(loss_name, class_counts, device=device,
                         focal_alpha=config.get("focal_alpha", 0.25),
                         focal_gamma=config.get("focal_gamma", 2.0),
                         cb_beta=config.get("cb_beta", 0.9999))

    # Expansion interventions: applied ONCE, before training 
    # These enlarge BOTH the message-passing graph and the supervised set, which
    # is intended: a synthetic transaction is a training example and also a real
    # edge of the augmented graph.
    # They are applied to the RAW graph (before any Multi-GAT augmentation), and
    # the enlarged graph is then prepared once, so the synthetic edges receive
    # their own reverse edges and port numbers exactly as real ones do.
    if intervention == "edge_smote":
        cont, cat, cyc = get_column_roles(include_timestamp=False)
        aug_ei, aug_ea, aug_y = edge_smote(
            tr_raw.edge_index, tr_raw.edge_attr, tr_raw.y,
            oversample_ratio=config.get("smote_ratio", 1.0),
            k_neighbours=config.get("smote_k", 5), generator=gen,
            continuous_cols=cont, categorical_cols=cat, cyclical_spec=cyc)
        graph_cache.free(tr)
        tr = graph_cache.prepare(tr_raw.x, aug_ei, aug_ea, aug_y, device, multi_gat)
    elif intervention == "duplicate":
        aug_ei, aug_ea, aug_y = minority_duplicate(
            tr_raw.edge_index, tr_raw.edge_attr, tr_raw.y,
            oversample_ratio=config.get("smote_ratio", 1.0), generator=gen)
        graph_cache.free(tr)
        tr = graph_cache.prepare(tr_raw.x, aug_ei, aug_ea, aug_y, device, multi_gat)
    tr_y = tr.y

    best_val_auprc, best_state = -1.0, None
    epochs_since_improve = 0
    patience = config.get("patience", 20)   # Ch3 3.9.3: halt after 20 without improvement
    retained_edges = None                   # Ch3 3.10.1: graph size actually trained on
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    t0 = time.perf_counter()
    for epoch in range(config.get("epochs", 50)):
        model.train()
        optimizer.zero_grad()

        # DropEdge contracts the MESSAGE-PASSING graph only, resampled each
        # epoch. The supervised set (tr_edge_index/tr_y) is untouched, so every
        # training transaction still contributes to the loss and contraction is
        # never confounded with random undersampling of positives.
        if intervention == "dropedge":
            # Mask the prepared tensors; nothing is rebuilt or re-transferred.
            ei, ea = tr.dropped(config.get("drop_rate", 0.2), generator=gen)
            sup = (tr.edge_index[:, :tr.n_real], tr.edge_attr[:tr.n_real])
        else:
            ei, ea, sup = None, None, None    # message-passing and supervision coincide

        # Size of the message-passing graph actually trained on (DropEdge shrinks
        # it, the expansion methods enlarge it) -- an efficiency measure.
        if retained_edges is None:
            retained_edges = int((ei if ei is not None else tr.edge_index).shape[1])
        out = _run_model(model, tr, edge_index=ei, edge_attr=ea, sup=sup)
        loss = loss_fn(out, tr_y)
        loss.backward()
        # Gradient clipping stabilises full-batch GNN training, which can
        # otherwise oscillate; a max-norm of 1.0 is a standard, safe default.
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=config.get("grad_clip", 1.0))
        optimizer.step()

        # ---- Validation-based model selection (on AUPRC) -------------------
        val_scores = _scores(model, va)
        from sklearn.metrics import average_precision_score
        val_auprc = average_precision_score(va_raw.y.numpy()[seed_idx["val"]],
                                            val_scores[seed_idx["val"]])
        if val_auprc > best_val_auprc:
            best_val_auprc = val_auprc
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            epochs_since_improve = 0
        else:
            epochs_since_improve += 1

        if epoch % max(1, config.get("epochs", 50) // 5) == 0:
            # .item() detaches before converting to a Python float, avoiding the
            # "converting a tensor with requires_grad=True" warning that float()
            # on the raw tensor triggers.
            logger.info("epoch %3d | loss %.4f | val AUPRC %.4f", epoch, loss.item(), val_auprc)

        # Early stopping (Ch3 3.9.3): stop once validation AUPRC has not improved
        # for `patience` epochs. Prevents over-fitting and saves compute.
        if epochs_since_improve >= patience:
            logger.info("early stop at epoch %d (no val improvement in %d epochs)",
                        epoch, patience)
            break

    train_seconds = time.perf_counter() - t0

    # Restore best-on-validation weights, then evaluate on test.
    if best_state is not None:
        model.load_state_dict(best_state)

    # Threshold selected on validation, applied to test (leakage-safe).
    val_scores = _scores(model, va)
    te_scores = _scores(model, te)
    y_val = va_raw.y.numpy()[seed_idx["val"]]
    y_te = te_raw.y.numpy()[seed_idx["test"]]
    thr = select_threshold_for_precision(y_val, val_scores[seed_idx["val"]],
                                         target_precision=target_precision)
    metrics = evaluate_scores(y_te, te_scores[seed_idx["test"]], thr)
    metrics["train_seconds"] = train_seconds
    metrics["best_val_auprc"] = float(best_val_auprc)
    metrics["n_params"] = int(sum(p.numel() for p in model.parameters()))
    # Efficiency measures promised in Ch3 3.10.1 / 3.9.6.
    # ---- Efficiency breakdown (Section 3.10.4) -----------------------------
    # Reported separately rather than as one number, because the interventions
    # load different stages: expansion pays at training time, contraction saves
    # there, and neither changes inference cost much.
    _t = time.perf_counter()
    _ = _scores(model, te)                             # timed inference pass
    infer_s = time.perf_counter() - _t
    n_test = int(te.n_real)
    metrics["inference_seconds"] = float(infer_s)
    metrics["inference_ms_per_1k_edges"] = float(1000.0 * infer_s / max(n_test, 1) * 1000.0)
    metrics["retained_train_edges"] = int(retained_edges or 0)
    metrics["epochs_run"] = int(epoch + 1)
    metrics["peak_memory_mb"] = (
        float(torch.cuda.max_memory_allocated(device) / 1024**2)
        if device.type == "cuda" else 0.0)

    # Retain the raw scores. The maximum attainable F1, the precision-recall
    # curve, a bootstrap over test transactions and performance restricted to
    # accounts unseen during training are all computable from predictions alone,
    # so persisting them allows those analyses without retraining.
    metrics["_scores"] = {
        "test": te_scores[seed_idx["test"]].astype(float).tolist(),
        "test_labels": y_te.astype(int).tolist(),
        "val": val_scores[seed_idx["val"]].astype(float).tolist(),
        "val_labels": y_val.astype(int).tolist(),
    }
    return metrics, model
