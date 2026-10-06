"""
tuning.py

GRID
----
    learning rate in {0.001, 0.003, 0.006}, dropout fixed at 0.1

"""

from __future__ import annotations

import gc
import logging

import torch

from models import build_model
from trainer import train_and_evaluate

logger = logging.getLogger(__name__)

LR_GRID = [0.001, 0.003, 0.006]
FIXED_DROPOUT = 0.1
BUDGET = len(LR_GRID)          # identical for every condition


def tune_at_depth(spec, graphs, base_config, device, n_node_f, n_edge_f, depth,
                  tune_seed=0, epochs=30, edge_updates=None):
    """Grid-search the learning rate at one depth; return (best, trials).

    ``edge_updates`` MUST match the setting used by the scoring experiments. If
    it does not, tuning selects a learning rate for a different architecture from
    the one actually evaluated, and -- because the per-layer edge MLP roughly
    triples activation memory -- a deep configuration that trains comfortably in
    the experiments can exhaust the GPU during tuning. It defaults to the
    project-wide setting rather than to the backbone, which is what keeps the two
    consistent.
    """
    if edge_updates is None:
        edge_updates = base_config.get("edge_updates", False)
    is_multi = spec["model"] == "multi_gat"
    trials, best = [], None

    for lr in LR_GRID:
        cfg = dict(base_config)
        cfg["n_gnn_layers"] = depth
        cfg["lr"] = lr
        cfg["dropout"] = FIXED_DROPOUT
        cfg["final_dropout"] = FIXED_DROPOUT
        cfg["edge_updates"] = edge_updates   # NOT tied to the backbone
        cfg["epochs"] = epochs

        model = build_model(spec["model"], n_node_f,
                            n_edge_f + (2 if is_multi else 0), cfg)
        metrics, trained = train_and_evaluate(
            model, graphs, cfg, device, seed=tune_seed,
            intervention=spec["intervention"], loss_name=spec["loss"],
            multi_gat=is_multi)

        val = metrics.get("best_val_auprc", 0.0)
        # Release the model and its activations before the next configuration.
        # Without this the tuning loop accumulates one model per configuration on
        # the GPU, and the deepest configuration -- the most memory-hungry -- runs
        # last, on top of everything already resident. Deleting the references and
        # emptying the allocator cache keeps peak memory at ONE model rather than
        # the whole grid.
        del model, trained, metrics
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
        trials.append({"depth": depth, "lr": lr, "dropout": FIXED_DROPOUT,
                       "val_auprc": val})
        logger.info("    depth %-2d lr=%.4f -> val AUPRC %.4f", depth, lr, val)
        if best is None or val > best["val_auprc"]:
            best = {"depth": depth, "lr": lr, "dropout": FIXED_DROPOUT,
                    "val_auprc": val}

    logger.info("  depth %d best: lr=%.4f (val AUPRC %.4f)",
                depth, best["lr"], best["val_auprc"])
    return best, trials


def reconcile(best_shallow, best_deep, lr_grid=LR_GRID):
    """Choose ONE learning rate from the two depth-specific winners.

    The rule is deliberate and stated in advance, so the choice is not made after
    seeing the experimental results:

      * identical winners      -> use that value; report the setting as
                                  empirically depth-robust.
      * adjacent grid values   -> take the LOWER (more conservative) value. The
                                  deep model is the fragile case; a slightly
                                  small learning rate costs the shallow model
                                  little, whereas a slightly large one can
                                  destabilise the deep model, so the asymmetry
                                  of risk favours the lower value.
      * non-adjacent winners   -> take the intermediate GRID value. Averaging is
                                  avoided because the mean of 0.001 and 0.006 is
                                  0.0035, a value that appears nowhere in the
                                  grid and was therefore never evaluated.

    Returns a dict recording the decision and the reason, which goes into the
    write-up as the justification for the value used.
    """
    lo, hi = best_shallow["lr"], best_deep["lr"]
    i_lo, i_hi = lr_grid.index(lo), lr_grid.index(hi)

    if lo == hi:
        chosen, rule = lo, ("Both depths selected the same learning rate; the "
                            "setting is empirically depth-robust.")
    elif abs(i_lo - i_hi) == 1:
        chosen = min(lo, hi)
        rule = ("Winners were adjacent grid values; the lower (more conservative) "
                "value was taken, because the deep model is the fragile case.")
    else:
        mid = (i_lo + i_hi) // 2
        chosen = lr_grid[mid]
        rule = ("Winners were non-adjacent; the intermediate grid value was taken "
                "rather than an arithmetic mean, which would fall outside the grid.")

    logger.info("Reconciled learning rate: %.4f (%s)", chosen, rule)
    return {"lr": chosen, "dropout": FIXED_DROPOUT,
            "shallow": best_shallow, "deep": best_deep,
            "rule": rule, "depth_robust": lo == hi}
