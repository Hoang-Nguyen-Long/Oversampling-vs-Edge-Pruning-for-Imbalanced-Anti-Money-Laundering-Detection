"""
EXPERIMENT 2 -- Hyperparameter Tuning, one depth per process

USAGE
-----
    python exp2_tuning.py --depth 2       # writes exp2_tuning_depth2.json
    python exp2_tuning.py --depth 8       # writes exp2_tuning_depth8.json
    python exp2_tuning.py --combine       # writes exp2_tuning.json (used by all later steps)

Restart the Python process (or simply let each command finish) between the two
depth runs. On a shared or memory-tight machine, run them in separate terminals.
"""
import argparse
import json
import os

import common as C
from data_pipeline import build_graphs
from data_understanding import load_transactions
from run_experiments import ABLATION, default_config
from tuning import tune_at_depth, reconcile

REPRESENTATIVE = {"gat": "E2", "multi_gat": "E6"}


def run_one_depth(depth, epochs):
    """Tune both backbones at a single depth and save the result."""
    df = load_transactions(C.subset_path())
    graphs = build_graphs(df)
    nf, ef = graphs["n_node_features"], graphs["n_edge_features"]

    C.logger.info("Tuning with edge_updates=%s (matching the scoring experiments)",
                  C.EDGE_UPDATES)
    out = {"depth": depth, "edge_updates": C.EDGE_UPDATES, "backbones": {}}
    for backbone, rep_eid in REPRESENTATIVE.items():
        spec = ABLATION[rep_eid]
        C.logger.info("=" * 64)
        C.logger.info("Depth %d | backbone '%s' (representative condition %s)",
                      depth, backbone, rep_eid)
        # edge_updates must match the scoring experiments exactly; passing the
        # project-wide setting is what guarantees that.
        base = default_config(depth, edge_updates=C.EDGE_UPDATES)
        best, trials = tune_at_depth(spec, graphs, base, C.DEVICE, nf, ef,
                                     depth=depth, epochs=epochs,
                                     edge_updates=C.EDGE_UPDATES)
        out["backbones"][backbone] = {"best": best, "trials": trials,
                                      "representative": rep_eid}
        C.save(out, f"exp2_tuning_depth{depth}")

    C.logger.info("=" * 64)
    C.logger.info("Depth %d complete. Winners:", depth)
    for backbone, r in out["backbones"].items():
        C.logger.info("   %-10s lr=%.4f (val AUPRC %.4f)",
                      backbone, r["best"]["lr"], r["best"]["val_auprc"])
    C.logger.info("Now run the OTHER depth, then: python exp2_tuning.py --combine")
    return out


def combine(shallow, deep):
    """Reconcile the two saved depth results into the final tuned values."""
    s = C.load(f"exp2_tuning_depth{shallow}")
    d = C.load(f"exp2_tuning_depth{deep}")
    missing = [f"depth {x}" for x, r in ((shallow, s), (deep, d)) if r is None]
    if missing:
        raise SystemExit(
            f"Missing tuning results for {', '.join(missing)}.\n"
            f"Run:  python exp2_tuning.py --depth {shallow}\n"
            f"      python exp2_tuning.py --depth {deep}\n"
            f"then re-run --combine.")

    final = {}
    C.logger.info("=" * 64)
    for backbone, rep_eid in REPRESENTATIVE.items():
        best_s = s["backbones"][backbone]["best"]
        best_d = d["backbones"][backbone]["best"]
        decision = reconcile(best_s, best_d)
        # Share the reconciled value with every condition on this backbone. The
        # MLP baseline follows the GAT settings, having no graph layers.
        for eid, sp in ABLATION.items():
            if sp["model"] == backbone or (backbone == "gat" and sp["model"] == "mlp"):
                final[eid] = {
                    "spec": sp,
                    "best": {"lr": decision["lr"], "dropout": decision["dropout"]},
                    "tuned_via": rep_eid,
                    "decision": decision,
                    "trials": s["backbones"][backbone]["trials"]
                              + d["backbones"][backbone]["trials"],
                    "budget_used": len(s["backbones"][backbone]["trials"])
                                   + len(d["backbones"][backbone]["trials"]),
                }
    C.save(final, "exp2_tuning")

    # Summary for the write-up: this is the justification for the value used.
    C.logger.info("=" * 64)
    C.logger.info("FINAL TUNED VALUES (used at every depth)")
    for backbone, rep_eid in REPRESENTATIVE.items():
        dec = final[rep_eid]["decision"]
        C.logger.info("  %-10s lr=%.4f   depth %d chose %.4f | depth %d chose %.4f   %s",
                      backbone, dec["lr"],
                      dec["shallow"]["depth"], dec["shallow"]["lr"],
                      dec["deep"]["depth"], dec["deep"]["lr"],
                      "[depth-robust]" if dec["depth_robust"] else "[reconciled]")
        C.logger.info("      rule: %s", dec["rule"])
    return final


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--depth", type=int,
                    help="tune at this depth only, and save the result")
    ap.add_argument("--combine", action="store_true",
                    help="reconcile the saved depth results into final values")
    ap.add_argument("--depths", nargs=2, type=int, default=[2, 8],
                    metavar=("SHALLOW", "DEEP"),
                    help="which two depths --combine should reconcile")
    ap.add_argument("--epochs", type=int, default=30,
                    help="reduced cap: ranking configurations, not final scoring")
    a = ap.parse_args()

    C.ensure_dirs()
    if a.combine:
        combine(a.depths[0], a.depths[1])
    elif a.depth is not None:
        run_one_depth(a.depth, a.epochs)
    else:
        ap.error("give either --depth N (to tune) or --combine (to reconcile). "
                 "Run each depth separately so the GPU is empty for the deep run.")


if __name__ == "__main__":
    C.require_gpu()
    main()
