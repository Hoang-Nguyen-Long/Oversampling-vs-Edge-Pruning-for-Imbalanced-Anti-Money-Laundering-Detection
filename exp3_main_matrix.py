"""
EXPERIMENT 3 -- Main Experiment Matrix E0-E8  (Chapter 3, Table 3.1)
The central comparison: all ten conditions at the reference depth, five seeds,
full epoch budget with early stopping. This produces the headline results table
of Chapter 5 (AUPRC, recall, precision, F1, top-k, efficiency, and the
over-smoothing profile at this depth).

Uses the hyperparameters chosen by Experiment 2 if present; otherwise falls back
to the defaults and says so, since reporting untuned results as tuned would be
misleading.

Priority order if compute is short (Section 3.9.1): E1, E2, E4, E5 first, then
E6-E8, then E0. Pass a subset via --only.

    python exp3_main_matrix.py
    python exp3_main_matrix.py --only E1 E2 E4 E5
"""
import argparse
import common as C
from run_experiments import ABLATION, run

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", nargs="+", default=list(ABLATION.keys()))
    ap.add_argument("--depth", type=int, default=C.REFERENCE_DEPTH)
    ap.add_argument("--edge-updates", dest="edge_updates", action="store_true",
                    help="enable the per-layer edge-update MLP for ALL conditions "
                         "(roughly triples activation memory; caps the feasible depth)")
    ap.add_argument("--no-edge-updates", dest="edge_updates", action="store_false",
                    help="disable edge updates for ALL conditions (default; allows deeper sweeps)")
    ap.set_defaults(edge_updates=C.EDGE_UPDATES)
    a = ap.parse_args()

    # The suffix records the setting in the filename so runs with and without
    # edge updates cannot silently overwrite one another.
    eu_tag = "_eu" if a.edge_updates else ""

    tuned = C.load("exp2_tuning")
    if tuned:
        C.logger.info("Using tuned hyperparameters from Experiment 2.")
    else:
        C.logger.warning("No exp2_tuning.json found -- using DEFAULT hyperparameters. "
                         "Report these results as untuned, or run exp2_tuning.py first.")

    run(csv_path=C.subset_path(), experiments=a.only, seeds=C.SEEDS,
        depth=a.depth, epochs=C.MAX_EPOCHS, device=C.DEVICE,
        out_path=f"{C.RESULTS_DIR}/exp3_main_depth{a.depth}{eu_tag}.json",
        edge_updates=a.edge_updates,
        tuned=tuned)

if __name__ == "__main__":
    C.ensure_dirs(); C.require_gpu(); main()
